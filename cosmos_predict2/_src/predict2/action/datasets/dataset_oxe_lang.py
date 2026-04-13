"""
OXE dataset with language action conditioning.

Loads videos + ZERO actions + language descriptions for on-the-fly text encoding.

The text encoder is NOT loaded here. Instead, the dataset sets data["ai_caption"]
to the language description string, and the model's forward pass encodes it
on-the-fly via compute_online=True (TextEncoder inside the model, one copy on GPU).

Stage 1 (pre-training): actions are ZERO, ai_caption has language descriptions.
Stage 2 (fine-tuning): uses standard Bridge/DROID datasets (actions real, caption empty).
"""

import json
import os
import random
import traceback
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from cosmos_predict2._src.predict2.action.datasets.dataset_oxe import Dataset_3D_OXE


class Dataset_3D_OXE_Lang(Dataset_3D_OXE):
    """
    OXE dataset with language action conditioning.

    Inherits from Dataset_3D_OXE for video loading and per-dataset handling.
    Overrides __getitem__ to:
      1. Set actions to ZERO (model conditions on language, not numerical actions)
      2. Load language annotation for the video chunk
      3. Set ai_caption to the language string (model encodes on-the-fly)

    Args:
        lang_annotations_dir: Path to directory containing {episode_id}_lang.json files.
        All other args are passed to Dataset_3D_OXE.
    """

    def __init__(self, lang_annotations_dir, **kwargs):
        super().__init__(**kwargs)
        self.lang_annotations_dir = lang_annotations_dir

        # Filter samples to only those with language annotations
        original_count = len(self.samples)
        self.samples = [
            s for s in self.samples
            if os.path.exists(self._get_lang_path(s["ann_file"]))
        ]
        print(
            f"OXE Lang ({self.dataset_name}, {self.mode}): "
            f"{len(self.samples)}/{original_count} episodes have language annotations"
        )

    def _get_lang_path(self, ann_file):
        """Get path to language annotation file for a given annotation file."""
        episode_id = os.path.basename(ann_file).replace(".json", "")
        return os.path.join(self.lang_annotations_dir, f"{episode_id}_lang.json")

    def _load_language_for_chunk(self, ann_file, frame_ids):
        """
        Load the language description for the chunk corresponding to frame_ids.

        The language annotation has multiple chunks per episode. We find the
        chunk whose frame range best overlaps with frame_ids.
        """
        lang_path = self._get_lang_path(ann_file)
        with open(lang_path) as f:
            lang_data = json.load(f)

        start_frame = frame_ids[0]
        # Find the chunk with matching start_frame
        for chunk in lang_data["chunks"]:
            if chunk["start_frame"] == start_frame:
                return chunk["language"]

        # Fallback: use the closest chunk by start_frame
        closest_chunk = min(
            lang_data["chunks"],
            key=lambda c: abs(c["start_frame"] - start_frame),
        )
        return closest_chunk["language"]

    def __getitem__(self, index, cam_id=None, return_video=False):
        if self.mode != "train":
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample["ann_file"]
            frame_ids = sample["frame_ids"]

            with open(ann_file, "r") as f:
                label = json.load(f)

            data = dict()

            # ZERO actions -- model conditions on language, not numerical actions
            data["action"] = torch.zeros(
                self.sequence_length - 1, 7, dtype=torch.float32
            )

            # Load video (same as parent class)
            video, cam_id = self._get_obs(label, frame_ids, cam_id, pre_encode=False)
            video = video.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]
            data["video"] = video.to(dtype=torch.uint8)

            data["annotation_file"] = ann_file

            # __key__ with dataset prefix to avoid ConcatDataset collisions
            if "episode_id" in label:
                data["__key__"] = str(f"{self.dataset_name}/{label['episode_id']}")
            else:
                try:
                    data["__key__"] = str(f"{self.dataset_name}/{label['original_path']}")
                except Exception:
                    try:
                        data["__key__"] = str(
                            f"{self.dataset_name}/{label['episode_metadata']['episode_id']}"
                        )
                    except Exception:
                        data["__key__"] = str(
                            f"{self.dataset_name}/{label['episode_metadata']['segment_id']}"
                        )

            # Language conditioning: set ai_caption to the language description.
            # The model's forward pass will encode this on-the-fly via compute_online=True.
            language = self._load_language_for_chunk(ann_file, frame_ids)
            data["ai_caption"] = language

            # Dummy t5_text_embeddings (overridden by model when compute_online=True)
            data["t5_text_embeddings"] = torch.zeros(
                512, 1024, dtype=torch.bfloat16
            ).cuda()
            data["t5_text_mask"] = torch.ones(512, dtype=torch.int64).cuda()

            data["fps"] = 4
            data["image_size"] = 256 * torch.ones(4).cuda()
            data["num_frames"] = self.sequence_length
            data["padding_mask"] = torch.zeros(1, 256, 256).cuda()

            return data

        except Exception:
            warnings.warn(
                f"Invalid data encountered: {self.samples[index]['ann_file']}. Skipped."
            )
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            self.wrong_number += 1
            print(self.wrong_number)
            return self[np.random.randint(len(self.samples))]
