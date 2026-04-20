"""
Generic ``action_load_fn`` for OXE LAM eval — dispatches on ``stacking_mode``
recorded in the annotation JSON (written by ``prepare_val_data.py``).

Covers all 8 oxe_lam datasets with a single loader:
  - None        (egodex / bridge / fractal / bc_z / taco_play)  single view
  - dreamzero   (droid / fmb)                                   3-view composite
  - horizontal  (furniture_bench)                               2-view composite
"""

import mediapy
import numpy as np

from video_cross_eval.compositor import composite


def load_lam_action_fn():
    def load_fn(json_data, video_path, args):
        views = [mediapy.read_video(p) for p in json_data["view_paths"]]
        T = min(len(v) for v in views)
        views = [v[:T] for v in views]

        video_array = composite(views, json_data.get("stacking_mode"))

        img = video_array[args.start_frame_idx]
        if args.resolution != "none":
            h, w = map(int, args.resolution.split(","))
            img = mediapy.resize_image(img, (h, w))

        actions = np.load(json_data["latent_actions_path"]).astype(np.float32)

        return {
            "actions": actions,
            "initial_frame": img,
            "video_array": video_array,
            "video_path": f"{json_data.get('dataset', 'composite')}/{json_data.get('rel', '?')}",
        }

    return load_fn
