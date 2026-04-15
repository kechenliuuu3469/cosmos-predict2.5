"""
Pre-download every remote asset needed to run
`ac_reason_embeddings_rectified_flow_2b_256_320_oxe_lam` offline.

Run this on a machine WITH internet access (e.g. della login node with
proxy/default loaded). It populates $HF_HOME so you can tar+ship the
cache to an offline machine.
"""

import os
import sys
import traceback

# Bump download timeouts and disable hf_transfer (less stable on slow mirrors).
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

# IMPORTANT: the checkpoint DB is populated lazily by register_checkpoints().
# Call it before touching download_checkpoint, otherwise every URI falls
# through to the local-path branch and fails.
from cosmos_oss.checkpoints_predict2 import register_checkpoints

register_checkpoints()

from cosmos_predict2._src.imaginaire.utils.checkpoint_db import (
    CheckpointConfig,
    CheckpointDirHf,
    CheckpointFileHf,
    download_checkpoint,
)
from huggingface_hub import snapshot_download, hf_hub_download


# S3 URIs that the training pipeline resolves via the checkpoint DB.
# Each one maps internally to a HuggingFace repo download.
S3_ASSETS = [
    # 1. Base 2B T2V rectified-flow backbone (load_path in the experiment)
    "s3://bucket/cosmos_diffusion_v2/official_runs_text2world/"
    "Stage-c_pt_4-reason_embeddings-v1p1-Index-26-Size-2B-Res-720-Fps-16-"
    "Note-T2V_high_sigma_loss_reweighted/checkpoints/iter_000010000/model",

    # 2. Reason1p1 7B text encoder (compute_online=True pulls this)
    "s3://bucket/cosmos_reasoning1/sft_exp700/"
    "sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/"
    "checkpoints/iter_000016000/model",

    # 3. Wan2.1 VAE tokenizer
    "s3://bucket/cosmos_diffusion_v2/pretrain_weights/tokenizer/wan2pt1/"
    "Wan2.1_VAE.pth",

    # 4. Qwen2.5-VL-7B-Instruct tokenizer/processor files (the reason1
    #    text encoder calls AutoProcessor on this path). Registered under
    #    the same s3 prefix in cosmos_oss/checkpoints.py.
    "s3://bucket/cosmos_reasoning1/pretrained/Qwen_tokenizer/"
    "Qwen/Qwen2.5-VL-7B-Instruct",
]

# HuggingFace repos that are downloaded directly via AutoProcessor /
# AutoConfig inside the reason1 text encoder.
HF_REPOS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
]


def _download_via_snapshot(uri: str) -> str:
    """Fallback: bypass the `uvx hf download` subprocess and call
    snapshot_download/hf_hub_download directly. Same HF repo + revision,
    in-process, so we can get proper exceptions and retries instead of
    a CalledProcessError."""
    cfg = CheckpointConfig.maybe_from_uri(uri)
    if cfg is None:
        raise ValueError(f"URI not in checkpoint DB: {uri}")
    hf = cfg.hf
    if isinstance(hf, CheckpointFileHf):
        path = hf_hub_download(
            repo_id=hf.repository,
            filename=hf.filename,
            revision=hf.revision,
            repo_type="model",
        )
    elif isinstance(hf, CheckpointDirHf):
        include = list(hf.include) or ["*"]
        exclude = list(hf.exclude)
        if hf.subdirectory:
            include = [os.path.join(hf.subdirectory, p) for p in include]
            exclude = [os.path.join(hf.subdirectory, p) for p in exclude]
        path = snapshot_download(
            repo_id=hf.repository,
            revision=hf.revision,
            repo_type="model",
            allow_patterns=include,
            ignore_patterns=exclude or None,
        )
        if hf.subdirectory:
            path = os.path.join(path, hf.subdirectory)
    else:
        raise TypeError(f"Unknown hf config type: {type(hf)}")
    return path


def main() -> int:
    print(f"HF_HOME                    = {os.environ.get('HF_HOME', '(unset)')}")
    print(f"HF_ENDPOINT                = {os.environ.get('HF_ENDPOINT', '(unset)')}")
    print(f"HF_TOKEN                   = {'SET' if os.environ.get('HF_TOKEN') else '(unset)'}")
    print(f"HF_HUB_DOWNLOAD_TIMEOUT    = {os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}")
    print(f"HF_HUB_ENABLE_HF_TRANSFER  = {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
    print()

    failed = []

    for uri in S3_ASSETS:
        print(f"[s3 asset] {uri}")
        try:
            # First attempt: the repo's own download path (uvx hf ... subprocess).
            path = download_checkpoint(uri)
            print(f"  -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  subprocess download failed: {exc}")
            print(f"  retrying in-process via snapshot_download...")
            try:
                path = _download_via_snapshot(uri)
                print(f"  -> {path}")
            except Exception as exc2:  # noqa: BLE001
                print(f"  !! FAILED: {exc2}")
                traceback.print_exc()
                failed.append(uri)
        print()

    for repo_id in HF_REPOS:
        print(f"[hf repo] {repo_id}")
        try:
            path = snapshot_download(repo_id=repo_id, repo_type="model")
            print(f"  -> {path}")
        except Exception as exc:  # noqa: BLE001
            print(f"  !! FAILED: {exc}")
            traceback.print_exc()
            failed.append(repo_id)
        print()

    if failed:
        print("FAILED assets:")
        for f in failed:
            print(f"  - {f}")
        return 1

    print("All assets cached successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
