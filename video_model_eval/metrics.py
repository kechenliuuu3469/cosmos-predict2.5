"""
Video quality metrics: PSNR, SSIM, LPIPS (per-frame) and FVD (distributional).

FVD uses torchvision's R3D-18 pretrained on Kinetics-400 as the feature
extractor. This is NOT the original I3D FVD, so absolute numbers are not
directly comparable to published FVD values — but for comparing our own
checkpoints against each other (the purpose here) it's valid.

If you later want paper-comparable FVD, swap `FVDFeatureExtractor` for
pytorchvideo's I3D without changing anything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import lpips
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn
from torchvision.models.video import R3D_18_Weights, r3d_18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class FrameMetrics:
    psnr: List[float]
    ssim: List[float]
    lpips: List[float]


class LPIPSWrapper:
    """Lazy-init LPIPS so importers that only need frame metrics don't pay the cost."""

    def __init__(self, net: str = "vgg"):
        self.net = net
        self._fn = None

    def __call__(self, gt_uint8: np.ndarray, gen_uint8: np.ndarray) -> float:
        if self._fn is None:
            self._fn = lpips.LPIPS(net=self.net).to(DEVICE).eval()
        with torch.no_grad():
            return self._fn(_to_lpips(gt_uint8), _to_lpips(gen_uint8)).item()


def _to_lpips(x: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 -> [1,3,H,W] in [-1, 1]."""
    t = torch.from_numpy(x).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(DEVICE)


def compute_frame_metrics(
    gt: np.ndarray, gen: np.ndarray, lpips_fn: LPIPSWrapper, skip_first: int = 1
) -> FrameMetrics:
    """
    gt, gen: [T, H, W, 3] uint8, same shape.
    skip_first: frames to drop from the start (frame 0 = GT conditioning).
    """
    T = min(len(gt), len(gen))
    psnrs, ssims, lps = [], [], []
    for t in range(skip_first, T):
        psnrs.append(psnr_fn(gt[t], gen[t], data_range=255))
        ssims.append(ssim_fn(gt[t], gen[t], channel_axis=2, data_range=255))
        lps.append(lpips_fn(gt[t], gen[t]))
    return FrameMetrics(psnrs, ssims, lps)


# ---------- FVD --------------------------------------------------------------

class FVDFeatureExtractor:
    """R3D-18 features on Kinetics-400 mean/std, 112x112, (B, C, T, H, W)."""

    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

    def __init__(self):
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        # Strip classifier, keep avgpool output as feature.
        model.fc = torch.nn.Identity()
        self.model = model.to(DEVICE).eval()
        self.mean = self.MEAN.to(DEVICE)
        self.std = self.STD.to(DEVICE)

    @torch.no_grad()
    def features(self, video_uint8: np.ndarray) -> torch.Tensor:
        """
        video_uint8: [T, H, W, 3] uint8.
        Returns: [512] feature vector (single clip).
        """
        # Center-crop to square then resize to 112; preserves composite content.
        v = torch.from_numpy(video_uint8).permute(3, 0, 1, 2).float() / 255.0  # [C, T, H, W]
        v = F.interpolate(v.unsqueeze(0), size=(v.shape[1], 112, 112), mode="trilinear", align_corners=False)
        v = v.to(DEVICE)
        v = (v - self.mean) / self.std
        feat = self.model(v)  # [1, 512]
        return feat.squeeze(0).cpu()


def frechet_distance(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    """
    Fréchet distance between two Gaussian fits. Standard FID/FVD formula.
    feats_*: [N, D] numpy arrays.
    """
    mu_r, mu_f = feats_real.mean(axis=0), feats_fake.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_f = np.cov(feats_fake, rowvar=False)

    diff = mu_r - mu_f
    # Matrix sqrt of product of covariances via eigendecomposition on symmetrized product.
    from scipy.linalg import sqrtm  # scipy is a dep of scikit-image so already present
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


def compute_fvd(gt_videos: List[np.ndarray], gen_videos: List[np.ndarray]) -> Tuple[float, int]:
    """
    gt_videos, gen_videos: lists of [T, H, W, 3] uint8 arrays (paired episodes).
    Returns (fvd, n_pairs_used).
    """
    assert len(gt_videos) == len(gen_videos)
    extractor = FVDFeatureExtractor()
    feats_r, feats_f = [], []
    for vr, vf in zip(gt_videos, gen_videos):
        T = min(len(vr), len(vf))
        feats_r.append(extractor.features(vr[:T]).numpy())
        feats_f.append(extractor.features(vf[:T]).numpy())
    feats_r = np.stack(feats_r)
    feats_f = np.stack(feats_f)
    return frechet_distance(feats_r, feats_f), len(feats_r)
