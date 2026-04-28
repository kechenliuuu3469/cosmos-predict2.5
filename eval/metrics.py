"""
Video quality metrics: PSNR / SSIM / LPIPS (per-frame) and FVD (distributional).

FVD uses torchvision's R3D-18 pretrained on Kinetics-400 as the feature
extractor — NOT the original I3D FVD, so absolute numbers are not directly
comparable to published FVD values. For internal checkpoint comparison
(the purpose here) that's fine.
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
    def __init__(self, net: str = "vgg"):
        self.net = net
        self._fn = None

    def __call__(self, gt_uint8: np.ndarray, gen_uint8: np.ndarray) -> float:
        if self._fn is None:
            self._fn = lpips.LPIPS(net=self.net).to(DEVICE).eval()
        with torch.no_grad():
            return self._fn(_to_lpips(gt_uint8), _to_lpips(gen_uint8)).item()


def _to_lpips(x: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(x).permute(2, 0, 1).float() / 127.5 - 1.0
    return t.unsqueeze(0).to(DEVICE)


def compute_frame_metrics(
    gt: np.ndarray, gen: np.ndarray, lpips_fn: LPIPSWrapper, skip_first: int = 1
) -> FrameMetrics:
    T = min(len(gt), len(gen))
    psnrs, ssims, lps = [], [], []
    for t in range(skip_first, T):
        psnrs.append(psnr_fn(gt[t], gen[t], data_range=255))
        ssims.append(ssim_fn(gt[t], gen[t], channel_axis=2, data_range=255))
        lps.append(lpips_fn(gt[t], gen[t]))
    return FrameMetrics(psnrs, ssims, lps)


class FVDFeatureExtractor:
    MEAN = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    STD = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)

    def __init__(self):
        weights = R3D_18_Weights.KINETICS400_V1
        model = r3d_18(weights=weights)
        model.fc = torch.nn.Identity()
        self.model = model.to(DEVICE).eval()
        self.mean = self.MEAN.to(DEVICE)
        self.std = self.STD.to(DEVICE)

    @torch.no_grad()
    def features(self, video_uint8: np.ndarray) -> torch.Tensor:
        v = torch.from_numpy(video_uint8).permute(3, 0, 1, 2).float() / 255.0
        v = F.interpolate(v.unsqueeze(0), size=(v.shape[1], 112, 112), mode="trilinear", align_corners=False)
        v = v.to(DEVICE)
        v = (v - self.mean) / self.std
        feat = self.model(v)
        return feat.squeeze(0).cpu()


def frechet_distance(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    mu_r, mu_f = feats_real.mean(axis=0), feats_fake.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_f = np.cov(feats_fake, rowvar=False)
    diff = mu_r - mu_f
    from scipy.linalg import sqrtm
    covmean, _ = sqrtm(sigma_r @ sigma_f, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))


def compute_fvd(gt_videos: List[np.ndarray], gen_videos: List[np.ndarray]) -> Tuple[float, int]:
    assert len(gt_videos) == len(gen_videos)
    n = len(gt_videos)
    if n < 2:
        # Covariance over a single sample is degenerate (NaNs in sqrtm).
        # Return NaN so the eval still emits summary.json with PSNR/SSIM/LPIPS.
        return float("nan"), n
    extractor = FVDFeatureExtractor()
    feats_r, feats_f = [], []
    for vr, vf in zip(gt_videos, gen_videos):
        T = min(len(vr), len(vf))
        feats_r.append(extractor.features(vr[:T]).numpy())
        feats_f.append(extractor.features(vf[:T]).numpy())
    feats_r = np.stack(feats_r)
    feats_f = np.stack(feats_f)
    return frechet_distance(feats_r, feats_f), n
