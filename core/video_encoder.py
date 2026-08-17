"""
EchoNet Video Encoder
=====================
3D-CNN encoder for echocardiogram video analysis.
Processes (B, 1, T, H, W) grayscale echo clips and produces
a fixed-size embedding vector suitable for:
  - Ejection fraction regression
  - RAG retrieval queries
  - Report generation soft-prompts

Architecture follows the R2+1D factored convolution approach
(Tran et al., 2018) adapted for single-channel medical ultrasound.
"""

import torch
import torch.nn as nn
from typing import Optional


class SpatioTemporalBlock(nn.Module):
    """Factored 3D convolution: spatial 2D conv followed by temporal 1D conv."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int,
                 spatial_kernel: int = 3, temporal_kernel: int = 3,
                 stride: int = 1, downsample_temporal: bool = False):
        super().__init__()
        s_pad = spatial_kernel // 2
        t_pad = temporal_kernel // 2
        t_stride = 2 if downsample_temporal else 1

        self.spatial = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch,
                      kernel_size=(1, spatial_kernel, spatial_kernel),
                      stride=(1, stride, stride),
                      padding=(0, s_pad, s_pad), bias=False),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.temporal = nn.Sequential(
            nn.Conv3d(mid_ch, out_ch,
                      kernel_size=(temporal_kernel, 1, 1),
                      stride=(t_stride, 1, 1),
                      padding=(t_pad, 0, 0), bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.temporal(self.spatial(x))


class EchoVideoEncoder(nn.Module):
    """
    3D-CNN encoder for echocardiogram videos.

    Input : (B, 1, T, H, W)  — grayscale echo clip
    Output: (B, embed_dim)    — video-level embedding
    """

    def __init__(self, embed_dim: int = 384, dropout: float = 0.3):
        super().__init__()
        self.embed_dim = embed_dim

        # Stem: quick spatial downscale
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
        )

        # Factored spatio-temporal blocks
        self.block1 = SpatioTemporalBlock(32,  48,  64,  stride=1, downsample_temporal=True)
        self.block2 = SpatioTemporalBlock(64,  96,  128, stride=2, downsample_temporal=True)
        self.block3 = SpatioTemporalBlock(128, 192, 256, stride=2, downsample_temporal=True)
        self.block4 = SpatioTemporalBlock(256, 320, 512, stride=2, downsample_temporal=False)

        # Global pooling → projection
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.projection = nn.Sequential(
            nn.Linear(512, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, T, H, W) — grayscale echocardiogram video tensor
        Returns:
            (B, embed_dim) — video embedding
        """
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x).flatten(1)
        x = self.projection(x)
        return x


class EFRegressor(nn.Module):
    """
    Predicts Ejection Fraction (EF) from video embeddings.
    Also provides classification into HFrEF / HFmrEF / Normal.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Returns predicted EF as (B, 1)."""
        return self.regressor(embeddings)

    @staticmethod
    def classify_ef(ef_value: float) -> str:
        if ef_value < 40:
            return "HFrEF"
        elif ef_value < 50:
            return "HFmrEF"
        else:
            return "Normal"


class EchoNetModel(nn.Module):
    """
    Full EchoNet analysis model: video encoder → EF regression + embedding.
    """

    def __init__(self, embed_dim: int = 384):
        super().__init__()
        self.encoder = EchoVideoEncoder(embed_dim=embed_dim)
        self.ef_head = EFRegressor(embed_dim=embed_dim)

    def forward(self, video: torch.Tensor) -> dict:
        """
        Args:
            video: (B, 1, T, H, W)
        Returns:
            dict with keys: embeddings, ef_predicted, ef_category
        """
        embeddings = self.encoder(video)
        ef_pred = self.ef_head(embeddings).squeeze(-1)  # (B,)
        # Physiological gating (peer-review M1): ejection fraction is a percentage;
        # clamp to [0, 100] so an impossible value (e.g. -0.1%) can never propagate
        # into a downstream risk score. See core/safety_gating.validate_measurement.
        ef_pred = ef_pred.clamp(0.0, 100.0)

        categories = [
            self.ef_head.classify_ef(ef.item()) for ef in ef_pred
        ]

        return {
            "embeddings": embeddings,
            "ef_predicted": ef_pred,
            "ef_categories": categories,
        }
