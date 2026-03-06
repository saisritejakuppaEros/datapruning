# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""REPA (Representation Alignment) utilities for optional projection alignment loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize

# Encoder preprocessing constants (mirrors REPA/train_t2i.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def build_repa_projectors(
    hidden_size: int,
    encoder_feat_dim: int,
    num_projectors: int,
) -> nn.ModuleList:
    """
    Build lightweight 2-layer MLP projectors to align hidden states -> encoder features.
    One projector per selected block layer.

    Args:
        hidden_size: Model hidden dimension
        encoder_feat_dim: Encoder feature dimension (e.g. DINOv2-B = 768)
        num_projectors: Number of projectors (one per aligned block)

    Returns:
        nn.ModuleList of projectors
    """
    projectors = nn.ModuleList()
    for _ in range(num_projectors):
        proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, encoder_feat_dim),
        )
        projectors.append(proj)
    return projectors


def _align_spatial_dims(z: torch.Tensor, z_tilde: torch.Tensor) -> torch.Tensor:
    """Interpolate z to match z_tilde's spatial (sequence) dimension if needed."""
    if z.shape[1] == z_tilde.shape[1]:
        return z
    B, T_enc, D = z.shape
    T_model = z_tilde.shape[1]
    side_enc = int(T_enc ** 0.5)
    side_model = int(T_model ** 0.5)
    if side_enc * side_enc == T_enc and side_model * side_model == T_model:
        z_2d = z.transpose(1, 2).reshape(B, D, side_enc, side_enc)
        z_resized = F.interpolate(
            z_2d, size=(side_model, side_model), mode="bilinear", align_corners=False
        )
        return z_resized.flatten(2).transpose(1, 2)
    if T_enc < T_model:
        return z.repeat(1, (T_model // T_enc) + 1, 1)[:, :T_model]
    return z[:, :: max(1, T_enc // T_model), :][:, :T_model]


def compute_proj_loss(zs: list, zs_tilde: list, device: torch.device = None) -> torch.Tensor:
    """
    Cosine alignment loss between encoder features (zs) and projected model hiddens (zs_tilde).

    zs:       list of [B, T_enc, D] encoder features (same z repeated for each layer, or one per encoder)
    zs_tilde: list of [B, T_model, D] projected model hiddens per selected layer
              Each element can be a tensor or list with one tensor (from model output format)
    device:   Device to use for zero tensor when count == 0 (defaults to zs[0].device or cuda:0)

    Handles spatial mismatch between z and z_tilde via interpolation when needed.
    """
    proj_loss = None
    count = 0
    for z, z_tilde in zip(zs, zs_tilde):
        if isinstance(z_tilde, list):
            z_tilde = z_tilde[0]
        z_tilde = z_tilde.float()
        z = z.float().to(z_tilde.device)
        z = _align_spatial_dims(z, z_tilde)
        z_tilde = F.normalize(z_tilde, dim=-1)
        z = F.normalize(z, dim=-1)
        term = -(z * z_tilde).sum(dim=-1).mean()
        proj_loss = term if proj_loss is None else proj_loss + term
        count += 1
    if count == 0:
        if device is None:
            device = zs[0].device if zs else torch.device("cuda:0")
        return torch.tensor(0.0, device=device, dtype=torch.float32)
    return proj_loss / count


def preprocess_for_encoder(
    images_pixel: torch.Tensor,
    enc_type: str,
    resolution: int = 256,
) -> torch.Tensor:
    """
    Normalize raw pixel images for the vision encoder.
    Mirrors REPA-main/train_t2i.py preprocess_raw_image().

    Args:
        images_pixel: [B, C, H, W] in [0, 255] range (float or uint8)
        enc_type: Encoder type string, e.g. "dinov2-vit-b", "clip-vit-b"
        resolution: Training resolution (256 or 512)

    Returns:
        Preprocessed tensor ready for encoder forward
    """
    x = images_pixel.float() / 255.0
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if "clip" in enc_type:
        x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        x = Normalize(CLIP_MEAN, CLIP_STD)(x)
    elif "dinov2" in enc_type:
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        enc_size = 224 * (resolution // 256)
        x = F.interpolate(x, size=enc_size, mode="bicubic", align_corners=False)
    elif any(k in enc_type for k in ("dinov1", "mocov3", "mae", "jepa")):
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        if "jepa" in enc_type or "clip" in enc_type:
            x = F.interpolate(x, size=224, mode="bicubic", align_corners=False)
        else:
            x = F.interpolate(x, size=224 * (resolution // 256), mode="bicubic", align_corners=False)
    return x


@torch.no_grad()
def load_repa_encoders(enc_type: str, device: torch.device, resolution: int = 256):
    """
    Load vision encoder(s) for REPA alignment.
    Adapted from REPA/utils.py load_encoders.

    Supports DINOv2 natively (no external REPA dependencies). For 512 resolution,
    only DINOv2 is supported per REPA implementation.

    Args:
        enc_type: Comma-separated encoder specs, e.g. "dinov2-vit-b" or "dinov2-vit-b,dinov2-vit-l"
        device: Target device
        resolution: 256 or 512

    Returns:
        encoders: List of encoder modules
        encoder_types: List of encoder type strings
        architectures: List of architecture strings
    """
    assert resolution in (256, 512), "Resolution must be 256 or 512"
    enc_names = enc_type.split(",")
    encoders = []
    encoder_types = []
    architectures = []

    try:
        import timm
    except ImportError:
        raise ImportError("timm is required for REPA encoders. Install with: pip install timm")

    def _resample_pos_embed(posemb, new_size, num_prefix_tokens=1):
        """Resample absolute position embedding to new spatial size."""
        import math

        num_pos_tokens = posemb.shape[1]
        num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
        if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
            return posemb
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = (hw, hw)
        if num_prefix_tokens:
            posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
        else:
            posemb_prefix, posemb = None, posemb
        embed_dim = posemb.shape[-1]
        orig_dtype = posemb.dtype
        posemb = posemb.float()
        posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
        posemb = F.interpolate(posemb, size=new_size, mode="bicubic", antialias=True)
        posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
        posemb = posemb.to(orig_dtype)
        if posemb_prefix is not None:
            posemb = torch.cat([posemb_prefix, posemb], dim=1)
        return posemb

    for enc_name in enc_names:
        enc_name = enc_name.strip()
        parts = enc_name.split("-")
        if len(parts) < 3:
            raise ValueError(
                f"enc_type '{enc_name}' must be format 'encoder_type-architecture-config' "
                "(e.g. dinov2-vit-b)"
            )
        encoder_type, architecture, model_config = parts[0], parts[1], parts[2]

        if resolution == 512 and encoder_type != "dinov2":
            raise NotImplementedError(
                "Currently, 512x512 resolution only supports DINOv2 encoders. "
                f"Got encoder_type={encoder_type}"
            )

        architectures.append(architecture)
        encoder_types.append(encoder_type)

        if "dinov2" in encoder_type:
            # In distributed training, only rank 0 downloads first to avoid torch.hub cache race
            if torch.distributed.is_initialized():
                if torch.distributed.get_rank() == 0:
                    if "reg" in encoder_type:
                        torch.hub.load(
                            "facebookresearch/dinov2", f"dinov2_vit{model_config}14_reg", trust_repo=True
                        )
                    else:
                        torch.hub.load(
                            "facebookresearch/dinov2", f"dinov2_vit{model_config}14", trust_repo=True
                        )
                torch.distributed.barrier()

            if "reg" in encoder_type:
                encoder = torch.hub.load(
                    "facebookresearch/dinov2", f"dinov2_vit{model_config}14_reg", trust_repo=True
                )
            else:
                encoder = torch.hub.load(
                    "facebookresearch/dinov2", f"dinov2_vit{model_config}14", trust_repo=True
                )
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = _resample_pos_embed(
                encoder.pos_embed.data,
                [patch_resolution, patch_resolution],
            )
            encoder.head = nn.Identity()
            if not hasattr(encoder, "embed_dim"):
                encoder.embed_dim = encoder.pos_embed.shape[-1]
            encoder = encoder.to(device)
            encoder.eval()
            encoders.append(encoder)
        else:
            raise NotImplementedError(
                f"Encoder type '{encoder_type}' is not supported in self-contained repa_utils. "
                "Only dinov2 is supported. For mocov3, mae, jepa, clip, dinov1, "
                "you would need to integrate REPA's models directory."
            )

    return encoders, encoder_types, architectures


@torch.no_grad()
def extract_encoder_features(
    encoders: list,
    encoder_types: list,
    pixel_imgs: torch.Tensor,
    resolution: int = 256,
) -> list:
    """
    Extract features from encoder(s) given preprocessed pixel images.

    Args:
        encoders: List of encoder modules
        encoder_types: List of encoder type strings (for output format)
        pixel_imgs: [B, C, H, W] preprocessed for encoder (from preprocess_for_encoder)
        resolution: Training resolution

    Returns:
        zs: List of [B, T, D] tensors, one per encoder
    """
    zs = []
    for enc, enc_type in zip(encoders, encoder_types):
        if hasattr(enc, "forward_features"):
            feat = enc.forward_features(pixel_imgs)
        else:
            feat = enc(pixel_imgs)
        if isinstance(feat, dict):
            if "x_norm_patchtokens" in feat:
                feat = feat["x_norm_patchtokens"]
            else:
                feat = feat.get("x", feat.get("last_hidden_state", feat))
        elif "mocov3" in enc_type:
            feat = feat[:, 1:]  # drop CLS
        zs.append(feat)
    return zs
