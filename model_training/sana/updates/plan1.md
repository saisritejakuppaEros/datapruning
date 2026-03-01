# REPA Integration into Sana Training

This guide walks you through integrating REPA (Representation Alignment) into the Sana training pipeline so you can train **with or without** it via a single config flag.

---

## What REPA Does

REPA adds a **projection alignment loss** that aligns intermediate hidden states of the diffusion transformer with features from a pretrained vision encoder (DINOv2, CLIP, JEPA, etc.). The total loss becomes:

```
total_loss = diffusion_loss + λ * proj_loss
```

Where `λ` (`repa_proj_coeff`) controls how much the alignment is weighted. Setting it to `0` disables REPA entirely.

---

## File-by-File Plan

### 1. `diffusion/utils/config.py` — Add REPA config block

Add a new `REPAConfig` dataclass and wire it into `SanaConfig`.

```python
# diffusion/utils/config.py

@dataclass
class REPAConfig(BaseConfig):
    enabled: bool = False                        # Master switch
    enc_type: str = "dinov2-vit-b"               # Encoder(s), comma-separated
    proj_coeff: float = 0.5                      # λ weight for proj_loss
    encoder_depth: int = 8                       # Which layer(s) to extract features from
    resolution: int = 256                        # Must match training resolution
    # Optional: path to cached encoder features
    precomputed_feats_path: Optional[str] = None
    extra: Any = None
```

Then inside `SanaConfig`:

```python
@dataclass
class SanaConfig(BaseConfig):
    ...
    repa: REPAConfig = field(default_factory=REPAConfig)  # ADD THIS
    ...
```

---

### 2. `diffusion/model/nets/sana.py` — Expose intermediate hidden states

REPA needs hidden states from inside the transformer blocks. Modify `Sana.forward()` to optionally collect and project them.

#### 2a. Add projector heads in `__init__`

```python
# In Sana.__init__, after self.blocks is built:

self.repa_enabled = False  # toggled externally

# Projector: maps hidden_size -> encoder feature dim
# Instantiate this only when REPA is active (done in train.py via setup)
self.repa_projectors = nn.ModuleList()
```

#### 2b. Modify `forward()` to optionally return hidden states

```python
def forward(self, x, timestep, y, mask=None, data_info=None, return_hidden=False, **kwargs):
    ...
    # existing embed code unchanged
    ...
    hidden_states = []
    for i, block in enumerate(self.blocks):
        x = auto_grad_checkpoint(block, x, y, t0, y_lens, image_pos_embed)
        if return_hidden and self.repa_enabled:
            hidden_states.append(x)  # collect ALL block outputs

    x = self.final_layer(x, t)
    x = self.unpatchify(x)

    if return_hidden and self.repa_enabled and self.repa_projectors:
        # Project selected hidden states
        zs_tilde = []
        for proj, h in zip(self.repa_projectors, hidden_states):
            zs_tilde.append([proj(h)])  # shape: [B, T, proj_dim]
        return x, zs_tilde

    return x
```

> **Note:** The `__call__` passthrough already forwards `**kwargs`, so `return_hidden` will flow through fine.

---

### 3. New file: `diffusion/model/repa_utils.py`

Create this file to hold REPA-specific helpers, keeping the main training script clean.

```python
# diffusion/model/repa_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_repa_projectors(hidden_size: int, encoder_feat_dim: int, num_projectors: int) -> nn.ModuleList:
    """
    Build lightweight 2-layer MLP projectors to align hidden states -> encoder features.
    One projector per selected block layer.
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


def compute_proj_loss(zs: list, zs_tilde: list) -> torch.Tensor:
    """
    Cosine alignment loss between encoder features (zs) and projected model hiddens (zs_tilde).
    
    zs:       list of [B, T, D] encoder features per selected layer
    zs_tilde: list of [B, T, D] projected model hiddens per selected layer
    """
    proj_loss = 0.0
    count = 0
    for z, z_tilde in zip(zs, zs_tilde):
        # z_tilde comes as list-of-lists from model, flatten
        if isinstance(z_tilde, list):
            z_tilde = z_tilde[0]
        z_tilde = F.normalize(z_tilde, dim=-1)
        z = F.normalize(z.to(z_tilde.device, z_tilde.dtype), dim=-1)
        proj_loss += -(z * z_tilde).sum(dim=-1).mean()
        count += 1
    return proj_loss / max(count, 1)


def preprocess_for_encoder(images_pixel: torch.Tensor, enc_type: str, resolution: int = 256) -> torch.Tensor:
    """
    Normalize raw pixel images [0,255] for the vision encoder.
    Mirrors REPA-main/train_t2i.py preprocess_raw_image().
    """
    from torchvision.transforms import Normalize
    import torch.nn.functional as F

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD  = (0.229, 0.224, 0.225)
    CLIP_MEAN     = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD      = (0.26862954, 0.26130258, 0.27577711)

    x = images_pixel / 255.0

    if "clip" in enc_type:
        x = F.interpolate(x, 224, mode="bicubic")
        x = Normalize(CLIP_MEAN, CLIP_STD)(x)
    elif "dinov2" in enc_type:
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        x = F.interpolate(x, 224 * (resolution // 256), mode="bicubic")
    elif any(k in enc_type for k in ("dinov1", "mocov3", "mae", "jepa")):
        x = Normalize(IMAGENET_MEAN, IMAGENET_STD)(x)
        if "jepa" in enc_type:
            x = F.interpolate(x, 224, mode="bicubic")
    return x
```

---

### 4. `train_scripts/train.py` — Wire it all together

This is where the biggest edits land. There are 4 zones to touch.

#### Zone A: Imports (top of file)

```python
# ADD after existing imports
from diffusion.model.repa_utils import build_repa_projectors, compute_proj_loss, preprocess_for_encoder
```

If you want to reuse REPA's encoder loader directly:
```python
import sys
sys.path.insert(0, "/path/to/REPA-main")  # adjust as needed
from utils import load_encoders  # REPA's encoder loader
```

Or copy `load_encoders` from `REPA-main/utils.py` into `diffusion/model/repa_utils.py` — recommended for a self-contained codebase.

---

#### Zone B: Setup section (inside `main()`, after model is built)

```python
# --- REPA Setup ---
repa_cfg = config.repa
encoders = []
if repa_cfg.enabled:
    logger.info(f"[REPA] Loading encoder(s): {repa_cfg.enc_type}")
    encoders = load_encoders(repa_cfg.enc_type, device=accelerator.device, resolution=repa_cfg.resolution)
    for enc in encoders:
        enc.eval()
        for p in enc.parameters():
            p.requires_grad_(False)

    # Pick which block indices to align (evenly spaced, like REPA paper)
    num_proj = len(encoders)  # one projector per encoder
    encoder_feat_dim = encoders[0].embed_dim  # DINOv2-B = 768, adjust per encoder

    projectors = build_repa_projectors(
        hidden_size=model.hidden_size,
        encoder_feat_dim=encoder_feat_dim,
        num_projectors=num_proj,
    ).to(accelerator.device)

    # Attach projectors to model so they get wrapped by accelerator
    model.repa_projectors = projectors
    model.repa_enabled = True

    logger.info(f"[REPA] Projectors built: {num_proj}x ({model.hidden_size} -> {encoder_feat_dim})")
else:
    logger.info("[REPA] Disabled — training with diffusion loss only.")
```

> Put this **after** `model = build_model(...)` but **before** `accelerator.prepare(model, optimizer, ...)`.

---

#### Zone C: Training loop — compute REPA features & loss

Find the existing loss block:

```python
# EXISTING
loss_term = train_diffusion.training_losses(
    model, clean_images, timesteps, model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
)
loss = loss_term["loss"].mean()
accelerator.backward(loss)
```

Replace with:

```python
# NEW — with REPA support
with accelerator.accumulate(model):
    optimizer.zero_grad()

    if repa_cfg.enabled:
        # 1. Get encoder features from raw pixel images
        with torch.no_grad():
            pixel_imgs = preprocess_for_encoder(
                raw_images,                  # <-- you need raw pixels here (see Zone D)
                enc_type=repa_cfg.enc_type,
                resolution=repa_cfg.resolution,
            )
            zs = []
            for enc in encoders:
                feats = enc(pixel_imgs)      # shape: [B, T_enc, D]
                zs.append(feats)

        # 2. Forward with hidden state collection
        _, zs_tilde = model(
            clean_images, timesteps,
            y=y, mask=y_mask, data_info=data_info,
            return_hidden=True,
        )

        # 3. Diffusion loss (standard path, but we already ran forward above)
        # Use a dedicated training_losses call that accepts precomputed output:
        loss_term = train_diffusion.training_losses(
            model, clean_images, timesteps,
            model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
        )
        diff_loss = loss_term["loss"].mean()

        # 4. REPA projection loss
        proj_loss = compute_proj_loss(zs, zs_tilde)
        loss = diff_loss + repa_cfg.proj_coeff * proj_loss

    else:
        loss_term = train_diffusion.training_losses(
            model, clean_images, timesteps,
            model_kwargs=dict(y=y, mask=y_mask, data_info=data_info)
        )
        loss = loss_term["loss"].mean()
        proj_loss = torch.tensor(0.0)

    accelerator.backward(loss)
    ...  # rest unchanged
```

---

#### Zone D: Logging

In the `logs` dict that goes to W&B / log buffer, add:

```python
logs = {
    args.loss_report_name: accelerator.gather(loss).mean().item(),
}
if repa_cfg.enabled:
    logs["proj_loss"]  = accelerator.gather(proj_loss).mean().item()
    logs["diff_loss"]  = accelerator.gather(diff_loss).mean().item()
```

---

### 5. Raw pixel images in the dataloader

REPA needs **raw pixel images** (`[0, 255]` uint8 or float). Sana's default pipeline loads pre-encoded VAE latents (`load_vae_feat: true`). You have two options:

**Option A — Load pixels alongside latents (recommended)**

Modify the dataset to also return raw images when `repa.enabled=True`. In your dataset class (under `diffusion/data/`), return a `raw_image` key alongside `img` (the VAE latent). In the training loop, pull it out:

```python
raw_images = batch.get("raw_image", None)
```

**Option B — Precompute encoder features offline**

Run all images through the encoder once before training and save `.npy` or `.pt` files alongside each sample. Load them as a dataset column. This is faster at training time. Use `REPA-main/preprocessing/dataset_tools.py` as a reference.

---

## YAML Config Example

Add to your training YAML under the `repa:` key:

```yaml
# configs/sana_600M_512px.yaml

repa:
  enabled: true
  enc_type: "dinov2-vit-b"    # or: "clip-vit-b,dinov2-vit-b" for multi-encoder
  proj_coeff: 0.5
  encoder_depth: 8
  resolution: 512
  precomputed_feats_path: null  # set to a path to use offline features
```

To train **without REPA**:

```yaml
repa:
  enabled: false
```

---

## Summary of New / Changed Files

| File | Change |
|---|---|
| `diffusion/utils/config.py` | Add `REPAConfig` dataclass; add `repa: REPAConfig` field to `SanaConfig` |
| `diffusion/model/nets/sana.py` | Add `repa_enabled`, `repa_projectors`, `return_hidden` in `forward()` |
| `diffusion/model/repa_utils.py` | **NEW** — projector builder, proj loss fn, encoder preprocessor |
| `train_scripts/train.py` | Import repa_utils; setup encoders+projectors; branch loss logic; extend logging |
| `configs/*.yaml` | Add `repa:` block |
| `diffusion/data/` dataset class | Add `raw_image` return if using Option A |

---

## Gotchas

- **Double forward pass**: The cleanest way to avoid running the model twice is to restructure `training_losses` to accept a precomputed output, or to compute the noisy input + target yourself in train.py and call `model()` once, then compute the MSE manually. Copy the relevant math from `diffusion/model/gaussian_diffusion.py` or the scheduler.
- **FSDP**: If using FSDP (`config.train.use_fsdp: true`), make sure `repa_projectors` is a proper `nn.ModuleList` on the model so FSDP wraps it correctly. Do **not** keep projectors as a separate module outside the model.
- **Encoder dtype**: Encoders should stay in `fp32` or `bf16` matching your main dtype. Cast encoder outputs before computing proj loss.
- **`enc_type` string format**: REPA's `load_encoders()` expects the format `"encoder_type-architecture-config"` e.g. `"dinov2-vit-b-patch14"`. Check `REPA-main/utils.py::load_encoders` for the exact expected strings.
- **Gradient flow**: Encoder weights must be frozen (`requires_grad_(False)`). Only the projector weights (and the diffusion model) train.