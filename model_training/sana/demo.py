
import os
os.environ['HF_HOME'] = '/mnt/data0/parth/hf_models_cache'

import torch
from app.sana_pipeline import SanaPipeline
from torchvision.utils import save_image

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = torch.Generator(device=device).manual_seed(42)

sana = SanaPipeline("configs/sana1-5_config/1024ms/Sana_1600M_1024px_allqknorm_bf16_lr2e5.yaml")
sana.from_pretrained("hf://Efficient-Large-Model/SANA1.5_1.6B_1024px/checkpoints/SANA1.5_1.6B_1024px.pth")

image = sana(
    prompt='a cyberpunk cat with a neon sign that says "Sana"',
    height=1024,
    width=1024,
    guidance_scale=4.5,
    pag_guidance_scale=1.0,
    num_inference_steps=100,
    generator=generator,
)
save_image(image, 'output/sana1.png', nrow=1, normalize=True, value_range=(-1, 1))