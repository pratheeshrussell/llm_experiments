import gc
import textwrap
from datetime import datetime

import torch
# from PIL import Image, ImageDraw, ImageChops, ImageEnhance
from diffusers import StableDiffusionPipeline

# https://github.com/gabacode/InspirationAI/blob/main/src/image/image.py
# epicrealism_naturalSinRC1VAE.safetensors
prompt = "A radiant sun illuminates a lush garden filled with blossoming flowers and vibrant greenery; its bountiful harvest seems to overflow like an endless stream, symbolizing the boundless potential of generous spirits who open their hearts to the beauty around them."
args = {
    "prompt":prompt,
    "width": 512,
    "height": 512,
    "negative_prompt": ["text", "bad anatomy", "bad hands", "unrealistic", "bad pose", "bad lighting"],
    "num_inference_steps": 50,
    "guidance_scale": 1,
}

model_path="./models/image_models/"
# emilianJR/epiCRealism
# runwayml/stable-diffusion-v1-5
repo_path="emilianJR/epiCRealism"

pipe = StableDiffusionPipeline.from_pretrained(
    repo_path,
    torch_dtype=torch.float32,
    use_safetensors=True,)

image = pipe(**args).images[0]

del pipe
gc.collect()
image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
image.save(image_path)
print("Saved to "+image_path)