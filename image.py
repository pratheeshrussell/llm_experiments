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

# To use from local path use this
model_path="models/epicrealism.safetensors"
# pipe=StableDiffusionPipeline.from_single_file(
#     model_path, 
#     use_safetensors=True,
#     load_safety_checker=False,
#     local_files_only=True)

# emilianJR/epiCRealism
# runwayml/stable-diffusion-v1-5
repo_path="emilianJR/epiCRealism"
# To download from hugging face use the following
pipe = StableDiffusionPipeline.from_pretrained(
    repo_path,
    torch_dtype=torch.float32,
    use_safetensors=True,)

pipe.enable_sequential_cpu_offload()
# if torch.cuda.is_available():
#     print("Using CUDA")
#     pipe = pipe.to("cuda")
#     pipe.enable_vae_slicing()
#     pipe.enable_xformers_memory_efficient_attention()
# else:
#     print("Using CPU")
#     pipe.enable_sequential_cpu_offload()

image = pipe(**args).images[0]

del pipe
torch.cuda.empty_cache()
gc.collect()
image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
image.save(image_path)
print("Saved to "+image_path)