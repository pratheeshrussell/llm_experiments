import gc
import textwrap
from datetime import datetime

import torch
# from PIL import Image, ImageDraw, ImageChops, ImageEnhance
from diffusers import StableDiffusionPipeline

# https://github.com/gabacode/InspirationAI/blob/main/src/image/image.py
# epicrealism_naturalSinRC1VAE.safetensors
prompt = "In this tranquil yet vibrant scene, we see several young kids engaged in various playful activities on a sunny day at their local public park. The lush green grass stretches out as far as the eye can see underfoot, with scattered patches of brightly colored wildflowers adding splashes of reds, yellows, blues, and purples throughout the expanse. Trees draped in thick foliage line one side of the open field where these little ones have come to enjoy themselves after school let out for the day. Their laughter fills the air along with the gentle rustling of leaves swaying gently overhead in response to the warm breeze that carries the scent of fresh cut grass across the park.\n\nThe children's ages span from about four years old up until early teenagers; they represent an array of genders, races, and cultures all coming together harmoniously through shared experiences during playtime. Some wear shorts and t-shirts while others don more casual attire such as jeans paired with hoodies or dresses depending upon personal preference  each outfit reflective of individual identity but united by their common purpose here today which seems primarily focused around enjoying each others company amidst nature's beauty"
args = {
    "prompt":prompt,
    "width": 512,
    "height": 512,
    "negative_prompt": ["text", "bad anatomy", "bad hands", "unrealistic", "bad pose", "bad lighting"],
    "num_inference_steps": 30,
    "guidance_scale": 1,
}

# To use from local path use this
model_path="models/epicrealism.safetensors"
pipe=StableDiffusionPipeline.from_single_file(
    model_path, 
    use_safetensors=True,
    load_safety_checker=False,
    local_files_only=True)

# emilianJR/epiCRealism
# runwayml/stable-diffusion-v1-5
repo_path="emilianJR/epiCRealism"
# To download from hugging face use the following
# pipe = StableDiffusionPipeline.from_pretrained(
#     repo_path,
#     torch_dtype=torch.float32,
#     use_safetensors=True,)

# pipe.enable_sequential_cpu_offload()
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
# torch.cuda.empty_cache()
gc.collect()
image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
image.save(image_path)
print("Saved to "+image_path)