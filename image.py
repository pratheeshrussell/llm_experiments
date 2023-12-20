import gc
import textwrap
from datetime import datetime

import torch
# from PIL import Image, ImageDraw, ImageChops, ImageEnhance
from diffusers import StableDiffusionPipeline

# https://github.com/gabacode/InspirationAI/blob/main/src/image/image.py
# epicrealism_naturalSinRC1VAE.safetensors
prompt = "Generate an image of a pristine night sky featuring a full moon suspended majestically amidst a backdrop of captivating scenery. Place the full moon centrally in the frame, radiating a soft, ethereal glow that illuminates the surroundings. Surround the moon with a panoramic landscape that incorporates silhouetted elements such as towering trees, distant mountains, or a serene body of water. Infuse the scene with a sense of tranquility and mystery, capturing the play of shadows and highlights to accentuate the moonlit details. Convey a seamless blend of realism and artistic flair, ensuring that the viewer feels the serene ambiance of a moonlit night in a picturesque environment."
# prompt = prompt 

negative_prompt="verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.8),cross-eyed,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, ((repeating hair))"
img_width=768
img_height=512
inference_steps=20
guidance_scale=7

max_length=77 # the size allowed - Dont change

# To use from local path use this
# model_path="models/epicrealism.safetensors"
# pipe=StableDiffusionPipeline.from_single_file(
#     model_path, 
#     use_safetensors=True,
#     load_safety_checker=False)

# emilianJR/epiCRealism
# runwayml/stable-diffusion-v1-5
# "Justin-Choo/epiCRealism-Natural_Sin_RC1_VAE"
repo_path="emilianJR/epiCRealism"
# To download from hugging face use the following
pipe = StableDiffusionPipeline.from_pretrained(
    repo_path,
    torch_dtype=torch.float32,
    use_safetensors=True,)

# pipe.enable_sequential_cpu_offload()
# if torch.cuda.is_available():
#     print("Using CUDA")
#     pipe = pipe.to("cuda")
#     pipe.enable_vae_slicing()
#     pipe.enable_xformers_memory_efficient_attention()
# else:
#     print("Using CPU")
#     pipe.enable_sequential_cpu_offload()

# Tokenizer long prompt issues
input_ids = pipe.tokenizer(
    prompt, 
    return_tensors="pt", 
    truncation=False
).input_ids
negative_ids = pipe.tokenizer(
    negative_prompt, 
    truncation=False, 
    padding="max_length",
    max_length=input_ids.shape[-1], 
    return_tensors="pt"
).input_ids
concat_embeds = []
neg_embeds = []
for i in range(0, input_ids.shape[-1], max_length):
    concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
    neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])

prompt_embeds = torch.cat(concat_embeds, dim=1)
negative_prompt_embeds = torch.cat(neg_embeds, dim=1)

print(prompt_embeds.shape)
print(negative_prompt_embeds.shape)
print('Tokens Generated...')
# generate args
# "negative_prompt_embeds": negative_prompt_embeds,
new_args = {
    "prompt_embeds":prompt_embeds,
    
    "width": img_width,
    "height": img_height,
    "guidance_scale":guidance_scale,
    "num_inference_steps":inference_steps,
    "num_images_per_prompt":1
}
image = pipe(**new_args).images[0]

del pipe
# torch.cuda.empty_cache()
gc.collect()
image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
image.save(image_path)
print("Saved to "+image_path)