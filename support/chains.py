from langchain.chains import LLMChain
from langchain.llms.base import LLM
from support.prompts import QUOTE_PROMPT, CAPTION_PROMPT, IMAGE_GEN_PROMPT

import gc
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

import os
import random
from compel import Compel
from transformers import pipeline, set_seed
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

def generate_image_prompt(starting_text: str) -> str:
    """
    Generates the image prompt using the LLM model.
    """
    gpt2_pipe = pipeline('text-generation', 
        model='Gustavosta/MagicPrompt-Stable-Diffusion', 
        tokenizer='gpt2')
    # hf = HuggingFacePipeline(pipeline=gpt2_pipe)
    seed = random.randint(100, 1000000)
    set_seed(seed)

    response = gpt2_pipe(starting_text,
                     max_length=1000, 
                     pad_token_id=gpt2_pipe.tokenizer.eos_token_id, 
                     num_return_sequences=1)
    return response[0].get('generated_text')

def generate_image_description(llm: LLM, desc: str) -> str:
    """
    Generates the scenery of the image using the LLM model.
    """
    chain = LLMChain(
        llm=llm,
        prompt=IMAGE_GEN_PROMPT
    )
    return chain.run({
        "description": desc
    }).strip().replace(" +", "+").replace(" -", "-")

def generate_image(description: str) -> str:
    """
    Generates an image based on the description and returns the path.
    """
    # https://huggingface.co/stablediffusionapi/epicrealism5
    # Atleast an empty "negative_prompt" is needed
    # Negative prompt taken from seaart ai
    negative_prompt="verybadimagenegative_v1.3, ng_deepnegative_v1_75t, (ugly face:0.8),cross-eyed,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, bad anatomy, DeepNegative, facing away, tilted head, {Multiple people}, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worstquality, low quality, normal quality, jpegartifacts, signature, watermark, username, blurry, bad feet, cropped, poorly drawn hands, poorly drawn face, mutation, deformed, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, extra fingers, fewer digits, extra limbs, extra arms,extra legs, malformed limbs, fused fingers, too many fingers, long neck, cross-eyed,mutated hands, polar lowres, bad body, bad proportions, gross proportions, text, error, missing fingers, missing arms, missing legs, extra digit, extra arms, extra leg, extra foot, ((repeating hair))"
    img_width=768
    img_height=768
    inference_steps=50
    guidance_scale=7
    seed=random.randint(100, 1000000)

    # Using repo path
    repo_path="digiplay/Juggernaut_final"
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_path,
        torch_dtype=torch.float32,
        use_safetensors=True,)
    pipe.safety_checker = None
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    
    # Tokenizer long prompt issues
    compel = Compel(
        tokenizer=pipe.tokenizer, 
        text_encoder=pipe.text_encoder,
        truncate_long_prompts=False
    )
    conditioning = compel.build_conditioning_tensor(description)
    negative_conditioning = compel.build_conditioning_tensor(negative_prompt)
    [prompt_embeds, negative_prompt_embeds] = compel.pad_conditioning_tensors_to_same_length([conditioning, negative_conditioning])

    #print(prompt_embeds.shape)
    #print(negative_prompt_embeds.shape)
    # generate args
    # Atleast an empty "negative_prompt_embeds" is needed
    generator = torch.Generator("cpu").manual_seed(seed)
    new_args = {
        "prompt_embeds":prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "width": img_width,
        "height": img_height,
        "guidance_scale":guidance_scale,
        "num_inference_steps":inference_steps,
        "num_images_per_prompt":1,
        "generator":generator
    }

    pipe.enable_sequential_cpu_offload()
    images = pipe(**new_args).images
    image=images[0]
    image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
    image.save(image_path)

    del pipe
    # torch.cuda.empty_cache()
    gc.collect()
    return {
        "path": image_path,
        "image":images
    }