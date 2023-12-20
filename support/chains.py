from langchain.chains import LLMChain
from langchain.llms.base import LLM
from support.prompts import QUOTE_PROMPT, CAPTION_PROMPT, IMAGE_GEN_PROMPT

import gc
import torch
from datetime import datetime
from diffusers import StableDiffusionPipeline

import os

def generate_quote(llm: LLM, topic: str) -> str:
    """
    Generates a quote using the LLM model.
    """
    chain = LLMChain(
        llm=llm,
        prompt=QUOTE_PROMPT
    )
    return chain.run({
        "topic": topic
    }).strip()


def generate_caption(llm: LLM, quote: str) -> str:
    """
    Generates a description of the image using the LLM model.
    """
    chain = LLMChain(
        llm=llm,
        prompt=CAPTION_PROMPT
    )
    return chain.run({
        "quote": quote
    }).strip()

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
    }).strip()

def generate_image(description: str) -> str:
    """
    Generates an image based on the description and returns the path.
    """

    negative_prompt="text, bad anatomy, bad hands, unrealistic, bad pose, bad lighting"
    img_width=512
    img_height=512
    inference_steps=30
    guidance_scale=1

    max_length=77 # the size allowed - Dont change

    args = {
        "prompt":description,
        "width": img_width,
        "height": img_height,
        "negative_prompt": negative_prompt,
        "num_inference_steps": inference_steps,
        "guidance_scale": guidance_scale,
        "num_images_per_prompt": 1,
    }

    # Using repo path
    # repo_path="emilianJR/epiCRealism"
    # pipe = StableDiffusionPipeline.from_pretrained(
    #     repo_path,
    #     torch_dtype=torch.float32,
    #     use_safetensors=True,)
    model_path="models/epicrealism.safetensors"
  
    pipe=StableDiffusionPipeline.from_single_file(
        model_path, 
        use_safetensors=True,
        load_safety_checker=False)

    # Tokenizer long prompt issues
    input_ids = pipe.tokenizer(
        description, 
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
        concat_embeds.append(
            pipe.text_encoder(input_ids[:, i: i + max_length])[0]
        )
        neg_embeds.append(
            pipe.text_encoder(negative_ids[:, i: i + max_length])[0]
        )

    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    # generate args
    new_args = {
        "prompt_embeds":prompt_embeds,
        "negative_prompt_embeds": negative_prompt_embeds,
        "width": img_width,
        "height": img_height,
        "guidance_scale":guidance_scale,
        "num_inference_steps":inference_steps,
        "num_images_per_prompt":1
    }

    # pipe.enable_sequential_cpu_offload()
    image = pipe(**new_args).images[0]

    del pipe
    # torch.cuda.empty_cache()
    gc.collect()
    image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
    image.save(image_path)
    return {
        "path": image_path
    }