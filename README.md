# Hugging Face LLM Experiments

## Install
```
pip install diffusers transformers omegaconf accelerate langchain compel  llama-cpp-python
```
To install torch use the following link (If you have Graphics card use cuda enabled version or cpu only version)  
https://pytorch.org/get-started/locally/  
   
In windows the next headache is installing llama-cpp-python  
https://github.com/abetlen/llama-cpp-python/releases  
you can use the following command - make sure python version matches
```
pip install LINK_FROM_RELEASE
```


## Env Vars
In windows I had to set the following env vars also
* HF_DATASETS_OFFLINE=1  
* TRANSFORMERS_OFFLINE=1
* HUGGINGFACE_HUB_CACHE 
The third might not be required but since I didnt want to store everything in C: I changed it. [Link](https://huggingface.co/docs/transformers/installation#offline-mode)

## Does this work??
throws random errors. Try running from repo first before using a local model  
In windows local model didnt run at all - maybe a path problem

## Model download links
Download and place it in models folder
* Q4_K_M - https://huggingface.co/TheBloke/Mistral-7B-Merge-14-v0.1-GGUF

-- Needed only if you download from sources other than hugging face -- Also local files doesn't seem to work in windows
* https://civitai.com/models/25694/epicrealism

## Links:
* https://github.com/gabacode/InspirationAI/tree/main
* https://github.com/huggingface/diffusers/issues/3694#issuecomment-1593845649
* https://github.com/huggingface/diffusers/issues/4194#issuecomment-1648741638
* https://medium.com/mlearning-ai/using-civitai-models-with-diffusers-package-45e0c475a67e

## Stable Diffusion
* https://jalammar.github.io/illustrated-stable-diffusion/?ref=blog.segmind.com

## Upscaler
* https://stable-diffusion-art.com/ai-upscaler/

## Issues
1. Truncated to 77 Tokens
* https://medium.com/mlearning-ai/using-long-prompts-with-the-diffusers-package-with-prompt-embeddings-819657943050
* https://github.com/huggingface/diffusers/issues/2136#issuecomment-1811947735