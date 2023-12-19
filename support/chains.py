from langchain.chains import LLMChain
from langchain.llms.base import LLM
from support.prompts import QUOTE_PROMPT, CAPTION_PROMPT


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

def generate_image(description: str) -> str:
    """
    Generates an image based on the description and returns the path.
    """
    args = {
        "prompt":prompt,
        "width": 512,
        "height": 512,
        "negative_prompt": ["text", "bad anatomy", "bad hands", "unrealistic", "bad pose", "bad lighting"],
        "num_inference_steps": 50,
        "guidance_scale": 1,
    }
    # emilianJR/epiCRealism
    # runwayml/stable-diffusion-v1-5
    repo_path="emilianJR/epiCRealism"
    # To download from hugging face use the following
    pipe = StableDiffusionPipeline.from_pretrained(
        repo_path,
        torch_dtype=torch.float32,
        use_safetensors=True,)

    pipe.enable_sequential_cpu_offload()
    image = pipe(**args).images[0]

    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    image_path = "image/" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".jpg"
    image.save(image_path)
    return chain.run({
        "path": image_path
    }).strip()