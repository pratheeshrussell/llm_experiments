from langchain.llms.base import LLM
from langchain.llms.llamacpp import LlamaCpp

base_config = {
    "n_gpu_layers": 64,
    "n_batch": 4096,
    "n_ctx": 4096,
    "f16_kv": True,
    "repeat_penalty": 1.3,
    "last_n_tokens_size": 2048,
    "temperature": 0.6,
    "top_p": 1,
    "verbose": False,
    "streaming": False
}

# mistral-7b-merge-14-v0.1.Q4_K_M
# phi-2.Q4_K_M
def load_llm() -> LLM:
    try:
        config = {
            "model_path": "./models/mistral-7b-merge-14-v0.1.Q4_K_M.gguf",
            **base_config
        }
        return LlamaCpp(**config)
    except Exception as e:
        print(e)
        exit(1)