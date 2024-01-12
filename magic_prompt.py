from transformers import pipeline, set_seed
import random

starting_text="dark night sky with full moon rising over the hills and trees"
gpt2_pipe = pipeline('text-generation', model='Gustavosta/MagicPrompt-Stable-Diffusion', tokenizer='gpt2')

seed = random.randint(100, 1000000)
set_seed(seed)

response = gpt2_pipe(starting_text,
                     max_length=500, 
                     pad_token_id=gpt2_pipe.tokenizer.eos_token_id, 
                     num_return_sequences=1)
print(response[0].get('generated_text'))

