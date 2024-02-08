import gc
from support import load_llm, generate_image_description,generate_image_prompt, generate_image

topic = "dark night sky with full moon rising over the hills and trees"
# llm = load_llm()

description_Text= generate_image_prompt(topic)
print(description_Text)

image_path = generate_image(description_Text)
# Just print the result
result={
    "topic": topic,
    "description": description_Text,
    "image_path": image_path['path']
}

for keys, value in result.items():
   print((keys + ' => ' + value).encode())

