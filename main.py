import gc
from support import load_llm, generate_quote, generate_caption,generate_image_description, generate_image

topic = "dark night sky with full moon rising over the hills and trees"
llm = load_llm()
# quote = generate_quote(llm, topic)
description = generate_image_description(llm, topic)

# description="A radiant sun illuminates a lush garden filled with blossoming flowers and vibrant greenery; its bountiful harvest seems to overflow like an endless stream, symbolizing the boundless potential of generous spirits who open their hearts to the beauty around them."

description_Text=description
print(description_Text)

# Free memory
del llm
gc.collect()
# exit(0)
image_path = generate_image(description_Text)

# Just print the result
result={
    "topic": topic,
    "description": description_Text,
    "image_path": image_path['path']
}

for keys, value in result.items():
   print((keys + ' => ' + value).encode())

