from support import load_llm, generate_quote, generate_caption, generate_image

topic = "A group of children playing in a park"
llm = load_llm()
# quote = generate_quote(llm, topic)
caption = generate_caption(llm, topic)
print(caption.encode('ascii', 'ignore'))
image_path = generate_image(caption)

print({
    "topic": topic,
    "caption": caption.encode('ascii', 'ignore'),
    "image_path": image_path
})