from support import load_llm, generate_quote, generate_caption

topic = "The secret of living is giving."
llm = load_llm()
quote = generate_quote(llm, topic)
caption = generate_caption(llm, quote)

print({
    "quote": quote.encode('ascii', 'ignore'),
    "caption": caption.encode('ascii', 'ignore')
})