import gradio as gr
from support import generate_image_prompt, generate_image

block = gr.Blocks(css=".container { max-width: 800px; margin: auto; }")

def infer(prompt):
    description_Text=generate_image_prompt(prompt)
    img_data=generate_image(description_Text)
    return img_data['image']

with block as demo:
    gr.Markdown("<h1><center>Image generation demo</center></h1>")
    gr.Markdown(
        "Uses the MagicPrompt-Stable-Diffusion model to refine the input prompt and then uses the Juggernaut model to generate an image"
    )
    with gr.Group():
        with gr.Group():
            with gr.Row():

                text = gr.Textbox(
                    label="Enter your prompt", show_label=False, max_lines=1
                )
                btn = gr.Button("Run")
               
        gallery = gr.Gallery(label="Generated images", show_label=False)
        text.submit(infer, inputs=[text], outputs=gallery)
        btn.click(infer, inputs=[text], outputs=gallery)

    gr.Markdown(
        """___
            <p style='text-align: center'>
                Created as a Demo by Pratheesh
            <br/>
            </p>""")

demo.launch(debug=True,share=True)