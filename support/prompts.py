from textwrap import dedent

from langchain.prompts import PromptTemplate

QUOTE_PROMPT = PromptTemplate(
    template_format="jinja2",
    input_variables=["topic"],
    template=dedent(
        """
        Use the topic provided to generate an original, deep inspirational quote.
        It should be brief, concise, and easy to understand.
        It has the potential of changing someone's life.
        Use maximum 25 words in your quote.
        ---
        Topic: {{ topic }}
        Inspirational Quote:
        """
    ).strip(),
)

CAPTION_PROMPT = PromptTemplate(
    template_format="jinja2",
    input_variables=["quote"],
    template=dedent(
        """
        Use the quote provided to imagine an image description that would go with it.
        Formulate a description of the image by using rethorical figures.
        You can use the following rethorical figures:
        - simile
        - metaphor
        - personification
        - hyperbole
        Use maximum 25 words in your description.
        ---
        Quote: {{ quote }}
        Image description:
        """
    ).strip(),
)


IMAGE_GEN_PROMPT = PromptTemplate(
    template_format="jinja2",
    input_variables=["description"],
    template=dedent(
        """
        Imagine a scene of {description} in full realistic detail. 
            - Describe whatever is the central focus or subject of the scene first. Write a very detailed multi-sentence verbose description
            - If there are people or characters present, describe any discernable details like age range, gender, ethnicity, number present, poses, facial emotions, clothing and accessories. 
            - If no people/characters describe any notable animals, objects, buildings, plantlife or landscape features like trees, forests, fields etc. Describe the arrangement and positions of subjects/features in the scene, lighting conditions, dominant colors and color temperature, any background setting details within the scene composition.
        ---
        Description: {{ description }}
        Image description:
        """
    ).strip(),
)