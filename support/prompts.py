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
        Imagine a scene of {description} in full realistic detail. Concisely describe the most salient elements of the scene first in one sentence.  
        Write an approximately 60 word highly detailed description emulating informative photo captions. 
        Only include descriptive details that vividly depict important aspects of the scene. 
        Do not use filler words or extraneous descriptors. 
        Focus on defining details about subjects/objects, background, positions, colors, lighting, expressions, backgrounds that efficiently create a clear visual of the scene. 
        The goal is to guide an AI image generation system to render the scene accurately without fluff or distraction. 
        Keep the description succinct within a 60 word limit. Do not use redundant or unnecessary sentences
        ---
        Description: {{ description }}
        Image description:
        """
    ).strip(),
)