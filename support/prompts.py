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
        Understanding the scene {description} and imagine the entire environment.
        Follow the steps given below 
        - Generate a list of realistic descriptive keywords to illustrate this scene, separated by commas. 
        - If there are people or characters present, add any discernable details like age range, gender, ethnicity, number present, poses, facial emotions, clothing and accessories as keyword. 
        - If no people/characters, then add any notable animals, objects, buildings, plantlife or landscape features like trees, forests, fields etc. 
        - As the descriptors increase in importance to emphazise the essence of the scene, attach additional '+' symbols after each keyword.
        - If a keyword should be deemphasized in the image, place a '-' symbol after it.
        - Don't add symbols like '#' and '~' only '+','-',',' are allowed
        - If an image frame/style is specified, emphasize keywords to depict the scene from that perspective.
        - Focus the list on pertinent keywords necessary to visualize the scene, with no less than 10 and no more than 25 descriptors
         
        The final output should be in this style:
        Description: group of children playing in park 
        Image description: trees++, playground, grass, young children+++, laughing++, running++, sunshine+
        
        Description: friends talking at dinner table as a wide angle photo  
        Image description: dinner table+++, friends laughing++, blurred restaurant interior+, glass of wine+, window
        
        Description: close up photo of a rabbit in a forest  
        Image description: close up photo+, rabbit++, forest++, haze, halation, bloom, dramatic atmosphere, centred, rule of thirds, 200mm 1.4f macro shot
        
        Description: offroad vehicle  
        Image description: offroad car++, forest, sunset, clouds

        Description: polaroid night photo of 24 y.o woman  
        Image description: polaroid photo++, night photo++, 24 y.o++, beautiful woman++, pale skin, bokeh, motion blur
        ---
        Description: {{ description }}
        Image description:
        """
    ).strip(),
)