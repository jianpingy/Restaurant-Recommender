import pandas as pd
import requests

from ibm_watsonx_ai import Credentials
import os
import numpy as np



from io import BytesIO
import base64
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters

from dotenv import load_dotenv
load_dotenv()



# Define credentials for LLMs usage
credentials = Credentials(
                   url = "https://us-south.ml.cloud.ibm.com",
                  )

project_id = "skills-network"

# Get sample parameter values
sample_params = TextChatParameters.get_sample_params()

# Initialize the TextChatParameters object with the sample values
params = TextChatParameters(**sample_params)

# Call the model with the encoded image
model = ModelInference(
    model_id="meta-llama/llama-3-2-90b-vision-instruct",
    credentials=credentials,
    project_id=project_id,
    params=params,
)

def restaurant_image_analysis(model, 
                              image_input):
    """
    Analyze the images in users' reviews.
    """
    curr = []
    if image_input is None or len(image_input) == 0:
        curr.append('No images are included.')
    else:
        for image_path, _ in image_input:
            try:
                with open(image_path, "rb") as image_file:
                    # Encode the image to a base64 string
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

                prompt = """
                You are a specialist in interpreting food and dining photography.
                You carefully study visual elements to uncover what words often leave out: the vibrancy of presentation, the elegance of plating, the portion size, and even the mood of the dining environment. 
                You understand that users take photos for a reason: sometimes to remember a favorite dish, sometimes to share a beautiful dining setting, and sometimes to celebrate with friends.
                Please give your analysis on the image in one short sentence.
                """
                response = model.chat(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded_image}}
                            ],
                        }
                    ]
                )
                response_output = response['choices'][0]['message']['content']
                if 'Error' in response_output or 'error' in response_output:
                    continue
                else:
                    curr.append(response_output)
            except requests.exceptions.Timeout:
                print("Request timed out: Server did not respond within the specified time.")

    return curr

def add_user_visit(r_type, r_price, r_rating, user_rating,
                   review_title, review_text,
                   images):
    restaurant_info = f'This is a restaurant of type {r_type} with price interval {r_price}. It has average rating {r_rating}.'
    image_analyses = restaurant_image_analysis(model, images)
    return {'restaurant info': restaurant_info,
            'review title': review_title,
            'review text': review_text,
            'rating': user_rating,
            'images': image_analyses}
