import base64
from http.client import HTTPException
import io
import boto3
from sagemaker import Session
from sagemaker.predictor import Predictor
import json
from pydantic import BaseModel
from typing import List

import sys

# Get the prompt from the command line
prompt = sys.argv[1]

class TextPrompt(BaseModel):
    text: str

class ImageGenerationPayload(BaseModel):
    text_prompts: List[TextPrompt]
    width: int = 1024
    height: int = 1024
    sampler: str
    cfg_scale: float
    steps: int
    seed: int
    use_refiner: bool
    refiner_steps: int
    refiner_strength: float

# Define the payload
payload = ImageGenerationPayload(
    text_prompts=[
        TextPrompt(text=prompt)
    ],
    width=1024,
    height=1024,
    sampler="DPMPP2MSampler",
    cfg_scale=7.0,
    steps=50,
    seed=123,
    use_refiner=True,
    refiner_steps=40,
    refiner_strength=0.2
)

# Create a SageMaker session
sagemaker_session = Session(boto3.Session())

# Define the name of the deployed model
endpoint_name = 'neuro-dev-sdxl-6szd-endpoint'

# Create a predictor
sdxl_model_predictor = Predictor(endpoint_name, sagemaker_session)

def generate_image(payload: ImageGenerationPayload):
    try:
        # Convert the Pydantic model to a dictionary
        sdxl_payload = payload.dict(by_alias=True)
        
        # Get prediction from SageMaker endpoint
        sdxl_response = sdxl_model_predictor.predict(sdxl_payload)
        prompt = sdxl_payload.get('text_prompts', [{}])[0].get('text', 'image')
        # Decode the response and return the image
        return decode_and_show(sdxl_response)
    except Exception as e:
        # Handle general exceptions
        raise e
    

def decode_and_show(response_bytes):
    # Parse the JSON response to get the base64-encoded string
    response_json = json.loads(response_bytes)
    image_base64 = response_json['generated_image']

    # Decode the base64 string
    image_data = base64.b64decode(image_base64)

    # Create an in-memory bytes buffer for the image data
    img_io = io.BytesIO(image_data)
    
    # Upload to S3
    try:
        # Ensure we're at the start of the image buffer before uploading
        with open('output/result.png', 'w') as f:
            f.write(img_io.seek(0))
        print(f"Image saved")
    except BotoCoreError as e:
        print(f"Failed to save: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

generate_image(payload)