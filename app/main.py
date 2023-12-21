import os
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from fastapi.responses import StreamingResponse, PlainTextResponse
from mangum import Mangum
import boto3
from botocore.exceptions import NoCredentialsError

from datetime import datetime
import hashlib
import base64
import io
from PIL import Image
import re

# Specific imports for SageMaker
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import BytesDeserializer
import sagemaker
import json
from typing import List

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

# The `decode_and_show` function remains the same as previously defined
class DownloadFileBody(BaseModel):
    bucket_name: str
    file_name: str

class SimplePromptRequest(BaseModel):
    prompt: str

# Default values
DEFAULT_CONTENT_TYPE = "application/json"
DEFAULT_HISTORY = [
]
DEFAULT_SYSTEM_PROMPT = "You are an expert in the field of cloud computing and will give helpful answers to questions about AWS."
DEFAULT_PARAMETERS = {
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "max_new_tokens": 512,
    "repetition_penalty": 1.03,
    "stop": ["###", "</s>"]
}

# Assume other necessary imports are already done

app = FastAPI()

# Initialize SageMaker session and predictor
sess = sagemaker.Session()
sdxl_endpoint_name = os.environ.get("SDXL_ENDPOINT_NAME", "endpoint-name-not-set")
s3_bucket = os.environ.get("S3_BUCKET", "s3-bucket-not-set")


# Define the predictor (this could also be done inside the endpoint call for a fresh setup each time)
sdxl_model_predictor = Predictor(
    endpoint_name=sdxl_endpoint_name, 
    sagemaker_session=sess,
    serializer=JSONSerializer(),
    deserializer=BytesDeserializer()
)


# Create a SageMaker client
smr = boto3.client('sagemaker-runtime')

s3_client = boto3.client('s3')

app = FastAPI()


@app.post("/prompt-mistral")
async def prompt_mistral(request_data: SimplePromptRequest):
    """
    FastAPI endpoint to generate responses using Mistral.
    """
    try:
        return generate_llm_response(request_data.prompt)
    except HTTPException as http_exception:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing the request: {str(e)}")


@app.post("/generate-image")
async def generate_image(payload: ImageGenerationPayload):
    try:
        # Convert the Pydantic model to a dictionary
        sdxl_payload = payload.dict(by_alias=True)
        
        # Get prediction from SageMaker endpoint
        sdxl_response = sdxl_model_predictor.predict(sdxl_payload)
        prompt = sdxl_payload.get('text_prompts', [{}])[0].get('text', 'image')
        # Decode the response and return the image
        filename = sanitize_filename(prompt)
        return decode_and_show(sdxl_response, s3_bucket, 'images/'+filename)
    except Exception as e:
        # Handle general exceptions
        raise HTTPException(status_code=500, detail=str(e))
    
def generate_llm_response(prompt: str):
    request = {
        'inputs': prompt,
        'parameters': DEFAULT_PARAMETERS
    }

    try:
        response = smr.invoke_endpoint(
            EndpointName=llm_endpoint_name,
            ContentType=DEFAULT_CONTENT_TYPE,
            Body=json.dumps(request)
        )
        return response['Body'].read().decode()
    except Exception as e:
        print(f"Error generating response: {str(e)}")
        raise

def sanitize_filename(text):
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Shorten the text if it's too long
    safe_prompt = (text[:10] + '..') if len(text) > 10 else text
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    hash_part = hashlib.md5(text.encode()).hexdigest()[:6]
    filename = f"{safe_prompt}_{timestamp}_{hash_part}.png"
    return filename

def upload_file_to_s3(response_bytes, filename):
    # Parse the JSON response to get the base64-encoded string
    response_json = json.loads(response_bytes)
    image_base64 = response_json['generated_image']

    # Decode the base64 string
    image_data = base64.b64decode(image_base64)

    # Create an image from the byte data
    image = Image.open(io.BytesIO(image_data))
    
    # Prepare the image data for streaming
    img_io = io.BytesIO()
    image.save(img_io, 'PNG')
    img_io.seek(0)
    image.close()
    # Upload the image to S3
    s3 = boto3.client('s3')
    s3.put_object(Body=image, Bucket=s3_bucket, Key='images/'+filename)

    # Return the S3 URL of the image
    s3_url = f"s3://{s3_bucket}/images/{image_name}"
    print(f"Uploaded image to {s3_url}")

def decode_and_show(response_bytes, s3_bucket, s3_key):
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
        img_io.seek(0)
        s3_client.put_object(Bucket=s3_bucket, Key=s3_key, Body=img_io, ContentType='image/png')
        print(f"Image uploaded to S3: {s3_bucket}/{s3_key}")
    except BotoCoreError as e:
        print(f"Failed to upload image to S3: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Generate the S3 URL for the uploaded image
    s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{s3_key}"
    return {"url": s3_url}

if os.getenv('AWS_EXECUTION_ENV') is not None:
    handler = Mangum(app)
else:
    handler = app
