from gradio_client import Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../../.env")

# Get the Openweather API key from environment variables
HF_APIKEY = os.getenv("HF_TOKEN")

client = Client("agents-course/decoding_visualizer", hf_token="HF_APIKEY")
result = client.predict(
    input_text="The square root of 50 is",
    api_name="/get_beam_search_html"
)
print(result)

