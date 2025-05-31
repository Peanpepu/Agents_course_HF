import datasets
from langchain.docstore.document import Document
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path="../../.env")

# Get the Openweather API key from environment variables
OPENWEATHER_APIKEY = os.getenv("OPENWEATHER_APIKEY")

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

search_tool = DuckDuckGoSearchRun()
results = search_tool.invoke("Who's the current President of France?")
print(results)

def get_weather_info(location: str) -> str:
    """Fetches real-time weather information for a given location."""
    if not OPENWEATHER_APIKEY:
        return "Weather API key not found. Please set OPENWEATHER_API_KEY in your .env file."

    try:
        # Call OpenWeatherMap API
        url = (
            f"https://api.openweathermap.org/data/2.5/weather?q={location}"
            f"&appid={OPENWEATHER_APIKEY}&units=metric"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        condition = data["weather"][0]["description"].capitalize()
        temp_c = data["main"]["temp"]
        city = data["name"]
        country = data["sys"]["country"]

        return f"Weather in {city}, {country}: {condition}, {temp_c}°C"
    except requests.exceptions.RequestException as e:
        return f"Error fetching weather data: {str(e)}"
    except KeyError:
        return "Unexpected response format from weather API."

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches real-time weather information for a given location. Input should be a city name (e.g., 'London')."
)

