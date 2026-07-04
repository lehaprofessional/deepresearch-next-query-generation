import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

base_url = os.getenv("OPENAI_BASE_URL")
api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

if not base_url:
    raise ValueError("OPENAI_BASE_URL is not set in .env")

client = OpenAI(
    base_url=base_url,
    api_key=api_key,
)

models = client.models.list()

print("Available models:")
for model in models.data:
    print("-", model.id)