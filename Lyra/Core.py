from openai import OpenAI
import os
from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv()

# Set cache environment variables (adjust the path if necessary)
API_KEY = os.getenv("DEEPSEEK_API_KEY")

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)