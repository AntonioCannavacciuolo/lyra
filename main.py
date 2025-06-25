# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("DEESEEK_API_KEY")
SUPER_PROMPT_FILE = r"C:\Users\canna\OneDrive\Desktop\Projects\AssetsAI\prompt.txt"

def load_super_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        raise RuntimeError(f"Super prompt file not found: {file_path}")

SUPER_PROMPT = load_super_prompt(SUPER_PROMPT_FILE)

client = OpenAI(api_key=API_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": SUPER_PROMPT},
        {"role": "user", "content": "Ciao, come ti chiami? che cosa sai fare?"},
    ],
    stream=False
)

print(response.choices[0].message.content)