# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI

API_KEY = "sk-74cfbb585ec54655ad3de5e54b5adeb5"
MODEL_ID = "deepseek-v4-pro"

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.deepseek.com")

def call_deepseek(prompt: str) -> tuple[str, str]:
    response = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        stream=False,
        extra_body={"thinking": {"type": "enabled", "reasoning_effort": "high"}},
    )
    msg = response.choices[0].message
    reasoning = (msg.reasoning_content or "").strip()
    label = (msg.content or "").strip().lower()
    return reasoning, label
