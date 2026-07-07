import os
import requests, base64
from pathlib import Path


invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False


def _load_api_key() -> str:
    if key := os.environ.get("NVIDIA_API_KEY"):
        return key
    for directory in [Path(__file__).parent, *Path(__file__).parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("NVIDIA_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "NVIDIA_API_KEY not found. Set it in the environment or in a .env file."
    )


API_KEY = _load_api_key()
MODEL_ID = "mistralai/mistral-medium-3.5-128b"


def format_prompt(text_cree: str, text_en: str, footnote_en: str) -> str:
    prompt = f"""
You are a helpful assistant for annotating Plains Cree texts, and your specialty are figurative language phenomena, such as metaphors, similes, idioms, and proverbs.
You are given a paragraph in Plains Cree, its English translation, and an optional footnote. 
Read all three carefully. If the footnote contains information about figurative language in the paragraph, use that information to inform your annotation.
Here is the paragraph to analyze:
Cree: {text_cree}
English: {text_en}
Footnote: {footnote_en if footnote_en else "None"}
Please return only one word, either "simile", "metaphor", "idiom", "proverb", or "none" if there is no figurative language.
    """

    return prompt

def call_llm(prompt: str) -> tuple[str, str]:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/json"
    }

    payload = {
        "model": MODEL_ID,
        "reasoning_effort": "high",
        "messages": [{"role":"user","content":prompt}],
        "max_tokens": 16384,
        "temperature": 0.70,
        "top_p": 1.00,
        "stream": stream
    }

    r = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    response = r.json()
    if "choices" not in response:
        raise RuntimeError(f"Unexpected API response: {response}")
    msg = response["choices"][0]["message"]
    reasoning = msg.get("reasoning") or ""
    label = msg.get("content", "").strip().lower()
    return reasoning, label