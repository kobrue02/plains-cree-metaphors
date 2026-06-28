import os
from pathlib import Path
from openai import OpenAI

MODEL_ID = "deepseek-v4-pro"


def _load_api_key() -> str:
    if key := os.environ.get("DEEPSEEK_API_KEY"):
        return key
    # Walk up from this file to find a .env
    for directory in [Path(__file__).parent, *Path(__file__).parents]:
        env_file = directory / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("DEEPSEEK_API_KEY="):
                    return line.split("=", 1)[1].strip()
    raise RuntimeError(
        "DEEPSEEK_API_KEY not found. Set it in the environment or in a .env file."
    )


client = OpenAI(api_key=_load_api_key(), base_url="https://api.deepseek.com")

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
