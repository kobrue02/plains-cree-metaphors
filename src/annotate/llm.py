"""
NVIDIA-hosted model access (https://integrate.api.nvidia.com), for comparing
other models against DeepSeek on the same figurative-language annotation
procedure. Uses the same shared prompt as src/annotate/deepseek.py (see
src/annotate/figurative_prompt.py) — no per-provider prompt drift.

NVIDIA's endpoint is OpenAI-compatible, so this uses the OpenAI SDK (matching
src/annotate/deepseek.py) rather than raw requests — the SDK retries transient
errors (429/5xx/timeouts) internally, which a hand-rolled requests.post did not.
"""

from __future__ import annotations
import os
from pathlib import Path

from openai import OpenAI

from src.annotate.figurative_prompt import SYSTEM_PROMPT, parse_label


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


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=_load_api_key(),
    max_retries=6,          # SDK backs off internally on 429/5xx/timeouts
    timeout=240.0,
)


def call_llm(model_id: str, system: str, prompt: str,
             max_tokens: int = 16384, reasoning: bool = True) -> tuple[str, str]:
    """Returns (reasoning, content). Set reasoning=False for plain instruct
    models that don't support reasoning_effort (e.g. meta/llama-3.3-70b-instruct) —
    sending it to a model that doesn't understand it risks a 400, not a no-op.

    max_tokens defaults high (was 8192) because reasoning_effort="high" can burn
    the entire budget on the reasoning chain before ever emitting the final
    answer, especially with the longer patched SYSTEM_PROMPT (figurative_prompt.py)
    — observed as an empty response silently defaulting to "literal" in
    make_annotate_fn below, which would systematically bias a large annotation
    run toward the majority class rather than failing loudly."""
    kwargs = {"extra_body": {"reasoning_effort": "high"}} if reasoning else {}
    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.70,
        top_p=1.00,
        stream=False,
        **kwargs,
    )
    msg = resp.choices[0].message
    reasoning_text = (getattr(msg, "reasoning_content", None) or "").strip()
    content        = (msg.content or "").strip()
    return reasoning_text, content


def make_annotate_fn(model_id: str, reasoning: bool = True, retries: int = 3):
    """Returns a (prompt) -> (label, reasoning) callable for the given NVIDIA
    model, matching src/annotate/deepseek.py's _deepseek_annotate signature —
    pass it as annotate_fn to scripts/annotate/deepseek_label_pool.py's
    annotate_pool()/​_annotate_one() to reuse that same pool-annotation loop.
    Pass reasoning=False for plain instruct models (see call_llm).

    An empty/unparseable response (reasoning_effort burning the whole token
    budget before emitting an answer) is NOT silently defaulted to "literal"
    here — that would get written into the cache as if it were a genuine
    result, permanently defeating annotate_pool()'s resume-safe retry (a
    sentence the cache thinks is "done" never gets revisited). Instead this
    retries a few times (temperature=0.70 means a retry can genuinely get a
    different, complete answer) and raises if still empty, so the caller's
    existing "don't cache, retry next run" exception handling takes over."""
    def annotate(prompt: str) -> tuple[str, str]:
        for attempt in range(retries + 1):
            reasoning_text, content = call_llm(model_id, SYSTEM_PROMPT, prompt, reasoning=reasoning)
            label = parse_label(content)
            if label is not None:
                full_reasoning = f"{reasoning_text}\n\n--- final answer ---\n{content}".strip()
                return label, full_reasoning
            print(f"  [{model_id}] empty/unparseable response on attempt {attempt + 1}/{retries + 1} "
                  f"(content={content[:80]!r})" + ("; retrying" if attempt < retries else "; giving up"))
        raise RuntimeError(f"{model_id}: empty/unparseable response after {retries + 1} attempts")
    return annotate
