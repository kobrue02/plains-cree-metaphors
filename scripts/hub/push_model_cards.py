"""
Push model cards to the HuggingFace Hub for all trained checkpoints.
Run locally: python scripts/push_model_cards.py
Requires: huggingface-cli login (or HF_TOKEN env var set)
"""

from __future__ import annotations
from huggingface_hub import ModelCard, ModelCardData

LANGUAGE = "crk"  # Plains Cree ISO 639-3
AUTHOR   = "KonradBRG"

# ── Model definitions ─────────────────────────────────────────────────────────

MODELS = [

    # ── TLM checkpoints ───────────────────────────────────────────────────────
    dict(
        repo_id    = f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-tlm",
        base_model = "FacebookAI/xlm-mlm-100-1280",
        tags       = ["translation-language-modeling", "plains-cree", "low-resource"],
        task       = "fill-mask",
        summary    = "XLM-MLM-100-1280 fine-tuned with the Translation Language Modeling (TLM) objective on a Plains Cree–English parallel corpus.",
        details    = """\
## Training

The model was fine-tuned using the TLM objective introduced in [Lample & Conneau (2019)](https://arxiv.org/abs/1901.07291).
Cree–English sentence pairs are concatenated and masked jointly, encouraging the model to use cross-lingual context when predicting masked tokens.

**Training data:** Bloomfield (1934) Plains Cree texts (scraped and sentence-aligned) + EdTeKLA parallel corpus (~9,000 sentence pairs total)
**Epochs:** 15
**Hardware:** 1× NVIDIA A100 40 GB

## Intended use

This checkpoint serves as the student encoder foundation for Cross-Lingual Knowledge Distillation (CLKD) of figurative language classifiers into Plains Cree.
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/glot500-base-plains-cree-en-tlm",
        base_model = "cis-lmu/glot500-base",
        tags       = ["translation-language-modeling", "plains-cree", "low-resource"],
        task       = "fill-mask",
        summary    = "Glot500-base fine-tuned with the TLM objective on a Plains Cree–English parallel corpus.",
        details    = """\
## Training

Fine-tuned using the TLM objective on ~9,000 Plains Cree–English sentence pairs.
TLM fine-tuning adapts [Glot500](https://huggingface.co/cis-lmu/glot500-base)'s broad multilingual representations to the Cree–English cross-lingual alignment needed for downstream CLKD.

**Epochs:** 15 | **Hardware:** 1× NVIDIA A100 40 GB

## Intended use

Student encoder foundation for CLKD figurative language transfer into Plains Cree.
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-v-base-plains-cree-en-tlm",
        base_model = "facebook/xlm-v-base",
        tags       = ["translation-language-modeling", "plains-cree", "low-resource"],
        task       = "fill-mask",
        summary    = "XLM-V-base fine-tuned with the TLM objective on a Plains Cree–English parallel corpus.",
        details    = """\
## Training

Fine-tuned using the TLM objective on ~9,000 Plains Cree–English sentence pairs.
[XLM-V](https://huggingface.co/facebook/xlm-v-base)'s 901K-token vocabulary provides strong subword coverage for morphologically rich low-resource languages like Plains Cree.

**Epochs:** 15 | **Batch size:** 16 | **Max length:** 128 (reduced due to large vocabulary memory footprint)
**Hardware:** 1× NVIDIA A100 40 GB

## Intended use

Student encoder foundation for CLKD figurative language transfer into Plains Cree.
""",
    ),

    # ── Figurative classifiers (English teacher / baselines) ──────────────────
    dict(
        repo_id    = f"{AUTHOR}/deberta-v3-base-figurative",
        base_model = "microsoft/deberta-v3-base",
        tags       = ["figurative-language", "text-classification", "english"],
        task       = "text-classification",
        language   = "en",
        summary    = "DeBERTa-v3-base fine-tuned on VUA20 + MAGPIE + FLUTE for 4-class figurative language detection (literal / idiom / metaphor / simile).",
        details    = """\
## Training

Fine-tuned on a combination of:
- **VUA20** — VU Amsterdam Metaphor Corpus (metaphor)
- **MAGPIE** — idiom dataset
- **FLUTE** — figurative language understanding

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
**Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

## Intended use

Frozen English teacher in the CLKD pipeline for Plains Cree figurative language detection.
The teacher produces soft-label distributions on English translations; these are distilled into a student encoder operating on Cree text.
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-r-plains-cree-en-tlm-figurative",
        base_model = "xlm-roberta-base",
        tags       = ["figurative-language", "text-classification", "plains-cree", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-R-base fine-tuned for 4-class figurative language detection, using a TLM-adapted encoder for Plains Cree.",
        details    = """\
## Training

Encoder: XLM-RoBERTa-base with TLM fine-tuning on Plains Cree–English pairs.
Classifier head trained on VUA20 + MAGPIE + FLUTE (English only).

**Labels:** `literal`, `idiom`, `metaphor`, `simile`
**Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-r-large-plains-cree-en-tlm-figurative",
        base_model = "xlm-roberta-large",
        tags       = ["figurative-language", "text-classification", "plains-cree", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-R-large fine-tuned for 4-class figurative language detection, using a TLM-adapted encoder for Plains Cree.",
        details    = """\
## Training

Encoder: XLM-RoBERTa-large with TLM fine-tuning on Plains Cree–English pairs.
Classifier head trained on VUA20 + MAGPIE + FLUTE (English only).

**Labels:** `literal`, `idiom`, `metaphor`, `simile`
**Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-figurative",
        base_model = "FacebookAI/xlm-mlm-100-1280",
        tags       = ["figurative-language", "text-classification", "plains-cree", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-MLM-100-1280 fine-tuned for 4-class figurative language detection, using a TLM-adapted encoder for Plains Cree.",
        details    = """\
## Training

Encoder: XLM-MLM-100-1280 with TLM fine-tuning on Plains Cree–English pairs.
Classifier head trained on VUA20 + MAGPIE + FLUTE (English only).

**Labels:** `literal`, `idiom`, `metaphor`, `simile`
**Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB
""",
    ),

    # ── CLKD models ───────────────────────────────────────────────────────────
    dict(
        repo_id    = f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-clkd-frozen12",
        base_model = f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-tlm",
        tags       = ["figurative-language", "text-classification", "plains-cree", "clkd", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-MLM-100-1280 adapted for Plains Cree figurative language detection via Cross-Lingual Knowledge Distillation (CLKD), with layers 0–11 frozen.",
        details    = """\
## Method

Inspired by [Cross-Lingual Knowledge Distillation (ACL 2023)](https://aclanthology.org/2023.findings-acl.885/).

**CLKD pipeline:**
1. Teacher: `KonradBRG/deberta-v3-base-figurative` (frozen, English)
2. Student base: `KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm` (TLM-warmed)
3. Layers 0–11 frozen to preserve cross-lingual alignment from TLM; layers 12–15 + classification head trained

The teacher predicts soft label distributions on English translations of Plains Cree sentences.
The student is trained to match these distributions on the Cree side via KL divergence.

**Training data:** ~9,000 Cree–English parallel sentence pairs
**Temperature:** 2.0 | **Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-clkd-full",
        base_model = f"{AUTHOR}/xlm-mlm-100-1280-plains-cree-en-tlm",
        tags       = ["figurative-language", "text-classification", "plains-cree", "clkd", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-MLM-100-1280 adapted for Plains Cree figurative language detection via CLKD, all layers trainable.",
        details    = """\
## Method

Same CLKD setup as `xlm-mlm-100-1280-plains-cree-en-clkd-frozen12` but with all 16 layers trainable.

**Teacher:** `KonradBRG/deberta-v3-base-figurative`
**Student base:** `KonradBRG/xlm-mlm-100-1280-plains-cree-en-tlm`
**Temperature:** 2.0 | **Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/glot500-base-plains-cree-en-clkd-direct",
        base_model = "cis-lmu/glot500-base",
        tags       = ["figurative-language", "text-classification", "plains-cree", "clkd", "low-resource"],
        task       = "text-classification",
        summary    = "Glot500-base adapted for Plains Cree figurative language detection via CLKD, without TLM warmup.",
        details    = """\
## Method

CLKD applied directly to the pretrained [Glot500-base](https://huggingface.co/cis-lmu/glot500-base) checkpoint (no TLM warmup).
Tests whether Glot500's broad multilingual representations are sufficient for cross-lingual figurative language transfer into Plains Cree without explicit alignment fine-tuning.

**Teacher:** `KonradBRG/deberta-v3-base-figurative`
**Temperature:** 2.0 | **Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/glot500-base-plains-cree-en-clkd-tlm",
        base_model = f"{AUTHOR}/glot500-base-plains-cree-en-tlm",
        tags       = ["figurative-language", "text-classification", "plains-cree", "clkd", "low-resource"],
        task       = "text-classification",
        summary    = "Glot500-base adapted for Plains Cree figurative language detection via CLKD, with TLM warmup.",
        details    = """\
## Method

CLKD applied to Glot500-base after TLM fine-tuning on Plains Cree–English pairs.

**Teacher:** `KonradBRG/deberta-v3-base-figurative`
**Student base:** `KonradBRG/glot500-base-plains-cree-en-tlm`
**Temperature:** 2.0 | **Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-v-base-plains-cree-en-clkd-direct",
        base_model = "facebook/xlm-v-base",
        tags       = ["figurative-language", "text-classification", "plains-cree", "clkd", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-V-base adapted for Plains Cree figurative language detection via CLKD, without TLM warmup.",
        details    = """\
## Method

CLKD applied directly to the pretrained [XLM-V-base](https://huggingface.co/facebook/xlm-v-base) checkpoint (no TLM warmup).
XLM-V's 901K-token vocabulary provides strong morphological coverage for Plains Cree.

**Teacher:** `KonradBRG/deberta-v3-base-figurative`
**Temperature:** 2.0 | **Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
""",
    ),

    dict(
        repo_id    = f"{AUTHOR}/xlm-v-base-plains-cree-en-clkd-tlm",
        base_model = f"{AUTHOR}/xlm-v-base-plains-cree-en-tlm",
        tags       = ["figurative-language", "text-classification", "plains-cree", "clkd", "low-resource"],
        task       = "text-classification",
        summary    = "XLM-V-base adapted for Plains Cree figurative language detection via CLKD, with TLM warmup.",
        details    = """\
## Method

CLKD applied to XLM-V-base after TLM fine-tuning on Plains Cree–English pairs.

**Teacher:** `KonradBRG/deberta-v3-base-figurative`
**Student base:** `KonradBRG/xlm-v-base-plains-cree-en-tlm`
**Temperature:** 2.0 | **Epochs:** 10 | **Hardware:** 1× NVIDIA A100 40 GB

**Labels:** `literal` (0), `idiom` (1), `metaphor` (2), `simile` (3)
""",
    ),
]


# ── Card template ─────────────────────────────────────────────────────────────

def make_card(m: dict) -> ModelCard:
    language = m.get("language", LANGUAGE)
    card_data = ModelCardData(
        language      = language if isinstance(language, list) else [language],
        license       = "cc-by-4.0",
        base_model    = m["base_model"],
        tags          = m.get("tags", []),
        pipeline_tag  = m["task"],
    )
    content = f"""\
---
{ card_data.to_yaml() }
---

# { m["repo_id"].split("/")[-1] }

{ m["summary"] }

{ m["details"] }

## Citation

If you use this model, please cite the associated thesis/paper (TBD).

## Data

Training data includes:
- [Bloomfield (1934) *Plains Cree Texts*](https://bloomfield.kiyanaw.net) (scraped and sentence-aligned)
- [EdTeKLA Indigenous Languages Corpora](https://github.com/EdTeKLA/IndigenousLanguages_Corpora)
"""
    return ModelCard(content)


# ── Push ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from huggingface_hub import HfApi
    api = HfApi()

    for m in MODELS:
        repo = m["repo_id"]
        try:
            # Check repo exists before pushing
            api.repo_info(repo_id=repo, repo_type="model")
            card = make_card(m)
            card.push_to_hub(repo)
            print(f"✓  {repo}")
        except Exception as exc:
            print(f"✗  {repo}  — {exc}")
