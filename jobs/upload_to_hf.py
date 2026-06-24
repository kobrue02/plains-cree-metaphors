"""
Upload the fine-tuned awesome-align model to the Hugging Face Hub.

Usage (from project root, with awesome-align venv active):
    source vendor/awesome-align-venv/bin/activate
    python jobs/upload_to_hf.py --repo <your-hf-username/repo-name>

You will be prompted to log in if no token is cached.
Run `huggingface-cli login` beforehand to avoid the prompt.
"""

import argparse
import os
import sys

MODEL_DIR = "data/awesome_align/model"

MODEL_CARD = """\
---
language:
  - cr
  - en
tags:
  - word-alignment
  - awesome-align
  - cree
  - low-resource
  - morphology
  - xlm-roberta
base_model: xlm-roberta-base
license: apache-2.0
---

# Cree–English Word Aligner (awesome-align / XLM-RoBERTa)

Fine-tuned from `xlm-roberta-base` using
[awesome-align](https://github.com/neulab/awesome-align) on a parallel
Cree–English corpus derived from Leonard Bloomfield's *Plains Cree Texts*
(~5,400 paragraph pairs, ~7,000 sentence pairs after splitting).

## Training

Objectives: Translation Language Modeling (TLM) + Contrastive Objective (CO).
No gold word alignments were used — training is fully unsupervised given the
parallel corpus.

```
awesome-train \\
  --model_name_or_path xlm-roberta-base \\
  --train_tlm --train_co \\
  --num_train_epochs 3 \\
  --learning_rate 2e-5
```

## Usage

```python
# Install awesome-align first: pip install git+https://github.com/neulab/awesome-align
import awesome_align

# Input: one sentence pair per line, src ||| tgt
# Output: Pharaoh-format alignments (i-j per line)
```

Or via CLI:
```bash
awesome-align \\
  --model_name_or_path <this-repo> \\
  --data_file input.txt \\
  --output_file alignments.txt \\
  --extraction softmax
```

## Language

Plains Cree (`cr`) is a polysynthetic Algonquian language spoken in Canada.
This model is intended for morpheme-level analysis and downstream use in the
INFOALIGN character-level segmentation pipeline.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HF repo id, e.g. username/cree-en-awesome-align")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    if not os.path.isdir(args.model_dir):
        print(f"Model directory not found: {args.model_dir}")
        print("Run jobs/train_awesome_align.sh first.")
        sys.exit(1)

    api = HfApi()

    # Log in if no cached token
    try:
        api.whoami()
    except Exception:
        print("No cached HF token found — logging in...")
        login()

    # Create repo if it doesn't exist
    api.create_repo(repo_id=args.repo, exist_ok=True, private=args.private)
    print(f"Repository: https://huggingface.co/{args.repo}")

    # Write model card
    card_path = os.path.join(args.model_dir, "README.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD)

    # Upload everything in the model directory
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo,
        commit_message="Upload fine-tuned Cree-English awesome-align model",
    )

    print(f"\nDone. Model available at https://huggingface.co/{args.repo}")
    print(f"Use it with: --model_name_or_path {args.repo}")


if __name__ == "__main__":
    main()
