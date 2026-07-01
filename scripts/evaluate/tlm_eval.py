"""
TLM intrinsic evaluation: held-out perplexity + bitext retrieval.

Reports per-model:
  - MLM pseudo-perplexity on held-out Cree and English text
  - Bitext retrieval Recall@1, Recall@5, MRR  (Cree→EN and EN→Cree)

Pass --baseline to compare against the raw base model side-by-side.

Usage:
  python scripts/evaluate/tlm_eval.py \\
      --model data/tlm_xlm-mlm \\
      --baseline FacebookAI/xlm-mlm-100-1280

  # Use the Hub checkpoint instead of local dir:
  python scripts/evaluate/tlm_eval.py \\
      --model KonradBRG/xlm-mlm-plains-cree-en-tlm \\
      --baseline FacebookAI/xlm-mlm-100-1280 \\
      --n 300
"""

from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

SENTENCES_FILE = "data/sentences.parquet"
DEFAULT_N      = 500   # evaluation pairs (randomly sampled from held-out split)
BATCH_SIZE     = 32
MAX_LENGTH     = 128
MASK_PROB      = 0.15
SEED           = 42


# ── Embedding ─────────────────────────────────────────────────────────────────

def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(model: AutoModel, tokenizer, sentences: list[str], device: str) -> np.ndarray:
    model.eval()
    vecs = []
    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=False)
        # works for both AutoModel and AutoModelForMaskedLM
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") \
                 else out.hidden_states[-1]
        vecs.append(_mean_pool(hidden, enc["attention_mask"]).cpu().float().numpy())
    return np.concatenate(vecs, axis=0)


# ── Bitext retrieval ──────────────────────────────────────────────────────────

def retrieval_metrics(src_embs: np.ndarray, tgt_embs: np.ndarray) -> dict[str, float]:
    src = src_embs / np.linalg.norm(src_embs, axis=1, keepdims=True).clip(1e-9)
    tgt = tgt_embs / np.linalg.norm(tgt_embs, axis=1, keepdims=True).clip(1e-9)
    sim   = src @ tgt.T                               # (N, N)
    ranks = np.argsort(-sim, axis=1)                  # descending similarity
    gold  = np.arange(len(src))
    rank_of_gold = (ranks == gold[:, None]).argmax(1) + 1  # 1-indexed

    return {
        "R@1":  float((rank_of_gold == 1).mean()),
        "R@5":  float((rank_of_gold <= 5).mean()),
        "MRR":  float((1.0 / rank_of_gold).mean()),
    }


def bitext_retrieval(model, tokenizer, cree: list[str], en: list[str], device: str) -> dict:
    cree_embs = embed(model, tokenizer, cree, device)
    en_embs   = embed(model, tokenizer, en,   device)
    return {
        "Cree→EN": retrieval_metrics(cree_embs, en_embs),
        "EN→Cree": retrieval_metrics(en_embs,   cree_embs),
    }


# ── Perplexity ────────────────────────────────────────────────────────────────

def pseudo_perplexity(model: AutoModelForMaskedLM, tokenizer, sentences: list[str], device: str) -> float:
    """Mean MLM loss (≈ pseudo log-perplexity) over 15%-masked inputs."""
    model.eval()
    special_ids = torch.tensor(tokenizer.all_special_ids, device=device)
    total_loss, total_masked = 0.0, 0

    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LENGTH, return_tensors="pt",
        ).to(device)

        ids    = enc["input_ids"].clone()
        labels = ids.clone()

        # Mask 15% of non-special, non-padding tokens
        mask_candidates = enc["attention_mask"].bool() \
                          & ~torch.isin(ids, special_ids)
        mask_positions  = mask_candidates & (torch.rand_like(ids.float()) < MASK_PROB)

        masked_ids = ids.clone()
        masked_ids[mask_positions] = tokenizer.mask_token_id

        # Only compute loss on masked positions
        labels[~mask_positions] = -100

        with torch.no_grad():
            loss = model(
                input_ids=masked_ids,
                attention_mask=enc["attention_mask"],
                labels=labels,
            ).loss

        n = int(mask_positions.sum())
        if n > 0:
            total_loss   += loss.item() * n
            total_masked += n

    return float(np.exp(total_loss / max(total_masked, 1)))


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(model_name: str, cree: list[str], en: list[str], device: str) -> dict:
    print(f"\n  Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Use ForMaskedLM so we get both hidden states (for retrieval) and MLM head (for perplexity)
    mlm_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)

    # Perplexity
    print("  Computing perplexity ...")
    ppl_cree = pseudo_perplexity(mlm_model, tokenizer, cree, device)
    ppl_en   = pseudo_perplexity(mlm_model, tokenizer, en,   device)

    # Retrieval — embeddings from the base model (strip MLM head)
    print("  Computing bitext retrieval ...")
    base_model = mlm_model.base_model  # removes the LM head
    retr = bitext_retrieval(base_model, tokenizer, cree, en, device)

    return {
        "ppl_cree": ppl_cree,
        "ppl_en":   ppl_en,
        "retrieval": retr,
    }


def print_results(label: str, res: dict) -> None:
    print(f"\n{'─'*58}")
    print(f"  {label}")
    print(f"{'─'*58}")
    print(f"  Perplexity   Cree : {res['ppl_cree']:.2f}")
    print(f"               EN   : {res['ppl_en']:.2f}")
    for direction, m in res["retrieval"].items():
        print(f"  {direction:<10}  R@1={m['R@1']:.3f}  R@5={m['R@5']:.3f}  MRR={m['MRR']:.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model",    required=True,
                   help="TLM checkpoint path or Hub ID")
    p.add_argument("--baseline", default=None,
                   help="Base model to compare against (e.g. FacebookAI/xlm-mlm-100-1280)")
    p.add_argument("--sentences-file", default=SENTENCES_FILE)
    p.add_argument("--n",        type=int, default=DEFAULT_N,
                   help="Number of held-out pairs to evaluate on (default: 500)")
    p.add_argument("--seed",     type=int, default=SEED)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load and split data — use a fixed held-out split so results are comparable
    df = pd.read_parquet(args.sentences_file)
    df = df.dropna(subset=["text_cree", "text_en"])
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    eval_df = df.tail(args.n)  # last N rows after shuffle = held-out
    cree = eval_df["text_cree"].astype(str).tolist()
    en   = eval_df["text_en"].astype(str).tolist()
    print(f"Evaluating on {len(cree)} held-out pairs from {args.sentences_file}")

    models_to_eval = [(args.model, "TLM")]
    if args.baseline:
        models_to_eval.append((args.baseline, "Baseline"))

    all_results = {}
    for model_name, label in models_to_eval:
        all_results[label] = evaluate(model_name, cree, en, device)
        print_results(label, all_results[label])

    # Delta summary when comparing
    if args.baseline and "Baseline" in all_results and "TLM" in all_results:
        base = all_results["Baseline"]
        tlm  = all_results["TLM"]
        print(f"\n{'─'*58}")
        print("  Δ TLM − Baseline")
        print(f"{'─'*58}")
        print(f"  Perplexity   Cree : {tlm['ppl_cree'] - base['ppl_cree']:+.2f}")
        print(f"               EN   : {tlm['ppl_en']   - base['ppl_en']:+.2f}")
        for direction in ("Cree→EN", "EN→Cree"):
            for k in ("R@1", "R@5", "MRR"):
                delta = tlm["retrieval"][direction][k] - base["retrieval"][direction][k]
                print(f"  {direction:<10}  {k}: {delta:+.3f}")
    print()


if __name__ == "__main__":
    main()
