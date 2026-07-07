"""
TLM intrinsic metrics: MLM pseudo-perplexity and cross-lingual bitext retrieval.

Shared by scripts/evaluate/tlm_eval.py (post-hoc comparison against a baseline)
and src/mt/tlm.py (per-epoch logging during training).
"""

from __future__ import annotations
import numpy as np
import torch

BATCH_SIZE = 32
MAX_LENGTH = 128
MASK_PROB  = 0.15


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def embed(model, tokenizer, sentences: list[str], device: str) -> np.ndarray:
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


def retrieval_metrics(src_embs: np.ndarray, tgt_embs: np.ndarray) -> dict[str, float]:
    src = src_embs / np.linalg.norm(src_embs, axis=1, keepdims=True).clip(1e-9)
    tgt = tgt_embs / np.linalg.norm(tgt_embs, axis=1, keepdims=True).clip(1e-9)
    sim   = src @ tgt.T
    ranks = np.argsort(-sim, axis=1)
    gold  = np.arange(len(src))
    rank_of_gold = (ranks == gold[:, None]).argmax(1) + 1  # 1-indexed

    return {
        "R@1": float((rank_of_gold == 1).mean()),
        "R@5": float((rank_of_gold <= 5).mean()),
        "MRR": float((1.0 / rank_of_gold).mean()),
    }


def bitext_retrieval(model, tokenizer, cree: list[str], en: list[str], device: str) -> dict:
    cree_embs = embed(model, tokenizer, cree, device)
    en_embs   = embed(model, tokenizer, en,   device)
    return {
        "Cree→EN": retrieval_metrics(cree_embs, en_embs),
        "EN→Cree": retrieval_metrics(en_embs,   cree_embs),
    }


def pseudo_perplexity(model, tokenizer, sentences: list[str], device: str) -> float:
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

        mask_candidates = enc["attention_mask"].bool() \
                          & ~torch.isin(ids, special_ids)
        mask_positions  = mask_candidates & (torch.rand_like(ids.float()) < MASK_PROB)

        masked_ids = ids.clone()
        masked_ids[mask_positions] = tokenizer.mask_token_id

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
