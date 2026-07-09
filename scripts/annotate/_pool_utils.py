"""
Shared pool-loading helper for the DeepSeek-label / predict / agreement-eval
scripts, so all three apply the exact same filtering and end up scoring the
same sentence set.
"""

from __future__ import annotations
import pandas as pd

POOL_FILE      = "data/bloomfield_texts_sentences.parquet"
EXCLUDE_SOURCE = "bloomfield_1930"


def load_pool(
    path:           str        = POOL_FILE,
    exclude_source: str | None = EXCLUDE_SOURCE,
    limit:          int | None = None,
) -> pd.DataFrame:
    """Load the sentence pool, optionally excluding a source_file, deduped by text_cree."""
    pool = pd.read_parquet(path).dropna(subset=["text_cree", "text_en"]).copy()
    if exclude_source:
        pool = pool[pool["source_file"] != exclude_source].copy()
    pool["text_cree"] = pool["text_cree"].str.strip()
    pool["text_en"]   = pool["text_en"].str.strip()
    pool = pool.drop_duplicates(subset=["text_cree"]).reset_index(drop=True)
    if limit:
        pool = pool.head(limit)
    return pool
