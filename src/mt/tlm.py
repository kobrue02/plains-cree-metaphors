"""
Fine-tune a masked language model on parallel data using Translation Language Modeling (TLM).

TLM (Lample & Conneau, 2019): sentence pairs are concatenated, tokens are masked from
both sides, and the model is trained to fill blanks by attending to both languages.
Running this on Cree-English pairs pulls Cree representations into the same embedding
space as English.
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.device import get_device, get_precision_kwargs


@dataclass
class TLMConfig:
    """Hyperparameters for TLM fine-tuning."""
    model_name:      str   = "xlm-roberta-base"
    output_dir:      str   = "data/tlm_model"
    max_length:      int   = 256     # tokens per concatenated pair; XLM-R max = 512
    mlm_probability: float = 0.15
    batch_size:      int   = 16
    grad_accum:      int   = 2
    epochs:          int   = 3
    learning_rate:   float = 2e-5
    warmup_ratio:    float = 0.06
    weight_decay:    float = 0.01
    dev_ratio:       float = 0.05
    seed:            int   = 42
    hub_model_id:    str | None = None
    wandb_project:   str | None = None  # e.g. "fnlp-tlm"; set to None to disable


class TLMDataset(Dataset):
    """Tokenised parallel sentence pairs for TLM.

    Each example encodes the pair as a single sequence:
        <s> cree_tokens </s></s> english_tokens </s>
    (XLM-R sentence-pair format). Masking is applied by the DataCollator at
    training time so masks are re-sampled each epoch.
    """

    def __init__(
        self,
        pairs:      list[tuple[str, str]],
        tokenizer,
        max_length: int,
    ):
        self.examples: list[dict] = []
        for src, tgt in pairs:
            enc = tokenizer(
                src, tgt,
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            example = {
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
            }
            # XLM-R produces all-zero token_type_ids (useless); XLM uses them to
            # distinguish the two language segments, so keep them when present.
            if enc.get("token_type_ids") and any(enc["token_type_ids"]):
                example["token_type_ids"] = enc["token_type_ids"]
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class TLMFinetuner:
    """Fine-tune XLM-R (or any masked LM) on Cree-English parallel data via TLM.

    Parameters
    ----------
    config : TLMConfig, optional
        Hyperparameters; defaults to ``TLMConfig()``.

    Example
    -------
    >>> prep   = AwesomeAlignPrep(df).split_to_sentences()
    >>> high_q = prep[prep.confidence > 0.6]          # filter noisy pairs
    >>> ft     = TLMFinetuner(TLMConfig(epochs=5))
    >>> ckpt   = ft.fit(high_q)
    # Use ckpt as encoder in metaphor experiments:
    >>> cfg = ExperimentConfig(encoder=ckpt, experiment_name="tlm_encoder")
    """

    def __init__(self, config: TLMConfig | None = None):
        self.config = config or TLMConfig()

    def fit(
        self,
        df:      pd.DataFrame,
        src_col: str = "text_cree",
        tgt_col: str = "text_en",
        dev_df:  pd.DataFrame | None = None,
    ) -> str:
        """Fine-tune on parallel sentence pairs using TLM loss.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with ``src_col`` and ``tgt_col`` columns.
            Pass a confidence-filtered slice for cleaner training signal.
        src_col, tgt_col : str
            Column names for the two languages.
        dev_df : pd.DataFrame, optional
            Explicit dev set. If None, ``config.dev_ratio`` of ``df`` is
            held out automatically.

        Returns
        -------
        str
            Path to the saved checkpoint.
        """
        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        pairs = list(zip(df[src_col].astype(str), df[tgt_col].astype(str)))

        if dev_df is None:
            rng = random.Random(cfg.seed)
            rng.shuffle(pairs)
            split       = max(1, int(len(pairs) * (1 - cfg.dev_ratio)))
            train_pairs = pairs[:split]
            dev_pairs   = pairs[split:]
        else:
            train_pairs = pairs
            dev_pairs   = list(zip(
                dev_df[src_col].astype(str), dev_df[tgt_col].astype(str)
            ))

        device = get_device()
        print(f"[TLM] model    : {cfg.model_name}")
        print(f"[TLM] device   : {device}")
        print(f"[TLM] train    : {len(train_pairs):,} pairs")
        print(f"[TLM] dev      : {len(dev_pairs):,} pairs")
        print(f"[TLM] output   : {cfg.output_dir}")

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        model     = AutoModelForMaskedLM.from_pretrained(
            cfg.model_name,
            torch_dtype=torch.float32,  # load in FP32; Trainer casts to BF16 if needed
        )
        if not getattr(model.config, "model_type", None):
            model.config.model_type = cfg.model_name.split("/")[-1].split("-")[0]

        train_ds = TLMDataset(train_pairs, tokenizer, cfg.max_length)
        dev_ds   = TLMDataset(dev_pairs,   tokenizer, cfg.max_length)

        collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=cfg.mlm_probability,
        )

        if cfg.wandb_project:
            os.environ["WANDB_PROJECT"] = cfg.wandb_project

        hub_kwargs = (
            {"push_to_hub": True, "hub_model_id": cfg.hub_model_id}
            if cfg.hub_model_id else {}
        )

        args = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            per_device_train_batch_size=cfg.batch_size,
            per_device_eval_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            learning_rate=cfg.learning_rate,
            warmup_ratio=cfg.warmup_ratio,
            weight_decay=cfg.weight_decay,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            report_to="wandb" if cfg.wandb_project else "none",
            **get_precision_kwargs(),
            seed=cfg.seed,
            **hub_kwargs,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=collator,
        )

        trainer.train()
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"[TLM] saved to {cfg.output_dir}")

        if cfg.hub_model_id:
            trainer.push_to_hub()
            print(f"[TLM] pushed to hub: {cfg.hub_model_id}")

        return cfg.output_dir