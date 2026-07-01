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
from dataclasses import dataclass, field

import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
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
    max_length:      int   = 256     # tokens per concatenated pair; xlm-r max = 512
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
    wandb_project:   str | None = None  # e.g. "fnlp-tlm"
    # infonce contrastive alignment — set alpha > 0 to enable
    contrastive_alpha:       float = 0.0   # weight on infonce term; 0 = pure tlm
    contrastive_temperature: float = 0.05  # softmax temperature for infonce


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
        with_contrastive: bool = False,
    ):
        self.examples: list[dict] = []
        mono_len = max_length // 2
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
            # xlm-r always emits zero token_type_ids; xlm uses them to distinguish segments
            if enc.get("token_type_ids") and any(enc["token_type_ids"]):
                example["token_type_ids"] = enc["token_type_ids"]

            if with_contrastive:
                # separate encodings so the model can't align via cross-lingual attention
                enc_src = tokenizer(src, max_length=mono_len, truncation=True, padding=False)
                enc_tgt = tokenizer(tgt, max_length=mono_len, truncation=True, padding=False)
                example["src_input_ids"]      = enc_src["input_ids"]
                example["src_attention_mask"] = enc_src["attention_mask"]
                example["tgt_input_ids"]      = enc_tgt["input_ids"]
                example["tgt_attention_mask"] = enc_tgt["attention_mask"]

            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class TLMContrastiveCollator:
    """Data collator that applies MLM masking to the concatenated pair and
    pads the individual src/tgt encodings without masking them."""

    def __init__(self, tokenizer, mlm_probability: float = 0.15):
        self._mlm = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability,
        )
        self._pad_id = tokenizer.pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        mlm_features = [
            {k: v for k, v in f.items()
             if k not in ("src_input_ids", "src_attention_mask",
                          "tgt_input_ids", "tgt_attention_mask")}
            for f in features
        ]
        batch = self._mlm(mlm_features)

        def _pad(seqs: list[list[int]], pad_val: int) -> torch.Tensor:
            return pad_sequence(
                [torch.tensor(s) for s in seqs],
                batch_first=True, padding_value=pad_val,
            )

        batch["src_input_ids"]      = _pad([f["src_input_ids"]      for f in features], self._pad_id)
        batch["src_attention_mask"] = _pad([f["src_attention_mask"]  for f in features], 0)
        batch["tgt_input_ids"]      = _pad([f["tgt_input_ids"]       for f in features], self._pad_id)
        batch["tgt_attention_mask"] = _pad([f["tgt_attention_mask"]  for f in features], 0)
        return batch


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


class TLMContrastiveTrainer(Trainer):
    """Trainer that adds an InfoNCE sentence-alignment loss to the MLM objective.

    total_loss = mlm_loss + alpha * infonce_loss
    """

    def __init__(self, *args, contrastive_alpha: float = 0.1,
                 contrastive_temperature: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_alpha       = contrastive_alpha
        self.contrastive_temperature = contrastive_temperature

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        src_ids  = inputs.pop("src_input_ids",      None)
        tgt_ids  = inputs.pop("tgt_input_ids",      None)
        src_mask = inputs.pop("src_attention_mask", None)
        tgt_mask = inputs.pop("tgt_attention_mask", None)

        outputs  = model(**inputs)
        mlm_loss = outputs.loss

        if self.contrastive_alpha > 0 and src_ids is not None:
            encoder = model.base_model  # strip the mlm head

            src_hidden = encoder(input_ids=src_ids, attention_mask=src_mask).last_hidden_state
            tgt_hidden = encoder(input_ids=tgt_ids, attention_mask=tgt_mask).last_hidden_state

            src_emb = F.normalize(_mean_pool(src_hidden, src_mask), dim=-1)
            tgt_emb = F.normalize(_mean_pool(tgt_hidden, tgt_mask), dim=-1)

            sim    = src_emb @ tgt_emb.T / self.contrastive_temperature
            labels = torch.arange(len(src_emb), device=sim.device)
            # symmetric infonce: average both directions
            info_nce = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2

            loss = mlm_loss + self.contrastive_alpha * info_nce
        else:
            loss = mlm_loss

        return (loss, outputs) if return_outputs else loss


class TLMFinetuner:
    """Fine-tune XLM-R (or any masked LM) on Cree-English parallel data via TLM.

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
        """Fine-tune on parallel sentence pairs using TLM loss. Returns the checkpoint path."""
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
            torch_dtype=torch.float32,  # load in fp32; trainer casts to bf16 if needed
        )
        if not getattr(model.config, "model_type", None):
            model.config.model_type = cfg.model_name.split("/")[-1].split("-")[0]

        use_contrastive = cfg.contrastive_alpha > 0
        if use_contrastive:
            print(f"[TLM] contrastive α={cfg.contrastive_alpha}  τ={cfg.contrastive_temperature}")

        train_ds = TLMDataset(train_pairs, tokenizer, cfg.max_length, with_contrastive=use_contrastive)
        dev_ds   = TLMDataset(dev_pairs,   tokenizer, cfg.max_length, with_contrastive=use_contrastive)

        collator = (
            TLMContrastiveCollator(tokenizer, mlm_probability=cfg.mlm_probability)
            if use_contrastive
            else DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=cfg.mlm_probability,
            )
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
            save_total_limit=2,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            report_to="wandb" if cfg.wandb_project else "none",
            **get_precision_kwargs(),
            seed=cfg.seed,
            **hub_kwargs,
        )

        trainer_cls = TLMContrastiveTrainer if use_contrastive else Trainer
        trainer_kwargs = (
            {"contrastive_alpha": cfg.contrastive_alpha,
             "contrastive_temperature": cfg.contrastive_temperature}
            if use_contrastive else {}
        )
        trainer = trainer_cls(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            data_collator=collator,
            **trainer_kwargs,
        )

        trainer.train()
        trainer.save_model(cfg.output_dir)
        tokenizer.save_pretrained(cfg.output_dir)
        print(f"[TLM] saved to {cfg.output_dir}")

        if cfg.hub_model_id:
            trainer.push_to_hub()
            print(f"[TLM] pushed to hub: {cfg.hub_model_id}")

        return cfg.output_dir