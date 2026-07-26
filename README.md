# Figurative Language Corpus Construction in Plains Cree

The first figurative-language corpus for **Plains Cree (nêhiyawêwin)** containing literals /
idioms / metaphors / similes, built by treating Leonard Bloomfield's 1930s documentary
commentary as gold evidence and scaling it with a dictionary-grounded LLM annotation
procedure. Two independent methods for producing a Cree figurative-language classifier are
compared: cross-lingual distillation from an English teacher (no Cree labels at all), and
direct fine-tuning on the LLM's silver labels.

## Corpus

- **Gold (228 sentences)** — Bloomfield's *Plains Cree Texts* (1934) footnotes, read by
  DeepSeek-V4-Pro alongside the Cree text, English translation, and a numbered sentence
  list, then manually reviewed and corrected against the original footnote (`src/annotate`,
  `scripts/annotate/annotate_bloomfield.py`, `scripts/annotate/apply_gold_corrections.py`).
  Plus 9 examples from a Cree-worldview thesis and a community idiom collection. Never used
  for training — reserved entirely for evaluation.
- **Silver (10,619 sentences)** — the remaining parallel-corpus sentences, labeled by a
  dictionary-grounded procedure inspired by the Metaphor Identification Procedure (MIP):
  each sentence is judged against itwêwina dictionary entries for its content words
  (`src/scrapers/itwewina.py`) plus its English gloss (`scripts/annotate/deepseek_label_pool.py`).
  Production labels come from Qwen3.5-122B-A10B; four other LLMs (DeepSeek-V4-Pro,
  GPT-OSS-120B, Llama-3.3-70B, Mistral Medium 3.5) are run through the same procedure for
  comparison (`scripts/annotate/ablation_llm_grounding.py`,
  `scripts/evals/llm_annotator_comparison.py`) via the NVIDIA NIM API.

Both subsets, plus a Cree–English parallel corpus (Bloomfield, EdTeKLA, Okimāsis, and an
Ojibwe bitext used only for TLM pretraining) are released on Hugging Face:
[`KonradBRG/plains-cree-figurative`](https://huggingface.co/datasets/KonradBRG/plains-cree-figurative)
(`scripts/hub/push_dataset.py`).

## Two pipelines, one shared encoder

Both start from the same TLM-adapted XLM-100 encoder (`sec:tlm`) and never see the gold set
during training.

1. **TLM** — continue-pretrain a multilingual encoder on the Cree–English parallel corpus
   with the Translation Language Modeling objective
   ([Lample & Conneau, 2019](https://arxiv.org/abs/1901.07291)), optionally adding an
   InfoNCE sentence-alignment loss (`--contrastive-alpha`) (`src/mt/tlm.py`).
2. **CLKD** (`sec:clkd-pipeline`) — zero-shot: no labeled Cree data at all. A frozen English
   teacher ([`deberta-v3-base-figurative`](https://huggingface.co/KonradBRG/deberta-v3-base-figurative),
   trained on VUA20 + MAGPIE + FLUTE) scores the English side of the parallel corpus; the
   Cree-side student is trained to match those soft labels via KL divergence
   (`src/figurative/distill.py`). Run via `pipeline.py --skip-calibrate` (see below).
3. **Silver-SFT** (`sec:silver-pipeline`) — fine-tune the TLM-adapted encoder directly on
   the full silver pool with a hierarchical head (binary literal-vs-figurative, then a
   conditional 3-way idiom/metaphor/simile head), swept over how many lower encoder layers
   are frozen (`src/figurative/silver_sft.py`, `scripts/train/train_silver.py`).

`pipeline.py` also has a `calibrate` stage (fine-tuning on Bloomfield's LLM-derived labels,
optionally under 5-fold CV — `src/figurative/calibrate.py`, `jobs/calibrate_cv.sh`) from an
earlier multi-encoder comparison track. It is not part of either pipeline as reported in the
current paper draft; `--skip-calibrate` stops after CLKD.

Trained checkpoints live in the
[Plains Cree Figurative Language](https://huggingface.co/collections/KonradBRG/plains-cree-figurative-language-6a65c3a9d6f1f86d3bef479d)
Hub collection (TLM encoder, CLKD student, Silver-SFT classifiers, English teacher).

## Repo layout

```
pipeline.py       TLM -> CLKD (-> optional Calibrate) driver
tune.py           Single-trial runner for wandb hyperparameter sweeps
funcs.py          Thin wrappers around src/ used by pipeline.py, tune.py, and notebooks

src/
  scrapers/       Bloomfield, EdTeKLA, itwewina dictionary scrapers
  parsers/        Okimāsis, Ojibwe, Bloomfield-1930 text parsers
  mt/             Sentence splitting/alignment + TLM (+InfoNCE) fine-tuning
  figurative/     Teacher training, CLKD distillation, Silver-SFT (hierarchical head), calibration, prediction
  annotate/       LLM clients (DeepSeek, NVIDIA NIM) + the shared dictionary-grounded prompt
  eda/            Exploratory data analysis / figure generation

scripts/
  data/           Corpus regeneration, CV fold construction
  annotate/       Gold annotation, silver pool labeling, grounding ablation, agreement eval
  train/          English teacher training, Silver-SFT CLI, sweep summarization
  evaluate/       TLM intrinsic evaluation (perplexity, bitext retrieval)
  evals/          Classifier results table, CV eval, LLM-annotator comparison table
  hub/            Model cards, collection sync, dataset push
  viz/            Figure/table generation for the paper

jobs/             SLURM job scripts (Tübingen cluster) — pipeline runs, Silver-SFT sweeps, ablations, evals
sweeps/           wandb sweep configs (pipeline / clkd / calibrate / tlm)
data/, figures/   Corpora, annotations, checkpoints (local), generated figures/tables
```

## Setup

```bash
uv sync
hf auth login                # or export HF_TOKEN=...
wandb login                  # or export WANDB_API_KEY=...
export DEEPSEEK_API_KEY=...  # gold annotation (scripts/annotate/annotate_bloomfield.py)
export NVIDIA_API_KEY=...    # silver labeling + LLM comparison (src/annotate/llm.py, NIM endpoints)
```

On the cluster, `jobs/*.sh` scripts `source .venv/bin/activate` and `uv sync` themselves — see
`PROJECT_ROOT` near the top of each script.

## Usage

### CLKD pipeline (zero-shot, no Cree labels)

```bash
python pipeline.py --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm --skip-calibrate

# resume from an existing TLM checkpoint, only run CLKD
python pipeline.py --model-id xlm-mlm --skip-tlm --skip-calibrate
```

On the cluster: `sbatch jobs/pipeline.sh --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm --skip-calibrate`.

### Silver-SFT pipeline (trained on silver labels)

```bash
python scripts/train/train_silver.py \
    --checkpoint KonradBRG/xlm-mlm-plains-cree-en-tlm \
    --hierarchical --freeze-n-layers 6 \
    --hub-model-id KonradBRG/xlm-mlm-plains-cree-en-silver-sft
```

On the cluster: `bash jobs/silver_sft_freeze_sweep.sh` (sweeps `freeze-n-layers` over 0..16,
pushes only the winner via `scripts/train/push_best_silver_sft.py`); `jobs/silver_sft_no_tlm.sh`
runs the raw-encoder (no TLM) ablation.

### Annotation

```bash
python scripts/annotate/annotate_bloomfield.py       # gold: DeepSeek reads Bloomfield's footnotes
python scripts/annotate/deepseek_label_pool.py       # silver: dictionary-grounded pool labeling
python scripts/annotate/ablation_llm_grounding.py --model <nim-model-id>  # 5-LLM comparison / grounding ablation
```

### Evaluation

```bash
sbatch jobs/tlm_eval.sh                              # TLM intrinsic eval (pseudo-perplexity, bitext retrieval)
python scripts/evals/eval_all.py                     # classifier eval against the 228-sentence gold set
python scripts/evals/llm_annotator_comparison.py     # builds Table: LLM annotator comparison
```

Figures and tables for the paper are regenerated with `python scripts/viz/generate_figures.py`
into `figures/`.
