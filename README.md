# Plains Cree Figurative Language Detection

Cross-lingual transfer of figurative language detection (literal / idiom / metaphor / simile)
into **Plains Cree (crk)**, a low-resource Indigenous language with no labeled figurative-language
data of its own.

## Approach

There's no Cree figurative-language training data, so the pipeline gets there via cross-lingual
transfer from English, in three stages:

1. **TLM** — warm up a multilingual encoder on a Plains Cree–English parallel corpus using the
   Translation Language Modeling objective ([Lample & Conneau, 2019](https://arxiv.org/abs/1901.07291)),
   so the encoder learns Cree↔English alignment.
2. **CLKD** — Cross-Lingual Knowledge Distillation: a frozen English teacher
   ([`deberta-v3-base-figurative`](https://huggingface.co/KonradBRG/deberta-v3-base-figurative), trained on
   VUA20 + MAGPIE + FLUTE) predicts soft figurative-language labels on the English side of the parallel
   corpus; the Cree-side student is trained to match those distributions via KL divergence.
3. **Calibrate** — a short low-LR fine-tune on Bloomfield (1934) Plains Cree sentences annotated by an
   LLM (DeepSeek), correcting for domain drift between the CLKD training corpus and real Cree text.

Optionally, a **Stage 0 (mono-MLM)** warmup on Cree-only text can precede TLM (`--mono-mlm`), and TLM
can add an InfoNCE alignment loss (`--contrastive-alpha`) — both are ablation conditions, see
[`jobs/ablation.sh`](jobs/ablation.sh).

Trained checkpoints are pushed to the [KonradBRG Hub namespace](https://huggingface.co/KonradBRG),
organized into collections by stage (TLM Encoders, CLKD Models, Calibrated Classifiers, Figurative
Language Classifiers).

## Repo layout

```
pipeline.py       End-to-end TLM → CLKD → Calibrate driver
tune.py           Single-trial runner for wandb hyperparameter sweeps
funcs.py          Thin wrappers around src/ used by pipeline.py, tune.py, and notebooks

src/
  scrapers/       Bloomfield, EdTeKLA, Ojibwe, itwewina, Okimāsis corpus scrapers
  mt/             Sentence splitting/alignment + TLM fine-tuning
  figurative/     Figurative classifier training, CLKD distillation, calibration, prediction
  annotate/       LLM (DeepSeek) annotation of Bloomfield sentences
  eda/            Exploratory data analysis / figure generation

scripts/
  data/           Corpus regeneration
  annotate/       Active-learning annotation loop
  train/          Sweep result summarization
  evaluate/       TLM intrinsic evaluation (perplexity, bitext retrieval)
  evals/          Full model evaluation sweep
  hub/            Model card + collection generation (also called from pipeline.py)
  viz/            Figure/table generation for the report

jobs/             SLURM job scripts (Tübingen cluster) — pipeline runs, ablations, sweeps, evals
sweeps/           wandb sweep configs (pipeline / clkd / calibrate / tlm)
data/, figures/   Corpora, annotations, checkpoints (local), generated figures/tables
```

## Setup

```bash
uv sync
huggingface-cli login       # or export HF_TOKEN=...
wandb login                 # or export WANDB_API_KEY=...
export DEEPSEEK_API_KEY=... # only needed for LLM annotation (src/annotate)
```

On the cluster, `jobs/*.sh` scripts `source .venv/bin/activate` and `uv sync` themselves — see
`PROJECT_ROOT` near the top of each script.

## Pipeline usage

```bash
# Full pipeline, xlm-mlm base
python pipeline.py --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm

# XLM-V needs a shorter max length to avoid OOM (large vocab)
python pipeline.py --base-model facebook/xlm-v-base --model-id xlm-v --max-length 128

# Glot500 is already multilingual — skip TLM warmup
python pipeline.py --base-model cis-lmu/glot500-base --model-id glot500 --skip-tlm

# Resume from an existing CLKD checkpoint, only run calibration
python pipeline.py --model-id xlm-mlm --skip-tlm --skip-clkd
```

Checkpoints land in `data/{stage}_{model-id}/`; the calibrated model is always pushed to the Hub as
`KonradBRG/{model-id}-plains-cree-en-calibrated` (TLM/CLKD checkpoints only if `--push-intermediates`
is passed). Each Hub push automatically gets a model card and is added to its collection — see
`scripts/hub/push_model_cards.py` / `scripts/hub/create_collections.py` for the full logic, or to
re-run it manually (e.g. to backfill a failed push).

On the cluster, submit via SLURM instead of running `pipeline.py` directly:

```bash
sbatch jobs/pipeline.sh --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm
bash jobs/ablation.sh                # submits all ablation conditions
bash jobs/ablation.sh --dry-run      # preview without submitting
```

### Hyperparameter sweeps

```bash
wandb sweep sweeps/pipeline.yaml            # prints <entity/project/sweep_id>
sbatch jobs/sweep_agent.sh <sweep_id>       # repeat for parallel agents
python tune.py --stage calibrate --show-best --sweep-id <sweep_id>
```

### Evaluation

```bash
sbatch jobs/tlm_eval.sh                                        # TLM intrinsic eval (perplexity, bitext retrieval)
python scripts/evals/eval_all.py                                # full model comparison sweep
python -c "from funcs import figurative_eval_idioms as f; f()"  # idiom golden-set eval
```

Figures and tables for the writeup are regenerated with `python scripts/viz/generate_figures.py`
into `figures/`.
