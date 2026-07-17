#!/bin/bash
#SBATCH --job-name=FinalModel
#SBATCH --partition=gpu_a100_il
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# One job, one encoder: TLM(+InfoNCE, alpha=0.2 by default) -> CLKD -> SFT,
# with the SFT stage's reported metric coming from genuine 5-fold
# cross-validation, not a single held-out checkpoint — a single calibrated
# checkpoint's own in-training eval picks the epoch that maximizes score on
# the same fixed gold test set it then reports on (checkpoint-selection
# leakage), which is exactly what jobs/calibrate_cv.sh + eval_cv.py's 5-fold
# protocol avoids: each fold's model predicts only the fold it never trained
# or model-selected on, so the concatenated predictions are genuinely
# held-out. This job now does that in-line instead of as a separate manual
# step, so nothing gets reported off the wrong (biased) checkpoint by
# accident.
#
# TLM and CLKD run exactly once and are reused (from local disk) across all
# 5 fold-calibration passes plus the final full-data production pass — they
# don't depend on the label pool, only the fold-calibration and eval steps
# do.
#
# Stages:
#   1. TLM(+InfoNCE) + CLKD          — once, pushed to the Hub
#   2. 5x fold-holdout calibration   — local only (never pushed), one per
#                                       jobs/calibrate_cv_worker.sh's holdout
#                                       convention
#   3. scripts/evals/eval_cv.py      — aggregates the 5 folds' out-of-fold
#                                       predictions into the honest macro-F1
#                                       (data/figurative/eval_cv_gold.parquet)
#   4. One more calibration pass on the FULL data (no holdout) — this is the
#      actual checkpoint that gets pushed to the Hub for downstream/serving
#      use. Its own in-training eval number (calibrate_results.parquet) is
#      informational only (early-stopping diagnostic) — the number to CITE
#      is the one from step 3.
#
# Usage:
#   sbatch jobs/train_final_model.sh --encoder xlm-mlm
#   sbatch jobs/train_final_model.sh --encoder xlm-v --contrastive-alpha 0.1
#   sbatch jobs/train_final_model.sh --encoder glot500 --contrastive-alpha 0
#
# Requires scripts/data/build_cv_folds.py to have been run already (once,
# not per-encoder) — same requirement as jobs/calibrate_cv.sh.

PROJECT_ROOT=/home/tu/tu_tu/tu_zxoqp65/work/plains-cree-metaphors

module load devel/cuda/12.8
module load devel/python/3.13.3-llvm-19.1
echo "CUDA: $CUDA_HOME"

export CUDA_VISIBLE_DEVICES=0
export TORCH_EXTENSIONS_DIR=$PROJECT_ROOT/.cache/torch_extensions
export HF_HOME=$PROJECT_ROOT/.cache/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p $TORCH_EXTENSIONS_DIR $HF_HOME

source $PROJECT_ROOT/.venv/bin/activate
cd $PROJECT_ROOT
uv sync
mkdir -p logs

ENCODER=""
CONTRASTIVE_ALPHA=0.2
ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --encoder)            ENCODER="$2"; shift 2 ;;
    --contrastive-alpha)  CONTRASTIVE_ALPHA="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$ENCODER" ]; then
  echo "Usage: sbatch jobs/train_final_model.sh --encoder {xlm-mlm|xlm-v|glot500|xlm-r|xlm-r-large} [extra pipeline.py flags]" >&2
  exit 1
fi

EXTRA=()
case "$ENCODER" in
  xlm-mlm)     BASE_MODEL="FacebookAI/xlm-mlm-100-1280"; CV_LABEL="XLM-MLM (base pipeline)" ;;
  xlm-v)       BASE_MODEL="facebook/xlm-v-base"; EXTRA+=(--max-length 128); CV_LABEL="XLM-V (base pipeline)" ;;
  glot500)     BASE_MODEL="cis-lmu/glot500-base"; CV_LABEL="Glot500 (base pipeline)" ;;
  xlm-r)       BASE_MODEL="xlm-roberta-base"; CV_LABEL="XLM-R (base pipeline)" ;;
  xlm-r-large) BASE_MODEL="xlm-roberta-large"; CV_LABEL="XLM-R-large (base pipeline)" ;;
  *)
    echo "Unknown --encoder '$ENCODER' — choose from xlm-mlm, xlm-v, glot500, xlm-r, xlm-r-large" >&2
    exit 1
    ;;
esac
MODEL_ID="$ENCODER"

if [ ! -f data/figurative/cv_folds.parquet ]; then
  echo "data/figurative/cv_folds.parquet not found — run scripts/data/build_cv_folds.py first" >&2
  exit 1
fi

echo "=== Stage 1: TLM(alpha=$CONTRASTIVE_ALPHA) -> CLKD  ($ENCODER / $BASE_MODEL) ==="
python3 pipeline.py \
    --base-model "$BASE_MODEL" --model-id "$MODEL_ID" \
    --contrastive-alpha "$CONTRASTIVE_ALPHA" \
    --skip-calibrate --push-intermediates \
    --wandb-project fnlp-pipeline \
    "${EXTRA[@]}" "${ARGS[@]}"
if [ $? -ne 0 ]; then
  echo "TLM/CLKD stage failed." && exit 1
fi

echo "=== Stage 2: 5-fold CV calibration (honest, leakage-free eval) ==="
for FOLD in 0 1 2 3 4; do
  echo "--- fold $FOLD ---"
  python3 pipeline.py \
      --base-model "$BASE_MODEL" --model-id "$MODEL_ID" \
      --skip-tlm --skip-clkd \
      --holdout-fold "$FOLD" \
      --calibrate-lr 5e-6 --calibrate-epochs 15 \
      "${EXTRA[@]}" "${ARGS[@]}"
  if [ $? -ne 0 ]; then
    echo "Fold $FOLD failed." && exit 1
  fi
done

echo "=== Stage 3: aggregating out-of-fold predictions ==="
python3 scripts/evals/eval_cv.py --condition "$CV_LABEL"
if [ $? -ne 0 ]; then
  echo "eval_cv.py failed." && exit 1
fi

echo "=== Stage 4: production calibration (full data, pushed to Hub) ==="
python3 pipeline.py \
    --base-model "$BASE_MODEL" --model-id "$MODEL_ID" \
    --skip-tlm --skip-clkd \
    --calibrate-lr 5e-6 --calibrate-epochs 15 \
    "${EXTRA[@]}" "${ARGS[@]}"
if [ $? -ne 0 ]; then
  echo "Production calibration failed." && exit 1
fi

echo "Done."
echo "Honest (leakage-free) macro-F1: data/figurative/eval_cv_gold.parquet, condition '${CV_LABEL}'"
echo "Deployed checkpoint pushed to: KonradBRG/${MODEL_ID}-plains-cree-en-figurative"
echo "(calibrate_results.parquet's number for this run is informational only — an"
echo " early-stopping diagnostic on the production checkpoint, not the reported metric)"
