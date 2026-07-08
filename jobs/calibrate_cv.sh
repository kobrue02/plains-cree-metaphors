#!/bin/bash
# Submit 5 fold-holdout calibration runs (cross-validation) for ONE condition,
# reusing an existing CLKD/TLM/base checkpoint — no need to redo TLM/CLKD.
#
# Each run excludes one cv_folds.parquet fold from training and is never
# pushed to the Hub (see pipeline.py --holdout-fold); they exist purely so
# scripts/evals/eval_cv.py can aggregate genuinely held-out predictions across
# the whole annotation pool. Run scripts/data/build_cv_folds.py once first.
#
# Usage:
#   bash jobs/calibrate_cv.sh --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm-abl-full --calibrate-from KonradBRG/xlm-mlm-plains-cree-en-clkd
#   bash jobs/calibrate_cv.sh ... --dry-run
#   bash jobs/calibrate_cv.sh ... --calibrate-lr 5e-6 --calibrate-epochs 15   # forwarded as-is

DRY_RUN=0
ARGS=()
BASE_MODEL=""
MODEL_ID=""
CALIBRATE_FROM=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --calibrate-from) CALIBRATE_FROM="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$BASE_MODEL" ] || [ -z "$MODEL_ID" ] || [ -z "$CALIBRATE_FROM" ]; then
  echo "Usage: bash jobs/calibrate_cv.sh --base-model <hf-id> --model-id <id> --calibrate-from <ckpt> [extra pipeline.py flags] [--dry-run]"
  exit 1
fi

run_or_print() {
  local label="$1"; shift
  echo ""
  echo "── $label ──"
  echo "  $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

for FOLD in 0 1 2 3 4; do
  run_or_print "${MODEL_ID} fold ${FOLD}" \
    sbatch \
      --job-name="cv_${MODEL_ID}_f${FOLD}" \
      --time=00:30:00 \
      jobs/pipeline.sh \
        --base-model "$BASE_MODEL" \
        --model-id "$MODEL_ID" \
        --skip-tlm \
        --skip-clkd \
        --calibrate-from "$CALIBRATE_FROM" \
        --holdout-fold "$FOLD" \
        "${ARGS[@]}"
done

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete — no jobs submitted."
else
  echo "All 5 CV calibration folds submitted for ${MODEL_ID}.  Monitor with: squeue -u \$USER"
fi
