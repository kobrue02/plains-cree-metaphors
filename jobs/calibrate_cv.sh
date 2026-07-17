#!/bin/bash
# Submit 5 independent SLURM jobs, one per fold-holdout calibration run
# (cross-validation) for ONE condition, reusing an existing CLKD/TLM/base
# checkpoint — no need to redo TLM/CLKD.
#
# Each run excludes one cv_folds.parquet fold from training and is never
# pushed to the Hub (see pipeline.py --holdout-fold); they exist purely so
# scripts/evals/eval_cv.py can aggregate genuinely held-out predictions across
# the whole annotation pool. Run scripts/data/build_cv_folds.py once first.
#
# Used to run all 5 folds sequentially inside jobs/calibrate_cv_worker.sh's
# single allocation, trading cross-fold parallelism for paying queue-wait +
# environment setup once instead of 5 times. Switched to 5 independent jobs
# because gpu_a100_short's walltime cap (30 min) doesn't leave room to run 5
# folds sequentially in one allocation anymore.
#
# Usage:
#   bash jobs/calibrate_cv.sh --base-model FacebookAI/xlm-mlm-100-1280 --model-id xlm-mlm-abl-full --calibrate-from KonradBRG/xlm-mlm-plains-cree-en-clkd
#   bash jobs/calibrate_cv.sh ... --dependency afterok:12345          # wait for a prerequisite job (e.g. fresh CLKD training)
#   bash jobs/calibrate_cv.sh ... --dry-run
#   bash jobs/calibrate_cv.sh ... --calibrate-lr 5e-6 --calibrate-epochs 15   # forwarded as-is
#
# Prints the 5 submitted job IDs to stdout (one per line, via sbatch
# --parsable), so callers (e.g. jobs/ablation.sh) can capture them —
# everything else goes to stderr. Nothing currently chains off these IDs
# (calibration is the last stage per condition), so this is informational.

DRY_RUN=0
ARGS=()
BASE_MODEL=""
MODEL_ID=""
CALIBRATE_FROM=""
DEPENDENCY=""

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --base-model) BASE_MODEL="$2"; shift 2 ;;
    --model-id) MODEL_ID="$2"; shift 2 ;;
    --calibrate-from) CALIBRATE_FROM="$2"; shift 2 ;;
    --dependency) DEPENDENCY="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [ -z "$BASE_MODEL" ] || [ -z "$MODEL_ID" ] || [ -z "$CALIBRATE_FROM" ]; then
  echo "Usage: bash jobs/calibrate_cv.sh --base-model <hf-id> --model-id <id> --calibrate-from <ckpt> [--dependency afterok:<jobid>] [extra pipeline.py flags] [--dry-run]" >&2
  exit 1
fi

echo "" >&2
echo "── ${MODEL_ID} (5 independent fold jobs)${DEPENDENCY:+ [depends on $DEPENDENCY]} ──" >&2

for FOLD in 0 1 2 3 4; do
  SBATCH_ARGS=(--job-name="cv_${MODEL_ID}_fold${FOLD}" --parsable)
  if [ -n "$DEPENDENCY" ]; then
    SBATCH_ARGS+=(--dependency="$DEPENDENCY")
  fi

  echo "  sbatch ${SBATCH_ARGS[*]} jobs/calibrate_cv_worker.sh --base-model $BASE_MODEL --model-id $MODEL_ID --calibrate-from $CALIBRATE_FROM ${ARGS[*]} --holdout-fold $FOLD" >&2

  if [ "$DRY_RUN" -eq 0 ]; then
    sbatch "${SBATCH_ARGS[@]}" jobs/calibrate_cv_worker.sh \
      --base-model "$BASE_MODEL" \
      --model-id "$MODEL_ID" \
      --calibrate-from "$CALIBRATE_FROM" \
      "${ARGS[@]}" \
      --holdout-fold "$FOLD"
  fi
done
