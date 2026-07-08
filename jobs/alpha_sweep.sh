#!/bin/bash
# Sweep the InfoNCE contrastive-alignment weight (--contrastive-alpha) during TLM,
# to see how alignment strength trades off against final calibrated performance.
#
# alpha=0.0  and alpha=0.1 already exist as the "full" and "tlm_contrastive"
# ablation conditions (jobs/ablation.sh) — this script only submits the new
# points needed to fill out the curve.
#
# Each run is a full TLM(contrastive) -> CLKD -> Calibrate pipeline from scratch
# (~3-4h in practice).
#
# Usage (run from project root on the cluster):
#   bash jobs/alpha_sweep.sh
#   bash jobs/alpha_sweep.sh --dry-run     # print commands without submitting

DRY_RUN=0
for arg in "$@"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

BASE_MODEL="FacebookAI/xlm-mlm-100-1280"
MODEL_ID="xlm-mlm"

run_or_print() {
  local label="$1"; shift
  echo ""
  echo "── $label ──"
  echo "  $*"
  if [ "$DRY_RUN" -eq 0 ]; then
    "$@"
  fi
}

for ALPHA in 0.05 0.15 0.2 0.3 0.4 0.5 0.75 1.0; do
  SUFFIX=$(echo "$ALPHA" | tr '.' 'p')   # 0.05 -> 0p05, matches Hub-safe naming
  run_or_print "alpha=${ALPHA} (TLM+InfoNCE -> CLKD -> calibrate)" \
    sbatch \
      --job-name="alpha_${SUFFIX}" \
      --time=04:00:00 \
      jobs/pipeline.sh \
        --base-model "$BASE_MODEL" \
        --model-id "${MODEL_ID}-alpha-${SUFFIX}" \
        --contrastive-alpha "$ALPHA" \
        --calibrate-lr 5e-6 \
        --calibrate-epochs 15
done

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete — no jobs submitted."
else
  echo "All alpha-sweep jobs submitted.  Monitor with: squeue -u \$USER"
fi
