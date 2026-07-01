#!/bin/bash
# Submit all four ablation conditions as separate SLURM jobs.
#
# The four conditions differ only in which checkpoint feeds calibration:
#
#   full      TLM → CLKD → Calibrate  (uses existing checkpoints)
#   no_tlm    base → CLKD → Calibrate (CLKD retrained from base model)
#   no_clkd   TLM → Calibrate         (calibrate from TLM output)
#   neither   base → Calibrate        (calibrate from base model directly)
#
# Usage (run from project root on the cluster):
#   bash jobs/ablation.sh
#   bash jobs/ablation.sh --dry-run     # print commands without submitting

DRY_RUN=0
for arg in "$@"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

BASE_MODEL="FacebookAI/xlm-mlm-100-1280"
MODEL_ID="xlm-mlm"
TLM_CKPT="data/tlm_xlm-mlm"
TLM_HUB="KonradBRG/xlm-mlm-plains-cree-en-tlm"
CLKD_CKPT="data/clkd_xlm-mlm"
CLKD_HUB="KonradBRG/xlm-mlm-plains-cree-en-clkd"

# Resolve checkpoint paths: prefer local dir, fall back to Hub.
if [ -d "$TLM_CKPT" ]; then
  TLM_RESOLVED="$TLM_CKPT"
else
  TLM_RESOLVED="$TLM_HUB"
fi

if [ -d "$CLKD_CKPT" ]; then
  CLKD_RESOLVED="$CLKD_CKPT"
else
  CLKD_RESOLVED="$CLKD_HUB"
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

# ── Condition A: full pipeline (TLM → CLKD → calibrate) ─────────────────────
# Calibrate from the existing CLKD checkpoint.  We use --calibrate-from to
# point at the real CLKD ckpt explicitly, since --model-id differs from the
# original run and the default path would resolve to a non-existent dir.
run_or_print "full (calibrate from existing CLKD checkpoint)" \
  sbatch \
    --job-name=abl_full \
    --time=01:00:00 \
    jobs/pipeline.sh \
      --base-model "$BASE_MODEL" \
      --model-id "${MODEL_ID}-abl-full" \
      --skip-tlm \
      --skip-clkd \
      --calibrate-from "$CLKD_RESOLVED" \
      --calibrate-lr 5e-6 \
      --calibrate-epochs 15

# ── Condition B: no TLM (base → CLKD → calibrate) ───────────────────────────
# CLKD starts from the raw base model instead of TLM output.
# This is the most expensive ablation (~2h CLKD + ~30min calibrate).
run_or_print "no_tlm (CLKD from base model, then calibrate)" \
  sbatch \
    --job-name=abl_no_tlm \
    --time=04:00:00 \
    jobs/pipeline.sh \
      --base-model "$BASE_MODEL" \
      --model-id "${MODEL_ID}-abl-no-tlm" \
      --skip-tlm \
      --clkd-from "$BASE_MODEL" \
      --calibrate-lr 5e-6 \
      --calibrate-epochs 15

# ── Condition C: no CLKD (TLM → calibrate) ───────────────────────────────────
# Skip CLKD entirely; calibrate directly from TLM output.
run_or_print "no_clkd (calibrate from TLM checkpoint)" \
  sbatch \
    --job-name=abl_no_clkd \
    --time=01:00:00 \
    jobs/pipeline.sh \
      --base-model "$BASE_MODEL" \
      --model-id "${MODEL_ID}-abl-no-clkd" \
      --skip-tlm \
      --skip-clkd \
      --calibrate-from "$TLM_RESOLVED" \
      --calibrate-lr 5e-6 \
      --calibrate-epochs 15

# ── Condition D: neither (base → calibrate) ───────────────────────────────────
# No TLM, no CLKD — calibrate directly from the base model.
run_or_print "neither (calibrate from base model directly)" \
  sbatch \
    --job-name=abl_neither \
    --time=01:00:00 \
    jobs/pipeline.sh \
      --base-model "$BASE_MODEL" \
      --model-id "${MODEL_ID}-abl-neither" \
      --skip-tlm \
      --skip-clkd \
      --calibrate-from "$BASE_MODEL" \
      --calibrate-lr 5e-6 \
      --calibrate-epochs 15

echo ""
if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run complete — no jobs submitted."
else
  echo "All ablation jobs submitted.  Monitor with: squeue -u \$USER"
fi
