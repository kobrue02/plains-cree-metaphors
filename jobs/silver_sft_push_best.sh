#!/bin/bash
#SBATCH --job-name=SilverSFTPushBest
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=konrad-rudolf.brueggemann@student.uni-tuebingen.de

# Submitted by jobs/silver_sft_freeze_sweep.sh with --dependency=afterok on
# every n-sweep job, so it only runs once all 17 have finished (successfully
# or not — see that script's note on afterok vs afterany).

set -euo pipefail
python scripts/train/push_best_silver_sft.py
