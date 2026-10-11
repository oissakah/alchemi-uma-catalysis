#!/usr/bin/env bash
#SBATCH --job-name=alchemi-m2-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
source .venv/bin/activate

python scripts/00_check_environment.py
python scripts/03_numerical_equivalence.py \
  --manifest results/pilot_500/dataset_manifest.csv \
  --lmdb-root data/oc20dense_ood_val \
  --n 10

python scripts/05_benchmark_fixed.py \
  --manifest results/pilot_500/dataset_manifest.csv \
  --lmdb-root data/oc20dense_ood_val \
  --batch-sizes 1 2 4 \
  --limit 20
