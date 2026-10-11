#!/usr/bin/env bash
#SBATCH --job-name=alchemi-m2-500
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
source .venv/bin/activate

python scripts/04_benchmark_serial.py \
  --manifest results/pilot_500/dataset_manifest.csv \
  --lmdb-root data/oc20dense_ood_val

python scripts/05_benchmark_fixed.py \
  --manifest results/pilot_500/dataset_manifest.csv \
  --lmdb-root data/oc20dense_ood_val \
  --batch-sizes 1 2 4 8 16 32

python scripts/06_benchmark_inflight.py \
  --manifest results/pilot_500/dataset_manifest.csv \
  --lmdb-root data/oc20dense_ood_val \
  --max-batch-size 16

python scripts/08_build_accuracy_csv.py \
  --targets data/oc20dense_val_targets.pkl \
  --relaxation-results results/pilot_500/relaxation_results.csv || true

python analysis/make_figures.py
