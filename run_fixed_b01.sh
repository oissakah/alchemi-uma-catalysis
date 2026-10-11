#!/bin/bash
#SBATCH --job-name=alchemi_b01
#SBATCH --account=ock
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=fixed_b01_%j.log
#SBATCH --error=fixed_b01_%j.err

set -euo pipefail

PROJECT="/lustre/work/ock/oissakah10/alchemi-uma-catalysis-milestone2"
ENV="$PROJECT/.conda-env"

cd "$PROJECT"

module purge
module load anaconda

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "============================================================"
echo " ALCHEMI + UMA — Fixed B1 / 500 Structures"
echo " Job ID: ${SLURM_JOB_ID:-unknown}"
echo " Node: $(hostname)"
echo " Started: $(date)"
echo "============================================================"

echo
echo "=== Environment ==="
which python
python --version

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is not available.")

print("GPU:", torch.cuda.get_device_name(0))
PY

echo
echo "=== Fixed B1 benchmark ==="

python scripts/05_benchmark_fixed.py \
    --manifest results/pilot_500/dataset_manifest.csv \
    --lmdb-root data/oc20dense_ood_val \
    --batch-sizes 1 \
    --replicates 3 \
    --limit 500 \
    --results-dir results/production_500/fixed_b01

echo
echo "============================================================"
echo " FIXED B1 BENCHMARK COMPLETED"
echo " Finished: $(date)"
echo "============================================================"
