#!/bin/bash
#SBATCH --job-name=alchemi_prod500
#SBATCH --account=ock
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=prod500_%j.log
#SBATCH --error=prod500_%j.err

set -euo pipefail

PROJECT="/lustre/work/ock/oissakah10/alchemi-uma-catalysis-milestone2"
ENV="$PROJECT/.conda-env"

cd "$PROJECT"

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV"

echo "============================================================"
echo " ALCHEMI + UMA — 500-Structure Production Benchmark"
echo " Job ID: ${SLURM_JOB_ID:-unknown}"
echo " Node: $(hostname)"
echo " Started: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# Environment / GPU checks
# ------------------------------------------------------------

python --version
nvidia-smi

python - <<'PY'
import torch
import nvalchemi

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("ALCHEMI:", nvalchemi.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable.")

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2,
    ),
    "GB",
)
PY

# ------------------------------------------------------------
# Syntax checks
# ------------------------------------------------------------

python -m py_compile scripts/05_benchmark_fixed.py
python -m py_compile scripts/06_benchmark_inflight.py

echo "PASS: benchmark scripts compile."

# ------------------------------------------------------------
# Common paths
# ------------------------------------------------------------

MANIFEST="results/pilot_500/dataset_manifest.csv"
LMDB="data/oc20dense_ood_val"

mkdir -p results/production_500

# ============================================================
# 1. FIXED B4 PRODUCTION BASELINE
# ============================================================

echo
echo "============================================================"
echo " 1/4 — Fixed B4 production baseline"
echo " 500 structures, 3 replicates"
echo "============================================================"

python scripts/05_benchmark_fixed.py \
    --manifest "$MANIFEST" \
    --lmdb-root "$LMDB" \
    --batch-sizes 4 \
    --replicates 3 \
    --limit 500 \
    --results-dir results/production_500/fixed_b04

echo
echo "Fixed B4 complete: $(date)"

# ============================================================
# 2. INFLIGHT B4 / 360 ATOMS
# ============================================================

echo
echo "============================================================"
echo " 2/4 — Inflight B4 / 360 atoms"
echo " 500 structures, 3 replicates"
echo "============================================================"

python scripts/06_benchmark_inflight.py \
    --manifest "$MANIFEST" \
    --lmdb-root "$LMDB" \
    --limit 500 \
    --max-batch-size 4 \
    --max-atoms 360 \
    --refill-frequency 1 \
    --fmax 0.05 \
    --max-steps 300 \
    --replicates 3 \
    --results-dir results/production_500/inflight_b04_a360

echo
echo "Inflight B4 complete: $(date)"

# ============================================================
# 3. INFLIGHT B8 / 720 ATOMS
# ============================================================

echo
echo "============================================================"
echo " 3/4 — Inflight B8 / 720 atoms"
echo " 500 structures, 3 replicates"
echo "============================================================"

python scripts/06_benchmark_inflight.py \
    --manifest "$MANIFEST" \
    --lmdb-root "$LMDB" \
    --limit 500 \
    --max-batch-size 8 \
    --max-atoms 720 \
    --refill-frequency 1 \
    --fmax 0.05 \
    --max-steps 300 \
    --replicates 3 \
    --results-dir results/production_500/inflight_b08_a720

echo
echo "Inflight B8 complete: $(date)"

# ============================================================
# 4. INFLIGHT B16 / 1440 ATOMS
# ============================================================

echo
echo "============================================================"
echo " 4/4 — Inflight B16 / 1440 atoms"
echo " 500 structures, 3 replicates"
echo "============================================================"

python scripts/06_benchmark_inflight.py \
    --manifest "$MANIFEST" \
    --lmdb-root "$LMDB" \
    --limit 500 \
    --max-batch-size 16 \
    --max-atoms 1440 \
    --refill-frequency 1 \
    --fmax 0.05 \
    --max-steps 300 \
    --replicates 3 \
    --results-dir results/production_500/inflight_b16_a1440

echo
echo "Inflight B16 complete: $(date)"

# ============================================================
# Final summary
# ============================================================

echo
echo "============================================================"
echo " PRODUCTION BENCHMARK COMPLETED"
echo " Finished: $(date)"
echo "============================================================"

echo
echo "Results:"
echo "  results/production_500/fixed_b04"
echo "  results/production_500/inflight_b04_a360"
echo "  results/production_500/inflight_b08_a720"
echo "  results/production_500/inflight_b16_a1440"
