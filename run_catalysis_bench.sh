#!/bin/bash
#SBATCH --job-name=alchemi_bench
#SBATCH --account=ock
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=bench_%j.log
#SBATCH --error=bench_%j.err

set -euo pipefail

PROJECT="/lustre/work/ock/oissakah10/alchemi-uma-catalysis-milestone2"
ENV="$PROJECT/.conda-env"

cd "$PROJECT"

echo "============================================================"
echo " ALCHEMI + UMA Milestone 2"
echo " Job ID: ${SLURM_JOB_ID:-unknown}"
echo " Node: $(hostname)"
echo " Started: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Load environment
# ------------------------------------------------------------

module purge
module load anaconda

# Make conda available in a non-interactive SLURM shell
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$ENV"

echo
echo "=== Python environment ==="
echo "Python executable: $(which python)"
python --version

# ------------------------------------------------------------
# 2. Verify Python development headers
# ------------------------------------------------------------

echo
echo "=== Checking Python.h ==="

python - <<'PY'
import sys
import sysconfig
from pathlib import Path

include = Path(sysconfig.get_paths()["include"])
header = include / "Python.h"

print("Python:", sys.executable)
print("Include directory:", include)
print("Python.h:", header)
print("Exists:", header.exists())

if not header.exists():
    raise RuntimeError(
        f"Python development header is missing: {header}"
    )

print("PASS: Python.h is available.")
PY

# ------------------------------------------------------------
# 3. GPU diagnostics
# ------------------------------------------------------------

echo
echo "=== NVIDIA GPU ==="

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("PyTorch cannot access the GPU.")

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
# 4. Test Triton before loading UMA
# ------------------------------------------------------------

echo
echo "=== Triton JIT test ==="

python - <<'PY'
import torch
from triton.runtime import driver

assert torch.cuda.is_available()

print("GPU:", torch.cuda.get_device_name(0))
print("Triton target:", driver.active.get_current_target())
print("PASS: Triton CUDA driver initialized.")
PY

# ------------------------------------------------------------
# 5. Numerical equivalence
# ------------------------------------------------------------

echo
echo "============================================================"
echo " Script 03: Numerical Equivalence"
echo "============================================================"

python scripts/03_numerical_equivalence.py \
    --manifest results/pilot_500/dataset_manifest.csv \
    --lmdb-root data/oc20dense_ood_val \
    --n 10

# ------------------------------------------------------------
# 6. Small fixed-batch benchmark
# ------------------------------------------------------------

echo
echo "============================================================"
echo " Script 05: Warmed Fixed-Batch Benchmark"
echo "============================================================"

python scripts/05_benchmark_fixed.py \
    --manifest results/pilot_500/dataset_manifest.csv \
    --lmdb-root data/oc20dense_ood_val \
    --batch-sizes 1 2 4 8 16 \
    --replicates 3 \
    --limit 20
echo
echo "============================================================"
echo " ALL TESTS COMPLETED SUCCESSFULLY"
echo " Finished: $(date)"
echo "============================================================"