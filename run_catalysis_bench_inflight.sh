#!/bin/bash
#SBATCH --job-name=alchemi_inflight
#SBATCH --account=ock
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=inflight_%j.log
#SBATCH --error=inflight_%j.err

set -euo pipefail

PROJECT="/lustre/work/ock/oissakah10/alchemi-uma-catalysis-milestone2"
ENV="$PROJECT/.conda-env"

cd "$PROJECT"

echo "============================================================"
echo " ALCHEMI + UMA Milestone 2 — Inflight Pilot"
echo " Job ID: ${SLURM_JOB_ID:-unknown}"
echo " Node: $(hostname)"
echo " Started: $(date)"
echo "============================================================"

# ------------------------------------------------------------
# 1. Load environment
# ------------------------------------------------------------

module purge
module load anaconda

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
    raise RuntimeError(f"Python development header is missing: {header}")

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
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
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
# 5. Verify ALCHEMI installation and Script 06 syntax
# ------------------------------------------------------------

echo
echo "=== ALCHEMI / Script 06 preflight ==="

python - <<'PY'
import importlib.metadata as metadata
import nvalchemi

try:
    version = nvalchemi.__version__
except AttributeError:
    try:
        version = metadata.version("nvalchemi-toolkit")
    except metadata.PackageNotFoundError:
        version = "unknown"

print("nvalchemi version:", version)
PY

python -m py_compile scripts/06_benchmark_inflight.py
echo "PASS: scripts/06_benchmark_inflight.py compiles."

# ------------------------------------------------------------
# 6. Inflight batching pilot
# ------------------------------------------------------------

echo
echo "============================================================"
echo " Script 06: Inflight Batching Benchmark"
echo "============================================================"
echo " Workload: 20 OC20-Dense structures"
echo " Capacity: max_batch_size=4, max_atoms=360"
echo " Refill frequency: 1 FIRE step"
echo " FIRE: fmax=0.05 eV/Angstrom, max_steps=300"
echo " Replicates: 3"
echo "============================================================"

python scripts/06_benchmark_inflight.py \
    --manifest results/pilot_500/dataset_manifest.csv \
    --lmdb-root data/oc20dense_ood_val \
    --limit 20 \
    --max-batch-size 4 \
    --max-atoms 360 \
    --refill-frequency 1 \
    --fmax 0.05 \
    --max-steps 300 \
    --replicates 3

echo
echo "============================================================"
echo " INFLIGHT PILOT COMPLETED SUCCESSFULLY"
echo " Finished: $(date)"
echo "============================================================"
