#!/bin/bash
#SBATCH --job-name=geom-resume
#SBATCH --account=ock
#SBATCH --partition=ock
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=geometry_resume_%j.log
#SBATCH --error=geometry_resume_%j.err

set -euo pipefail

PROJECT=/lustre/work/ock/oissakah10/alchemi-uma-catalysis-milestone2
cd "$PROJECT"

module purge
module load anaconda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/.conda-env"

echo "============================================================"
echo " ALCHEMI + UMA — Geometry Validation Resume"
echo " Job ID: ${SLURM_JOB_ID:-NA}"
echo " Node: $(hostname)"
echo " Started: $(date)"
echo " Python: $(which python)"
python --version
echo "============================================================"

python - <<'PY'
import torch, nvalchemi
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("nvalchemi:", getattr(nvalchemi, "__version__", "unknown"))
PY

python scripts/09_geometry_validation_resume.py --reuse-b1

echo "============================================================"
echo " Finished: $(date)"
echo "============================================================"
