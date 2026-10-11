#!/usr/bin/env python
from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess

import torch


def ver(name):
    try:
        return md.version(name)
    except Exception:
        return "not installed"


print("Python:", platform.python_version())
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("GPU capability:", torch.cuda.get_device_capability(0))

for pkg in ["nvalchemi-toolkit", "fairchem-core", "ase", "pandas", "lmdb"]:
    print(f"{pkg}: {ver(pkg)}")

if shutil.which("nvidia-smi"):
    print("\nnvidia-smi:")
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        check=False,
    )
else:
    print("nvidia-smi: not found")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA GPU not visible.")
print("\nEnvironment check passed.")
