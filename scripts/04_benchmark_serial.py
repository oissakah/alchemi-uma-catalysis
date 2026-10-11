#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.optimize import FIRE as ASEFIRE
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alchemi_uma.dataset import OCPDenseLMDB
from alchemi_uma.io import replace_method_rows
from alchemi_uma.models import load_fairchem_uma
from alchemi_uma.telemetry import NvidiaSMITelemetry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--lmdb-root", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default="uma-s-1p2p1")
    ap.add_argument("--task", default="oc20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--results-dir", default="results/pilot_500")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit)
    ds = OCPDenseLMDB(args.lmdb_root)
    _, calc = load_fairchem_uma(args.model, args.task, args.device, "default")

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    telemetry_path = outdir / "gpu_samples.csv"
    run_id = f"ase_serial_n{len(manifest)}"

    rows = []
    total_atoms = int(manifest.n_atoms.sum())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    with NvidiaSMITelemetry(run_id, telemetry_path, interval_s=0.5) as telem:
        for _, rec in tqdm(manifest.iterrows(), total=len(manifest), desc="ASE serial"):
            atoms = ds.get_atoms(int(rec.lmdb_index))
            atoms.calc = calc
            initial_e = float(atoms.get_potential_energy())
            s0 = time.perf_counter()
            opt = ASEFIRE(atoms, logfile=None)
            converged = bool(opt.run(fmax=args.fmax, steps=args.max_steps))
            wall = time.perf_counter() - s0
            final_e = float(atoms.get_potential_energy())
            forces = atoms.get_forces()
            fmax = float(np.linalg.norm(forces, axis=1).max())
            rows.append({
                "sample_idx": int(rec.sample_idx),
                "lmdb_index": int(rec.lmdb_index),
                "sid": rec.sid,
                "system_id": rec.system_id,
                "method": "ase_serial",
                "batch_size": 1,
                "n_atoms": len(atoms),
                "converged": converged,
                "n_steps": int(opt.nsteps),
                "initial_energy_eV": initial_e,
                "final_energy_eV": final_e,
                "final_fmax_eV_A": fmax,
                "wall_s": wall,
                "status": "ok",
            })

    torch.cuda.synchronize()
    total_wall = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2

    replace_method_rows(outdir / "relaxation_results.csv", rows, "ase_serial")
    summary = [{
        "run_id": run_id,
        "method": "ase_serial",
        "batch_size": 1,
        "n_structures": len(manifest),
        "n_atoms": total_atoms,
        "wall_s": total_wall,
        "structures_per_s": len(manifest) / total_wall,
        "atoms_per_s": total_atoms / total_wall,
        "speedup_vs_serial": 1.0,
        "peak_torch_memory_mb": peak_mb,
        "peak_nvidia_smi_memory_mb": telem.max_memory_mb,
        "mean_gpu_util_pct": telem.mean_utilization,
        "model": args.model,
        "task": args.task,
        "inference_settings": "default",
        "gpu_name": torch.cuda.get_device_name(0),
        "host": platform.node(),
        "status": "ok",
    }]
    replace_method_rows(outdir / "performance_summary.csv", summary, "ase_serial")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
