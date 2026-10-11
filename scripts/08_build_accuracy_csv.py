#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


ENERGY_KEYS = (
    "energy", "adsorption_energy", "ads_energy", "target", "y",
    "relaxed_energy", "dft_energy"
)


def scalar_energy(value):
    if np.isscalar(value):
        return float(value)
    if isinstance(value, dict):
        for key in ENERGY_KEYS:
            if key in value and np.isscalar(value[key]):
                return float(value[key])
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.asarray(value, dtype=object).reshape(-1)
        # Prefer a directly scalar numeric member.
        for item in arr:
            if np.isscalar(item):
                try:
                    return float(item)
                except Exception:
                    pass
    return None


def flatten_targets(obj):
    rows = []
    if not isinstance(obj, dict):
        raise TypeError(
            "Expected DFT target pickle to be a dict. Run 07_inspect_targets.py "
            "and send its output if your release differs."
        )
    for key, value in obj.items():
        energy = scalar_energy(value)
        if energy is not None:
            rows.append({"system_id": str(key), "dft_energy_eV": energy})
    if not rows:
        raise ValueError(
            "Could not infer scalar target energies. Run scripts/07_inspect_targets.py "
            "and send the first entries so the adapter can be patched safely."
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--relaxation-results", required=True)
    ap.add_argument("--method", default=None,
                    help="Method to analyze. Default: largest successful fixed batch.")
    ap.add_argument("--out", default="results/pilot_500/accuracy_vs_dft.csv")
    args = ap.parse_args()

    with open(args.targets, "rb") as f:
        targets_obj = pickle.load(f)
    targets = flatten_targets(targets_obj)

    relax = pd.read_csv(args.relaxation_results)
    relax = relax[relax.status == "ok"].copy()
    fixed = relax[relax.method.astype(str).str.startswith("alchemi_fixed")]
    if args.method:
        fixed = fixed[fixed.method == args.method]
    elif len(fixed):
        # Select the largest batch size with rows.
        bs = int(fixed.batch_size.max())
        fixed = fixed[fixed.batch_size == bs]

    if not len(fixed):
        raise ValueError("No successful ALCHEMI fixed-batch relaxation rows found.")

    # Pilot estimate: minimum UMA energy among sampled configurations per system.
    pred = (
        fixed.groupby("system_id", as_index=False)
        .agg(
            uma_min_energy_eV=("final_energy_eV", "min"),
            n_sampled_configs=("sample_idx", "count"),
            n_converged=("converged", "sum"),
        )
    )
    pred["system_id"] = pred.system_id.astype(str)
    targets["system_id"] = targets.system_id.astype(str)
    out = pred.merge(targets, on="system_id", how="inner")
    out["error_eV"] = out.uma_min_energy_eV - out.dft_energy_eV
    out["abs_error_eV"] = out.error_eV.abs()

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Matched {len(out)} systems to DFT targets.")
    if len(out):
        print("Pilot system-level MAE:", out.abs_error_eV.mean(), "eV")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
