#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alchemi_uma.conversion import atoms_list_to_batch
from alchemi_uma.dataset import OCPDenseLMDB
from alchemi_uma.models import load_alchemi_uma, load_fairchem_uma


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare FAIRChem/ASE UMA predictions against "
            "ALCHEMI UMA predictions on OC20-Dense structures."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to dataset_manifest.csv",
    )

    parser.add_argument(
        "--lmdb-root",
        required=True,
        help="Path to OC20-Dense LMDB directory",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of structures to compare",
    )

    parser.add_argument(
        "--model",
        default="uma-s-1p2p1",
        help="UMA checkpoint name",
    )

    parser.add_argument(
        "--task",
        default="oc20",
        help="UMA task name",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to use",
    )

    parser.add_argument(
        "--out",
        default="results/pilot_500/numerical_equivalence.csv",
        help="Output CSV",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Environment checks
    # -----------------------------------------------------------------

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False.\n"
            "Run this script inside a GPU SLURM allocation rather than "
            "directly on the Swan login node."
        )

    print()
    print("=" * 72)
    print(" UMA Numerical Equivalence Test")
    print("=" * 72)
    print(f"Model:               {args.model}")
    print(f"Task:                {args.task}")
    print(f"Device:              {args.device}")
    print(f"Requested structures:{args.n}")

    if torch.cuda.is_available():
        print(f"GPU:                 {torch.cuda.get_device_name(0)}")
        print(
            f"GPU memory:          "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print("=" * 72)
    print()

    # -----------------------------------------------------------------
    # Load manifest
    # -----------------------------------------------------------------

    manifest_path = Path(args.manifest)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = pd.read_csv(manifest_path)

    if len(manifest) == 0:
        raise RuntimeError("Manifest contains no structures.")

    if "lmdb_index" not in manifest.columns:
        raise RuntimeError(
            "Manifest must contain an 'lmdb_index' column."
        )

    n_requested = min(args.n, len(manifest))
    manifest = manifest.head(n_requested).copy()

    print(f"Manifest:            {manifest_path}")
    print(f"Structures selected: {len(manifest)}")

    # -----------------------------------------------------------------
    # Load OC20-Dense LMDB
    # -----------------------------------------------------------------

    print()
    print("Loading OC20-Dense LMDB...")

    ds = OCPDenseLMDB(args.lmdb_root)

    print(f"Dataset length:      {len(ds)}")

    # -----------------------------------------------------------------
    # Load UMA through both execution paths
    #
    # Path 1:
    #   FAIRChem UMA + ASE calculator
    #
    # Path 2:
    #   ALCHEMI UMA tensor-native wrapper
    # -----------------------------------------------------------------

    print()
    print("Loading FAIRChem/ASE UMA...")

    predictor, calc = load_fairchem_uma(
        args.model,
        args.task,
        args.device,
        "default",
    )

    print("Loading ALCHEMI UMA...")

    model = load_alchemi_uma(
        args.model,
        args.task,
        args.device,
        "default",
    )

    model.eval()

    print("Models loaded successfully.")
    print()

    # -----------------------------------------------------------------
    # Numerical equivalence loop
    # -----------------------------------------------------------------

    rows = []

    iterator = tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Equivalence",
    )

    for row_number, (_, rec) in enumerate(iterator):

        lmdb_index = int(rec.lmdb_index)

        # -------------------------------------------------------------
        # Load structure
        # -------------------------------------------------------------

        atoms = ds.get_atoms(lmdb_index)

        n_atoms = len(atoms)

        # -------------------------------------------------------------
        # FAIRChem + ASE
        # -------------------------------------------------------------

        atoms.calc = calc

        e_ase = float(
            atoms.get_potential_energy()
        )

        # IMPORTANT:
        #
        # ASE normally applies attached constraints when get_forces()
        # is called. OC20 catalyst structures commonly contain fixed
        # slab atoms.
        #
        # ALCHEMI returns raw model forces. Therefore, for a valid
        # numerical comparison, obtain the raw ASE forces too.
        #
        # Without apply_constraint=False, fixed atoms may appear to
        # have zero force in the ASE result while ALCHEMI retains their
        # raw predicted force.

        f_ase = np.asarray(
            atoms.get_forces(
                apply_constraint=False
            ),
            dtype=np.float64,
        )

        # -------------------------------------------------------------
        # ALCHEMI
        # -------------------------------------------------------------

        batch = atoms_list_to_batch(
            [atoms],
            args.device,
        )

        # IMPORTANT:
        #
        # DO NOT wrap this call in:
        #
        #     torch.no_grad()
        #
        # or:
        #
        #     torch.inference_mode()
        #
        # UMA computes forces from derivatives of the energy using
        # torch.autograd.grad(). Autograd therefore has to remain
        # enabled during the forward pass.

        outputs = model(batch)

        # -------------------------------------------------------------
        # Extract ALCHEMI energy
        # -------------------------------------------------------------

        if "energy" not in outputs:
            raise RuntimeError(
                "ALCHEMI output does not contain an 'energy' field. "
                f"Available fields: {list(outputs.keys())}"
            )

        e_al = float(
            outputs["energy"]
            .detach()
            .cpu()
            .reshape(-1)[0]
        )

        # -------------------------------------------------------------
        # Extract ALCHEMI forces
        # -------------------------------------------------------------

        if "forces" not in outputs:
            raise RuntimeError(
                "ALCHEMI output does not contain a 'forces' field. "
                f"Available fields: {list(outputs.keys())}"
            )

        f_al = (
            outputs["forces"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        # -------------------------------------------------------------
        # Shape checks
        # -------------------------------------------------------------

        if f_ase.shape != f_al.shape:
            raise RuntimeError(
                "\nForce shape mismatch.\n"
                f"  Manifest row: {row_number}\n"
                f"  LMDB index:   {lmdb_index}\n"
                f"  Number atoms: {n_atoms}\n"
                f"  ASE shape:    {f_ase.shape}\n"
                f"  ALCHEMI shape:{f_al.shape}"
            )

        # -------------------------------------------------------------
        # Numerical differences
        # -------------------------------------------------------------

        signed_delta_energy = e_al - e_ase
        delta_energy = abs(signed_delta_energy)

        force_diff = f_al - f_ase

        force_mae = float(
            np.mean(
                np.abs(force_diff)
            )
        )

        force_rmse = float(
            np.sqrt(
                np.mean(
                    force_diff**2
                )
            )
        )

        force_max = float(
            np.max(
                np.abs(force_diff)
            )
        )

        # -------------------------------------------------------------
        # Optional manifest metadata
        # -------------------------------------------------------------

        sample_idx = (
            int(rec.sample_idx)
            if "sample_idx" in manifest.columns
            else row_number
        )

        sid = (
            rec.sid
            if "sid" in manifest.columns
            else lmdb_index
        )

        system_id = (
            rec.system_id
            if "system_id" in manifest.columns
            else ""
        )

        config_id = (
            rec.config_id
            if "config_id" in manifest.columns
            else ""
        )

        # -------------------------------------------------------------
        # Save row
        # -------------------------------------------------------------

        rows.append(
            {
                "sample_idx": sample_idx,
                "lmdb_index": lmdb_index,
                "sid": sid,
                "system_id": system_id,
                "config_id": config_id,
                "n_atoms": n_atoms,

                "ase_energy_eV": e_ase,
                "alchemi_energy_eV": e_al,

                "signed_delta_energy_eV": signed_delta_energy,
                "delta_energy_eV": delta_energy,

                "force_mae_eV_A": force_mae,
                "force_rmse_eV_A": force_rmse,
                "force_max_eV_A": force_max,
            }
        )

    # -----------------------------------------------------------------
    # Create dataframe
    # -----------------------------------------------------------------

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------
    # Write CSV
    # -----------------------------------------------------------------

    out = Path(args.out)

    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        out,
        index=False,
    )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    summary_cols = [
        "delta_energy_eV",
        "force_mae_eV_A",
        "force_rmse_eV_A",
        "force_max_eV_A",
    ]

    print()
    print("=" * 72)
    print(" Numerical Equivalence Summary")
    print("=" * 72)

    print(
        df[summary_cols]
        .describe()
        .to_string()
    )

    # -----------------------------------------------------------------
    # Aggregate metrics
    # -----------------------------------------------------------------

    mean_delta_e = float(
        df["delta_energy_eV"].mean()
    )

    max_delta_e = float(
        df["delta_energy_eV"].max()
    )

    mean_force_mae = float(
        df["force_mae_eV_A"].mean()
    )

    mean_force_rmse = float(
        df["force_rmse_eV_A"].mean()
    )

    maximum_force_difference = float(
        df["force_max_eV_A"].max()
    )

    print()
    print("-" * 72)

    print(
        f"Mean |Delta E|:       "
        f"{mean_delta_e:.8e} eV"
    )

    print(
        f"Max  |Delta E|:       "
        f"{max_delta_e:.8e} eV"
    )

    print(
        f"Mean force MAE:       "
        f"{mean_force_mae:.8e} eV/Angstrom"
    )

    print(
        f"Mean force RMSE:      "
        f"{mean_force_rmse:.8e} eV/Angstrom"
    )

    print(
        f"Maximum |Delta F|:    "
        f"{maximum_force_difference:.8e} eV/Angstrom"
    )

    print("-" * 72)

    # -----------------------------------------------------------------
    # Diagnostic interpretation
    #
    # These thresholds are intentionally diagnostic rather than used
    # to abort the job. The CSV should always be retained for analysis.
    # -----------------------------------------------------------------

    energy_pass = mean_delta_e < 1.0e-4
    force_pass = mean_force_mae < 1.0e-4

    print()

    if energy_pass and force_pass:

        print(
            "PASS: FAIRChem/ASE and ALCHEMI predictions are "
            "numerically equivalent within the diagnostic thresholds."
        )

    elif energy_pass and not force_pass:

        print(
            "WARNING: Energy predictions agree closely, but the "
            "force differences remain larger than expected."
        )

        print(
            "Do not start the full benchmark yet. Inspect the "
            "per-structure force differences in the output CSV."
        )

    elif not energy_pass and force_pass:

        print(
            "WARNING: Force predictions agree closely, but the "
            "energy differences are larger than expected."
        )

    else:

        print(
            "WARNING: Both energy and force differences are larger "
            "than expected for numerical equivalence."
        )

    # -----------------------------------------------------------------
    # Largest discrepancies
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print(" Largest Energy Differences")
    print("=" * 72)

    energy_display_cols = [
        "lmdb_index",
        "n_atoms",
        "delta_energy_eV",
    ]

    print(
        df.nlargest(
            min(5, len(df)),
            "delta_energy_eV",
        )[energy_display_cols].to_string(
            index=False
        )
    )

    print()
    print("=" * 72)
    print(" Largest Force Differences")
    print("=" * 72)

    force_display_cols = [
        "lmdb_index",
        "n_atoms",
        "force_mae_eV_A",
        "force_rmse_eV_A",
        "force_max_eV_A",
    ]

    print(
        df.nlargest(
            min(5, len(df)),
            "force_mae_eV_A",
        )[force_display_cols].to_string(
            index=False
        )
    )

    # -----------------------------------------------------------------
    # Finish
    # -----------------------------------------------------------------

    print()
    print("=" * 72)
    print(f"Wrote: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()