#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.geometry import find_mic
from ase.io import read as ase_read, write as ase_write

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alchemi_uma.conversion import atoms_list_to_batch, atoms_to_atomic_data, per_graph_fmax
from alchemi_uma.dataset import OCPDenseLMDB
from alchemi_uma.models import load_alchemi_uma

from nvalchemi.data import Batch
from nvalchemi.dynamics import FIRE, SizeAwareSampler
from nvalchemi.dynamics.base import ConvergenceHook, FusedStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook
from nvalchemi.dynamics.sinks import HostMemory


class GeometryDataset:
    """Manifest-backed dataset for the inflight geometry-capture run."""

    def __init__(self, manifest: pd.DataFrame, lmdb_root: str):
        self.manifest = manifest.reset_index(drop=True)
        self.ds = OCPDenseLMDB(lmdb_root)
        self.counts = self.manifest.n_atoms.astype(int).tolist()

    def __len__(self):
        return len(self.manifest)

    def get_metadata(self, idx: int):
        return self.counts[idx], 0

    def __getitem__(self, idx: int):
        rec = self.manifest.iloc[idx]
        atoms = self.ds.get_atoms(int(rec.lmdb_index))
        data = atoms_to_atomic_data(atoms)

        # These graph-level IDs let us map final inflight snapshots back to
        # the original 500-structure manifest regardless of refill order.
        data.add_system_property(
            "sample_idx",
            torch.tensor([int(rec.sample_idx)], dtype=torch.long),
        )
        data.add_system_property(
            "lmdb_index",
            torch.tensor([int(rec.lmdb_index)], dtype=torch.long),
        )
        return data, {}


class HostMemoryWithIds(HostMemory):
    """HostMemory that preserves graph-level IDs across unbatch/rebatch."""

    _ID_KEYS = ("system_id", "sample_idx", "lmdb_index")

    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        num_total = batch.num_graphs or 0
        if num_total == 0:
            return

        if mask is not None:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )
            num_selected = int(mask.sum().item())
            if num_selected == 0:
                return
            if num_selected < num_total:
                indices = torch.nonzero(mask, as_tuple=True)[0]
                _ = batch.batch_ptr
                batch = batch.index_select(indices)

        saved = {}
        for key in self._ID_KEYS:
            try:
                saved[key] = batch[key].detach().to("cpu")
            except Exception:
                if key != "system_id":
                    raise

        data_list = batch.to_data_list()
        if len(self._data_list) + len(data_list) > self._capacity:
            raise RuntimeError(
                f"Buffer is full. Cannot add {len(data_list)} samples "
                f"to buffer with {len(self._data_list)}/{self._capacity} samples."
            )

        for i, data in enumerate(data_list):
            data_cpu = data.to(self._device)
            for key, values in saved.items():
                data_cpu.add_system_property(key, values[i : i + 1])
            self._data_list.append(data_cpu)


def as_scalar(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().reshape(-1)
        return x[0].item()
    arr = np.asarray(x).reshape(-1)
    return arr[0].item()


def make_atoms(ds, rec, positions, method, converged, final_energy, final_fmax):
    atoms = ds.get_atoms(int(rec.lmdb_index))
    atoms.set_positions(np.asarray(positions, dtype=float))
    atoms.info.update(
        {
            "sample_idx": int(rec.sample_idx),
            "sid": str(rec.sid),
            "system_id": str(rec.system_id),
            "method": method,
            "converged": bool(converged),
            "final_energy_eV": float(final_energy),
            "final_fmax_eV_A": float(final_fmax),
        }
    )
    return atoms


def run_b1(manifest, ds, model, device, fmax, max_steps, dt, outdir):
    print("\n=== Geometry capture: B1 ===")
    rows = []
    frames = []
    positions_by_sample = {}

    for count, (_, rec) in enumerate(manifest.iterrows(), start=1):
        atoms = ds.get_atoms(int(rec.lmdb_index))
        batch = atoms_list_to_batch([atoms], device)

        fire = FIRE(
            model=model,
            dt=dt,
            n_steps=max_steps,
            convergence_hook=ConvergenceHook.from_forces(threshold=fmax),
            hooks=[FreezeAtomsHook()],
        )
        batch = fire.run(batch)
        torch.cuda.synchronize()

        final_fmax = float(per_graph_fmax(batch).detach().cpu()[0])
        final_energy = float(batch.energy.detach().cpu().reshape(-1)[0])
        positions = batch.positions.detach().cpu().numpy().copy()
        converged = final_fmax <= fmax

        positions_by_sample[int(rec.sample_idx)] = positions
        frames.append(
            make_atoms(
                ds, rec, positions, "b1", converged, final_energy, final_fmax
            )
        )
        rows.append(
            {
                "sample_idx": int(rec.sample_idx),
                "lmdb_index": int(rec.lmdb_index),
                "sid": rec.sid,
                "system_id": rec.system_id,
                "method": "b1",
                "n_atoms": int(rec.n_atoms),
                "converged": bool(converged),
                "n_steps": int(fire.step_count),
                "final_energy_eV": final_energy,
                "final_fmax_eV_A": final_fmax,
            }
        )

        if count % 50 == 0 or count == len(manifest):
            print(f"B1: {count}/{len(manifest)} structures complete")

    ase_write(outdir / "b1_final_structures.extxyz", frames, format="extxyz")
    pd.DataFrame(rows).to_csv(outdir / "b1_geometry_results.csv", index=False)
    return pd.DataFrame(rows), positions_by_sample


def run_inflight(
    manifest,
    model,
    lmdb_root,
    device,
    fmax,
    max_steps,
    dt,
    max_batch_size,
    max_atoms,
    refill_frequency,
    outdir,
):
    print("\n=== Geometry capture: inflight B4 ===")
    dataset = GeometryDataset(manifest, lmdb_root)

    # Warm-up before the captured run.
    warm_data, _ = dataset[0]
    warm = Batch.from_data_list([warm_data]).to(device)
    _ = model(warm)
    torch.cuda.synchronize()

    sampler = SizeAwareSampler(
        dataset,
        max_atoms=max_atoms,
        max_edges=None,
        max_batch_size=max_batch_size,
    )

    # FusedStage writes each graduated system (converged OR max-step) to its
    # sinks exactly once. Do not also attach ConvergedSnapshotHook here: using
    # both paths duplicates converged structures and can fill a capacity=N sink
    # before the N-structure campaign is finished.
    sink = HostMemoryWithIds(capacity=len(dataset))
    fire = FIRE(
        model=model,
        dt=dt,
        n_steps=max_steps,
        convergence_hook=ConvergenceHook.from_forces(threshold=fmax),
        hooks=[FreezeAtomsHook()],
    )

    fused = FusedStage(
        sub_stages=[(0, fire)],
        sampler=sampler,
        sinks=[sink],
        refill_frequency=refill_frequency,
    )

    result = fused.run(batch=None)
    torch.cuda.synchronize()

    if result is not None and getattr(result, "num_graphs", 0):
        raise RuntimeError(
            f"Inflight run returned {result.num_graphs} active graphs instead "
            "of exhausting the dataset."
        )

    if len(sink) != len(manifest):
        raise RuntimeError(
            f"Expected {len(manifest)} final inflight snapshots, captured {len(sink)}."
        )

    final_batch = sink.drain()
    sample_ids = final_batch["sample_idx"].detach().cpu().reshape(-1).numpy().astype(int)
    lmdb_ids = final_batch["lmdb_index"].detach().cpu().reshape(-1).numpy().astype(int)

    ptr = final_batch.batch_ptr.detach().cpu().numpy().astype(int)
    energies = final_batch.energy.detach().cpu().reshape(-1).numpy()
    graph_fmax = per_graph_fmax(final_batch).detach().cpu().numpy()

    rows = []
    frames = []
    positions_by_sample = {}
    ds = dataset.ds
    manifest_by_sample = manifest.set_index("sample_idx", drop=False)

    if len(set(sample_ids.tolist())) != len(manifest):
        raise RuntimeError(
            "Inflight snapshots do not contain one unique sample_idx per structure."
        )

    for graph_idx, sample_idx in enumerate(sample_ids):
        start, stop = ptr[graph_idx], ptr[graph_idx + 1]
        positions = final_batch.positions[start:stop].detach().cpu().numpy().copy()
        rec = manifest_by_sample.loc[int(sample_idx)]

        if int(rec.lmdb_index) != int(lmdb_ids[graph_idx]):
            raise RuntimeError(
                f"LMDB mismatch for sample_idx={sample_idx}: "
                f"manifest={rec.lmdb_index}, sink={lmdb_ids[graph_idx]}"
            )

        final_energy = float(energies[graph_idx])
        final_fmax = float(graph_fmax[graph_idx])
        converged = final_fmax <= fmax

        positions_by_sample[int(sample_idx)] = positions
        frames.append(
            make_atoms(
                ds,
                rec,
                positions,
                "inflight_b4",
                converged,
                final_energy,
                final_fmax,
            )
        )
        rows.append(
            {
                "sample_idx": int(sample_idx),
                "lmdb_index": int(rec.lmdb_index),
                "sid": rec.sid,
                "system_id": rec.system_id,
                "method": "inflight_b4",
                "n_atoms": int(rec.n_atoms),
                "converged": bool(converged),
                "final_energy_eV": final_energy,
                "final_fmax_eV_A": final_fmax,
            }
        )

    rows_df = pd.DataFrame(rows).sort_values("sample_idx").reset_index(drop=True)
    ase_write(
        outdir / "inflight_b04_final_structures.extxyz",
        frames,
        format="extxyz",
    )
    rows_df.to_csv(outdir / "inflight_b04_geometry_results.csv", index=False)
    return rows_df, positions_by_sample


def rmsd_from_vectors(vectors):
    if len(vectors) == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.sum(vectors * vectors, axis=1))))


def compare_geometries(
    manifest,
    ds,
    b1_df,
    inflight_df,
    b1_positions,
    inflight_positions,
    outdir,
):
    print("\n=== Comparing final geometries ===")
    b1_meta = b1_df.set_index("sample_idx")
    inf_meta = inflight_df.set_index("sample_idx")
    rows = []

    for _, rec in manifest.iterrows():
        sample_idx = int(rec.sample_idx)
        p1 = b1_positions[sample_idx]
        p2 = inflight_positions[sample_idx]

        if p1.shape != p2.shape:
            raise RuntimeError(
                f"Position-shape mismatch for sample_idx={sample_idx}: "
                f"{p1.shape} vs {p2.shape}"
            )

        atoms0 = ds.get_atoms(int(rec.lmdb_index))
        raw_delta = p2 - p1

        # Use minimum-image displacements so periodic wrapping does not look
        # like a large structural difference.
        mic_delta, _ = find_mic(
            raw_delta,
            cell=atoms0.cell,
            pbc=atoms0.pbc,
        )

        tags = np.asarray(atoms0.get_tags())
        if len(tags) != len(atoms0):
            tags = np.ones(len(atoms0), dtype=int)

        mobile = tags != 0
        adsorbate = tags == 2

        norms = np.linalg.norm(mic_delta, axis=1)
        mobile_norms = norms[mobile]
        ads_norms = norms[adsorbate]

        e1 = float(b1_meta.loc[sample_idx, "final_energy_eV"])
        e2 = float(inf_meta.loc[sample_idx, "final_energy_eV"])
        c1 = bool(b1_meta.loc[sample_idx, "converged"])
        c2 = bool(inf_meta.loc[sample_idx, "converged"])

        rows.append(
            {
                "sample_idx": sample_idx,
                "sid": rec.sid,
                "system_id": rec.system_id,
                "n_atoms": int(rec.n_atoms),
                "b1_converged": c1,
                "inflight_converged": c2,
                "same_convergence": c1 == c2,
                "b1_final_energy_eV": e1,
                "inflight_final_energy_eV": e2,
                "abs_delta_energy_eV": abs(e2 - e1),
                "abs_delta_energy_meV_atom": abs(e2 - e1) / int(rec.n_atoms) * 1000.0,
                "rmsd_all_A": rmsd_from_vectors(mic_delta),
                "rmsd_mobile_A": rmsd_from_vectors(mic_delta[mobile]),
                "rmsd_adsorbate_A": rmsd_from_vectors(mic_delta[adsorbate]),
                "max_displacement_all_A": float(norms.max()) if len(norms) else np.nan,
                "max_displacement_mobile_A": (
                    float(mobile_norms.max()) if len(mobile_norms) else np.nan
                ),
                "max_displacement_adsorbate_A": (
                    float(ads_norms.max()) if len(ads_norms) else np.nan
                ),
            }
        )

    out = pd.DataFrame(rows).sort_values("sample_idx").reset_index(drop=True)
    out.to_csv(outdir / "geometry_comparison.csv", index=False)

    # Focused table for the structures that disagreed in the earlier
    # three-replicate production campaign, if that analysis file exists.
    disagreement_path = (
        ROOT
        / "results"
        / "production_500"
        / "analysis"
        / "b1_vs_inflight_b04_convergence_disagreements.csv"
    )
    if disagreement_path.exists():
        old = pd.read_csv(disagreement_path)
        target_ids = sorted(old.sample_idx.unique().astype(int).tolist())
        out[out.sample_idx.isin(target_ids)].to_csv(
            outdir / "previous_disagreement_structures_geometry.csv",
            index=False,
        )

    return out


def write_summary(comp, wall_b1, wall_inflight, outdir):
    both_conv = comp[comp.b1_converged & comp.inflight_converged]
    same = comp.same_convergence.mean()

    lines = [
        "Geometry validation — B1 vs inflight B4",
        "========================================",
        "",
        f"Structures compared: {len(comp)}",
        f"B1 coordinate-capture wall time: {wall_b1/60:.2f} min" if np.isfinite(wall_b1) else "B1 coordinate-capture wall time: reused existing capture",
        f"Inflight B4 coordinate-capture wall time: {wall_inflight/60:.2f} min",
        f"Convergence agreement in this capture run: {same*100:.2f}%",
        "",
        "Geometry differences, all structures:",
        f"  median mobile-atom RMSD: {comp.rmsd_mobile_A.median():.6f} A",
        f"  95th percentile mobile-atom RMSD: {comp.rmsd_mobile_A.quantile(0.95):.6f} A",
        f"  maximum mobile-atom RMSD: {comp.rmsd_mobile_A.max():.6f} A",
        "",
        "Geometry differences, cases where both methods converged:",
        f"  n: {len(both_conv)}",
        f"  median mobile-atom RMSD: {both_conv.rmsd_mobile_A.median():.6f} A",
        f"  95th percentile mobile-atom RMSD: {both_conv.rmsd_mobile_A.quantile(0.95):.6f} A",
        f"  maximum mobile-atom RMSD: {both_conv.rmsd_mobile_A.max():.6f} A",
        f"  median adsorbate RMSD: {both_conv.rmsd_adsorbate_A.median():.6f} A",
        f"  95th percentile adsorbate RMSD: {both_conv.rmsd_adsorbate_A.quantile(0.95):.6f} A",
        "",
        "Largest mobile-atom RMSD cases:",
    ]

    top = comp.nlargest(10, "rmsd_mobile_A")[
        [
            "sample_idx",
            "same_convergence",
            "rmsd_mobile_A",
            "rmsd_adsorbate_A",
            "max_displacement_mobile_A",
            "abs_delta_energy_meV_atom",
        ]
    ]
    lines.append(top.to_string(index=False))

    text = "\n".join(lines) + "\n"
    (outdir / "geometry_validation_summary.txt").write_text(text)
    print("\n" + text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="results/pilot_500/dataset_manifest.csv",
    )
    ap.add_argument(
        "--lmdb-root",
        default="data/oc20dense_ood_val",
    )
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--model", default="uma-s-1p2p1")
    ap.add_argument("--task", default="oc20")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--max-batch-size", type=int, default=4)
    ap.add_argument("--max-atoms", type=int, default=360)
    ap.add_argument("--refill-frequency", type=int, default=1)
    ap.add_argument(
        "--results-dir",
        default="results/production_500/geometry_validation",
    )
    ap.add_argument(
        "--reuse-b1",
        action="store_true",
        help="Reuse existing B1 CSV/extxyz and run only inflight + comparison.",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This validation requires a CUDA GPU.")

    manifest = pd.read_csv(args.manifest)
    if args.limit:
        manifest = manifest.head(args.limit).copy()

    if manifest.sample_idx.duplicated().any():
        raise RuntimeError("Manifest sample_idx values must be unique.")

    outdir = Path(args.results_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Structures: {len(manifest)}")
    print(f"Atoms: {int(manifest.n_atoms.sum())}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        "Settings: "
        f"fmax={args.fmax}, max_steps={args.max_steps}, dt={args.dt}, "
        f"inflight B{args.max_batch_size}, max_atoms={args.max_atoms}, "
        f"refill_frequency={args.refill_frequency}"
    )

    ds = OCPDenseLMDB(args.lmdb_root)
    model = load_alchemi_uma(args.model, args.task, args.device, "default")
    model.eval()

    # Shared warm-up.
    warm_atoms = ds.get_atoms(int(manifest.iloc[0].lmdb_index))
    warm_batch = atoms_list_to_batch([warm_atoms], args.device)
    _ = model(warm_batch)
    torch.cuda.synchronize()

    gc.collect()
    torch.cuda.empty_cache()

    if args.reuse_b1:
        b1_csv = outdir / "b1_geometry_results.csv"
        b1_xyz = outdir / "b1_final_structures.extxyz"
        if not b1_csv.exists() or not b1_xyz.exists():
            raise FileNotFoundError(
                "--reuse-b1 requested, but existing B1 CSV/extxyz were not found."
            )
        print("\n=== Reusing existing B1 geometry capture ===")
        b1_df = pd.read_csv(b1_csv)
        frames = ase_read(b1_xyz, index=":")
        if len(b1_df) != len(manifest) or len(frames) != len(manifest):
            raise RuntimeError(
                f"Expected {len(manifest)} saved B1 structures; "
                f"CSV has {len(b1_df)}, extxyz has {len(frames)}."
            )
        b1_positions = {}
        for atoms in frames:
            if "sample_idx" not in atoms.info:
                raise RuntimeError("Saved B1 extxyz frame is missing sample_idx.")
            sample_idx = int(atoms.info["sample_idx"])
            if sample_idx in b1_positions:
                raise RuntimeError(f"Duplicate B1 sample_idx={sample_idx} in extxyz.")
            b1_positions[sample_idx] = atoms.get_positions().copy()
        expected = set(manifest.sample_idx.astype(int).tolist())
        if set(b1_positions) != expected:
            raise RuntimeError("Saved B1 extxyz sample_idx set does not match manifest.")
        wall_b1 = float("nan")
        print(f"Reused {len(b1_positions)} saved B1 structures; B1 will not be recomputed.")
    else:
        t0 = time.perf_counter()
        b1_df, b1_positions = run_b1(
            manifest,
            ds,
            model,
            args.device,
            args.fmax,
            args.max_steps,
            args.dt,
            outdir,
        )
        torch.cuda.synchronize()
        wall_b1 = time.perf_counter() - t0

    gc.collect()
    torch.cuda.empty_cache()

    t0 = time.perf_counter()
    inflight_df, inflight_positions = run_inflight(
        manifest,
        model,
        args.lmdb_root,
        args.device,
        args.fmax,
        args.max_steps,
        args.dt,
        args.max_batch_size,
        args.max_atoms,
        args.refill_frequency,
        outdir,
    )
    torch.cuda.synchronize()
    wall_inflight = time.perf_counter() - t0

    comp = compare_geometries(
        manifest,
        ds,
        b1_df,
        inflight_df,
        b1_positions,
        inflight_positions,
        outdir,
    )
    write_summary(comp, wall_b1, wall_inflight, outdir)

    print(f"Outputs: {outdir}")
    print("PASS: geometry capture and comparison completed.")


if __name__ == "__main__":
    main()
