#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alchemi_uma.conversion import atoms_list_to_batch, per_graph_fmax
from alchemi_uma.dataset import OCPDenseLMDB
from alchemi_uma.io import replace_method_rows
from alchemi_uma.models import load_alchemi_uma
from alchemi_uma.telemetry import NvidiaSMITelemetry

from nvalchemi.dynamics import FIRE
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.hooks import FreezeAtomsHook
from nvalchemi.hooks import StageTimingHook


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def chunks(df, size):
    """Yield deterministic fixed-size chunks."""
    for start in range(0, len(df), size):
        yield start, df.iloc[start:start + size]


def sync(device):
    """Synchronize CUDA before/after timing."""
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_gpu(device):
    """Clear transient allocations and reset peak-memory statistics."""
    gc.collect()

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


# ---------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------

def warm_up(
    model,
    ds,
    manifest,
    batch_size,
    device,
):
    """
    Untimed warm-up over every exact chunk used by this batch size.

    OC20-Dense is heterogeneous. A single one-structure warm-up is not
    sufficient because different chunks can have different graph sizes
    and compositions.

    We therefore make one untimed UMA call for every exact chunk that
    will later appear in the measured workload.
    """

    print()
    print(
        f"Warm-up: batch size {batch_size} "
        "(excluded from benchmark timing)"
    )

    n_chunks = (
        len(manifest) + batch_size - 1
    ) // batch_size

    for _, chunk in tqdm(
        chunks(manifest, batch_size),
        total=n_chunks,
        desc=f"warmup_b{batch_size:02d}",
    ):

        atoms_list = [
            ds.get_atoms(int(rec.lmdb_index))
            for _, rec in chunk.iterrows()
        ]

        batch = atoms_list_to_batch(
            atoms_list,
            device,
        )

        # IMPORTANT:
        #
        # Do NOT use:
        #
        #   torch.no_grad()
        #   torch.inference_mode()
        #
        # UMA computes forces through autograd.

        outputs = model(batch)

        # Touch both outputs before synchronization.
        _ = outputs["energy"].detach()
        _ = outputs["forces"].detach()

        del outputs
        del batch
        del atoms_list

    sync(device)

    print(
        f"Warm-up complete for batch size {batch_size}."
    )


# ---------------------------------------------------------------------
# One measured replicate
# ---------------------------------------------------------------------

def run_replicate(
    model,
    ds,
    manifest,
    batch_size,
    replicate,
    args,
    outdir,
    timing_dir,
    session_id,
):

    method = (
        f"alchemi_fixed_b{batch_size:02d}"
    )

    run_id = (
        f"{method}"
        f"_n{len(manifest)}"
        f"_r{replicate:02d}"
        f"_job{session_id}"
    )

    total_atoms = int(
        manifest.n_atoms.sum()
    )

    rows = []

    status = "ok"
    error_message = ""
    total_wall = np.nan

    print()
    print("=" * 78)

    print(
        f"{method} | "
        f"replicate {replicate}/{args.replicates}"
    )

    print(
        f"Structures: {len(manifest)}"
    )

    print(
        f"run_id: {run_id}"
    )

    print("=" * 78)

    # -------------------------------------------------------------
    # Reset measurement state
    # -------------------------------------------------------------

    reset_gpu(args.device)

    # -------------------------------------------------------------
    # GPU telemetry
    # -------------------------------------------------------------

    telemetry = NvidiaSMITelemetry(
        run_id,
        outdir / "gpu_samples.csv",
        args.telemetry_interval,
    )

    try:

        with telemetry:

            # -----------------------------------------------------
            # Start timing
            # -----------------------------------------------------

            sync(args.device)

            t0 = time.perf_counter()

            n_chunks = (
                len(manifest)
                + batch_size
                - 1
            ) // batch_size

            # -----------------------------------------------------
            # Fixed batches
            # -----------------------------------------------------

            for chunk_no, (_, chunk) in enumerate(

                tqdm(
                    chunks(
                        manifest,
                        batch_size,
                    ),
                    total=n_chunks,
                    desc=(
                        f"{method}"
                        f"_r{replicate:02d}"
                    ),
                )

            ):

                # -------------------------------------------------
                # Load structures
                # -------------------------------------------------

                atoms_list = [

                    ds.get_atoms(
                        int(rec.lmdb_index)
                    )

                    for _, rec
                    in chunk.iterrows()

                ]

                # -------------------------------------------------
                # Build ALCHEMI batch
                # -------------------------------------------------

                batch = atoms_list_to_batch(
                    atoms_list,
                    args.device,
                )

                # -------------------------------------------------
                # Initial UMA prediction
                #
                # This remains inside the measured region because
                # it is part of the end-to-end relaxation workflow.
                # -------------------------------------------------

                initial_output = model(batch)

                initial_energy = (

                    initial_output[
                        "energy"
                    ]
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .numpy()

                )

                del initial_output

                # -------------------------------------------------
                # Stage timing
                # -------------------------------------------------

                timing_path = (

                    timing_dir
                    / (
                        f"{method}"
                        f"_r{replicate:02d}"
                        f"_chunk{chunk_no:04d}.csv"
                    )

                )

                profiler = StageTimingHook(

                    "detailed",

                    log_path=timing_path,

                    timer_backend="auto",

                    enable_nvtx=True,

                    show_console=False,

                )

                # -------------------------------------------------
                # FIRE relaxation
                # -------------------------------------------------

                fire = FIRE(

                    model=model,

                    dt=args.dt,

                    n_steps=args.max_steps,

                    convergence_hook=(
                        ConvergenceHook.from_forces(
                            threshold=args.fmax
                        )
                    ),

                    hooks=[
                        FreezeAtomsHook(),
                        profiler,
                    ],

                )

                chunk_t0 = (
                    time.perf_counter()
                )

                try:

                    batch = fire.run(
                        batch
                    )

                    sync(args.device)

                    chunk_wall = (
                        time.perf_counter()
                        - chunk_t0
                    )

                finally:

                    profiler.close()

                # -------------------------------------------------
                # Final values
                # -------------------------------------------------

                final_energy = (

                    batch.energy
                    .detach()
                    .cpu()
                    .reshape(-1)
                    .numpy()

                )

                final_fmax = (

                    per_graph_fmax(
                        batch
                    )
                    .detach()
                    .cpu()
                    .numpy()

                )

                # -------------------------------------------------
                # Sanity check
                # -------------------------------------------------

                if (
                    len(final_energy)
                    != len(chunk)
                ):

                    raise RuntimeError(

                        "Returned "
                        f"{len(final_energy)} "
                        "graphs for "
                        f"{len(chunk)} inputs."

                    )

                # -------------------------------------------------
                # Per-structure results
                # -------------------------------------------------

                per_structure_wall = (
                    chunk_wall
                    / len(chunk)
                )

                for j, (_, rec) in enumerate(
                    chunk.iterrows()
                ):

                    rows.append(
                        {
                            "sample_idx":
                                int(
                                    rec.sample_idx
                                ),

                            "lmdb_index":
                                int(
                                    rec.lmdb_index
                                ),

                            "sid":
                                rec.sid,

                            "system_id":
                                rec.system_id,

                            "method":
                                method,

                            "batch_size":
                                batch_size,

                            "replicate":
                                replicate,

                            "run_id":
                                run_id,

                            "n_atoms":
                                int(
                                    rec.n_atoms
                                ),

                            "converged":
                                bool(
                                    final_fmax[j]
                                    <= args.fmax
                                ),

                            "n_steps":
                                int(
                                    fire.step_count
                                ),

                            "initial_energy_eV":
                                float(
                                    initial_energy[j]
                                ),

                            "final_energy_eV":
                                float(
                                    final_energy[j]
                                ),

                            "final_fmax_eV_A":
                                float(
                                    final_fmax[j]
                                ),

                            "wall_s":
                                per_structure_wall,

                            "status":
                                "ok",
                        }
                    )

                # -------------------------------------------------
                # Cleanup
                # -------------------------------------------------

                del batch
                del atoms_list
                del final_energy
                del final_fmax
                del initial_energy
                del fire

            # -----------------------------------------------------
            # End timing
            # -----------------------------------------------------

            sync(args.device)

            total_wall = (
                time.perf_counter()
                - t0
            )

    # -----------------------------------------------------------------
    # OOM handling
    # -----------------------------------------------------------------

    except torch.cuda.OutOfMemoryError as exc:

        status = "OOM"

        error_message = str(exc)

        print(
            f"[OOM] batch={batch_size}, "
            f"replicate={replicate}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # Other errors
    # -----------------------------------------------------------------

    except Exception as exc:

        status = "error"

        error_message = repr(exc)

        raise

    finally:

        try:
            sync(args.device)
        except Exception:
            pass

    # -----------------------------------------------------------------
    # Memory
    # -----------------------------------------------------------------

    if torch.cuda.is_available():

        peak_torch_memory_mb = (

            torch.cuda
            .max_memory_allocated()
            / 1024**2

        )

    else:

        peak_torch_memory_mb = (
            np.nan
        )

    successful = (
        status == "ok"
    )

    # -----------------------------------------------------------------
    # Replicate summary
    # -----------------------------------------------------------------

    summary = {

        "run_id":
            run_id,

        "method":
            method,

        "batch_size":
            batch_size,

        "replicate":
            replicate,

        "n_structures":
            (
                len(manifest)
                if successful
                else len(rows)
            ),

        "n_atoms":
            total_atoms,

        "wall_s":
            (
                float(total_wall)
                if successful
                else np.nan
            ),

        "structures_per_s":
            (
                len(manifest)
                / total_wall
                if successful
                else np.nan
            ),

        "atoms_per_s":
            (
                total_atoms
                / total_wall
                if successful
                else np.nan
            ),

        "peak_torch_memory_mb":
            peak_torch_memory_mb,

        "peak_nvidia_smi_memory_mb":
            telemetry.max_memory_mb,

        "mean_gpu_util_pct":
            telemetry.mean_utilization,

        "model":
            args.model,

        "task":
            args.task,

        "inference_settings":
            args.inference_settings,

        "gpu_name":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else ""
            ),

        "host":
            platform.node(),

        "status":
            status,

        "error_message":
            error_message,
    }

    return rows, summary


# ---------------------------------------------------------------------
# Aggregate replicate statistics
# ---------------------------------------------------------------------

def aggregate(
    replicate_rows,
    batch_sizes,
    serial_wall,
):

    df = pd.DataFrame(
        replicate_rows
    )

    summaries = []

    def sample_std(series):

        series = (
            series
            .dropna()
        )

        if len(series) > 1:

            return float(
                series.std(
                    ddof=1
                )
            )

        return 0.0

    # -------------------------------------------------------------
    # Aggregate each batch size
    # -------------------------------------------------------------

    for batch_size in batch_sizes:

        group = df[
            df.batch_size
            == batch_size
        ]

        successful = group[
            group.status
            == "ok"
        ]

        method = (
            f"alchemi_fixed_b"
            f"{batch_size:02d}"
        )

        # ---------------------------------------------------------
        # Failed batch size
        # ---------------------------------------------------------

        if len(successful) == 0:

            summaries.append(
                {
                    "run_id":
                        f"{method}_aggregate",

                    "method":
                        method,

                    "batch_size":
                        batch_size,

                    "n_replicates":
                        len(group),

                    "n_successful_replicates":
                        0,

                    "status":
                        "failed",
                }
            )

            continue

        # ---------------------------------------------------------
        # Successful batch size
        # ---------------------------------------------------------

        mean_wall = float(
            successful.wall_s.mean()
        )

        summaries.append(
            {
                "run_id":
                    f"{method}_aggregate",

                "method":
                    method,

                "batch_size":
                    batch_size,

                "n_replicates":
                    len(group),

                "n_successful_replicates":
                    len(successful),

                "n_structures":
                    int(
                        successful
                        .n_structures
                        .iloc[0]
                    ),

                "n_atoms":
                    int(
                        successful
                        .n_atoms
                        .iloc[0]
                    ),

                # Mean wall time
                "wall_s":
                    mean_wall,

                # Standard deviation
                "wall_s_std":
                    sample_std(
                        successful.wall_s
                    ),

                # Median wall time
                "wall_s_median":
                    float(
                        successful
                        .wall_s
                        .median()
                    ),

                # Mean throughput
                "structures_per_s":
                    float(
                        successful
                        .structures_per_s
                        .mean()
                    ),

                # Throughput uncertainty
                "structures_per_s_std":
                    sample_std(
                        successful
                        .structures_per_s
                    ),

                # Mean atom throughput
                "atoms_per_s":
                    float(
                        successful
                        .atoms_per_s
                        .mean()
                    ),

                "atoms_per_s_std":
                    sample_std(
                        successful
                        .atoms_per_s
                    ),

                # Filled after batch-1
                # reference is known.
                "speedup_vs_batch1":
                    np.nan,

                # Existing ASE baseline,
                # if available.
                "speedup_vs_serial":
                    (
                        serial_wall
                        / mean_wall
                        if serial_wall
                        else np.nan
                    ),

                # Conservative memory reporting:
                # maximum across replicates.
                "peak_torch_memory_mb":
                    float(
                        successful
                        .peak_torch_memory_mb
                        .max()
                    ),

                "peak_nvidia_smi_memory_mb":
                    (
                        float(
                            successful
                            .peak_nvidia_smi_memory_mb
                            .max()
                        )
                        if successful
                        .peak_nvidia_smi_memory_mb
                        .notna()
                        .any()
                        else np.nan
                    ),

                # Mean utilization
                "mean_gpu_util_pct":
                    (
                        float(
                            successful
                            .mean_gpu_util_pct
                            .mean()
                        )
                        if successful
                        .mean_gpu_util_pct
                        .notna()
                        .any()
                        else np.nan
                    ),

                "mean_gpu_util_pct_std":
                    sample_std(
                        successful
                        .mean_gpu_util_pct
                    ),

                "model":
                    successful
                    .model
                    .iloc[0],

                "task":
                    successful
                    .task
                    .iloc[0],

                "inference_settings":
                    successful
                    .inference_settings
                    .iloc[0],

                "gpu_name":
                    successful
                    .gpu_name
                    .iloc[0],

                "host":
                    successful
                    .host
                    .iloc[0],

                "status":
                    "ok",
            }
        )

    # -------------------------------------------------------------
    # Fixed-batch speedup relative to warmed batch 1
    # -------------------------------------------------------------

    batch1 = next(

        (
            row
            for row in summaries
            if (
                row["batch_size"] == 1
                and row["status"] == "ok"
            )
        ),

        None,

    )

    if batch1:

        reference_throughput = (
            batch1[
                "structures_per_s"
            ]
        )

        for row in summaries:

            if row["status"] == "ok":

                row[
                    "speedup_vs_batch1"
                ] = (

                    row[
                        "structures_per_s"
                    ]
                    / reference_throughput

                )

    return summaries


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Steady-state fixed-batch "
            "ALCHEMI + UMA relaxation benchmark."
        )

    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--lmdb-root",
        required=True,
    )

    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[
            1,
            2,
            4,
            8,
            16,
            32,
        ],
    )

    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--limit",
        type=int,
    )

    parser.add_argument(
        "--model",
        default="uma-s-1p2p1",
    )

    parser.add_argument(
        "--task",
        default="oc20",
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--inference-settings",
        default="default",
    )

    parser.add_argument(
        "--fmax",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--telemetry-interval",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--skip-warmup",
        action="store_true",
    )

    parser.add_argument(
        "--results-dir",
        default="results/pilot_500",
    )

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    if args.replicates < 1:

        raise ValueError(
            "--replicates must be >= 1"
        )

    if any(
        batch_size < 1
        for batch_size
        in args.batch_sizes
    ):

        raise ValueError(
            "All batch sizes must be >= 1."
        )

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(

            "No CUDA GPU is visible. "
            "Run this script inside a "
            "GPU SLURM allocation."

        )

    # -----------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------

    manifest = pd.read_csv(
        args.manifest
    )

    if args.limit:

        manifest = (
            manifest
            .head(args.limit)
            .copy()
        )

    if len(manifest) == 0:

        raise RuntimeError(
            "No structures selected."
        )

    required_columns = {

        "sample_idx",
        "lmdb_index",
        "sid",
        "system_id",
        "n_atoms",

    }

    missing = (
        required_columns
        - set(
            manifest.columns
        )
    )

    if missing:

        raise RuntimeError(

            "Manifest missing columns: "
            f"{sorted(missing)}"

        )

    # -----------------------------------------------------------------
    # Batch sizes
    # -----------------------------------------------------------------

    batch_sizes = list(
        dict.fromkeys(
            args.batch_sizes
        )
    )

    oversized = [

        batch_size

        for batch_size
        in batch_sizes

        if batch_size
        > len(manifest)

    ]

    if oversized:

        print(

            "WARNING: skipping batch sizes "
            f"> {len(manifest)} selected "
            f"structures: {oversized}"

        )

        batch_sizes = [

            batch_size

            for batch_size
            in batch_sizes

            if batch_size
            <= len(manifest)

        ]

    if not batch_sizes:

        raise ValueError(
            "No valid batch sizes remain."
        )

    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------

    ds = OCPDenseLMDB(
        args.lmdb_root
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------

    print()
    print("Loading ALCHEMI UMA...")

    model = load_alchemi_uma(

        args.model,

        args.task,

        args.device,

        args.inference_settings,

    )

    model.eval()

    print("Model loaded.")

    # -----------------------------------------------------------------
    # Output directories
    # -----------------------------------------------------------------

    outdir = Path(
        args.results_dir
    )

    timing_dir = (
        outdir
        / "stage_timing"
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timing_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------------------
    # Existing ASE serial baseline
    # -----------------------------------------------------------------

    serial_wall = None

    performance_summary_path = (
        outdir
        / "performance_summary.csv"
    )

    if performance_summary_path.exists():

        previous = pd.read_csv(
            performance_summary_path
        )

        serial = previous[

            (
                previous.method
                == "ase_serial"
            )

            & (
                previous.n_structures
                == len(manifest)
            )

            & (
                previous.status
                == "ok"
            )

        ]

        if len(serial):

            serial_wall = float(
                serial.iloc[-1].wall_s
            )

    # -----------------------------------------------------------------
    # Unique job/session identifier
    # -----------------------------------------------------------------

    session_id = os.environ.get(

        "SLURM_JOB_ID",

        time.strftime(
            "%Y%m%d%H%M%S"
        ),

    )

    replicate_summaries = []

    all_relaxation_rows = []

    scientific_relaxation_rows = []

    # -----------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------

    print()
    print("=" * 78)

    print(
        " ALCHEMI + UMA "
        "steady-state fixed-batch benchmark"
    )

    print(
        f" Structures: {len(manifest)}"
    )

    print(
        f" Total atoms: "
        f"{int(manifest.n_atoms.sum())}"
    )

    print(
        f" Batch sizes: {batch_sizes}"
    )

    print(
        f" Replicates: {args.replicates}"
    )

    print(
        f" Inference settings: "
        f"{args.inference_settings}"
    )

    print(
        f" GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )

    print("=" * 78)

    # -----------------------------------------------------------------
    # Benchmark batch sizes
    # -----------------------------------------------------------------

    for batch_size in batch_sizes:

        print()
        print("=" * 78)

        print(
            f" Batch size {batch_size}"
        )

        print("=" * 78)

        # -------------------------------------------------------------
        # Per-batch warm-up
        # -------------------------------------------------------------

        if not args.skip_warmup:

            warm_up(

                model=model,

                ds=ds,

                manifest=manifest,

                batch_size=batch_size,

                device=args.device,

            )

        # Warm-up is explicitly outside measurement.
        reset_gpu(
            args.device
        )

        # -------------------------------------------------------------
        # Replicates
        # -------------------------------------------------------------

        for replicate in range(

            1,
            args.replicates + 1,

        ):

            rows, summary = run_replicate(

                model=model,

                ds=ds,

                manifest=manifest,

                batch_size=batch_size,

                replicate=replicate,

                args=args,

                outdir=outdir,

                timing_dir=timing_dir,

                session_id=session_id,

            )

            replicate_summaries.append(
                summary
            )

            # ---------------------------------------------------------
            # Successful run
            # ---------------------------------------------------------

            if summary["status"] == "ok":

                # Keep every replicate separately.
                all_relaxation_rows.extend(
                    rows
                )

                # Downstream DFT analysis should not count the
                # same configuration three times. Replicate 1 is
                # therefore the canonical scientific relaxation.
                if replicate == 1:

                    scientific_relaxation_rows.extend(
                        rows
                    )

            # ---------------------------------------------------------
            # OOM
            # ---------------------------------------------------------

            elif summary["status"] == "OOM":

                print(

                    f"Stopping batch "
                    f"{batch_size} "
                    "after OOM."

                )

                break

    # -----------------------------------------------------------------
    # Aggregate replicates
    # -----------------------------------------------------------------

    summaries = aggregate(

        replicate_rows=replicate_summaries,

        batch_sizes=batch_sizes,

        serial_wall=serial_wall,

    )

    # -----------------------------------------------------------------
    # Write CSVs
    # -----------------------------------------------------------------

    # Every measured replicate.
    replace_method_rows(

        outdir
        / "performance_replicates.csv",

        replicate_summaries,

        "alchemi_fixed",

    )

    # Mean / std / median for plotting.
    replace_method_rows(

        outdir
        / "performance_summary.csv",

        summaries,

        "alchemi_fixed",

    )

    # Every structure from every replicate.
    replace_method_rows(

        outdir
        / "relaxation_results_replicates.csv",

        all_relaxation_rows,

        "alchemi_fixed",

    )

    # Replicate 1 only for downstream scientific/DFT analysis.
    replace_method_rows(

        outdir
        / "relaxation_results.csv",

        scientific_relaxation_rows,

        "alchemi_fixed",

    )

    # -----------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------

    replicate_df = pd.DataFrame(
        replicate_summaries
    )

    aggregate_df = pd.DataFrame(
        summaries
    )

    print()
    print("=" * 78)
    print(" Replicate-level results")
    print("=" * 78)

    replicate_columns = [

        "method",
        "batch_size",
        "replicate",
        "wall_s",
        "structures_per_s",
        "atoms_per_s",
        "peak_nvidia_smi_memory_mb",
        "mean_gpu_util_pct",
        "status",

    ]

    print(

        replicate_df[
            replicate_columns
        ].to_string(
            index=False
        )

    )

    print()
    print("=" * 78)
    print(" Aggregate steady-state results")
    print("=" * 78)

    aggregate_columns = [

        "method",
        "batch_size",
        "n_successful_replicates",
        "wall_s",
        "wall_s_std",
        "structures_per_s",
        "structures_per_s_std",
        "speedup_vs_batch1",
        "peak_nvidia_smi_memory_mb",
        "mean_gpu_util_pct",
        "status",

    ]

    print(

        aggregate_df
        .reindex(
            columns=aggregate_columns
        )
        .to_string(
            index=False
        )

    )

    print()
    print("=" * 78)
    print(" Output files")
    print("=" * 78)

    print(
        outdir
        / "performance_replicates.csv"
    )

    print(
        outdir
        / "performance_summary.csv"
    )

    print(
        outdir
        / "relaxation_results_replicates.csv"
    )

    print(
        outdir
        / "relaxation_results.csv"
    )

    print(
        outdir
        / "gpu_samples.csv"
    )

    print(
        timing_dir
    )

    print()
    print(
        "Benchmark complete."
    )


if __name__ == "__main__":
    main()