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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alchemi_uma.conversion import atoms_list_to_batch
from alchemi_uma.dataset import OCPDenseLMDB
from alchemi_uma.io import replace_method_rows
from alchemi_uma.models import load_alchemi_uma
from alchemi_uma.telemetry import NvidiaSMITelemetry

from nvalchemi.dynamics import FIRE, SizeAwareSampler
from nvalchemi.dynamics.base import (
    ConvergenceHook,
    DynamicsStage,
    FusedStage,
)
from nvalchemi.dynamics.hooks import (
    FreezeAtomsHook,
    StageTimingHook,
)
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import DynamicsContext


# =====================================================================
# CUDA helpers
# =====================================================================

def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_gpu(device: str) -> None:
    gc.collect()

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def scalar(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(
            x.detach()
            .reshape(-1)[0]
            .cpu()
            .item()
        )

    return int(x)


# =====================================================================
# Step clock
# =====================================================================

class StepClock:
    """
    Lightweight global FIRE-step clock.

    This performs no GPU -> CPU tensor copies, so it should add
    negligible overhead to the benchmark.
    """

    stage = DynamicsStage.BEFORE_STEP
    frequency = 1

    def __init__(self):
        self.current_step = -1

    def __call__(
        self,
        ctx: DynamicsContext,
        stage: DynamicsStage,
    ) -> None:

        self.current_step = int(
            ctx.step_count
        )


# =====================================================================
# OC20-Dense -> SizeAwareSampler adapter
# =====================================================================

class OC20InflightDataset:
    """
    Adapter exposing the interface required by SizeAwareSampler.

    Required methods:
        __len__()
        __getitem__()
        get_metadata()
    """

    def __init__(
        self,
        ds,
        manifest,
        device,
        clock,
    ):

        self.ds = ds

        self.manifest = (
            manifest
            .reset_index(drop=True)
            .copy()
        )

        self.device = device
        self.clock = clock

        # source_idx -> global step when the structure entered
        self.load_step: dict[int, int] = {}

    def __len__(self):

        return len(
            self.manifest
        )

    def get_metadata(
        self,
        idx,
    ):

        # UMA constructs its graph internally.
        # Therefore edge-count gating is unnecessary.

        n_atoms = int(
            self.manifest
            .iloc[idx]
            .n_atoms
        )

        return n_atoms, 0

    def __getitem__(
        self,
        idx,
    ):

        rec = self.manifest.iloc[idx]

        atoms = self.ds.get_atoms(
            int(rec.lmdb_index)
        )

        # -------------------------------------------------------------
        # IMPORTANT
        #
        # Reuse exactly the same conversion function that passed
        # Script 03 and Script 05.
        #
        # This preserves the OC20 fixed-atom categories needed by
        # FreezeAtomsHook.
        # -------------------------------------------------------------

        batch = atoms_list_to_batch(
            [atoms],
            self.device,
        )

        data = batch.to_data_list()[0]

        # -------------------------------------------------------------
        # Stable identity
        #
        # SizeAwareSampler creates its own system_id. We additionally
        # store source_idx so results can always be mapped back to the
        # manifest even if the sampler changes processing order.
        # -------------------------------------------------------------

        data.add_system_property(
            "source_idx",
            torch.tensor(
                [[idx]],
                dtype=torch.long,
                device=data.device,
            ),
        )

        # Initial structures are loaded before step 0.
        # Replacement structures enter on the following step.

        self.load_step[idx] = max(
            0,
            self.clock.current_step + 1,
        )

        metadata = {
            "source_idx": idx,
            "lmdb_index": int(
                rec.lmdb_index
            ),
        }

        return data, metadata


# =====================================================================
# HostMemory sink preserving IDs
# =====================================================================

class HostMemoryWithIDs(HostMemory):
    """
    HostMemory that preserves:

        system_id
        source_idx

    across Batch -> AtomicData conversion.
    """

    def write(
        self,
        batch,
        mask=None,
    ) -> None:

        n = batch.num_graphs or 0

        if n == 0:
            return

        # -------------------------------------------------------------
        # Apply selection mask if supplied
        # -------------------------------------------------------------

        if mask is not None:

            mask = mask.to(
                device=batch.device,
                dtype=torch.bool,
            )

            if mask.shape[0] != n:

                raise ValueError(
                    f"mask length "
                    f"{mask.shape[0]} "
                    f"!= num_graphs {n}"
                )

            selected = int(
                mask.sum().item()
            )

            if selected == 0:
                return

            if selected < n:

                indices = torch.nonzero(
                    mask,
                    as_tuple=True,
                )[0]

                # Trigger lazy batch_ptr creation before selection.
                _ = batch.batch_ptr

                batch = batch.index_select(
                    indices
                )

        # -------------------------------------------------------------
        # Save IDs before unbatching
        # -------------------------------------------------------------

        system_ids = (
            batch["system_id"]
            .detach()
            .cpu()
        )

        source_ids = (
            batch["source_idx"]
            .detach()
            .cpu()
        )

        items = batch.to_data_list()

        # -------------------------------------------------------------
        # Capacity check
        # -------------------------------------------------------------

        if (
            len(self._data_list)
            + len(items)
            > self._capacity
        ):

            raise RuntimeError(
                "HostMemoryWithIDs "
                "capacity exceeded."
            )

        # -------------------------------------------------------------
        # Store CPU copies
        # -------------------------------------------------------------

        for i, data in enumerate(items):

            data = data.to(
                self._device
            )

            data.add_system_property(
                "system_id",
                system_ids[
                    i : i + 1
                ],
            )

            data.add_system_property(
                "source_idx",
                source_ids[
                    i : i + 1
                ],
            )

            self._data_list.append(
                data
            )


# =====================================================================
# Convergence tracker
# =====================================================================

class ConvergenceTracker:
    """
    Record the number of FIRE steps used by structures that converge.

    This hook only performs CPU transfers when a convergence event
    occurs, rather than on every FIRE step.
    """

    stage = DynamicsStage.ON_CONVERGE
    frequency = 1

    def __init__(
        self,
        dataset,
    ):

        self.dataset = dataset

        self.steps: dict[
            int,
            int,
        ] = {}

    def __call__(
        self,
        ctx: DynamicsContext,
        stage: DynamicsStage,
    ) -> None:

        mask = ctx.converged_mask

        if mask is None:
            return

        # -------------------------------------------------------------
        # Convert convergence result to graph indices
        # -------------------------------------------------------------

        if mask.dtype == torch.bool:

            indices = torch.nonzero(
                mask,
                as_tuple=True,
            )[0]

        else:

            indices = (
                mask
                .long()
                .reshape(-1)
            )

        if indices.numel() == 0:
            return

        source = (
            ctx.batch["source_idx"]
            .reshape(-1)
        )

        # ctx.step_count is zero-based at hook time.
        completed_step = (
            int(ctx.step_count)
            + 1
        )

        # -------------------------------------------------------------
        # Record local steps since each structure entered the live batch
        # -------------------------------------------------------------

        for graph_idx in (
            indices
            .detach()
            .cpu()
            .tolist()
        ):

            src = int(
                source[
                    graph_idx
                ]
                .detach()
                .cpu()
                .item()
            )

            entered = (
                self.dataset
                .load_step
                .get(
                    src,
                    0,
                )
            )

            self.steps[src] = max(
                1,
                completed_step
                - entered,
            )


# =====================================================================
# Warm-up
# =====================================================================

def warm_up(
    model,
    ds,
    manifest,
    args,
):

    """
    Trigger UMA compilation before benchmark timing.
    """

    atoms_list = []
    atom_total = 0

    # -------------------------------------------------------------
    # Construct a representative capacity-limited batch
    # -------------------------------------------------------------

    for _, rec in manifest.iterrows():

        n_atoms = int(
            rec.n_atoms
        )

        if (
            len(atoms_list)
            >= args.max_batch_size
        ):
            break

        if (
            atoms_list
            and atom_total + n_atoms
            > args.max_atoms
        ):
            continue

        if (
            not atoms_list
            and n_atoms
            > args.max_atoms
        ):

            raise RuntimeError(
                f"Structure has "
                f"{n_atoms} atoms but "
                f"--max-atoms="
                f"{args.max_atoms}."
            )

        atoms_list.append(
            ds.get_atoms(
                int(
                    rec.lmdb_index
                )
            )
        )

        atom_total += n_atoms

    if not atoms_list:

        raise RuntimeError(
            "Could not construct "
            "warm-up batch."
        )

    batch = atoms_list_to_batch(
        atoms_list,
        args.device,
    )

    print()
    print(
        "=== Inflight model warm-up "
        "(excluded from timing) ==="
    )

    print(
        f"Warm-up batch: "
        f"{batch.num_graphs} structures, "
        f"{batch.num_nodes} atoms"
    )

    # -------------------------------------------------------------
    # Two passes:
    #
    # 1. compilation
    # 2. steady-state confirmation
    #
    # DO NOT use inference_mode/no_grad:
    # UMA forces require autograd.
    # -------------------------------------------------------------

    for i in range(2):

        outputs = model(
            batch
        )

        _ = (
            outputs["energy"]
            .detach()
        )

        _ = (
            outputs["forces"]
            .detach()
        )

        sync(
            args.device
        )

        del outputs

        print(
            f"Warm-up pass "
            f"{i + 1}/2 complete."
        )

    del batch

    reset_gpu(
        args.device
    )


# =====================================================================
# Data helpers
# =====================================================================

def get_prop(
    data,
    key,
):

    try:
        return data[key]

    except Exception:
        return getattr(
            data,
            key,
        )


# =====================================================================
# Extract final structures
# =====================================================================

def collect_results(
    sink,
    tracker,
    manifest,
    replicate,
    run_id,
    args,
):

    rows = []

    for data in sink._data_list:

        source_idx = scalar(
            get_prop(
                data,
                "source_idx",
            )
        )

        inflight_system_id = scalar(
            get_prop(
                data,
                "system_id",
            )
        )

        rec = manifest.iloc[
            source_idx
        ]

        # -------------------------------------------------------------
        # Final energy
        # -------------------------------------------------------------

        final_energy = float(
            data.energy
            .detach()
            .reshape(-1)[0]
            .cpu()
            .item()
        )

        # -------------------------------------------------------------
        # Final fmax
        #
        # FreezeAtomsHook zeros frozen-atom forces, so this represents
        # the active relaxation degrees of freedom.
        # -------------------------------------------------------------

        final_fmax = float(
            torch.linalg.vector_norm(
                data.forces.detach(),
                dim=-1,
            )
            .max()
            .cpu()
            .item()
        )

        converged = (
            final_fmax
            <= args.fmax
        )

        # If it did not fire the convergence hook, it graduated through
        # the per-system 300-step ceiling.

        n_steps = (
            tracker.steps.get(
                source_idx,
                args.max_steps,
            )
        )

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

                "source_idx":
                    source_idx,

                "inflight_system_id":
                    inflight_system_id,

                "method":
                    "alchemi_inflight",

                "batch_size":
                    args.max_batch_size,

                "max_batch_size":
                    args.max_batch_size,

                "max_atoms":
                    args.max_atoms,

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
                        converged
                    ),

                "n_steps":
                    int(
                        n_steps
                    ),

                "final_energy_eV":
                    final_energy,

                "final_fmax_eV_A":
                    final_fmax,

                "exit_reason":
                    (
                        "converged"
                        if converged
                        else "max_steps"
                    ),

                "status":
                    "ok",
            }
        )

    rows.sort(
        key=lambda row:
        row["source_idx"]
    )

    return rows


# =====================================================================
# One inflight replicate
# =====================================================================

def run_replicate(
    model,
    ds,
    manifest,
    replicate,
    args,
    outdir,
    timing_dir,
    job_id,
):

    # -------------------------------------------------------------
    # Every replicate gets a fresh sampler.
    #
    # SizeAwareSampler is stateful: consumed samples cannot be reused.
    # -------------------------------------------------------------

    clock = StepClock()

    dataset = OC20InflightDataset(
        ds,
        manifest,
        args.device,
        clock,
    )

    sampler = SizeAwareSampler(
        dataset,
        max_atoms=args.max_atoms,
        max_edges=None,
        max_batch_size=(
            args.max_batch_size
        ),
        shuffle=False,
    )

    # -------------------------------------------------------------
    # Sink
    # -------------------------------------------------------------

    sink = HostMemoryWithIDs(
        capacity=(
            len(dataset)
            + 16
        )
    )

    # -------------------------------------------------------------
    # Tracking
    # -------------------------------------------------------------

    tracker = ConvergenceTracker(
        dataset
    )

    # -------------------------------------------------------------
    # Stage timing
    # -------------------------------------------------------------

    timing_path = (
        timing_dir
        / (
            f"alchemi_inflight"
            f"_b{args.max_batch_size:02d}"
            f"_a{args.max_atoms}"
            f"_r{replicate:02d}.csv"
        )
    )

    timing_hook = StageTimingHook(
        "detailed",
        log_path=timing_path,
        timer_backend="auto",
        enable_nvtx=True,
        show_console=False,
    )

    # -------------------------------------------------------------
    # FIRE
    # -------------------------------------------------------------

    fire = FIRE(
        model=model,
        dt=args.dt,

        # IMPORTANT:
        # This is a PER-SYSTEM FIRE budget inside FusedStage.
        n_steps=args.max_steps,

        convergence_hook=(
            ConvergenceHook.from_forces(
                threshold=args.fmax
            )
        ),

        hooks=[
            FreezeAtomsHook(),
            clock,
            tracker,
            timing_hook,
        ],
    )

    # -------------------------------------------------------------
    # FusedStage inflight pipeline
    #
    # A one-stage FusedStage is intentional:
    #
    #      FIRE -> graduate -> refill
    #
    # -------------------------------------------------------------

    fused = FusedStage(
        sub_stages=[
            (
                0,
                fire,
            )
        ],

        sampler=sampler,

        sinks=[
            sink
        ],

        refill_frequency=(
            args.refill_frequency
        ),
    )

    # -------------------------------------------------------------
    # Run ID
    # -------------------------------------------------------------

    run_id = (
        f"alchemi_inflight"
        f"_b{args.max_batch_size:02d}"
        f"_a{args.max_atoms}"
        f"_n{len(manifest)}"
        f"_r{replicate:02d}"
        f"_job{job_id}"
    )

    # -------------------------------------------------------------
    # GPU telemetry
    # -------------------------------------------------------------

    telemetry = NvidiaSMITelemetry(
        run_id,
        outdir
        / "gpu_samples.csv",
        args.telemetry_interval,
    )

    # -------------------------------------------------------------
    # Global safety ceiling
    #
    # This is deliberately much larger than the expected run.
    #
    # FusedStage terminates early once:
    #     sampler exhausted
    #     AND
    #     all active structures graduated.
    #
    # Worst case:
    # every structure runs alone for its full 300-step budget.
    # -------------------------------------------------------------

    global_budget = (
        args.max_steps
        * len(dataset)
        + len(dataset)
        + 10
    )

    print()
    print(
        "=" * 78
    )

    print(
        f"Inflight replicate "
        f"{replicate}/"
        f"{args.replicates}"
    )

    print(
        f"run_id: "
        f"{run_id}"
    )

    print(
        f"max_batch_size: "
        f"{args.max_batch_size}"
    )

    print(
        f"max_atoms: "
        f"{args.max_atoms}"
    )

    print(
        f"refill_frequency: "
        f"{args.refill_frequency}"
    )

    print(
        f"per-system FIRE budget: "
        f"{args.max_steps}"
    )

    print(
        f"global safety budget: "
        f"{global_budget}"
    )

    print(
        "=" * 78
    )

    # -------------------------------------------------------------
    # Reset GPU statistics
    # -------------------------------------------------------------

    reset_gpu(
        args.device
    )

    status = "ok"
    error_message = ""
    result = None

    # -------------------------------------------------------------
    # Timed run
    # -------------------------------------------------------------

    try:

        with telemetry:

            sync(
                args.device
            )

            t0 = (
                time.perf_counter()
            )

            with fused:

                result = fused.run(
                    batch=None,
                    n_steps=global_budget,
                )

            sync(
                args.device
            )

            wall_s = (
                time.perf_counter()
                - t0
            )

    except torch.cuda.OutOfMemoryError as exc:

        wall_s = np.nan

        status = "OOM"

        error_message = str(
            exc
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    finally:

        timing_hook.close()

        try:
            sync(
                args.device
            )
        except Exception:
            pass

    # -------------------------------------------------------------
    # Peak Torch memory
    # -------------------------------------------------------------

    peak_torch_memory_mb = (

        torch.cuda
        .max_memory_allocated()
        / 1024**2

        if torch.cuda.is_available()

        else np.nan

    )

    # -------------------------------------------------------------
    # Extract graduated structures
    # -------------------------------------------------------------

    if status == "ok":

        relaxation_rows = (
            collect_results(
                sink,
                tracker,
                manifest,
                replicate,
                run_id,
                args,
            )
        )

    else:

        relaxation_rows = []

    # -------------------------------------------------------------
    # Completeness checks
    # -------------------------------------------------------------

    n_collected = len(
        relaxation_rows
    )

    n_converged = sum(
        int(
            row["converged"]
        )
        for row
        in relaxation_rows
    )

    if (
        status == "ok"
        and n_collected
        != len(dataset)
    ):

        status = "incomplete"

        error_message = (
            f"Collected "
            f"{n_collected}/"
            f"{len(dataset)} "
            f"systems; "
            f"active="
            f"{getattr(result, 'num_graphs', None)}"
        )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------

    total_atoms = int(
        manifest
        .n_atoms
        .sum()
    )

    summary = {

        "run_id":
            run_id,

        "method":
            "alchemi_inflight",

        # Keep batch_size for compatibility with existing figure code.
        "batch_size":
            args.max_batch_size,

        "max_batch_size":
            args.max_batch_size,

        "max_atoms":
            args.max_atoms,

        "refill_frequency":
            args.refill_frequency,

        "replicate":
            replicate,

        "n_structures":
            len(dataset),

        "n_collected":
            n_collected,

        "n_converged":
            n_converged,

        "convergence_fraction":
            (
                n_converged
                / len(dataset)
            ),

        "n_atoms":
            total_atoms,

        "wall_s":
            wall_s,

        "structures_per_s":
            (
                len(dataset)
                / wall_s

                if (
                    status == "ok"
                    and wall_s > 0
                )

                else np.nan
            ),

        "atoms_per_s":
            (
                total_atoms
                / wall_s

                if (
                    status == "ok"
                    and wall_s > 0
                )

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
                torch.cuda
                .get_device_name(0)
            ),

        "host":
            platform.node(),

        "status":
            status,

        "error_message":
            error_message,
    }

    return (
        relaxation_rows,
        summary,
    )


# =====================================================================
# Aggregate replicates
# =====================================================================

def aggregate(
    replicate_rows,
):

    df = pd.DataFrame(
        replicate_rows
    )

    ok = df[
        df.status
        == "ok"
    ]

    if ok.empty:

        return {
            "run_id":
                "alchemi_inflight_aggregate",

            "method":
                "alchemi_inflight",

            "n_replicates":
                len(df),

            "n_successful_replicates":
                0,

            "status":
                "failed",
        }

    def std(
        column,
    ):

        if len(ok) <= 1:
            return 0.0

        return float(
            ok[column]
            .std(
                ddof=1
            )
        )

    return {

        "run_id":
            "alchemi_inflight_aggregate",

        "method":
            "alchemi_inflight",

        "batch_size":
            int(
                ok.max_batch_size
                .iloc[0]
            ),

        "max_batch_size":
            int(
                ok.max_batch_size
                .iloc[0]
            ),

        "max_atoms":
            int(
                ok.max_atoms
                .iloc[0]
            ),

        "refill_frequency":
            int(
                ok.refill_frequency
                .iloc[0]
            ),

        "n_replicates":
            len(df),

        "n_successful_replicates":
            len(ok),

        "n_structures":
            int(
                ok.n_structures
                .iloc[0]
            ),

        "n_atoms":
            int(
                ok.n_atoms
                .iloc[0]
            ),

        "wall_s":
            float(
                ok.wall_s
                .mean()
            ),

        "wall_s_std":
            std(
                "wall_s"
            ),

        "wall_s_median":
            float(
                ok.wall_s
                .median()
            ),

        "structures_per_s":
            float(
                ok.structures_per_s
                .mean()
            ),

        "structures_per_s_std":
            std(
                "structures_per_s"
            ),

        "atoms_per_s":
            float(
                ok.atoms_per_s
                .mean()
            ),

        "atoms_per_s_std":
            std(
                "atoms_per_s"
            ),

        "peak_torch_memory_mb":
            float(
                ok.peak_torch_memory_mb
                .max()
            ),

        "peak_nvidia_smi_memory_mb":
            float(
                ok.peak_nvidia_smi_memory_mb
                .max()
            ),

        "mean_gpu_util_pct":
            float(
                ok.mean_gpu_util_pct
                .mean()
            ),

        "mean_gpu_util_pct_std":
            std(
                "mean_gpu_util_pct"
            ),

        "convergence_fraction":
            float(
                ok.convergence_fraction
                .mean()
            ),

        "model":
            ok.model.iloc[0],

        "task":
            ok.task.iloc[0],

        "inference_settings":
            ok.inference_settings
            .iloc[0],

        "gpu_name":
            ok.gpu_name.iloc[0],

        "host":
            ok.host.iloc[0],

        "status":
            "ok",
    }


# =====================================================================
# Main
# =====================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "ALCHEMI + UMA inflight "
            "FIRE benchmark on OC20-Dense."
        )
    )

    # -------------------------------------------------------------
    # Data
    # -------------------------------------------------------------

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--lmdb-root",
        required=True,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Inflight capacity
    # -------------------------------------------------------------

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--max-atoms",
        type=int,
        default=360,
    )

    parser.add_argument(
        "--refill-frequency",
        type=int,
        default=1,
    )

    # -------------------------------------------------------------
    # FIRE
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------

    parser.add_argument(
        "--replicates",
        type=int,
        default=3,
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

    # -------------------------------------------------------------
    # CUDA check
    # -------------------------------------------------------------

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):

        raise RuntimeError(
            "CUDA is not visible. "
            "Run Script 06 through "
            "a GPU SLURM allocation."
        )

    # -------------------------------------------------------------
    # Argument validation
    # -------------------------------------------------------------

    if args.max_batch_size < 1:
        raise ValueError(
            "--max-batch-size "
            "must be >= 1."
        )

    if args.max_atoms < 1:
        raise ValueError(
            "--max-atoms "
            "must be >= 1."
        )

    if args.refill_frequency < 1:
        raise ValueError(
            "--refill-frequency "
            "must be >= 1."
        )

    if args.replicates < 1:
        raise ValueError(
            "--replicates "
            "must be >= 1."
        )

    # -------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------

    manifest = pd.read_csv(
        args.manifest
    )

    if args.limit is not None:

        manifest = (
            manifest
            .head(
                args.limit
            )
            .copy()
        )

    manifest = (
        manifest
        .reset_index(
            drop=True
        )
    )

    required = {
        "sample_idx",
        "lmdb_index",
        "sid",
        "system_id",
        "n_atoms",
    }

    missing = (
        required
        - set(
            manifest.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing)}"
        )

    if manifest.empty:

        raise RuntimeError(
            "No structures selected."
        )

    largest = int(
        manifest
        .n_atoms
        .max()
    )

    if largest > args.max_atoms:

        raise RuntimeError(
            f"Largest structure has "
            f"{largest} atoms, "
            f"larger than "
            f"--max-atoms="
            f"{args.max_atoms}."
        )

    # -------------------------------------------------------------
    # Directories
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------

    ds = OCPDenseLMDB(
        args.lmdb_root
    )

    # -------------------------------------------------------------
    # Model
    # -------------------------------------------------------------

    print(
        "Loading ALCHEMI UMA..."
    )

    model = load_alchemi_uma(
        args.model,
        args.task,
        args.device,
        args.inference_settings,
    )

    model.eval()

    # -------------------------------------------------------------
    # Header
    # -------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        " ALCHEMI + UMA "
        "INFLIGHT PILOT"
    )

    print(
        "=" * 78
    )

    print(
        f"Structures:       "
        f"{len(manifest)}"
    )

    print(
        f"Total atoms:      "
        f"{int(manifest.n_atoms.sum())}"
    )

    print(
        f"Max batch size:   "
        f"{args.max_batch_size}"
    )

    print(
        f"Max atoms:        "
        f"{args.max_atoms}"
    )

    print(
        f"Refill frequency: "
        f"{args.refill_frequency}"
    )

    print(
        f"Fmax:             "
        f"{args.fmax} eV/Angstrom"
    )

    print(
        f"Per-system steps: "
        f"{args.max_steps}"
    )

    print(
        f"Replicates:       "
        f"{args.replicates}"
    )

    print(
        f"GPU:              "
        f"{torch.cuda.get_device_name(0)}"
    )

    print(
        "=" * 78
    )

    # -------------------------------------------------------------
    # Warm-up
    # -------------------------------------------------------------

    if not args.skip_warmup:

        warm_up(
            model,
            ds,
            manifest,
            args,
        )

    # -------------------------------------------------------------
    # Job ID
    # -------------------------------------------------------------

    job_id = os.environ.get(
        "SLURM_JOB_ID",
        time.strftime(
            "%Y%m%d%H%M%S"
        ),
    )

    replicate_rows = []
    relaxation_rows = []
    canonical_rows = []

    # -------------------------------------------------------------
    # Run replicates
    # -------------------------------------------------------------

    for replicate in range(
        1,
        args.replicates + 1,
    ):

        rows, summary = (
            run_replicate(
                model,
                ds,
                manifest,
                replicate,
                args,
                outdir,
                timing_dir,
                job_id,
            )
        )

        replicate_rows.append(
            summary
        )

        relaxation_rows.extend(
            rows
        )

        # Replicate 1 is the canonical scientific output.
        if replicate == 1:

            canonical_rows.extend(
                rows
            )

    # -------------------------------------------------------------
    # Aggregate
    # -------------------------------------------------------------

    aggregate_row = aggregate(
        replicate_rows
    )

    # -------------------------------------------------------------
    # Campaign-wide CSVs
    # -------------------------------------------------------------

    replace_method_rows(
        outdir
        / "performance_replicates.csv",

        replicate_rows,

        "alchemi_inflight",
    )

    replace_method_rows(
        outdir
        / "performance_summary.csv",

        [
            aggregate_row
        ],

        "alchemi_inflight",
    )

    replace_method_rows(
        outdir
        / "relaxation_results_replicates.csv",

        relaxation_rows,

        "alchemi_inflight",
    )

    replace_method_rows(
        outdir
        / "relaxation_results.csv",

        canonical_rows,

        "alchemi_inflight",
    )

    # -------------------------------------------------------------
    # Dedicated inflight CSVs
    # -------------------------------------------------------------

    pd.DataFrame(
        replicate_rows
    ).to_csv(
        outdir
        / "inflight_performance_replicates.csv",
        index=False,
    )

    pd.DataFrame(
        [
            aggregate_row
        ]
    ).to_csv(
        outdir
        / "inflight_performance_summary.csv",
        index=False,
    )

    pd.DataFrame(
        relaxation_rows
    ).to_csv(
        outdir
        / "inflight_relaxation_results_replicates.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Console summaries
    # -------------------------------------------------------------

    rep = pd.DataFrame(
        replicate_rows
    )

    print()
    print(
        "=" * 78
    )

    print(
        " Inflight replicate-level results"
    )

    print(
        "=" * 78
    )

    columns = [
        "replicate",
        "wall_s",
        "structures_per_s",
        "atoms_per_s",
        "n_collected",
        "n_converged",
        "convergence_fraction",
        "peak_nvidia_smi_memory_mb",
        "mean_gpu_util_pct",
        "status",
    ]

    print(
        rep[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        " Inflight aggregate result"
    )

    print(
        "=" * 78
    )

    print(
        pd.DataFrame(
            [
                aggregate_row
            ]
        ).to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Exit behavior
    # -------------------------------------------------------------

    if relaxation_rows:

        relax_df = pd.DataFrame(
            relaxation_rows
        )

        print()
        print(
            "=" * 78
        )

        print(
            " Per-system exit summary"
        )

        print(
            "=" * 78
        )

        print(
            relax_df
            .groupby(
                [
                    "replicate",
                    "exit_reason",
                ]
            )
            .size()
            .rename(
                "count"
            )
            .reset_index()
            .to_string(
                index=False
            )
        )

    # -------------------------------------------------------------
    # Output paths
    # -------------------------------------------------------------

    print()
    print(
        "=" * 78
    )

    print(
        " Output files"
    )

    print(
        "=" * 78
    )

    print(
        outdir
        / "inflight_performance_replicates.csv"
    )

    print(
        outdir
        / "inflight_performance_summary.csv"
    )

    print(
        outdir
        / "inflight_relaxation_results_replicates.csv"
    )

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
        "Inflight benchmark complete."
    )


if __name__ == "__main__":
    main()