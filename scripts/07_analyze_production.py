#!/usr/bin/env python3
"""
Phase 3 analysis for the ALCHEMI + UMA 500-structure production benchmark.

Reads the production benchmark CSV files and creates:
  - one summary table
  - B1 vs inflight B4 comparison tables
  - publication-ready diagnostic figures

Run from the repository root:
    python scripts/07_analyze_production.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results/production_500")
OUT = ROOT / "analysis"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

METHODS = [
    {
        "label": "B1",
        "perf": ROOT / "fixed_b01" / "performance_replicates.csv",
        "relax": ROOT / "fixed_b01" / "relaxation_results_replicates.csv",
    },
    {
        "label": "Fixed B4",
        "perf": ROOT / "fixed_b04" / "performance_replicates.csv",
        "relax": ROOT / "fixed_b04" / "relaxation_results_replicates.csv",
    },
    {
        "label": "Inflight B4",
        "perf": ROOT / "inflight_b04_a360" / "inflight_performance_replicates.csv",
        "relax": ROOT / "inflight_b04_a360" / "inflight_relaxation_results_replicates.csv",
    },
    {
        "label": "Inflight B8",
        "perf": ROOT / "inflight_b08_a720" / "inflight_performance_replicates.csv",
        "relax": ROOT / "inflight_b08_a720" / "inflight_relaxation_results_replicates.csv",
    },
    {
        "label": "Inflight B16",
        "perf": ROOT / "inflight_b16_a1440" / "inflight_performance_replicates.csv",
        "relax": ROOT / "inflight_b16_a1440" / "inflight_relaxation_results_replicates.csv",
    },
]


def require(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return path


def bool_series(s):
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False})


# ---------------------------------------------------------------------
# 1. Build one production summary table
# ---------------------------------------------------------------------
summary_rows = []

for item in METHODS:
    perf = pd.read_csv(require(item["perf"]))
    relax = pd.read_csv(require(item["relax"]))
    relax["converged"] = bool_series(relax["converged"])

    per_rep_conv = relax.groupby("replicate")["converged"].mean()

    summary_rows.append(
        {
            "method": item["label"],
            "n_replicates": int(perf["replicate"].nunique()),
            "wall_s_mean": perf["wall_s"].mean(),
            "wall_s_std": perf["wall_s"].std(ddof=1),
            "wall_min_mean": perf["wall_s"].mean() / 60.0,
            "structures_per_s_mean": perf["structures_per_s"].mean(),
            "structures_per_s_std": perf["structures_per_s"].std(ddof=1),
            "atoms_per_s_mean": perf["atoms_per_s"].mean(),
            "gpu_util_pct_mean": perf["mean_gpu_util_pct"].mean(),
            "gpu_util_pct_std": perf["mean_gpu_util_pct"].std(ddof=1),
            "peak_gpu_memory_gb": perf["peak_nvidia_smi_memory_mb"].max() / 1024.0,
            "convergence_fraction_mean": per_rep_conv.mean(),
            "convergence_fraction_std": per_rep_conv.std(ddof=1),
        }
    )

summary = pd.DataFrame(summary_rows)

b1_rate = summary.loc[summary["method"] == "B1", "structures_per_s_mean"].iloc[0]
b1_time = summary.loc[summary["method"] == "B1", "wall_s_mean"].iloc[0]
summary["speedup_vs_b1"] = summary["structures_per_s_mean"] / b1_rate
summary["time_saved_vs_b1_pct"] = (1.0 - summary["wall_s_mean"] / b1_time) * 100.0

summary.to_csv(OUT / "production_summary.csv", index=False)


# ---------------------------------------------------------------------
# 2. B1 vs inflight B4: structure-by-structure comparison
# ---------------------------------------------------------------------
b1 = pd.read_csv(require(ROOT / "fixed_b01" / "relaxation_results_replicates.csv"))
inf = pd.read_csv(
    require(
        ROOT
        / "inflight_b04_a360"
        / "inflight_relaxation_results_replicates.csv"
    )
)

b1["converged"] = bool_series(b1["converged"])
inf["converged"] = bool_series(inf["converged"])

keys = ["sample_idx", "replicate"]

comparison = b1.merge(
    inf,
    on=keys,
    how="inner",
    suffixes=("_b1", "_inflight"),
    validate="one_to_one",
)

if len(comparison) != 1500:
    raise RuntimeError(
        f"Expected 1500 B1/inflight matched rows, found {len(comparison)}."
    )

comparison["same_convergence"] = (
    comparison["converged_b1"] == comparison["converged_inflight"]
)
comparison["same_steps"] = (
    comparison["n_steps_b1"] == comparison["n_steps_inflight"]
)
comparison["delta_steps"] = (
    comparison["n_steps_inflight"] - comparison["n_steps_b1"]
)
comparison["delta_energy_eV"] = (
    comparison["final_energy_eV_inflight"]
    - comparison["final_energy_eV_b1"]
)
comparison["abs_delta_energy_eV"] = comparison["delta_energy_eV"].abs()
comparison["abs_delta_energy_meV_atom"] = (
    comparison["abs_delta_energy_eV"]
    / comparison["n_atoms_b1"]
    * 1000.0
)
comparison["delta_fmax_eV_A"] = (
    comparison["final_fmax_eV_A_inflight"]
    - comparison["final_fmax_eV_A_b1"]
)

comparison.to_csv(OUT / "b1_vs_inflight_b04_all.csv", index=False)

disagreements = comparison.loc[~comparison["same_convergence"]].copy()
disagreements.to_csv(
    OUT / "b1_vs_inflight_b04_convergence_disagreements.csv",
    index=False,
)

both_converged = comparison.loc[
    comparison["converged_b1"] & comparison["converged_inflight"]
].copy()
both_converged = both_converged.sort_values(
    "abs_delta_energy_meV_atom", ascending=False
)
both_converged.to_csv(
    OUT / "b1_vs_inflight_b04_both_converged.csv",
    index=False,
)

# Per-structure stability across the 3 repeats.
per_structure = comparison.groupby("sample_idx").agg(
    sid=("sid_b1", "first"),
    system_id=("system_id_b1", "first"),
    n_atoms=("n_atoms_b1", "first"),
    convergence_agreement_fraction=("same_convergence", "mean"),
    exact_step_agreement_fraction=("same_steps", "mean"),
    mean_abs_delta_energy_meV_atom=("abs_delta_energy_meV_atom", "mean"),
    max_abs_delta_energy_meV_atom=("abs_delta_energy_meV_atom", "max"),
)
per_structure.to_csv(
    OUT / "b1_vs_inflight_b04_per_structure.csv",
    index=False,
)


# ---------------------------------------------------------------------
# 3. Plain-text numerical summary
# ---------------------------------------------------------------------
agreement = comparison["same_convergence"].mean()
step_agreement = comparison["same_steps"].mean()

text_lines = [
    "ALCHEMI + UMA production analysis",
    "==================================",
    "",
    "Performance summary:",
    summary.to_string(index=False),
    "",
    "B1 vs inflight B4:",
    f"Matched comparisons: {len(comparison)}",
    f"Convergence agreement: {agreement * 100:.2f}%",
    f"Exact step-count agreement: {step_agreement * 100:.2f}%",
    f"Convergence disagreements: {len(disagreements)}",
    f"Both converged comparisons: {len(both_converged)}",
    (
        "Mean |energy difference| for both-converged cases: "
        f"{both_converged['abs_delta_energy_eV'].mean():.6g} eV"
    ),
    (
        "Mean |energy difference| per atom for both-converged cases: "
        f"{both_converged['abs_delta_energy_meV_atom'].mean():.6g} meV/atom"
    ),
    (
        "95th percentile |energy difference| per atom: "
        f"{both_converged['abs_delta_energy_meV_atom'].quantile(0.95):.6g} meV/atom"
    ),
    (
        "Maximum |energy difference| per atom: "
        f"{both_converged['abs_delta_energy_meV_atom'].max():.6g} meV/atom"
    ),
]
(OUT / "analysis_summary.txt").write_text("\n".join(text_lines) + "\n")


# ---------------------------------------------------------------------
# 4. Figures
# ---------------------------------------------------------------------
labels = summary["method"].tolist()
x = np.arange(len(labels))

# Figure 1: throughput
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x,
    summary["structures_per_s_mean"],
    yerr=summary["structures_per_s_std"].fillna(0),
    capsize=4,
)
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("Structures per second")
ax.set_title("How fast each method processes structures")
fig.tight_layout()
fig.savefig(FIG / "fig01_throughput.png", dpi=300)
plt.close(fig)

# Figure 2: time to finish 500 structures
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x,
    summary["wall_min_mean"],
    yerr=summary["wall_s_std"].fillna(0) / 60.0,
    capsize=4,
)
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("Minutes for 500 structures")
ax.set_title("Time needed to finish the 500-structure workload")
fig.tight_layout()
fig.savefig(FIG / "fig02_runtime.png", dpi=300)
plt.close(fig)

# Figure 3: GPU utilization
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x,
    summary["gpu_util_pct_mean"],
    yerr=summary["gpu_util_pct_std"].fillna(0),
    capsize=4,
)
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("Mean GPU utilization (%)")
ax.set_ylim(0, 100)
ax.set_title("A busier GPU is not always a faster workflow")
fig.tight_layout()
fig.savefig(FIG / "fig03_gpu_utilization.png", dpi=300)
plt.close(fig)

# Figure 4: peak GPU memory
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x, summary["peak_gpu_memory_gb"])
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("Peak GPU memory (GB)")
ax.set_title("Peak GPU memory use")
fig.tight_layout()
fig.savefig(FIG / "fig04_gpu_memory.png", dpi=300)
plt.close(fig)

# Figure 5: convergence fraction
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(
    x,
    summary["convergence_fraction_mean"] * 100.0,
    yerr=summary["convergence_fraction_std"].fillna(0) * 100.0,
    capsize=4,
)
ax.set_xticks(x, labels, rotation=20, ha="right")
ax.set_ylabel("Structures reaching the force target (%)")
ax.set_ylim(0, 100)
ax.set_title("Fraction of structures that finish within 300 steps")
fig.tight_layout()
fig.savefig(FIG / "fig05_convergence_fraction.png", dpi=300)
plt.close(fig)

# Figure 6: B1 vs inflight step counts
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    comparison["n_steps_b1"],
    comparison["n_steps_inflight"],
    s=12,
    alpha=0.45,
)
ax.plot([0, 300], [0, 300], linestyle="--", linewidth=1)
ax.set_xlim(0, 305)
ax.set_ylim(0, 305)
ax.set_xlabel("B1 optimization steps")
ax.set_ylabel("Inflight B4 optimization steps")
ax.set_title("B1 and inflight usually stop at the same step")
fig.tight_layout()
fig.savefig(FIG / "fig06_step_agreement.png", dpi=300)
plt.close(fig)

# Figure 7: energy differences when both methods converge
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(
    both_converged["abs_delta_energy_meV_atom"],
    bins=40,
)
ax.set_xlabel("|B1 - inflight energy| (meV/atom)")
ax.set_ylabel("Number of comparisons")
ax.set_title("Energy differences when both methods successfully relax")
fig.tight_layout()
fig.savefig(FIG / "fig07_energy_difference.png", dpi=300)
plt.close(fig)

print("\nProduction analysis complete.")
print(f"Summary: {OUT / 'production_summary.csv'}")
print(f"Text summary: {OUT / 'analysis_summary.txt'}")
print(f"Comparison table: {OUT / 'b1_vs_inflight_b04_all.csv'}")
print(f"Disagreements: {OUT / 'b1_vs_inflight_b04_convergence_disagreements.csv'}")
print(f"Figures: {FIG}")
print()
print(f"B1 vs inflight convergence agreement: {agreement * 100:.2f}%")
print(f"Exact step-count agreement: {step_agreement * 100:.2f}%")