#!/usr/bin/env python3
"""
Phase 3b: final analysis figures and review tables for the
ALCHEMI + UMA 500-structure production benchmark.

Run from repository root:
    python scripts/08_make_final_figures.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results/production_500")
ANALYSIS = ROOT / "analysis"
OUT = ANALYSIS / "final"
FIG = OUT / "figures"

OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(ANALYSIS / "production_summary.csv")
comp = pd.read_csv(ANALYSIS / "b1_vs_inflight_b04_all.csv")

# ------------------------------------------------------------------
# Correct per-structure table: preserve sample_idx explicitly.
# ------------------------------------------------------------------
per_structure = (
    comp.groupby("sample_idx", as_index=False)
    .agg(
        sid=("sid_b1", "first"),
        system_id=("system_id_b1", "first"),
        n_atoms=("n_atoms_b1", "first"),
        convergence_agreement_fraction=("same_convergence", "mean"),
        exact_step_agreement_fraction=("same_steps", "mean"),
        mean_abs_delta_energy_meV_atom=("abs_delta_energy_meV_atom", "mean"),
        max_abs_delta_energy_meV_atom=("abs_delta_energy_meV_atom", "max"),
    )
)
per_structure.to_csv(OUT / "b1_vs_inflight_per_structure_corrected.csv", index=False)

# ------------------------------------------------------------------
# Review the convergence disagreements.
# ------------------------------------------------------------------
dis = comp.loc[~comp["same_convergence"]].copy()

direction = np.where(
    dis["converged_b1"] & ~dis["converged_inflight"],
    "B1 converged; inflight did not",
    "Inflight converged; B1 did not",
)
dis.insert(0, "disagreement_direction", direction)

review_cols = [
    "disagreement_direction",
    "sample_idx",
    "sid_b1",
    "system_id_b1",
    "replicate",
    "n_atoms_b1",
    "converged_b1",
    "n_steps_b1",
    "final_fmax_eV_A_b1",
    "converged_inflight",
    "n_steps_inflight",
    "final_fmax_eV_A_inflight",
    "abs_delta_energy_meV_atom",
]
dis[review_cols].to_csv(OUT / "convergence_disagreements_to_review.csv", index=False)

# Structure-level outlier list.
outlier_ids = sorted(dis["sample_idx"].unique())
outliers = per_structure.loc[per_structure["sample_idx"].isin(outlier_ids)].copy()
outliers = outliers.sort_values(
    ["convergence_agreement_fraction", "max_abs_delta_energy_meV_atom"]
)
outliers.to_csv(OUT / "structures_to_review.csv", index=False)

# ------------------------------------------------------------------
# Key numbers.
# ------------------------------------------------------------------
def row(name):
    return summary.loc[summary["method"] == name].iloc[0]

b1 = row("B1")
fixed4 = row("Fixed B4")
inf4 = row("Inflight B4")
inf8 = row("Inflight B8")
inf16 = row("Inflight B16")

both = comp.loc[comp["converged_b1"] & comp["converged_inflight"]].copy()

agreement = comp["same_convergence"].mean()
exact_steps = comp["same_steps"].mean()
within_one_step = (comp["delta_steps"].abs() <= 1).mean()
within_ten_steps = (comp["delta_steps"].abs() <= 10).mean()

exact_energy = (both["abs_delta_energy_eV"] == 0).mean()
energy_95 = both["abs_delta_energy_meV_atom"].quantile(0.95)
energy_max = both["abs_delta_energy_meV_atom"].max()

key = f"""ALCHEMI + UMA — key production results
========================================

Workload
--------
500 OC20-Dense structures
39,123 atoms
3 replicates per method

Performance
-----------
B1:
  {b1.wall_min_mean:.2f} min
  {b1.structures_per_s_mean:.6f} structures/s

Fixed B4:
  {fixed4.wall_min_mean:.2f} min
  {fixed4.structures_per_s_mean:.6f} structures/s

Inflight B4:
  {inf4.wall_min_mean:.2f} min
  {inf4.structures_per_s_mean:.6f} structures/s
  {inf4.speedup_vs_b1:.3f}x B1 throughput
  {inf4.time_saved_vs_b1_pct:.2f}% less time than B1
  {(inf4.structures_per_s_mean/fixed4.structures_per_s_mean - 1)*100:.2f}% higher throughput than fixed B4

B1 vs inflight B4 scientific agreement
---------------------------------------
Matched comparisons: {len(comp)}
Convergence agreement: {agreement*100:.2f}%
Exact step-count agreement: {exact_steps*100:.2f}%
Within 1 optimization step: {within_one_step*100:.2f}%
Within 10 optimization steps: {within_ten_steps*100:.2f}%
Convergence disagreements: {len(dis)}
Unique structures with a disagreement: {dis.sample_idx.nunique()}

When both methods converged
---------------------------
Matched successful comparisons: {len(both)}
Exactly identical reported final energy: {exact_energy*100:.2f}%
Mean |energy difference|: {both.abs_delta_energy_meV_atom.mean():.6f} meV/atom
95th percentile: {energy_95:.6f} meV/atom
Maximum: {energy_max:.6f} meV/atom

Important limitation
--------------------
These benchmark files do not contain final atomic coordinates.
Energy/force/step agreement is strong, but geometric equivalence
has not yet been directly measured.
"""
(OUT / "key_results.txt").write_text(key)

# ------------------------------------------------------------------
# Figure A: throughput.
# ------------------------------------------------------------------
labels = summary["method"].tolist()
x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    x,
    summary["structures_per_s_mean"],
    yerr=summary["structures_per_s_std"].fillna(0),
    capsize=4,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Structures per second")
ax.set_title("Inflight B4 processes the workload fastest")

for bar, speedup in zip(bars, summary["speedup_vs_b1"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{speedup:.2f}× B1",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(FIG / "figA_throughput.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# Figure B: GPU utilization vs useful throughput.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    summary["gpu_util_pct_mean"],
    summary["structures_per_s_mean"],
    s=70,
)

for _, r in summary.iterrows():
    ax.annotate(
        r["method"],
        (r["gpu_util_pct_mean"], r["structures_per_s_mean"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xlabel("Mean GPU utilization (%)")
ax.set_ylabel("Structures per second")
ax.set_title("A busier GPU does not always finish more structures")
fig.tight_layout()
fig.savefig(FIG / "figB_gpu_util_vs_throughput.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# Figure C: fraction ending below the force threshold.
# Wording deliberately avoids over-interpreting fixed-batch behavior.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(
    x,
    summary["convergence_fraction_mean"] * 100,
    yerr=summary["convergence_fraction_std"].fillna(0) * 100,
    capsize=4,
)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right")
ax.set_ylabel("Structures ending below force threshold (%)")
ax.set_ylim(0, 50)
ax.set_title("B1 and inflight retain similar relaxation outcomes")

for bar, value in zip(bars, summary["convergence_fraction_mean"] * 100):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontsize=9,
    )

fig.tight_layout()
fig.savefig(FIG / "figC_relaxation_outcomes.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# Figure D: step agreement.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    comp["n_steps_b1"],
    comp["n_steps_inflight"],
    s=12,
    alpha=0.40,
)
ax.plot([0, 300], [0, 300], linestyle="--", linewidth=1)
ax.set_xlim(0, 305)
ax.set_ylim(0, 305)
ax.set_xlabel("B1 optimization steps")
ax.set_ylabel("Inflight B4 optimization steps")
ax.set_title(f"Stopping behavior agrees closely ({exact_steps*100:.1f}% exact)")
fig.tight_layout()
fig.savefig(FIG / "figD_step_agreement.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# Figure E: cumulative energy agreement.
# This is more readable than the original histogram because many
# comparisons have exactly zero reported energy difference.
# ------------------------------------------------------------------
energy = np.sort(both["abs_delta_energy_meV_atom"].to_numpy())
cdf = np.arange(1, len(energy) + 1) / len(energy) * 100

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(energy, cdf, linewidth=2)
ax.axhline(95, linestyle="--", linewidth=1)
ax.axvline(energy_95, linestyle="--", linewidth=1)
ax.set_xlim(left=0)
ax.set_ylim(0, 101)
ax.set_xlabel("|B1 - inflight final energy| (meV/atom)")
ax.set_ylabel("Comparisons within this difference (%)")
ax.set_title("Final energies agree closely when both methods converge")
ax.text(
    energy_95,
    8,
    f"95% ≤ {energy_95:.3f} meV/atom",
    rotation=90,
    va="bottom",
    ha="right",
    fontsize=9,
)
fig.tight_layout()
fig.savefig(FIG / "figE_energy_agreement.png", dpi=300)
plt.close(fig)

# ------------------------------------------------------------------
# Figure F: memory cost.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(
    summary["peak_gpu_memory_gb"],
    summary["structures_per_s_mean"],
    s=70,
)

for _, r in summary.iterrows():
    ax.annotate(
        r["method"],
        (r["peak_gpu_memory_gb"], r["structures_per_s_mean"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )

ax.set_xlabel("Peak GPU memory (GB)")
ax.set_ylabel("Structures per second")
ax.set_title("Larger inflight capacity costs memory without improving speed")
fig.tight_layout()
fig.savefig(FIG / "figF_memory_vs_throughput.png", dpi=300)
plt.close(fig)

print(key)
print("Final analysis outputs written to:", OUT)
print("Figures written to:", FIG)
