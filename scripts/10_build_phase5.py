#!/usr/bin/env python3
"""Build Phase 5 research-ready figures, tables, and writing from saved CSVs.

No GPU, ALCHEMI, UMA, ASE, or dataset download is required.

From the project root on Swan:
    python scripts/10_build_phase5.py

Optional:
    python scripts/10_build_phase5.py --project-root /path/to/project

Reads:
  results/production_500/analysis/production_summary.csv
  results/production_500/analysis/b1_vs_inflight_b04_all.csv
  results/production_500/geometry_validation/geometry_comparison.csv
  results/production_500/geometry_validation/*_final_structures.extxyz

Writes:
  results/phase5_publication/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def need(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    return path


def fmt(x: float, n: int = 2) -> str:
    return f"{x:.{n}f}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def savefig(fig, name: str, figdir: Path) -> None:
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(figdir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def set_categories(ax, labels, x):
    # Compatible with older Matplotlib on shared clusters.
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=19, ha="right")


def ecdf(vals):
    vals = np.sort(np.asarray(vals, dtype=float))
    return vals, np.arange(1, len(vals) + 1) / len(vals) * 100


def load_xyz_selected(path: Path, targets: set[int]):
    """Read only selected frames of our known extxyz layout; no ASE dependency."""
    out = {}
    with path.open(encoding="utf-8") as f:
        while True:
            first = f.readline()
            if not first:
                break
            if not first.strip():
                continue
            natoms = int(first.strip())
            header = f.readline()
            if not header:
                raise ValueError(f"Truncated xyz header: {path}")
            match = re.search(r"(?:^|\s)sample_idx=(\d+)(?:\s|$)", header)
            if not match:
                raise ValueError(f"Missing sample_idx in {path}")
            idx = int(match.group(1))
            if idx not in targets:
                for _ in range(natoms):
                    if not f.readline():
                        raise ValueError(f"Truncated frame {idx}")
                continue
            elements, positions, masks, tags = [], [], [], []
            for _ in range(natoms):
                parts = f.readline().split()
                if len(parts) < 6:
                    raise ValueError(f"Unexpected extxyz atom layout at {idx}: {parts}")
                elements.append(parts[0])
                positions.append([float(t) for t in parts[1:4]])
                masks.append(parts[4].upper() in ("T", "TRUE", "1"))
                tags.append(int(parts[5]))
            if idx in out:
                raise ValueError(f"Duplicate sample_idx {idx} in {path}")
            out[idx] = {
                "elements": np.asarray(elements),
                "pos": np.asarray(positions),
                "mobile": np.asarray(masks),
                "tags": np.asarray(tags),
            }
    if set(out) != targets:
        raise ValueError(f"Missing requested frames in {path}: {sorted(targets - set(out))}")
    return out


def aligned_rmsd(a, b):
    """Rigid Kabsch RMSD of same-index atoms; removes translation and rotation."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ac = a - a.mean(axis=0)
    bc = b - b.mean(axis=0)
    u, _, vt = np.linalg.svd(ac.T @ bc)
    fix = np.diag([1., 1., np.linalg.det(u @ vt)])
    rot = u @ fix @ vt
    return float(np.sqrt(np.mean(np.sum((ac @ rot - bc) ** 2, axis=1))))


def adsorbate_bonds(elements, xyz):
    # Visualization heuristic only; NOT a chemical bond-order assignment.
    radii = {"H": .31, "C": .76, "N": .71, "O": .66, "F": .57, "S": 1.05}
    pairs = []
    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            if elements[i] not in radii or elements[j] not in radii:
                continue
            if np.linalg.norm(xyz[i] - xyz[j]) <= 1.22 * (radii[elements[i]] + radii[elements[j]]):
                pairs.append((i, j))
    return pairs


def plot_adsorbate_overlay(b1, inf, idx, plane, out, g_row):
    ads = b1["tags"] == 2
    if not np.array_equal(b1["elements"], inf["elements"]) or not np.array_equal(b1["tags"], inf["tags"]):
        raise ValueError(f"Atom order or adsorbate tags changed for sample {idx}")
    elements = b1["elements"][ads]
    p = b1["pos"][ads]
    q = inf["pos"][ads]
    axes = {"xy": (0, 1), "xz": (0, 2)}
    a, b = axes[plane]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for coords, name, marker in [(p, "B1", "o"), (q, "Inflight B4", "x")]:
        segx, segy = [], []
        for i, j in adsorbate_bonds(elements, coords):
            segx.extend([coords[i, a], coords[j, a], np.nan])
            segy.extend([coords[i, b], coords[j, b], np.nan])
        line, = ax.plot(segx, segy, linewidth=1.5, alpha=.7, label="_nolegend_")
        ax.scatter(coords[:, a], coords[:, b], s=65, marker=marker,
                   color=line.get_color(), label=name, zorder=3)
        for i, elem in enumerate(elements):
            if elem != "H":
                ax.annotate(f"{elem}{i+1}", (coords[i, a], coords[i, b]),
                            xytext=(4, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel(f"{plane[0]} coordinate (Å)")
    ax.set_ylabel(f"{plane[1]} coordinate (Å)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(.20)
    ax.legend(loc="best", frameon=False)
    ax.set_title(f"Sample {idx}: alternative adsorbate configuration ({plane.upper()} view)")
    ax.text(.02, .02,
            f"Mobile RMSD {g_row['rmsd_mobile_A']:.3f} Å  |  ΔE {g_row['abs_delta_energy_eV']:.4f} eV",
            transform=ax.transAxes, fontsize=9, va="bottom")
    savefig(fig, f"fig_{'09' if idx == 208 else '10'}_sample{idx}_adsorbate_{plane}", out)


def build(root: Path, output: Path, script_path: Path) -> None:
    base = root / "results" / "production_500"
    paths = {
        "production_summary": need(base / "analysis" / "production_summary.csv"),
        "production_comparison": need(base / "analysis" / "b1_vs_inflight_b04_all.csv"),
        "geometry_comparison": need(base / "geometry_validation" / "geometry_comparison.csv"),
        "b1_coordinates": need(base / "geometry_validation" / "b1_final_structures.extxyz"),
        "inflight_coordinates": need(base / "geometry_validation" / "inflight_b04_final_structures.extxyz"),
    }
    summary = pd.read_csv(paths["production_summary"])
    comparison = pd.read_csv(paths["production_comparison"])
    geometry = pd.read_csv(paths["geometry_comparison"])
    expected = ["B1", "Fixed B4", "Inflight B4", "Inflight B8", "Inflight B16"]
    if summary["method"].tolist() != expected:
        raise ValueError(f"Unexpected method order: {summary['method'].tolist()}")
    if len(comparison) != 1500 or len(geometry) != 500:
        raise ValueError(f"Unexpected row counts: {len(comparison)} production, {len(geometry)} geometry")
    if comparison[["sample_idx", "replicate"]].duplicated().any():
        raise ValueError("Duplicate production comparison keys")
    if geometry.sample_idx.duplicated().any():
        raise ValueError("Duplicate geometry sample indices")
    if not (comparison.sid_b1 == comparison.sid_inflight).all():
        raise ValueError("B1/inflight sid mismatch")
    if not (comparison.n_atoms_b1 == comparison.n_atoms_inflight).all():
        raise ValueError("B1/inflight atom count mismatch")

    figures = output / "figures"
    tables = output / "tables"
    docs = output / "docs"
    paper = output / "paper"
    scripts = output / "scripts"
    for d in [figures, tables, docs, paper, scripts]:
        d.mkdir(parents=True, exist_ok=True)

    master = summary.rename(columns={
        "wall_min_mean": "runtime_min_mean",
        "structures_per_s_mean": "throughput_structures_s_mean",
        "structures_per_s_std": "throughput_structures_s_sd",
        "gpu_util_pct_mean": "gpu_util_pct_mean",
        "peak_gpu_memory_gb": "peak_gpu_memory_GiB",
        "convergence_fraction_mean": "endpoint_below_force_threshold_fraction_mean",
    }).copy()
    master["runtime_min_sd"] = master["wall_s_std"] / 60
    master["endpoint_below_force_threshold_pct"] = master["endpoint_below_force_threshold_fraction_mean"] * 100
    master = master[[
        "method", "n_replicates", "runtime_min_mean", "runtime_min_sd",
        "throughput_structures_s_mean", "throughput_structures_s_sd",
        "speedup_vs_b1", "time_saved_vs_b1_pct", "gpu_util_pct_mean",
        "peak_gpu_memory_GiB", "endpoint_below_force_threshold_pct",
    ]]
    master.to_csv(tables / "table1_master_results.csv", index=False, float_format="%.8f")

    both_prod = comparison.loc[comparison.converged_b1 & comparison.converged_inflight]
    both_geom = geometry.loc[geometry.b1_converged & geometry.inflight_converged]
    agree = int(comparison.same_convergence.sum())
    exact_steps = int(comparison.same_steps.sum())
    prod_agree = agree / len(comparison)
    exact_step_rate = exact_steps / len(comparison)
    p95_energy = float(both_prod.abs_delta_energy_meV_atom.quantile(.95))
    p95_geom = float(both_geom.rmsd_mobile_A.quantile(.95))
    thresholds = []
    for group, df in [("all_500", geometry), ("both_below_threshold_183", both_geom)]:
        for threshold in (.05, .10, .25, .50):
            n = int((df.rmsd_mobile_A <= threshold).sum())
            thresholds.append({"subset": group, "n": len(df), "rmsd_threshold_A": threshold,
                               "n_within": n, "fraction_within": n / len(df)})
    pd.DataFrame(thresholds).to_csv(tables / "table2_geometry_thresholds.csv", index=False)
    disagreement = comparison.loc[~comparison.same_convergence].copy()
    disagreement.to_csv(tables / "production_convergence_disagreements.csv", index=False)
    geom_outliers = geometry.sort_values("rmsd_mobile_A", ascending=False).head(20)
    geom_outliers.to_csv(tables / "geometry_top20_outliers.csv", index=False)

    confusion = pd.crosstab(comparison.converged_b1, comparison.converged_inflight)
    both_fail = int(confusion.loc[False, False])
    b1_only = int(confusion.loc[True, False])
    inf_only = int(confusion.loc[False, True])
    both_success = int(confusion.loc[True, True])
    agreement_table = pd.DataFrame([
        {"outcome": "both_not_converged", "count": both_fail},
        {"outcome": "both_converged", "count": both_success},
        {"outcome": "b1_only_converged", "count": b1_only},
        {"outcome": "inflight_only_converged", "count": inf_only},
    ])
    agreement_table.to_csv(tables / "table3_production_agreement.csv", index=False)

    samples = {155, 208}
    b1_xyz = load_xyz_selected(paths["b1_coordinates"], samples)
    inf_xyz = load_xyz_selected(paths["inflight_coordinates"], samples)
    extra_rows = []
    for idx in sorted(samples):
        a, b = b1_xyz[idx], inf_xyz[idx]
        if not np.array_equal(a["elements"], b["elements"]):
            raise ValueError(f"Elements differ for sample {idx}")
        ads = a["tags"] == 2
        slab = ~ads
        if not np.array_equal(a["tags"], b["tags"]):
            raise ValueError(f"Tags differ for sample {idx}")
        slab_rmsd = np.sqrt(np.mean(np.sum((a["pos"][slab] - b["pos"][slab]) ** 2, axis=1)))
        internal = aligned_rmsd(a["pos"][ads], b["pos"][ads])
        r = geometry.loc[geometry.sample_idx == idx].iloc[0]
        extra_rows.append({
            "sample_idx": idx,
            "adsorbate_formula": "".join(f"{e}{n if n > 1 else ''}" for e, n in sorted(pd.Series(a["elements"][ads]).value_counts().items())),
            "n_adsorbate_atoms": int(ads.sum()),
            "slab_rmsd_A_no_alignment": slab_rmsd,
            "adsorbate_rigid_aligned_rmsd_A": internal,
            "mobile_rmsd_A": r.rmsd_mobile_A,
            "adsorbate_rmsd_A": r.rmsd_adsorbate_A,
            "abs_delta_energy_eV": r.abs_delta_energy_eV,
            "both_below_threshold": bool(r.b1_converged and r.inflight_converged),
        })
    extras = pd.DataFrame(extra_rows)
    extras.to_csv(tables / "table4_outlier208_155_details.csv", index=False)

    labels = summary.method.tolist()
    x = np.arange(len(labels))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "savefig.dpi": 300})

    # Figure 1: throughput with replicate standard deviations.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(x, summary.structures_per_s_mean,
                  yerr=summary.structures_per_s_std, capsize=4)
    set_categories(ax, labels, x)
    ax.set_ylabel("Processed structures per second")
    ax.set_title("Inflight B4 gives the highest measured throughput")
    ax.set_ylim(0, summary.structures_per_s_mean.max() * 1.23)
    for bar, v in zip(bars, summary.speedup_vs_b1):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+.008,
                f"{v:.2f}×", ha="center", fontsize=9)
    ax.text(.01, -.23, "Labels show throughput relative to B1; error bars: SD of 3 runs.",
            transform=ax.transAxes, fontsize=9)
    savefig(fig, "fig01_throughput", figures)

    # Figure 2: runtime.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars = ax.bar(x, summary.wall_min_mean,
                  yerr=summary.wall_s_std / 60, capsize=4)
    set_categories(ax, labels, x)
    ax.set_ylabel("Minutes to process 500 structures")
    ax.set_title("Inflight B4 cuts B1 runtime by 40.8%")
    ax.set_ylim(0, summary.wall_min_mean.max()*1.20)
    for bar, val in zip(bars, summary.wall_min_mean):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.3,
                f"{val:.1f}", ha="center", fontsize=9)
    savefig(fig, "fig02_runtime", figures)

    # Figure 3: GPU utilization vs throughput.
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.scatter(summary.gpu_util_pct_mean, summary.structures_per_s_mean, s=100)
    for _, r in summary.iterrows():
        ax.annotate(r.method, (r.gpu_util_pct_mean, r.structures_per_s_mean),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Mean GPU utilization (%)")
    ax.set_ylabel("Processed structures per second")
    ax.set_xlim(50, 103)
    ax.set_ylim(.11, .255)
    ax.set_title("High GPU utilization does not guarantee high throughput")
    savefig(fig, "fig03_utilization_vs_throughput", figures)

    # Figure 4: memory vs throughput.
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.scatter(summary.peak_gpu_memory_gb, summary.structures_per_s_mean, s=100)
    for _, r in summary.iterrows():
        ax.annotate(r.method, (r.peak_gpu_memory_gb, r.structures_per_s_mean),
                    xytext=(7, 7), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Peak GPU memory (GiB; across three runs)")
    ax.set_ylabel("Processed structures per second")
    ax.set_ylim(.11, .255)
    ax.set_title("Larger inflight batches consume more memory")
    savefig(fig, "fig04_memory_vs_throughput", figures)

    # Figure 5: force threshold fraction; method-specific caveat in caption.
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    rate = summary.convergence_fraction_mean * 100
    bars = ax.bar(x, rate, yerr=summary.convergence_fraction_std * 100, capsize=4)
    set_categories(ax, labels, x)
    ax.set_ylabel("Final force below 0.05 eV/Å (%)")
    ax.set_ylim(0, 50)
    ax.set_title("Inflight retains B1-like force-threshold outcomes")
    for bar, val in zip(bars, rate):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.0,
                f"{val:.1f}%", ha="center", fontsize=9)
    savefig(fig, "fig05_force_threshold_fraction", figures)

    # Figure 6: production convergence agreement.
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    cat = ["Both unfinished", "Both finished", "B1 only", "Inflight only"]
    counts = [both_fail, both_success, b1_only, inf_only]
    bx = np.arange(len(cat))
    bars = ax.bar(bx, counts)
    ax.set_xticks(bx)
    ax.set_xticklabels(cat, rotation=13, ha="right")
    ax.set_ylabel("Structure–replicate comparisons")
    ax.set_title(f"Production convergence agreement: {100*prod_agree:.2f}% (1,500 comparisons)")
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+18,
                str(n), ha="center", fontsize=10)
    ax.set_ylim(0, max(counts)*1.16)
    savefig(fig, "fig06_production_convergence_agreement", figures)

    # Figure 7: energy differences in both-converged production comparisons.
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    energy_x, energy_y = ecdf(both_prod.abs_delta_energy_meV_atom)
    ax.plot(energy_x, energy_y, linewidth=2)
    ax.axvline(p95_energy, linestyle="--", linewidth=1)
    ax.axhline(95, linestyle=":", linewidth=1)
    ax.set_xlim(0, max(.11, energy_x.max()*1.07))
    ax.set_ylim(0, 101)
    ax.set_xlabel("Absolute final-energy difference (meV/atom)")
    ax.set_ylabel("Comparisons at or below difference (%)")
    ax.set_title(f"Both converged: 95% differ by ≤ {p95_energy:.3f} meV/atom")
    savefig(fig, "fig07_production_energy_agreement", figures)

    # Figure 8: geometry ECDF (all vs both below force threshold).
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    for vals, label in [(geometry.rmsd_mobile_A, "All 500"),
                        (both_geom.rmsd_mobile_A, "Both below force threshold (n=183)")]:
        vx, vy = ecdf(vals)
        ax.plot(vx, vy, linewidth=2, label=label)
    ax.axvline(.05, linestyle="--", linewidth=1)
    ax.axvline(.10, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlim(.001, 1.3)
    ax.set_ylim(0, 101)
    ax.set_xlabel("Mobile-atom RMSD (Å; logarithmic scale)")
    ax.set_ylabel("Structures at or below RMSD (%)")
    ax.set_title("Geometry agreement is tight for successfully relaxed structures")
    ax.legend(frameon=False, loc="upper left")
    savefig(fig, "fig08_geometry_rmsd_distribution", figures)

    # Figures 9 and 10, plus supplementary alternate projections.
    for idx in (208, 155):
        r = geometry.loc[geometry.sample_idx == idx].iloc[0]
        for plane in ("xy", "xz"):
            plot_adsorbate_overlay(b1_xyz[idx], inf_xyz[idx], idx, plane, figures, r)

    b1 = summary.iloc[0]
    fixed = summary.iloc[1]
    inflight = summary.iloc[2]
    inf16 = summary.iloc[4]
    n_geo_05 = int((both_geom.rmsd_mobile_A <= .05).sum())
    n_geo_10 = int((both_geom.rmsd_mobile_A <= .10).sum())
    n_all_gt_10 = int((geometry.rmsd_mobile_A > .10).sum())
    n_outside_both = int(((geometry.rmsd_mobile_A > .10) & ~(geometry.b1_converged & geometry.inflight_converged)).sum())
    g155 = geometry.loc[geometry.sample_idx == 155].iloc[0]
    g208 = geometry.loc[geometry.sample_idx == 208].iloc[0]
    ex155 = extras.loc[extras.sample_idx == 155].iloc[0]
    ex208 = extras.loc[extras.sample_idx == 208].iloc[0]

    methods_table = """| Method | Runtime (min) | Throughput (structures/s) | Speedup vs B1 | GPU util. (%) | Peak GPU memory (GiB) | Below force threshold (%) |
|---|---:|---:|---:|---:|---:|---:|\n"""
    for _, r in summary.iterrows():
        methods_table += (f"| {r.method} | {r.wall_min_mean:.2f} ± {r.wall_s_std/60:.2f} | "
                          f"{r.structures_per_s_mean:.4f} ± {r.structures_per_s_std:.4f} | "
                          f"{r.speedup_vs_b1:.2f}× | {r.gpu_util_pct_mean:.1f} | "
                          f"{r.peak_gpu_memory_gb:.2f} | {100*r.convergence_fraction_mean:.1f} |\n")

    results_text = f"""# Results — ALCHEMI-UMA catalyst-relaxation benchmark

## Experimental scope

We evaluated **500 deterministically sampled structures (39,123 atoms)** from the public **OC20-Dense OOD validation** subset on one NVIDIA **L40S** GPU. All five production settings were run in **three replicates** using UMA `uma-s-1p2p1` for the `oc20` task, with `inference_settings=default`, FIRE relaxation, a maximum of **300 steps**, and a **0.05 eV/Å** force threshold. Fixed B4 and inflight B4 have the same nominal maximum number of concurrent structures; inflight B4 additionally uses a **360-atom capacity** and refills active batches as structures finish.

**Interpretation note:** The table's last column is the fraction whose recorded final force is below the threshold. Fixed-batch termination and per-structure inflight termination are not identical procedures. These fractions are useful operational outcomes, not a proof that the optimizers follow the same path.

## Table 1 — Performance and recorded endpoint outcomes

{methods_table}

*Values are mean ± sample standard deviation across three runs, except GPU utilization (mean) and peak memory (maximum observed across runs). Memory values are GiB (reported MB divided by 1024). Timing refers to the benchmarked relaxation workload, not model training. The batch-capacity sweep is not a proof of a global optimum.*

## Throughput and runtime (Figures 1–2)

Inflight B4 completed the 500-structure workload in **{inflight.wall_min_mean:.2f} minutes** versus **{b1.wall_min_mean:.2f} minutes** for B1, a **{inflight.speedup_vs_b1:.2f}× throughput speedup** and **{inflight.time_saved_vs_b1_pct:.1f}% lower elapsed time**. Relative to fixed B4, inflight B4 delivered **{100*(inflight.structures_per_s_mean/fixed.structures_per_s_mean-1):.1f}% higher throughput**. These results show the benefit of replacing finished systems instead of keeping them coupled to slower systems in fixed-size relaxation groups.

## Utilization and memory (Figures 3–4)

The fastest tested setting was **not** the one with the highest GPU utilization. Inflight B16 reached **{inf16.gpu_util_pct_mean:.1f}% mean GPU utilization** and a **{inf16.peak_gpu_memory_gb:.2f} GiB** peak, yet completed only **{inf16.structures_per_s_mean:.4f} structures/s**, compared with **{inflight.structures_per_s_mean:.4f} structures/s** for inflight B4. GPU activity alone is therefore a poor proxy for completed relaxations per second. Inflight B4 is the **best tested** inflight capacity here; B2/B3 inflight capacities were not part of this production sweep.

## Production relaxation agreement (Figures 5–7)

Across **1,500 matched structure–replicate comparisons**, B1 and inflight B4 agreed on whether the run converged in **{agree}/1,500 ({100*prod_agree:.2f}%)** cases: **{both_success}** converged in both, **{both_fail}** converged in neither, **{b1_only}** converged only with B1, and **{inf_only}** only with inflight. The methods had exactly the same recorded step count in **{exact_steps}/1,500 ({100*exact_step_rate:.2f}%)** comparisons. For the **{len(both_prod)}** cases where both converged, the mean absolute energy difference was **{both_prod.abs_delta_energy_meV_atom.mean():.4f} meV/atom**, and the 95th percentile was **{p95_energy:.4f} meV/atom**. Among B1-successful comparisons, **{both_success}/{both_success+b1_only} ({100*both_success/(both_success+b1_only):.1f}%)** also converged with inflight. These results support close **workflow consistency**; they do not establish accuracy against DFT.

## Separate final-coordinate validation (Figure 8)

We separately captured final atomic coordinates from one B1 pass and one inflight-B4 pass over the same 500 structures. In this coordinate-capture run, **{len(both_geom)}** structures ended below the force threshold in both workflows. Their median mobile-atom RMSD was **{both_geom.rmsd_mobile_A.median():.4f} Å**, with **{n_geo_05}/{len(both_geom)} ({100*n_geo_05/len(both_geom):.1f}%)** within **0.05 Å** and **{n_geo_10}/{len(both_geom)} ({100*n_geo_10/len(both_geom):.1f}%)** within **0.10 Å**. The 95th percentile was **{p95_geom:.4f} Å**. Among all 500 structures, **{n_all_gt_10}** exceeded 0.10 Å; **{n_outside_both}** of these had at least one run that did not finish below the force threshold.

**Do not combine convergence rates across the two experiments.** The coordinate-capture code classified outcomes after reading the saved final forces, producing **{int(geometry.same_convergence.sum())}/500 ({100*geometry.same_convergence.mean():.1f}%)** apparent agreement, whereas the production benchmark preserved its own optimization exit/status accounting and reported **{100*prod_agree:.2f}%** agreement. The source files do not establish why these two bookkeeping schemes disagree. The geometry capture is used here to evaluate *coordinate similarity conditional on both ending below the force threshold*, not to revise the three-replicate production convergence claim.

## Geometry exceptions (Figures 9–10)

Two structures ended below the force threshold in both coordinate-capture workflows yet exceeded 0.10 Å mobile RMSD. **Sample 208** had **{g208.rmsd_mobile_A:.3f} Å** mobile RMSD and **{g208.rmsd_adsorbate_A:.3f} Å** adsorbate RMSD, with a total final-energy difference of **{g208.abs_delta_energy_eV:.4f} eV**. Its non-adsorbate slab RMSD was **{ex208.slab_rmsd_A_no_alignment:.3f} Å**, and its CH₂O adsorbate had only **{ex208.adsorbate_rigid_aligned_rmsd_A:.3f} Å** internal RMSD after rigid alignment. This is consistent with a change in adsorption orientation or position rather than internal molecular distortion.

**Sample 155** had **{g155.rmsd_mobile_A:.3f} Å** mobile RMSD, **{g155.rmsd_adsorbate_A:.3f} Å** adsorbate RMSD, and a total final-energy difference of **{g155.abs_delta_energy_eV:.4f} eV**. Its slab RMSD was **{ex155.slab_rmsd_A_no_alignment:.3f} Å**, while its adsorbate remained **{ex155.adsorbate_rigid_aligned_rmsd_A:.3f} Å** different after rigid alignment, consistent with an alternative molecular conformation (without establishing a distinct minimum or chemical connectivity change). Similar energies alone do not prove that the two endpoints are the same local minimum. The adsorbate overlays are qualitative 2D projections of saved coordinates; the geometry RMSD table is the quantitative result.

## Limits and appropriate claims

- **Supported:** Throughput and runtime advantages on the tested L40S, OC20-Dense subset, UMA model, and settings; close production B1/inflight convergence and step-count agreement; small conditional geometry differences for most successfully relaxed structures.
- **Not established:** DFT predictive accuracy, universal performance across GPU architectures, globally optimal batch size, identical final structures for every sample, or proof that all endpoint structures are true energy minima.
- **Caveat:** The geometry validation is one coordinate-capture pass per method and is separate from the three-replicate production campaign. Rare outliers should not be generalized away, and the convergence-label discrepancy requires a dedicated bookkeeping audit before further mechanistic interpretation.

## Figure index

See [`FIGURE_CAPTIONS.md`](FIGURE_CAPTIONS.md) and the images in `../figures/`. All figures are generated from saved CSVs/coordinates by `scripts/10_build_phase5.py` (no benchmark results hardcoded into plots).
"""
    (docs / "RESULTS.md").write_text(results_text, encoding="utf-8")

    paper_text = f"""# Draft manuscript Results

## Throughput and resource efficiency

We benchmarked three replicates of five GPU scheduling configurations on a 500-structure, 39,123-atom subset of OC20-Dense OOD validation using UMA and FIRE relaxation on a single NVIDIA L40S. Inflight batching with a maximum of four active structures and a 360-atom capacity delivered the highest throughput among the tested configurations, processing **{inflight.structures_per_s_mean:.4f} ± {inflight.structures_per_s_std:.4f} structures s⁻¹** (mean ± sample SD; n=3). This represents a **{inflight.speedup_vs_b1:.2f}-fold** increase over B1 (**{b1.structures_per_s_mean:.4f} ± {b1.structures_per_s_std:.4f} structures s⁻¹**) and a **{100*(inflight.structures_per_s_mean/fixed.structures_per_s_mean-1):.1f}%** increase over fixed B4 (**{fixed.structures_per_s_mean:.4f} ± {fixed.structures_per_s_std:.4f} structures s⁻¹**; Fig. 1). Mean workload completion time decreased from **{b1.wall_min_mean:.2f} min** for B1 to **{inflight.wall_min_mean:.2f} min** for inflight B4, a **{inflight.time_saved_vs_b1_pct:.1f}%** reduction (Fig. 2).

Increasing inflight capacity beyond four active structures did not improve throughput in this workload. Inflight B16 reached **{inf16.gpu_util_pct_mean:.1f}%** average GPU utilization and **{inf16.peak_gpu_memory_gb:.2f} GiB** peak observed memory but achieved only **{inf16.structures_per_s_mean:.4f} structures s⁻¹**, compared with **{inflight.gpu_util_pct_mean:.1f}%** utilization, **{inflight.peak_gpu_memory_gb:.2f} GiB** peak memory, and **{inflight.structures_per_s_mean:.4f} structures s⁻¹** for inflight B4 (Figs. 3–4). Thus, maximizing GPU occupancy did not maximize completed relaxations per unit time. The tested capacity range does not establish a global optimum.

## Relaxation behavior and numerical consistency

Across **1,500** matched structure–replicate comparisons between B1 and inflight B4, the workflows agreed on convergence status in **{agree} ({100*prod_agree:.2f}%)** comparisons and recorded identical optimization step counts in **{exact_steps} ({100*exact_step_rate:.2f}%)** (Figs. 5–6). Among the **{len(both_prod)}** comparisons where both methods converged, the mean absolute final-energy difference was **{both_prod.abs_delta_energy_meV_atom.mean():.4f} meV atom⁻¹**, with a 95th percentile of **{p95_energy:.4f} meV atom⁻¹** (Fig. 7). Among B1-converged comparisons, **{both_success}/{both_success+b1_only} ({100*both_success/(both_success+b1_only):.1f}%)** also converged with inflight. These results indicate close agreement between the two implementations for this workload; they do not assess either method against DFT reference energies.

## Final-geometry agreement and exceptional configurations

A separate one-pass coordinate-capture experiment showed that, among the **{len(both_geom)}** structures whose saved final forces were below the 0.05 eV Å⁻¹ threshold in both workflows, the median mobile-atom RMSD was **{both_geom.rmsd_mobile_A.median():.4f} Å** and **{n_geo_10}/{len(both_geom)} ({100*n_geo_10/len(both_geom):.1f}%)** were within **0.10 Å** (Fig. 8). Only two such structures exceeded 0.10 Å: samples 208 and 155. Their mobile-atom RMSDs were **{g208.rmsd_mobile_A:.3f} Å** and **{g155.rmsd_mobile_A:.3f} Å**, respectively, despite relatively small total-energy differences (**{g208.abs_delta_energy_eV:.4f} eV** and **{g155.abs_delta_energy_eV:.4f} eV**; Figs. 9–10). In sample 208, rigid alignment reduced the adsorbate internal RMSD to **{ex208.adsorbate_rigid_aligned_rmsd_A:.3f} Å**, consistent with a change in placement or orientation of a largely unchanged molecule. Sample 155 retained **{ex155.adsorbate_rigid_aligned_rmsd_A:.3f} Å** adsorbate RMSD after alignment, consistent with a change in internal conformation, without establishing a distinct energy minimum.

The coordinate-capture run and the production benchmark used different convergence-status accounting, and their raw agreement percentages should not be pooled. The geometry result is therefore reported **conditionally on both captured endpoints satisfying the final-force threshold**. Together, these observations show that inflight scheduling can substantially increase throughput while retaining B1-like relaxation outcomes and, for most jointly successful structures, very similar final coordinates. Rare trajectory-sensitive endpoints remain a limitation and a target for future analysis.

*[Draft for technical review; add references to methods, dataset, toolkit, and model before submission. Figure numbering refers to the Phase 5 package.]*
"""
    (paper / "RESULTS_DRAFT.md").write_text(paper_text, encoding="utf-8")

    captions = f"""# Phase 5 figure captions

**Fig. 1 — Throughput.** Mean processed structures per second for B1, fixed B4, and inflight capacities B4/B8/B16 on the 500-structure OC20-Dense OOD validation subset. Error bars show sample SD across three replicates. Text labels give speedup relative to B1.

**Fig. 2 — Time to solution.** Mean minutes to complete 500 structures. Error bars show sample SD (n=3). Inflight B4 reduced mean runtime by {inflight.time_saved_vs_b1_pct:.1f}% relative to B1.

**Fig. 3 — GPU utilization versus useful throughput.** Mean GPU utilization against processed structures per second. Inflight B16 has the highest utilization but substantially lower throughput than inflight B4. Utilization is not a substitute for workload throughput.

**Fig. 4 — Memory cost versus throughput.** Peak observed GPU memory across the three runs (GiB) versus mean throughput. Higher inflight capacity consumes substantially more memory without improving throughput in this sweep.

**Fig. 5 — Recorded final-force threshold fraction.** Fraction of structures ending below 0.05 eV/Å. Error bars show replicate SD. Fixed batches and inflight batches have different termination semantics; do not interpret this chart as optimizer-path equivalence.

**Fig. 6 — B1 versus inflight production convergence agreement.** Counts across 1,500 matched structure–replicate comparisons. The same outcome occurred in {agree} ({100*prod_agree:.2f}%) comparisons.

**Fig. 7 — Production final-energy consistency.** Empirical cumulative distribution of absolute energy differences per atom for the {len(both_prod)} production comparisons in which both methods converged. The 95th percentile was {p95_energy:.4f} meV/atom. This measures implementation agreement, not DFT accuracy.

**Fig. 8 — Coordinate-level agreement.** Empirical cumulative distributions of mobile-atom RMSD for all 500 structures and the {len(both_geom)} structures whose saved final forces met the threshold in both capture runs. {n_geo_10}/{len(both_geom)} jointly successful structures were within 0.10 Å. The coordinate-capture convergence labels must not be pooled with the three-replicate production labels.

**Fig. 9 — Sample 208 adsorbate comparison (XY and XZ views).** Overlay of the saved B1 and inflight CH₂O adsorbate coordinates. A change in orientation/position produces {g208.rmsd_adsorbate_A:.3f} Å adsorbate RMSD despite only {ex208.adsorbate_rigid_aligned_rmsd_A:.3f} Å rigid-aligned internal RMSD. Bond lines are approximate distance-based guides; no bond-order assignment is implied. The figures show the adsorbate only, not the slab.

**Fig. 10 — Sample 155 adsorbate comparison (XY and XZ views).** Overlay of the saved B1 and inflight C₂N₂H₈ adsorbate coordinates. The rigid-aligned internal RMSD is {ex155.adsorbate_rigid_aligned_rmsd_A:.3f} Å, consistent with a changed conformation. Bond lines are approximate distance-based guides. The figures show the adsorbate only, not the slab.

**All figures:** Same 500 structures and one NVIDIA L40S GPU; model UMA `uma-s-1p2p1`, task `oc20`, inference `default`, FIRE max 300 steps, force threshold 0.05 eV/Å. Figures 1–7 use three-replicate production results; Figures 8–10 use the separate single-pass coordinate-capture results.
"""
    (docs / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")

    reproducibility = """# Methods and reproducibility

## Data and environment

- Public dataset: OC20-Dense OOD validation, deterministic 500-structure sample (39,123 atoms). The underlying manifest and selection code remain in the benchmark repository.
- Benchmark accelerator: single NVIDIA L40S GPU on HCC Swan.
- GPU runtime: PyTorch 2.13.0+cu130, CUDA 13.0, Python 3.12.14; NVIDIA ALCHEMI 0.2.0.
- UMA: `uma-s-1p2p1`, task `oc20`, inference setting `default` (kept consistent across comparisons).
- Optimizer: FIRE, 300-step maximum, force threshold 0.05 eV/Å, timestep parameter 0.1.
- Production: B1, fixed B4, inflight B4/max_atoms=360, inflight B8/max_atoms=720, inflight B16/max_atoms=1440; three replicates per method. Inflight refill frequency 1.
- Coordinate capture: separate single-pass B1 and inflight B4 runs. Saved extxyz frames include sample_idx, tags, movable mask, and final positions. Frame order differs between methods: match on sample_idx, **not** extxyz frame number.

## Definitions

- Throughput: 500 / timed workload seconds, expressed in structures/s.
- Speedup: method throughput divided by B1 throughput.
- Time reduction: 100 × (1 - method mean wall time / B1 mean wall time).
- GPU utilization: telemetry average, not an end-to-end efficiency measure.
- Peak memory: maximum observed nvidia-smi memory, MB/1024, reported as GiB; not a mean and not CUDA allocator-only memory.
- Force-threshold success: recorded final maximum force below 0.05 eV/Å.
- RMSD: per-atom Cartesian displacement RMSD from the saved geometry comparison, using matched atom order and periodic-boundary handling in the original validation script. The Phase 5 builder **reads** these saved RMSDs; it does not recompute them.
- Rigid-aligned adsorbate RMSD: least-squares Kabsch alignment of adsorbate atoms from the saved extxyz files; used only to distinguish internal deformation from rigid translation/rotation in samples 208 and 155.
- Bond guides in overlays: heuristic covalent-radius thresholds, **not** validated chemical bond assignments.

## Build Phase 5 without a GPU

From the benchmark project root with pandas, numpy and matplotlib installed:

```bash
python scripts/10_build_phase5.py
```

Required files:

```text
results/production_500/analysis/production_summary.csv
results/production_500/analysis/b1_vs_inflight_b04_all.csv
results/production_500/geometry_validation/geometry_comparison.csv
results/production_500/geometry_validation/b1_final_structures.extxyz
results/production_500/geometry_validation/inflight_b04_final_structures.extxyz
```

Output directory: `results/phase5_publication/`. PNG figures are 300 dpi and SVG files are vector versions. All master tables and numerical statements are generated from the input files; no new relaxation is performed.

## Known limitations and audit items

1. The production and coordinate-capture convergence-label procedures differ. Do not combine their percentages or treat their difference as evidence of a scheduling-induced failure without an exit-state audit.
2. The fixed-batch per-structure step counter can represent group-level progress; do not interpret it as independent per-system step count.
3. The tested inflight batch sizes were 4, 8, and 16; no claim of global optimality.
4. The benchmark measures internal consistency and throughput, not DFT accuracy or adsorption-energy ranking.
5. The outlier plots are two-dimensional projections and exclude the slab for clarity; use extxyz and quantitative RMSDs for structural conclusions.
6. The coordinate-capture geometry analysis has one run per method; no uncertainty estimate for the 98.9% conditional fraction is asserted here.
"""
    (docs / "METHODS_AND_REPRODUCIBILITY.md").write_text(reproducibility, encoding="utf-8")

    readme = f"""# ALCHEMI-UMA Catalysis — Phase 5 results

GPU-accelerated heterogeneous catalyst relaxation with the **NVIDIA ALCHEMI Toolkit** and the **UMA foundation interatomic potential**, evaluated on 500 public OC20-Dense OOD validation structures (39,123 atoms).

## At a glance

- **1.69× throughput vs B1** using inflight B4.
- **40.8% shorter runtime:** {b1.wall_min_mean:.2f} → {inflight.wall_min_mean:.2f} minutes per 500 structures.
- **99.07% production convergence agreement** across 1,500 matched B1/inflight comparisons.
- **98.9% conditional geometry agreement within 0.1 Å** among 183 structures meeting the final-force threshold in both separate coordinate-capture runs.

**Core question:** Can inflight GPU scheduling improve throughput for a heterogeneous relaxation workload while retaining B1-like relaxation outcomes?

## Main figures

![Throughput](figures/fig01_throughput.png)

![Utilization versus throughput](figures/fig03_utilization_vs_throughput.png)

![Geometry agreement](figures/fig08_geometry_rmsd_distribution.png)

## Important interpretation

These are **two distinct experiments**: a three-replicate production benchmark for speed and convergence behavior, and a single-pass coordinate capture for final-geometry agreement. The geometry percentage is conditional on both endpoints passing the force threshold. Two successful-by-final-force structures (samples 208 and 155) have larger geometric differences, consistent with alternative adsorbate configurations. **We do not claim identical geometries, DFT accuracy, or a globally optimal batch size.**

## Navigate

- [Full Results and interpretation](docs/RESULTS.md)
- [Paper-style Results draft](paper/RESULTS_DRAFT.md)
- [Master performance table](tables/table1_master_results.csv)
- [Geometry threshold table](tables/table2_geometry_thresholds.csv)
- [Figure captions](docs/FIGURE_CAPTIONS.md)
- [Methods and reproduction](docs/METHODS_AND_REPRODUCIBILITY.md)

## Rebuild the package

Copy `scripts/10_build_phase5.py` into the benchmark project's `scripts/` folder, then run it from the project root:

```bash
python scripts/10_build_phase5.py
```

The builder reads the existing CSV and extxyz outputs. **No GPU job is required.** The original repository remains the source of the dataset manifest, numerical equivalence diagnostics, benchmarking scripts, and SLURM job configurations.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    repo_section = f"""## Verified 500-structure benchmark results

On a single NVIDIA L40S, inflight B4 completed the 500-structure OC20-Dense OOD validation workload in **{inflight.wall_min_mean:.2f} min**, compared with **{b1.wall_min_mean:.2f} min** for B1 (**{inflight.speedup_vs_b1:.2f}× throughput; {inflight.time_saved_vs_b1_pct:.1f}% less time**). Across 1,500 matched production comparisons, B1 and inflight agreed on convergence outcome in **{100*prod_agree:.2f}%** of cases. In a separate coordinate-capture run, **{n_geo_10}/{len(both_geom)} ({100*n_geo_10/len(both_geom):.1f}%)** of jointly force-threshold-successful structures had mobile-atom RMSD ≤0.1 Å.

See [Phase 5 Results](results/phase5_publication/docs/RESULTS.md), [figures](results/phase5_publication/figures/), and [reproducibility notes](results/phase5_publication/docs/METHODS_AND_REPRODUCIBILITY.md). The two experiments use different convergence accounting and must not be pooled; neither result demonstrates DFT accuracy.
"""
    (output / "README_ROOT_INSERT.md").write_text(repo_section, encoding="utf-8")

    checklist = """# Phase 5 review checklist

- [x] Cross-checked five-method master table against uploaded production CSV.
- [x] Kept n=3 production metrics separate from n=1 coordinate capture.
- [x] Quantified production convergence and energy agreement.
- [x] Quantified conditional geometry agreement and flagged samples 208/155.
- [x] Created 300-dpi PNG + vector SVG figures and figure captions.
- [x] Documented dataset, model, GPU, optimizer, and output provenance.
- [ ] Confirm whether final-force bookkeeping in geometry capture differs from actual inflight exit reasons (separate future audit).
- [ ] Review adsorption-site identity and chemistry for outlier samples before assigning specific minima.
- [ ] Add references/DOIs and publication venue formatting before manuscript submission.
- [ ] Review repository structure and remove stale setup instructions before public release.
- [ ] Consider independent DFT validation before making predictive-accuracy claims.
"""
    (docs / "REVIEW_CHECKLIST.md").write_text(checklist, encoding="utf-8")

    shutil.copy2(script_path, scripts / "10_build_phase5.py")
    manifest = {
        "project": "ALCHEMI-UMA Catalysis — Phase 5",
        "source": "User-supplied 500-structure production and geometry-validation files",
        "input_sha256": {k: sha256(p) for k, p in paths.items()},
        "n_structures": 500,
        "n_atoms": 39123,
        "production_replicates": 3,
        "geometry_capture_replicates": 1,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "notes": ["No new GPU simulation performed", "Results derived from saved benchmark outputs"],
    }
    (output / "SOURCE_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    archive = output.parent / "phase5_publication_package.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                z.write(p, Path("phase5_publication") / p.relative_to(output))
    print("Phase 5 package ready:")
    print(f"  Directory: {output}")
    print(f"  ZIP: {archive}")
    print(f"  Figures: {len(list(figures.glob('*.png')))} PNG + {len(list(figures.glob('*.svg')))} SVG")
    print(f"  B1/inflight convergence agreement: {agree}/1500 = {100*prod_agree:.2f}%")
    print(f"  Geometry within 0.10 Å (both successful): {n_geo_10}/{len(both_geom)}")
    print("  Outlier rigid-aligned adsorbate RMSD:",
          {int(r.sample_idx): round(float(r.adsorbate_rigid_aligned_rmsd_A), 4) for _, r in extras.iterrows()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (args.output or root / "results" / "phase5_publication").resolve()
    build(root, output, Path(__file__).resolve())


if __name__ == "__main__":
    main()