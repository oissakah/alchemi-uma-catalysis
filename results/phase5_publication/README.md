# Phase 5 — figures and derived results

This folder contains **12 figures in both PNG and SVG**, six selected derived tables, and the paper-style Results draft in `paper/RESULTS_DRAFT.md`.

## Key visuals

- [Throughput](figures/fig01_throughput.png)
- [GPU utilization versus throughput](figures/fig03_utilization_vs_throughput.png)
- [Production convergence agreement](figures/fig06_production_convergence_agreement.png)
- [Geometry RMSD distribution](figures/fig08_geometry_rmsd_distribution.png)
- [Sample 208 adsorbate comparison](figures/fig_09_sample208_adsorbate_xy.png)
- [Sample 155 adsorbate comparison](figures/fig_10_sample155_adsorbate_xy.png)

See `tables/table1_master_results.csv` for the master performance table.

**Reproducibility note:** the figures are saved outputs from the Swan benchmark. Most numerical figures can be regenerated from the derived CSVs committed in this repository; the outlier atom-position visualizations require final-structure `extxyz` files that remain off GitHub pending redistribution review. Do not interpret the figures as an independent validation of the published benchmark scripts.
