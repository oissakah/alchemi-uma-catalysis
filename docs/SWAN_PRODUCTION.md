# Swan production workflow and provenance

**Scope:** This folder contains scripts transferred from the completed Swan benchmark, not an independent re-execution in a new environment. The original SLURM launchers contain a site-specific absolute project path, account and partition; edit those fields for another cluster. Do not submit the launchers without checking their settings.

## Measured environment

- Single NVIDIA L40S GPU on HCC Swan (node type observed: c0615.swan.hcc.unl.edu)
- Python 3.12.14; PyTorch 2.13.0+cu130; CUDA runtime 13.0
- NVIDIA ALCHEMI Toolkit 0.2.0; UMA `uma-s-1p2p1`, task `oc20`, inference settings `default`
- FIRE optimizer: `fmax=0.05 eV/Å`, `max_steps=300`, `dt=0.1`
- OC20-Dense OOD validation sample: 500 structures, 39,123 atoms, deterministic manifest at `results/pilot_500/dataset_manifest.csv`

## Source entry points

- `run_production_500.sh`: fixed B4, inflight B4/360, B8/720, B16/1440, three repeats each
- `run_fixed_b01.sh`: fixed B1 reference, three repeats
- `scripts/05_benchmark_fixed.py`: fixed-batch production implementation
- `scripts/06_benchmark_inflight.py`: inflight production implementation
- `scripts/03_numerical_equivalence.py`: ASE/ALCHEMI energy and unconstrained force diagnostic
- `scripts/09_geometry_validation_resume.py` and `slurm/run_geometry_validation_resume.sbatch`: corrected geometry-capture continuation
- `scripts/07_analyze_production.py`, `scripts/08_make_final_figures.py`, `scripts/10_build_phase5.py`: analysis and publication figure generation

The **earlier** `scripts/09_geometry_validation.py` and `slurm/run_geometry_validation.sbatch` are deliberately omitted because their duplicate sink-storage path caused `RuntimeError: All sinks are full`. The corrected resume script is retained. The original `run_catalysis_bench.sh`, `run_catalysis_bench_inflight.sh` and pilot scripts are included as *historical pilot entry points*, not as the production methods used to obtain the headline 500-structure results.

## Reproducing the results

1. Obtain the OC20-Dense OOD validation LMDB and mapping metadata from the original dataset provider, subject to its access and redistribution terms. Place them at the paths expected by the scripts, or pass equivalent CLI arguments.
2. Create a compatible GPU environment. `pyproject.toml` defines the lightweight local package but is **not** a fully pinned reproduction environment. On Swan the environment additionally used `nvalchemi-toolkit[uma]`, `torch`, `torch_geometric`, `ase`, `fairchem`, `nvidia-ml-py`, `pandas`, `lmdb`, `matplotlib`, `pyyaml` and `tqdm`. Versions beyond those listed above must be pinned before a public reproducibility claim.
3. Review the launchers' absolute project path, account, partition, environment path and wall-time limits.
4. Run `run_fixed_b01.sh` and `run_production_500.sh` with `sbatch` on an allocated GPU; the scripts' fixed paths assume the Swan project layout.
5. Run `python scripts/07_analyze_production.py` and `python scripts/10_build_phase5.py` from the repository root after source results are available.

**No GPU run has been executed as part of this GitHub synchronization.** The published tables are the saved Swan outputs, not new independently verified experiments.

## Interpreting the benchmark

The production runs tracked relaxation outcome during the optimization. The separate coordinate-capture run labeled success afterward using the final stored force; those classifications should not be pooled. See `docs/VALIDATION_NOTES.md`.
