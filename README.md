# ALCHEMI-UMA Catalysis

**GPU-accelerated catalyst relaxation with UMA and the NVIDIA ALCHEMI Toolkit.**

> **Status:** Internal research preview. The repository is private while the validated Swan production scripts, data provenance, and convergence bookkeeping are audited. This is **not yet a complete, independently reproducible public release**.

## Research question

Can inflight GPU batching accelerate heterogeneous catalyst-relaxation workloads without substantially changing the behavior of individual-structure relaxation?

We benchmarked **500 OC20-Dense OOD validation structures (39,123 atoms)** using UMA `uma-s-1p2p1` with task `oc20` on a single NVIDIA L40S GPU.

## Main results

| Scheduling method | Runtime for 500 | Throughput (structures/s) | Mean GPU utilization |
|---|---:|---:|---:|
| B1 (one structure at a time) | 62.81 min | 0.1327 | 56.3% |
| Fixed B4 | 43.35 min | 0.1922 | 87.2% |
| **Inflight B4** | **37.17 min** | **0.2242** | **85.1%** |
| Inflight B8 | 42.84 min | 0.1945 | 92.6% |
| Inflight B16 | 60.16 min | 0.1385 | 96.9% |

- **1.69× B1 throughput**, or **40.8% less wall time**, with inflight B4 (three production replicates per method).
- **99.07%** matching B1/inflight convergence outcomes across **1,500** matched production comparisons.
- In a **separate, single-pass coordinate-capture experiment**, **181/183 (98.9%)** structures ending below the force threshold in both workflows agreed within **0.10 Å** mobile-atom RMSD.
- Higher GPU utilization did **not** necessarily translate into more completed structures per second.

## Scientific interpretation

The production and coordinate-capture experiments used **different convergence-status bookkeeping**. Their raw convergence percentages must not be pooled or directly compared. Rare geometric outliers (notably samples 208 and 155) show that similar endpoint energies do not guarantee identical atomic configurations. This study does **not** establish DFT accuracy, universal batch-size optimality, or identical geometries for every system.

## Repository contents

- [`paper/MANUSCRIPT_V0_1.md`](paper/MANUSCRIPT_V0_1.md) — working research manuscript; **not submission-ready**
- [`docs/CLAIMS_AND_LIMITATIONS.md`](docs/CLAIMS_AND_LIMITATIONS.md) — what the experiments do and do not support
- [`results/phase5_publication/`](results/phase5_publication/) — selected derived results, tables and publication material
- [`scripts/`](scripts/) — analysis and coordinate-capture scripts from the conversation, **not yet a full validated production workflow**

## Reproducibility status

The exact production benchmark scripts as run on Swan have not all been transferred into this repository. The original early scaffold differs from the validated production versions, so it is **not being presented as the exact executed source**. Regenerating all Phase 5 figures also requires five source CSV/coordinate files retained on Swan; redistribution rights and provenance must be reviewed before a public data release.

**Runtime context:** Python 3.12.14; PyTorch 2.13.0+cu130; CUDA 13.0; NVIDIA ALCHEMI 0.2.0; UMA `uma-s-1p2p1` with inference setting `default`; FIRE `fmax=0.05 eV/Å`, `max_steps=300`, `dt=0.1`.

No unpublished PFAS research data are included.

## Before making the repository public

1. Sync the **exact** final production benchmark scripts and environment configuration from Swan.
2. Audit the difference between production exit statuses and coordinate-capture post-hoc force-threshold labels.
3. Check source-data redistribution permissions and document hashes / dataset manifest.
4. Verify all figures, references, dependencies, authorship and licensing.

This repository documents research in progress and does not imply peer review or publication.
