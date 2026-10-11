# Data provenance and release boundary

**Source dataset:** OC20-Dense OOD validation subset; 500 deterministically selected initial configurations. The sample manifest is `results/pilot_500/dataset_manifest.csv` (500 rows; seed 42). The benchmark is an application of UMA (`uma-s-1p2p1`, task `oc20`) through the NVIDIA ALCHEMI Toolkit 0.2.0.

**Results:** `results/production_500/analysis/` contains derived production summaries and the B1/inflight matched comparison table. `results/production_500/geometry_validation/` contains derived coordinate-comparison tables, but no atomic coordinate files. `results/phase5_publication/` contains figures and derived tables.

**Not redistributed:** original OC20-Dense LMDB, mapping/target pickles, UMA checkpoint/model weights, large raw logs, Conda environment, and the two final-structure `extxyz` files. These were deliberately excluded from the GitHub synchronization package. Dataset and model licensing must be checked before any redistribution.

**Archive reviewed:** `alchemi_github_sync_review.zip` uploaded from Swan on 2026-10-11. Files were checked for Python syntax, file counts, and common embedded credential patterns. This is a basic static review, not a security audit or a new end-to-end reproduction.
