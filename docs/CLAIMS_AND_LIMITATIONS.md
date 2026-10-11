# Claim boundaries for public release

## Supported by three-replicate production measurements

- Inflight B4 achieved **0.2242 ± 0.0001 structures/s** and **37.17 min** average workload time for 500 structures on one L40S.
- Relative to B1, inflight B4 achieved **1.69× throughput** and **40.8% shorter time**.
- B1–inflight convergence outcomes agreed in **1486/1500 = 99.07%** matched production comparisons.
- Optimization step counts matched exactly in **1409/1500 = 93.93%** comparisons.
- Higher measured GPU utilization did **not** imply higher completed-structure throughput in the tested inflight capacity sweep.

## Supported by a *separate*, single-pass coordinate capture

- Among **183** structures with final saved forces below the target in both methods, **181 (98.9%)** had mobile-atom RMSD below 0.10 Å.
- Samples **208** and **155** are geometric exceptions among these jointly force-threshold-successful endpoints.
- These percentages are **conditional on both captured endpoints meeting the force threshold**. They do not describe all 500 structures, and uncertainty across geometry-capture replicates has not been quantified.

## Unresolved

- The production workflow's recorded convergence status and the geometry-capture script's post-hoc final-force classification differ. The apparent **92.8%** coordinate-capture status agreement is not directly comparable to the **99.07%** production figure. Do not attribute the difference to inflight scheduling without an exit-reason audit.
- The outliers may reflect different endpoint configurations. They have **not** been proven to be distinct local minima or validated by DFT.

## Do not claim

- Identical final geometries for all systems.
- Improved DFT accuracy, adsorption-energy prediction accuracy, or catalyst ranking.
- Global optimality of batch size 4, or performance across all GPUs/models/datasets.
- A peer-reviewed publication or a public GitHub release until those actions occur.