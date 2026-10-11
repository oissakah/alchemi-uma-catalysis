# Validation notes and scientific boundaries

## Numerical equivalence

The committed diagnostic CSV contains **10** structures, not 50. It compares ASE/FAIRChem and ALCHEMI predictions using raw ASE forces (`apply_constraint=False`) so constrained slab atoms do not create an artificial discrepancy. The earlier pilot README proposed 50 but the uploaded output contains 10; do not represent 50 as measured.

## Production agreement

Across 1,500 matched comparisons (500 structures × 3 repeats), B1 and inflight B4 agreed on convergence outcome in **1,486/1,500 (99.07%)**. Their recorded step counts were identical in **1,409/1,500 (93.93%)**. Both methods converged in 567 comparisons; mean absolute final-energy difference was **0.00361 meV/atom** for those cases. These results come from `results/production_500/analysis/b1_vs_inflight_b04_all.csv`.

## Coordinate capture is a separate experiment

The geometry capture contains 500 paired endpoints, with 183 structures ending below `0.05 eV/Å` in both workflows. Among these, **181/183 (98.9%)** have mobile-atom RMSD below `0.10 Å`, with median `0.00998 Å`. The capture-run post-hoc convergence agreement is **92.8%**, unlike the production outcome agreement; its classification discrepancy remains unresolved. Do not claim the production experiment achieved 98.9% geometric agreement on the same runs, and do not claim all final geometries are identical.

Sample 208 is an alternative adsorbate pose with near-preserved internal adsorbate geometry. Sample 155 is a larger internal adsorbate rearrangement. Both end below the force threshold in both capture runs; their differences show that near-equal energies do not guarantee identical atomic coordinates.

## Limitations

- The measured B4 inflight configuration is the best **tested** capacity; B2/B3 inflight were not included in the 500-structure production sweep.
- No DFT accuracy or catalyst ranking performance is established by the throughput and agreement figures.
- Only one GPU model and a fixed sampled workload were measured.
- Original final `extxyz` coordinates remain off GitHub pending dataset/checkpoint licensing and redistribution review; tables and images do not by themselves reproduce all coordinate-dependent figures.
- Figures were generated from saved Swan results. New users must obtain source dataset/checkpoints separately.
