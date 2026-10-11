# GPU-Native Inflight Scheduling for Foundation-Model Catalyst Relaxation: Throughput and Endpoint Consistency

**Status:** working manuscript v0.1, for internal scientific review; **not** submission-ready.  
**Authors, affiliations, venue, and references:** to be confirmed.  
**Data:** public OC20-Dense OOD validation structures; no unpublished PFAS results.

## Abstract

Foundation interatomic potentials make large-scale atomistic screening practical, but relaxation workloads are heterogeneous: different structures contain different numbers of atoms and require different numbers of optimization steps. We evaluated fixed and inflight GPU scheduling for UMA-based catalyst relaxations using the NVIDIA ALCHEMI Toolkit on 500 OC20-Dense out-of-domain validation structures (39,123 atoms) and a single NVIDIA L40S. Across three production replicates, inflight scheduling with capacity for four active structures processed 0.2242 ± 0.0001 structures/s, compared with 0.1327 ± 0.0004 structures/s for individual B1 processing and 0.1922 ± 0.0001 structures/s for fixed B4. Inflight B4 therefore increased throughput by 69.0% relative to B1 and reduced workload time from 62.81 to 37.17 minutes (40.8%). Larger inflight capacities raised mean GPU utilization but reduced useful throughput. In 1,500 matched B1–inflight comparisons, convergence status agreed in 99.07% and recorded step counts were identical in 93.93%. In a separate single-pass coordinate-capture experiment, 181 of 183 structures meeting the final-force threshold in both workflows had mobile-atom RMSD below 0.10 Å; two exceptions showed larger adsorbate rearrangements. These results demonstrate a favorable throughput–consistency tradeoff for the tested scheduling settings, without establishing DFT accuracy, global batch-size optimality, or identical geometries for every structure.

## 1. Introduction

Atomistic catalyst screening often involves relaxing many initial adsorbate–surface structures to low-force endpoints. A foundation interatomic potential can reduce the cost of force evaluations, but fast force prediction does not guarantee a fast screening campaign. A GPU may spend time waiting on small structures or processing groups whose members need very different numbers of optimization steps.

Conventional fixed batching groups structures for processing together. In heterogeneous relaxation workloads, the most difficult members can affect how long the group remains active. Inflight batching instead refills available capacity when a structure leaves the active workload. The central question of this study is whether this scheduling strategy improves completed relaxations per second **without materially changing the behavior of the underlying scientific workflow**.

We focus on two distinct kinds of evidence: (i) production-run throughput, resource use, and agreement in relaxation outcomes; and (ii) final-coordinate similarity in a separate geometry-capture experiment. This separation matters because the two experiments used different outcome bookkeeping, and energy agreement alone cannot establish geometric agreement.

**References needed before submission:** OC20/OC20-Dense dataset; UMA model and checkpoint; NVIDIA ALCHEMI Toolkit and inflight algorithm; FIRE optimizer; ASE; foundational work on heterogeneous GPU batching.

## 2. Methods

### 2.1 Dataset and model

We used a deterministic 500-structure subset of the public OC20-Dense OOD validation collection (39,123 atoms). The original manifest and sampling script in the benchmark repository define the exact selection. Force and energy evaluations used UMA `uma-s-1p2p1`, task `oc20`, and inference setting `default`. The benchmark was performed on one NVIDIA L40S GPU on HCC Swan, with ALCHEMI 0.2.0, Python 3.12.14, PyTorch 2.13.0+cu130, and CUDA 13.0.

### 2.2 Scheduling and relaxation

The five production configurations were B1, fixed B4, inflight B4 (maximum 360 atoms), inflight B8 (720 atoms), and inflight B16 (1440 atoms). Inflight refill frequency was one. FIRE relaxation used a 0.05 eV/Å force threshold, a 300-step maximum, and a timestep parameter of 0.1. Each production configuration was repeated three times on the same workload. The benchmark did not include inflight B2/B3; the best setting reported here is the best **tested** setting.

### 2.3 Performance and consistency metrics

Throughput was defined as the number of structures divided by the measured workload time. GPU utilization and peak memory came from GPU telemetry; utilization is not itself a measure of completed scientific work. B1 and inflight B4 production results were matched by structure and replicate (1,500 comparisons). We compared recorded convergence outcomes, optimization step counts, and final energies for jointly converged cases.

### 2.4 Separate coordinate-capture experiment

A separate one-pass B1 and one-pass inflight B4 experiment saved final atomic positions. Structures were matched by sample index, not frame order. Mobile-atom RMSD was calculated using the saved periodic-boundary-aware comparison. Adsorbate-only overlays and rigid-alignment analysis were used to characterize exceptional cases. The geometry-capture code recomputed a final-force threshold classification after saving endpoints, whereas the production workflow recorded relaxation outcomes during optimization. **The two experiments' raw convergence percentages must not be pooled or compared as though they used identical status accounting.**

## 3. Results

### 3.1 Inflight scheduling improves throughput

Inflight B4 was the fastest of the five tested configurations, reaching **0.2242 ± 0.0001 structures/s** (mean ± sample SD, n=3). B1 reached **0.1327 ± 0.0004 structures/s** and fixed B4 **0.1922 ± 0.0001 structures/s**. The resulting speedup was **1.69×** relative to B1 and **1.17×** relative to fixed B4. Mean time to complete 500 structures fell from **62.81 min** (B1) to **37.17 min** (inflight B4), a **40.8% reduction**. [Insert Figs. 1–2 and master performance table.]

### 3.2 GPU utilization is not throughput

Inflight B16 averaged **96.9% GPU utilization**, compared with **85.1%** for inflight B4, but processed only **0.1385 structures/s** compared with **0.2242 structures/s** for B4. Its peak observed GPU memory reached **41.71 GiB**, versus **14.15 GiB** for inflight B4. Higher occupancy therefore did not yield higher throughput in this heterogeneous workload. These data identify B4 as the best tested capacity, not a universal optimum. [Insert Figs. 3–4.]

### 3.3 Production relaxation behavior remains closely matched

Among **1,500** matched B1–inflight production comparisons, **1,486 (99.07%)** had the same recorded convergence status, and **1,409 (93.93%)** had the same recorded number of optimization steps. For the **567** comparisons in which both workflows converged, mean absolute final-energy difference was **0.0036 meV/atom** and the 95th percentile was **0.0227 meV/atom**. This measures workflow agreement, **not** accuracy relative to DFT. [Insert Figs. 5–7.]

### 3.4 Most jointly successful structures have similar final coordinates

In the separate geometry-capture experiment, **183** structures ended below the 0.05 eV/Å force threshold in both runs. Their median mobile-atom RMSD was **0.0100 Å**, and **181/183 (98.9%)** were below **0.10 Å**. The 95th percentile was **0.0462 Å**. This is a conditional, single-pass result and does not quantify uncertainty over independent coordinate-capture replicates. [Insert Fig. 8.]

Two jointly force-threshold-successful exceptions had larger geometric differences. Sample **208** had **1.005 Å** mobile-atom RMSD; rigid alignment of its CH₂O adsorbate reduced internal RMSD to **0.008 Å**, consistent with changed placement or orientation of an otherwise similar molecule. Sample **155** had **0.617 Å** mobile-atom RMSD and **0.798 Å** adsorbate RMSD after rigid alignment, consistent with internal rearrangement. Their total final-energy differences were **0.0206 eV** and **0.0407 eV**, respectively. These observations suggest alternative endpoints but do **not** prove distinct stable minima. [Insert Figs. 9–10.]

## 4. Discussion

The key engineering result is that a GPU scheduling policy can change campaign throughput substantially even when the force model and nominal relaxation settings are held fixed. Fixed batching improves device activity over processing one structure at a time, but it can couple systems with different relaxation lengths. Inflight scheduling can replace finished systems and better match active GPU work to a heterogeneous stream of structures. The throughput peak at the tested B4 capacity illustrates that scheduling must balance occupancy, per-step computational cost, and memory rather than maximizing utilization alone.

The scientific-consistency result is conditional but encouraging. Production convergence and step-count agreement between B1 and inflight B4 are high. Independently, the great majority of endpoints that satisfy the force criterion in both coordinate-capture workflows are geometrically close. Rare structures show much larger adsorbate changes despite small energy differences. Such cases merit targeted chemistry-aware analysis, especially if future screening uses final adsorption geometry or energy ranking to make decisions.

**Unresolved status-accounting issue:** The geometry-capture experiment produced a different apparent convergence agreement from the three-replicate production benchmark. Because endpoint labels were obtained differently, this manuscript reports the geometry result only for structures whose *saved final forces* satisfy the threshold in both runs. A dedicated exit-reason audit would be required before drawing conclusions about the discrepancy itself.

## 5. Limitations and future work

This study uses one GPU type, one UMA checkpoint, one 500-structure dataset subset, one FIRE configuration, and three production replicates. The B2/B3 inflight capacities were not tested, and no multi-GPU scaling or cross-hardware results are claimed. The geometry experiment has one pass per method. The workflow has not yet been evaluated against DFT reference energies for adsorption-energy accuracy or catalyst ranking. A future study should test wider workloads and GPUs, preserve per-structure exit reasons and full optimizer states, and evaluate predictive accuracy against independent reference calculations.

## 6. Conclusion

For a heterogeneous 500-structure OC20-Dense catalyst-relaxation workload on a single NVIDIA L40S, inflight B4 increased completed structures per second by **69.0%** and reduced elapsed time by **40.8%** relative to B1. It closely matched B1 production convergence behavior and produced similar final geometries for nearly all structures satisfying the force criterion in both coordinate-capture runs. The results support inflight scheduling as a promising workflow optimization for this setting, with clearly identified limitations around rare trajectory-sensitive structures and convergence bookkeeping.

---

## Required before external submission

- [ ] Verify citations/DOIs for OC20-Dense, UMA, ALCHEMI, FIRE, ASE, and benchmarking context.
- [ ] Confirm authors, contributions, affiliations, acknowledgments, funding, and hardware credit.
- [ ] Audit convergence exit-reason handling and update Methods/Discussion as warranted.
- [ ] Review raw sample 208 and 155 structures with chemistry expertise; do not label them proven distinct minima.
- [ ] Verify data redistribution rights and checkpoint access instructions.
- [ ] Verify code and results are reproducible from a fresh environment; pin dependencies.
- [ ] Select venue and apply its manuscript/data/code policies.