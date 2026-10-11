#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alchemi_uma.dataset import OCPDenseLMDB, load_pickle, lookup_mapping, resolve_mapping_path


def round_robin_sample(groups: dict[str, list[dict]], n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    keys = sorted(groups)
    shuffled = {}
    for key in keys:
        vals = list(groups[key])
        rng.shuffle(vals)
        shuffled[key] = vals

    selected = []
    depth = 0
    while len(selected) < n:
        added = False
        for key in keys:
            vals = shuffled[key]
            if depth < len(vals):
                selected.append(vals[depth])
                added = True
                if len(selected) == n:
                    break
        if not added:
            break
        depth += 1
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb-root", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/pilot_500/dataset_manifest.csv")
    args = ap.parse_args()

    ds = OCPDenseLMDB(args.lmdb_root)
    mapping_path = resolve_mapping_path(args.mapping)
    mapping = load_pickle(mapping_path)
    print(f"LMDB records: {len(ds):,}")
    print(f"Mapping: {mapping_path}")

    groups = defaultdict(list)
    missing_mapping = 0

    for idx in tqdm(range(len(ds)), desc="Indexing OC20-Dense"):
        summary = ds.record_summary(idx)
        meta = lookup_mapping(mapping, summary["sid"])
        system_id = meta.get("system_id")
        if system_id is None:
            missing_mapping += 1
            system_id = f"UNMAPPED_{summary['sid']}"

        row = {
            "lmdb_index": idx,
            "sid": summary["sid"],
            "system_id": system_id,
            "config_id": meta.get("config_id"),
            "mpid": meta.get("mpid"),
            "adsorbate": meta.get("adsorbate"),
            "miller_idx": str(meta.get("miller_idx")),
            "n_atoms": summary["n_atoms"],
            "n_fixed": summary["n_fixed"],
            "split": "oc20dense_ood_val",
        }
        groups[str(system_id)].append(row)

    if args.n > len(ds):
        raise ValueError(f"Requested {args.n} samples from only {len(ds)} records.")

    selected = round_robin_sample(groups, args.n, args.seed)
    for i, row in enumerate(selected):
        row["sample_idx"] = i
        row["seed"] = args.seed

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "sample_idx", "lmdb_index", "sid", "system_id", "config_id", "mpid",
        "adsorbate", "miller_idx", "n_atoms", "n_fixed", "split", "seed"
    ]
    pd.DataFrame(selected)[cols].to_csv(out, index=False)

    print(f"Selected {len(selected)} structures across "
          f"{len(set(r['system_id'] for r in selected))} systems.")
    if missing_mapping:
        print(f"WARNING: {missing_mapping} LMDB records had no mapping entry.")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
