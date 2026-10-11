from __future__ import annotations

from pathlib import Path
import pandas as pd


def replace_method_rows(path: str | Path, new_rows: list[dict], method_prefix: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(new_rows)
    if path.exists():
        old = pd.read_csv(path)
        if "method" in old.columns:
            old = old[~old["method"].astype(str).str.startswith(method_prefix)]
        out = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        out = new
    out.to_csv(path, index=False)


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def append_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
