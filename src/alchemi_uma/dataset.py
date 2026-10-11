from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import lmdb
import numpy as np
import torch
from ase import Atoms
from ase.constraints import FixAtoms


_METADATA_KEYS = {b"length", b"metadata", b"__len__", b"__keys__"}


def _scalar(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.numpy()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.reshape(-1)[0].item()
    return value


def _get(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _numpy(value: Any, default=None):
    if value is None:
        return default
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class RecordRef:
    db_index: int
    key: bytes


class OCPDenseLMDB:
    """Small read-only adapter for the legacy OC20-Dense LMDB release.

    The 2023 OC20-Dense validation archive predates fairchem's current ASE-DB
    dataset layer. This reader intentionally uses only LMDB + pickle and converts
    each stored PyG-like object to ASE Atoms.
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.paths = self._discover_lmdbs(self.root)
        if not self.paths:
            raise FileNotFoundError(
                f"No LMDB files/directories found under {self.root}. "
                "Run scripts/download_oc20dense_val.sh first."
            )

        self.envs = []
        self.refs: list[RecordRef] = []
        for db_index, path in enumerate(self.paths):
            env = lmdb.open(
                str(path),
                subdir=path.is_dir(),
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=64,
            )
            self.envs.append(env)
            with env.begin(write=False) as txn:
                cursor = txn.cursor()
                keys = [k for k, _ in cursor if k not in _METADATA_KEYS]
            # OCP records normally use integer ASCII keys. Keep only values that
            # successfully unpickle when accessed; metadata keys are filtered above.
            keys.sort(key=self._sort_key)
            self.refs.extend(RecordRef(db_index, k) for k in keys)

    @staticmethod
    def _sort_key(key: bytes):
        try:
            return (0, int(key.decode("ascii")))
        except Exception:
            return (1, key)

    @staticmethod
    def _discover_lmdbs(root: Path) -> list[Path]:
        candidates: list[Path] = []
        if root.is_file():
            return [root]
        # File-style LMDBs (common in OCP).
        for p in sorted(root.rglob("*.lmdb")):
            if p.is_file():
                candidates.append(p)
        # Directory-style LMDBs contain data.mdb.
        for p in sorted(root.rglob("data.mdb")):
            d = p.parent
            if d not in candidates:
                candidates.append(d)
        # Sometimes extraction gives a single extensionless file.
        if not candidates:
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name not in {"lock.mdb", "metadata.npz"}:
                    try:
                        env = lmdb.open(
                            str(p), subdir=False, readonly=True, lock=False,
                            readahead=False, meminit=False
                        )
                        with env.begin() as txn:
                            if txn.stat()["entries"] > 0:
                                candidates.append(p)
                        env.close()
                    except Exception:
                        pass
        return candidates

    def __len__(self) -> int:
        return len(self.refs)

    def raw(self, index: int):
        ref = self.refs[index]
        with self.envs[ref.db_index].begin(write=False) as txn:
            payload = txn.get(ref.key)
        if payload is None:
            raise KeyError(f"Missing LMDB key {ref.key!r}")
        return pickle.loads(payload)

    def sid(self, index: int):
        obj = self.raw(index)
        sid = _get(obj, "sid", index)
        sid = _scalar(sid)
        if isinstance(sid, bytes):
            sid = sid.decode()
        return sid

    def get_atoms(self, index: int) -> Atoms:
        obj = self.raw(index)
        pos = _numpy(_get(obj, "pos", _get(obj, "positions")))
        numbers = _numpy(_get(obj, "atomic_numbers", _get(obj, "numbers")))
        if pos is None or numbers is None:
            raise ValueError(
                f"Record {index} does not expose pos/positions and atomic_numbers/numbers."
            )

        cell = _numpy(_get(obj, "cell"))
        if cell is not None:
            cell = np.asarray(cell)
            if cell.ndim == 3:
                cell = cell[0]

        pbc = _numpy(_get(obj, "pbc"))
        if pbc is None:
            pbc = [True, True, True] if cell is not None else [False, False, False]
        pbc = np.asarray(pbc).reshape(-1)[:3].astype(bool)

        atoms = Atoms(
            numbers=np.asarray(numbers).reshape(-1).astype(int),
            positions=np.asarray(pos, dtype=float),
            cell=cell,
            pbc=pbc,
        )

        tags = _numpy(_get(obj, "tags"))
        if tags is not None:
            tags = np.asarray(tags).reshape(-1).astype(int)
            if len(tags) == len(atoms):
                atoms.set_tags(tags)

        fixed = _numpy(_get(obj, "fixed"))
        if fixed is not None:
            fixed = np.asarray(fixed).reshape(-1).astype(bool)
            if len(fixed) == len(atoms) and fixed.any():
                atoms.set_constraint(FixAtoms(mask=fixed))
        elif tags is not None:
            # OC20 convention: tag 0=subsurface/fixed, 1=surface, 2=adsorbate.
            mask = tags == 0
            if mask.any():
                atoms.set_constraint(FixAtoms(mask=mask))

        sid = self.sid(index)
        atoms.info["sid"] = sid
        return atoms

    def record_summary(self, index: int) -> dict[str, Any]:
        atoms = self.get_atoms(index)
        tags = atoms.get_tags()
        return {
            "lmdb_index": index,
            "sid": atoms.info["sid"],
            "n_atoms": len(atoms),
            "n_fixed": int((tags == 0).sum()) if len(tags) else 0,
        }

    def close(self):
        for env in self.envs:
            env.close()


def load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def lookup_mapping(mapping: dict, sid) -> dict:
    candidates = [sid, str(sid)]
    try:
        candidates.append(int(sid))
    except Exception:
        pass
    for key in candidates:
        if key in mapping:
            value = mapping[key]
            return value if isinstance(value, dict) else {"value": value}
    return {}


def resolve_mapping_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_file():
        return p
    matches = list(p.rglob("oc20dense_mapping.pkl"))
    if not matches:
        raise FileNotFoundError(f"Could not find oc20dense_mapping.pkl under {p}")
    return matches[0]
