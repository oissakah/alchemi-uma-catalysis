from __future__ import annotations

import numpy as np
import torch
from ase import Atoms

from nvalchemi._typing import AtomCategory
from nvalchemi.data import AtomicData, Batch


def atoms_to_atomic_data(atoms: Atoms) -> AtomicData:
    """Convert ASE atoms to ALCHEMI dynamics-ready AtomicData.

    OC20 tags: 0=subsurface/fixed, 1=surface, 2=adsorbate.
    FreezeAtomsHook freezes AtomCategory.SPECIAL.
    """
    data = AtomicData.from_atoms(atoms)
    n = data.num_nodes
    data.forces = torch.zeros(n, 3, dtype=torch.float32)
    data.energy = torch.zeros(1, 1, dtype=torch.float32)
    data.add_node_property("velocities", torch.zeros(n, 3, dtype=torch.float32))

    tags = np.asarray(atoms.get_tags())
    if len(tags) == n:
        tag_tensor = torch.as_tensor(tags, dtype=torch.long)
        data.atom_categories = torch.where(
            tag_tensor == 0,
            torch.tensor(AtomCategory.SPECIAL.value, dtype=torch.long),
            torch.tensor(AtomCategory.GAS.value, dtype=torch.long),
        )
    return data


def atoms_list_to_batch(atoms_list: list[Atoms], device: str = "cuda") -> Batch:
    batch = Batch.from_data_list([atoms_to_atomic_data(a) for a in atoms_list])
    return batch.to(device)


def per_graph_fmax(batch: Batch) -> torch.Tensor:
    norms = batch.forces.norm(dim=-1)
    out = torch.zeros(batch.num_graphs, device=norms.device, dtype=norms.dtype)
    out.scatter_reduce_(
        0, batch.batch_idx, norms, reduce="amax", include_self=True
    )
    return out
