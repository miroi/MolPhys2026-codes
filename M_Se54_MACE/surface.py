#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Input-structure handling, substrate constraints, and adsorbate adsorption geometry."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read


def read_vasp_structure(filename):
    """Read one structure from a VASP/ASE-readable structure file."""
    atoms = read(filename, format="vasp")
    if atoms is None or len(atoms) == 0:
        raise ValueError(f"Could not read a non-empty structure from {filename}")
    atoms.set_constraint([])
    return atoms


def _substrate_layers(atoms, substrate_symbol):
    """Return substrate atom indices, bottom-to-top layer z values and members."""
    symbols = np.asarray(atoms.get_chemical_symbols())
    indices = np.where(symbols == substrate_symbol)[0]
    if indices.size == 0:
        raise ValueError(
            f"No substrate atoms with symbol {substrate_symbol!r} were found"
        )

    z = np.asarray(atoms.positions[indices, 2], dtype=float)
    order = np.argsort(z)
    sorted_indices = indices[order]
    z_sorted = z[order]

    # Intra-layer relaxations are much smaller than the separation between
    # successive slab layers for the supplied slab structure.
    split_points = np.where(np.diff(z_sorted) > 0.5)[0] + 1
    groups = np.split(np.arange(len(sorted_indices)), split_points)
    layer_indices = [sorted_indices[g] for g in groups]
    layer_z = np.array([np.mean(z_sorted[g]) for g in groups], dtype=float)
    return indices, layer_z, layer_indices


def freeze_bottom_layers(atoms, n_layers, substrate_symbol):
    """Freeze only the bottom n substrate layers."""
    indices, layers, layer_indices = _substrate_layers(atoms, substrate_symbol)

    if n_layers < 0 or n_layers >= len(layers):
        raise ValueError(
            f"Invalid n_layers={n_layers}; slab has {len(layers)} "
            f"{substrate_symbol} layers"
        )

    mask = np.zeros(len(atoms), dtype=bool)
    for layer_atoms in layer_indices[:n_layers]:
        mask[np.asarray(layer_atoms, dtype=int)] = True

    atoms.set_constraint(FixAtoms(mask=mask))
    return atoms


def get_frozen_mask(atoms):
    """Return a boolean mask from all FixAtoms constraints."""
    mask = np.zeros(len(atoms), dtype=bool)
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            if hasattr(constraint, "get_indices"):
                idx = np.asarray(constraint.get_indices(), dtype=int)
            elif hasattr(constraint, "index"):
                idx = np.asarray(constraint.index, dtype=int)
            elif hasattr(constraint, "indices"):
                idx = np.asarray(constraint.indices, dtype=int)
            else:
                continue
            mask[idx] = True
    return mask


def get_mobile_indices(atoms):
    return np.where(~get_frozen_mask(atoms))[0].tolist()


def count_substrate_layers(atoms, substrate_symbol):
    return len(_substrate_layers(atoms, substrate_symbol)[1])


def get_adsorbate_surface_distance(atoms, substrate_symbol, adsorbate_symbol="Hg"):
    """Return the vertical adsorbate-to-top-substrate distance in Angstrom."""
    symbols = atoms.get_chemical_symbols()
    substrate_indices = [
        i for i, s in enumerate(symbols) if s == substrate_symbol
    ]
    adsorbate_indices = [i for i, s in enumerate(symbols) if s == adsorbate_symbol]

    if not substrate_indices:
        raise ValueError(f"No {substrate_symbol} atoms found")
    if len(adsorbate_indices) != 1:
        raise ValueError(
            f"Expected exactly one adsorbate atom, found {len(adsorbate_indices)}"
        )

    top_z = float(np.max(atoms.positions[substrate_indices, 2]))
    adsorbate_z = float(atoms.positions[adsorbate_indices[0], 2])
    return adsorbate_z - top_z



def get_nearest_adsorbate_surface_distance(
    atoms, substrate_symbol, adsorbate_symbol="Hg"
):
    """Return the minimum periodic 3D distance from the adsorbate to a top-layer substrate atom.

    The surface is defined as the highest substrate layer in z, using the same
    layer grouping as the rest of this module. Minimum-image distances are used
    when periodic boundary conditions are enabled.
    Returns (distance_A, adsorbate_index, substrate_index).
    """
    symbols = atoms.get_chemical_symbols()
    adsorbate_indices = [
        i for i, symbol in enumerate(symbols) if symbol == adsorbate_symbol
    ]
    if len(adsorbate_indices) != 1:
        raise ValueError(
            f"Expected exactly one adsorbate atom with symbol {adsorbate_symbol!r}, "
            f"found {len(adsorbate_indices)}"
        )

    _, layer_z, layer_indices = _substrate_layers(atoms, substrate_symbol)
    if not layer_indices:
        raise ValueError(f"No substrate atoms with symbol {substrate_symbol!r} were found")

    adsorbate_index = int(adsorbate_indices[0])
    top_layer_indices = np.asarray(layer_indices[-1], dtype=int)
    distances = np.asarray(
        atoms.get_distances(adsorbate_index, top_layer_indices, mic=True),
        dtype=float,
    )
    nearest_local_index = int(np.argmin(distances))
    return (
        float(distances[nearest_local_index]),
        adsorbate_index,
        int(top_layer_indices[nearest_local_index]),
    )

def print_surface_details(atoms, n_layers, substrate_symbol, adsorbate_symbol="Hg"):
    symbols = atoms.get_chemical_symbols()
    indices, layers, layer_indices = _substrate_layers(atoms, substrate_symbol)
    frozen = get_frozen_mask(atoms)

    print("\nSURFACE DETAILS")
    print("-" * 60)
    print("Total atoms:", len(atoms))
    print(f"{substrate_symbol} atoms:", len(indices))
    print(f"{substrate_symbol} layers:", len(layers))
    print("Frozen layers:", n_layers)

    for i, (z, members) in enumerate(zip(layers, layer_indices)):
        n_frozen = int(np.sum(frozen[members]))
        state = "FROZEN" if n_frozen == len(members) else "FREE/PARTIAL"
        print(
            f"Layer {i+1}: z={z:.6f} Å  {len(members)} {substrate_symbol}  {state}"
        )

    if adsorbate_symbol in symbols:
        print(
            f"\nadsorbate–{substrate_symbol} surface distance:",
            f"{get_adsorbate_surface_distance(atoms, substrate_symbol, adsorbate_symbol):.6f} Å",
        )
