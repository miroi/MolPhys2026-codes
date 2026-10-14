#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Geometry optimization utilities."""

from __future__ import annotations

import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import BFGS, FIRE


def get_mobile_indices(atoms):
    frozen = np.zeros(len(atoms), dtype=bool)

    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            if hasattr(constraint, "get_indices"):
                idx = constraint.get_indices()
            elif hasattr(constraint, "index"):
                idx = constraint.index
            elif hasattr(constraint, "indices"):
                idx = constraint.indices
            else:
                continue

            frozen[np.asarray(idx, dtype=int)] = True

    return np.where(~frozen)[0].tolist()


def optimize_structure(
    atoms,
    calc,
    name="opt",
    fmax=0.01,
    optimizer="FIRE",
    trajectory=True,
    steps=1000,
):
    """
    Optimize mobile atoms and require actual convergence.

    The convergence flag returned by ASE is checked against the requested
    force threshold.  The final maximum force on mobile atoms is also
    reported and checked explicitly.
    """
    if fmax <= 0:
        raise ValueError("fmax must be > 0")
    if steps <= 0:
        raise ValueError("steps must be > 0")

    atoms.calc = calc

    traj = f"{name}.traj" if trajectory else None
    logfile = f"{name}.log"

    optimizer_name = optimizer.upper()

    if optimizer_name == "FIRE":
        opt = FIRE(
            atoms,
            trajectory=traj,
            logfile=logfile,
        )
    elif optimizer_name == "BFGS":
        opt = BFGS(
            atoms,
            trajectory=traj,
            logfile=logfile,
        )
    else:
        raise ValueError(
            f"Unsupported optimizer: {optimizer}"
        )

    converged = bool(opt.run(fmax=fmax, steps=steps))

    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    mobile = get_mobile_indices(atoms)

    if mobile:
        max_force = float(
            np.max(np.linalg.norm(forces[mobile], axis=1))
        )
    elif len(forces):
        max_force = float(
            np.max(np.linalg.norm(forces, axis=1))
        )
    else:
        max_force = 0.0

    print("\nOptimization finished")
    print(f"Energy: {energy:.10f} eV")
    print(f"Max force on mobile atoms: {max_force:.6f} eV/Å")
    print(f"Converged: {'YES' if converged else 'NO'}")

    if not converged or max_force > fmax * (1.0 + 1e-8):
        raise RuntimeError(
            f"{name} did not converge to fmax={fmax:g} eV/Å "
            f"within {steps} steps (final max force={max_force:.6g} eV/Å)."
        )

    return atoms
