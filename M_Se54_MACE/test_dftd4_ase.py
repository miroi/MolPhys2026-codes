#!/usr/bin/env python3
"""Standalone DFT-D4/ASE and additive-energy/force smoke tests."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
from dftd4.ase import DFTD4


def main():
    # Official DFT-D4 example: DFTD4 can be combined with another ASE
    # calculator through SumCalculator.
    water = molecule("H2O")
    d4 = DFTD4(method="PBE")
    water.calc = d4
    e_d4 = float(water.get_potential_energy())
    f_d4 = np.asarray(water.get_forces(), dtype=float)
    assert np.isfinite(e_d4)
    assert f_d4.shape == (3, 3)
    assert np.all(np.isfinite(f_d4))

    water2 = molecule("H2O")
    mace_like = EMT()
    d4_2 = DFTD4(method="PBE")
    water2.calc = SumCalculator([d4_2, mace_like])
    e_total = float(water2.get_potential_energy())
    f_total = np.asarray(water2.get_forces(), dtype=float)
    children = (water2.calc.mixer.calcs
                if hasattr(water2.calc, "mixer") and hasattr(water2.calc.mixer, "calcs")
                else water2.calc.calcs)
    e_parts = [float(c.get_potential_energy(water2)) for c in children]
    f_parts = [np.asarray(c.get_forces(water2), dtype=float) for c in children]

    assert np.allclose(e_total, sum(e_parts), atol=1e-10, rtol=1e-10)
    assert np.allclose(f_total, np.sum(f_parts, axis=0), atol=1e-8, rtol=1e-8)

    # A separated two-body system must have a nonzero D4 interaction at a
    # chemically relevant distance. This catches an imported-but-inactive D4.
    pair = Atoms("ArAr", positions=[[0, 0, 0], [0, 0, 4.0]], pbc=False)
    pair.calc = DFTD4(method="PBE")
    e_pair = float(pair.get_potential_energy())
    assert abs(e_pair) > 1e-12, f"unexpectedly zero D4 energy: {e_pair!r}"

    print("DFT-D4 ASE smoke test: OK")
    print(f"D4(PBE) H2O energy       : {e_d4:.12f} eV")
    print(f"D4(PBE) Ar2 4.0 A energy: {e_pair:.12f} eV")
    print(f"SumCalculator energy     : {e_total:.12f} eV")
    print("Energy/force additivity  : verified")


if __name__ == "__main__":
    main()
