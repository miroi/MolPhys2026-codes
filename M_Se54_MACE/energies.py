#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Electronic adsorption energies and component decomposition."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from constants import EV_TO_KJ
from calculator import get_calculator_component_energies


def isolated_adsorbate_energy(calc, adsorbate_symbol):
    atom = Atoms(adsorbate_symbol, positions=[[0.0, 0.0, 0.0]])
    atom.calc = calc
    return float(atom.get_potential_energy())


def adsorption_energy(E_adsorbateSlab, E_Slab, E_adsorbate):
    values = [E_adsorbateSlab, E_Slab, E_adsorbate]
    if not all(np.isfinite(x) for x in values):
        raise ValueError("Non-finite energy detected")
    return float(E_adsorbateSlab - E_Slab - E_adsorbate)


def component_adsorption_energies(adsorbate_slab, slab, adsorbate, calc):
    """Return adsorption energies for each calculator component."""
    e_adsorbates = get_calculator_component_energies(adsorbate_slab, calc)
    e_slab = get_calculator_component_energies(slab, calc)
    e_adsorbate = get_calculator_component_energies(adsorbate, calc)
    labels = sorted(set(e_adsorbates) | set(e_slab) | set(e_adsorbate))
    return {
        label: float(e_adsorbates.get(label, 0.0) - e_slab.get(label, 0.0) - e_adsorbate.get(label, 0.0))
        for label in labels
    }


def ev_to_kjmol(energy):
    return float(energy) * EV_TO_KJ


def print_energy_summary(E_adsorbateSlab, E_Slab, E_adsorbate, E_ads, slab_name="slab", components=None, adsorbate_symbol="adsorbate"):

    print("\n" + "=" * 70)
    print("ENERGY SUMMARY")
    print("=" * 70)
    print(f"E({adsorbate_symbol}/{slab_name}) : {E_adsorbateSlab: .10f} eV")
    print(f"E({slab_name})    : {E_Slab: .10f} eV")
    print(f"E({adsorbate_symbol})            : {E_adsorbate: .10f} eV")
    print(f"ΔE_ads            : {E_ads: .10f} eV")
    print(f"                  : {ev_to_kjmol(E_ads): .4f} kJ/mol")
    if components:
        print("\nCalculator energy decomposition")
        for label, value in components.items():
            print(f"{label:<18}: {value: .12f} eV")
        print(f"{'SUM components':<18}: {sum(components.values()): .12f} eV")
    print("=" * 70)
