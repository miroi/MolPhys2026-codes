#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Complete single-atom adsorption workflow on a supplied slab."""

from __future__ import annotations

import gc
import json
import math

import numpy as np

import calculator as calculator_module
import phonons as phonons_module
import thermodynamics as thermodynamics_module
from calculator import (
    clear_cuda_cache,
    get_calculator_component_energies,
    validate_calculator_decomposition,
    validate_d4_calculator_active,
    validate_d4_adsorption_component,
    evaluate_adsorbate_substrate_d4_pair,
)
from energies import (
    adsorption_energy,
    isolated_adsorbate_energy,
    component_adsorption_energies,
    print_energy_summary,
)
from optimization import optimize_structure
from phonons import calculate_phonons
from surface import (
    freeze_bottom_layers,
    get_adsorbate_surface_distance,
    get_nearest_adsorbate_surface_distance,
    print_surface_details,
    read_vasp_structure,
)
from thermodynamics import (
    calculate_adsorption_thermodynamics,
    print_ase_adsorbateslab_thermo_batch,
)


def _json_safe(value):
    """Recursively convert NumPy objects to JSON-serializable Python objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _save_json(filename, data):
    """Save results with recursive NumPy-to-JSON conversion."""
    clean = _json_safe(data)
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2)


def _g_ads(Eads, Slab_freq, AdsorbateSlab_freq, temperature, pressure, include_zpe, reference_pressure=1.0, method="custom"):
    """Evaluate Delta G_ads at one temperature."""
    thermo = calculate_adsorption_thermodynamics(
        Eads,
        AdsorbateSlab_freq,
        Slab_freq,
        float(temperature),
        pressure,
        include_zpe=include_zpe,
        reference_pressure=reference_pressure,
        method=method,
    )
    return float(thermo["G_ads"]), thermo


def _thermo_scan(Eads, Slab_freq, AdsorbateSlab_freq, config):
    """Evaluate the thermodynamics on the configured temperature grid."""
    rows = []

    for T in config.temperatures:
        g_ads, thermo = _g_ads(
            Eads,
            Slab_freq,
            AdsorbateSlab_freq,
            T,
            config.pressure,
            config.include_zpe,
            config.gas_reference_pressure,
            config.thermo_method,
        )
        rows.append({"T_K": float(T), **thermo})

    filename = config.output_prefix + "_thermodynamics_scan.csv"
    keys = [
        "T_K",
        "E_ads",
        "ZPE_ads",
        "delta_U_vib_thermal",
    "delta_U_vib_total",
        "delta_F_vib",
        "S_gas",
        "S_ads_vib",
        "S_surface_vib",
        "S_ads",
        "H_gas_thermal",
        "H_ads",
        "G_ads",
    ]

    with open(filename, "w", encoding="utf-8") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(
                ",".join(f"{row.get(k, float('nan')):.12g}" for k in keys)
                + "\n"
            )

    # Always show the temperature scan in the terminal when mode=scan.
    # The scan reuses the already calculated phonon frequencies; no new
    # electronic-structure or phonon calculations are performed here.
    print("\n" + "=" * 70)
    print("THERMODYNAMICS TEMPERATURE SCAN")
    print("-" * 70)
    print(
        f"{'T (K)':>12}  {'ΔH_ads (eV)':>16}  "
        f"{'ΔS_ads (eV/K)':>18}  {'ΔG_ads (eV)':>16}"
    )
    print("-" * 70)
    for row in rows:
        print(
            f"{row['T_K']:12.2f}  "
            f"{row['H_ads']:16.8f}  "
            f"{row['S_ads']:18.8e}  "
            f"{row['G_ads']:16.8f}"
        )
    print("-" * 70)
    print(
        "Scan temperatures: "
        + ", ".join(f"{float(T):.2f}" for T in config.temperatures)
        + " K"
    )
    print(f"Scan output: {filename}")
    print("=" * 70)

    return rows


def _find_g_ads_zero(
    Eads,
    Slab_freq,
    AdsorbateSlab_freq,
    config,
    temperatures=None,
):
    """
    Find the temperature at which Delta G_ads(T) = 0.

    The search first checks the supplied temperature grid. If no adjacent
    grid points bracket a root, it searches a continuous interval using
    config.temperature_root_min/max_K.  A safeguarded bisection solve is
    then used, so no electronic-structure or phonon calculations are needed.
    """
    if temperatures is None:
        temperatures = np.asarray(config.temperatures, dtype=float)
    else:
        temperatures = np.asarray(temperatures, dtype=float)

    temperatures = temperatures[np.isfinite(temperatures)]
    temperatures = temperatures[temperatures > 0.0]

    if len(temperatures) < 2:
        raise ValueError(
            "At least two positive temperatures are required in the "
            "[calculation] temperatures setting to locate a ΔG_ads = 0 root."
        )

    # The existing [calculation] temperatures define the root-search interval.
    # No additional temperature-range keywords are required.
    tmin = float(np.min(temperatures))
    tmax = float(np.max(temperatures))

    if tmin <= 0.0 or tmax <= tmin:
        raise ValueError(
            "The [calculation] temperatures setting must contain at least "
            "two distinct positive temperatures."
        )

    # Include the configured scan points that fall inside the root interval.
    grid = temperatures[(temperatures >= tmin) & (temperatures <= tmax)]
    grid = np.unique(np.concatenate(([tmin], grid, [tmax])))
    grid.sort()

    values = []
    for T in grid:
        g, _ = _g_ads(
            Eads, Slab_freq, AdsorbateSlab_freq, T,
            config.pressure, config.include_zpe,
            config.gas_reference_pressure, config.thermo_method
        )
        values.append(g)
    values = np.asarray(values, dtype=float)

    # Exact grid-point root.
    exact = np.where(np.isclose(values, 0.0, atol=1e-14, rtol=0.0))[0]
    if len(exact):
        idx = int(exact[0])
        T = float(grid[idx])
        g, thermo = _g_ads(
            Eads, Slab_freq, AdsorbateSlab_freq, T,
            config.pressure, config.include_zpe,
            config.gas_reference_pressure, config.thermo_method
        )
        return {
            "found": True,
            "temperature_K": T,
            "G_ads_eV": g,
            "method": "exact_grid_point",
            "bracket_K": [T, T],
            "iterations": 0,
            "thermodynamics": thermo,
        }

    # Locate the first sign change in ascending temperature.
    bracket = None
    for i in range(len(grid) - 1):
        if values[i] * values[i + 1] < 0.0:
            bracket = (float(grid[i]), float(grid[i + 1]))
            break

    if bracket is None:
        return {
            "found": False,
            "temperature_K": None,
            "G_ads_eV": None,
            "method": "no_sign_change",
            "bracket_K": None,
            "iterations": 0,
            "search_min_K": tmin,
            "search_max_K": tmax,
            "G_ads_at_min_eV": float(values[0]),
            "G_ads_at_max_eV": float(values[-1]),
            "message": (
                "No G_ads=0 sign change was found in the configured "
                "temperature interval."
            ),
        }

    a, b = bracket
    fa = float(values[np.where(grid == a)[0][0]])
    fb = float(values[np.where(grid == b)[0][0]])

    # Bisection: robust and fully deterministic.
    # Fixed numerical tolerances for the one-dimensional bisection.
    temperature_tol = 1.0e-4
    g_tol = 1.0e-10
    max_iter = 100

    iterations = 0
    mid = 0.5 * (a + b)
    fm = None
    thermo_mid = None

    while iterations < max_iter:
        iterations += 1
        mid = 0.5 * (a + b)
        fm, thermo_mid = _g_ads(
            Eads, Slab_freq, AdsorbateSlab_freq, mid,
            config.pressure, config.include_zpe,
            config.gas_reference_pressure, config.thermo_method
        )

        if abs(fm) <= g_tol or (b - a) <= temperature_tol:
            break

        if fa * fm < 0.0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm

    # Re-evaluate at the reported midpoint for a self-consistent result.
    g_final, thermo_final = _g_ads(
        Eads, Slab_freq, AdsorbateSlab_freq, mid,
        config.pressure, config.include_zpe,
        config.gas_reference_pressure, config.thermo_method
    )

    return {
        "found": True,
        "temperature_K": float(mid),
        "G_ads_eV": float(g_final),
        "method": "bisection",
        "bracket_K": [float(a), float(b)],
        "iterations": iterations,
        "temperature_tolerance_K": temperature_tol,
        "G_ads_tolerance_eV": g_tol,
        "thermodynamics": thermo_final,
    }


def run_workflow(config, calc):
    calculator_module.ADSORBATE_SYMBOL = config.adsorbate_symbol
    phonons_module.ADSORBATE_SYMBOL = config.adsorbate_symbol
    thermodynamics_module.ADSORBATE_SYMBOL = config.adsorbate_symbol
    print("\n" + "=" * 70)
    print("SINGLE-ATOM ADSORPTION WORKFLOW")
    print("=" * 70)

    # ---------------------------------------------------------
    # Read the two supplied VASP structures
    # ---------------------------------------------------------
    slab = read_vasp_structure(config.slab_file)
    adsorbate_slab = read_vasp_structure(config.adsorbate_file)

    for name, atoms in (("slab-only", slab), ("adsorbate/slab", adsorbate_slab)):
        symbols = atoms.get_chemical_symbols()
        if config.substrate_symbol not in symbols:
            raise ValueError(
                f"{name} file {config.slab_file if name == 'slab-only' else config.adsorbate_file} "
                f"contains no {config.substrate_symbol} atoms."
            )

    adsorbate_count = adsorbate_slab.get_chemical_symbols().count(config.adsorbate_symbol)
    if adsorbate_count != 1:
        raise ValueError(
            f"Expected exactly one {config.adsorbate_symbol} atom in "
            f"{config.adsorbate_file}; found {adsorbate_count}."
        )

    if adsorbate_slab.get_chemical_symbols().count(config.substrate_symbol) != slab.get_chemical_symbols().count(config.substrate_symbol):
        raise ValueError("The slab-only and adsorbate/slab structures do not contain the same number of substrate atoms.")

    # Apply the same bottom-layer relaxation constraints to both input systems.
    slab = freeze_bottom_layers(
        slab, config.n_frozen_layers, config.substrate_symbol
    )
    adsorbate_slab = freeze_bottom_layers(
        adsorbate_slab, config.n_frozen_layers, config.substrate_symbol
    )

    slab.calc = calc
    adsorbate_slab.calc = calc

    # Validate the calculator stack before any geometry optimization.  In
    # particular, this prevents a silently inactive D4 child calculator from
    # producing a result that is mislabeled as MACE+D4.
    validate_calculator_decomposition(slab, calc, label=f"initial {config.substrate_symbol} slab")
    validate_calculator_decomposition(adsorbate_slab, calc, label=f"initial {config.adsorbate_symbol}/{config.substrate_symbol} slab")
    from ase import Atoms
    adsorbate_reference_initial = Atoms(config.adsorbate_symbol, positions=[[0.0, 0.0, 0.0]])
    adsorbate_reference_initial.calc = calc
    validate_calculator_decomposition(adsorbate_reference_initial, calc, label=f"isolated {config.adsorbate_symbol}")
    if config.d4_enabled:
        # Check D4 itself with an independent known interacting pair.
        # Do not require the adsorbate/Se reaction combination to be nonzero here:
        # that quantity is a physical result, not an API activity test.
        validate_d4_calculator_active(
            calc, tolerance_eV=config.d4_zero_tolerance_eV
        )
        validate_d4_adsorption_component(
            adsorbate_slab, slab, adsorbate_reference_initial, calc,
            tolerance_eV=config.d4_zero_tolerance_eV,
            fail_if_zero=False,
            adsorbate_symbol=config.adsorbate_symbol,
            substrate_symbol=config.substrate_symbol,
        )

    print(f"\nReading slab-only structure: {config.slab_file}")
    print_surface_details(slab, config.n_frozen_layers, config.substrate_symbol, config.adsorbate_symbol)

    slab = optimize_structure(
        slab,
        calc,
        name=f"{config.substrate_symbol}_slab_opt",
        fmax=config.fmax_slab,
        optimizer=config.optimizer,
        trajectory=config.save_trajectories,
    )
    E_slab = float(slab.get_potential_energy())
    slab_components = get_calculator_component_energies(slab, calc)

    print(f"\nReading adsorbate-on-slab structure: {config.adsorbate_file}")
    print_surface_details(
        adsorbate_slab, config.n_frozen_layers, config.substrate_symbol, config.adsorbate_symbol
    )
    adsorbate_surface_distance_before = get_adsorbate_surface_distance(
        adsorbate_slab, config.substrate_symbol, config.adsorbate_symbol
    )
    print(
        f"adsorbate–{config.substrate_symbol} surface distance BEFORE optimization: "
        f"{adsorbate_surface_distance_before:.6f} Å"
    )
    (
        nearest_surface_distance_before,
        adsorbate_index_before,
        nearest_surface_index_before,
    ) = get_nearest_adsorbate_surface_distance(
        adsorbate_slab, config.substrate_symbol, config.adsorbate_symbol
    )
    print(
        f"Minimum periodic {config.adsorbate_symbol}–{config.substrate_symbol} "
        f"distance to a top-layer surface atom BEFORE optimization: "
        f"{nearest_surface_distance_before:.6f} Å "
        f"(indices {adsorbate_index_before}–{nearest_surface_index_before})"
    )

    adsorbate_slab = optimize_structure(
        adsorbate_slab,
        calc,
        name=f"{config.adsorbate_symbol}_{config.substrate_symbol}_opt",
        fmax=config.fmax_adsorbate,
        optimizer=config.optimizer,
        trajectory=config.save_trajectories,
    )
    E_adsorbate_slab = float(adsorbate_slab.get_potential_energy())
    adsorbate_slab_components = get_calculator_component_energies(adsorbate_slab, calc)

    adsorbate_surface_distance_after = get_adsorbate_surface_distance(
        adsorbate_slab, config.substrate_symbol, config.adsorbate_symbol
    )
    print(
        f"adsorbate–{config.substrate_symbol} surface distance AFTER optimization: "
        f"{adsorbate_surface_distance_after:.6f} Å"
    )
    print(
        f"adsorbate–{config.substrate_symbol} surface distance change: "
        f"{adsorbate_surface_distance_after - adsorbate_surface_distance_before:+.6f} Å"
    )
    (
        nearest_surface_distance_after,
        adsorbate_index_after,
        nearest_surface_index_after,
    ) = get_nearest_adsorbate_surface_distance(
        adsorbate_slab, config.substrate_symbol, config.adsorbate_symbol
    )
    print(
        f"Minimum periodic {config.adsorbate_symbol}–{config.substrate_symbol} "
        f"distance to a top-layer surface atom AFTER optimization: "
        f"{nearest_surface_distance_after:.6f} Å "
        f"(indices {adsorbate_index_after}–{nearest_surface_index_after})"
    )
    print(
        f"Nearest {config.adsorbate_symbol}–{config.substrate_symbol} distance change: "
        f"{nearest_surface_distance_after - nearest_surface_distance_before:+.6f} Å"
    )

    # Re-validate the active calculator after relaxation. This is important
    # because the phonon and optimization stages depend on the same combined
    # energy/force surface.
    validate_calculator_decomposition(slab, calc, label=f"optimized {config.substrate_symbol} slab")
    validate_calculator_decomposition(adsorbate_slab, calc, label=f"optimized {config.adsorbate_symbol}/{config.substrate_symbol} slab")

    # ---------------------------------------------------------
    # Isolated adsorbate atom and electronic adsorption energy
    # ---------------------------------------------------------
    E_adsorbate = isolated_adsorbate_energy(calc, config.adsorbate_symbol)
    adsorbate_reference = Atoms(config.adsorbate_symbol, positions=[[0.0, 0.0, 0.0]])
    adsorbate_reference.calc = calc
    adsorbate_components = get_calculator_component_energies(adsorbate_reference, calc)

    Eads = adsorption_energy(E_adsorbate_slab, E_slab, E_adsorbate)
    component_adsorption = component_adsorption_energies(
        adsorbate_slab, slab, adsorbate_reference, calc
    )

    # Explicitly evaluate the element-specific adsorbate--Se D4 pair interaction at
    # the actual optimized nearest-neighbour distance.  This is a diagnostic
    # quantity and is deliberately kept separate from the full periodic
    # slab-level D4 adsorption correction reported above.
    adsorbate_substrate_d4_pair = {"enabled": False, "adsorbateX_D4_pair_interaction_eV": 0.0}
    if config.d4_enabled:
        adsorbate_substrate_d4_pair = evaluate_adsorbate_substrate_d4_pair(
            adsorbate_slab,
            calc,
            substrate_symbol=config.substrate_symbol,
            method=config.d4_method,
            scale=config.d4_scale,
            params_tweaks={
                "s6": config.d4_s6, "s9": config.d4_s9,
                "alp": config.d4_alp, "s8": config.d4_s8,
                "a1": config.d4_a1, "a2": config.d4_a2,
            },
        )

    # The component sum must reconstruct the total adsorption energy.
    component_sum = float(sum(component_adsorption.values()))
    if not np.isclose(Eads, component_sum, atol=1e-10, rtol=1e-10):
        raise RuntimeError(
            "Calculator decomposition does not reconstruct ΔE_ads: "
            f"total={Eads:.16e} eV, components={component_sum:.16e} eV"
        )

    if config.d4_enabled and abs(component_adsorption.get("DFT-D4", 0.0)) <= config.d4_zero_tolerance_eV:
        print(
            "\nWARNING: the optimized adsorbate adsorption DFT-D4 contribution is "
            f"{component_adsorption.get('DFT-D4', 0.0):+.6e} eV. "
            "The independent D4 activity check passed, so this is treated as "
            "the calculated physical D4 contribution rather than as an inactive "
            "calculator."
        )

    print_energy_summary(
        E_adsorbate_slab, E_slab, E_adsorbate, Eads,
        slab_name=f"{config.substrate_symbol} slab",
        adsorbate_symbol=config.adsorbate_symbol,
        components={
            "MACE": component_adsorption.get("MACE", 0.0),
            "DFT-D4": component_adsorption.get("DFT-D4", 0.0),
        },
    )
    print(f"Total MACE contribution to ΔE_ads : {component_adsorption.get('MACE', 0.0): .10f} eV")
    print(f"Total DFT-D4 contribution to ΔE_ads: {component_adsorption.get('DFT-D4', 0.0): .10f} eV")
    if config.d4_enabled:
        print(
            f"Explicit adsorbate–{config.substrate_symbol} D4 pair interaction "
            f"at optimized distance: "
            f"{adsorbate_substrate_d4_pair.get('adsorbateX_D4_pair_interaction_eV', 0.0):+.15e} eV"
        )
        print(
            f"All adsorbate–{config.substrate_symbol} additive D4 pair terms: "
            f"{adsorbate_substrate_d4_pair.get('adsorbate_substrate_D4_all_pair_terms_eV', 0.0):+.15e} eV"
        )
        print(
            f"adsorbate-involving non-additive D4 contribution: "
            f"{adsorbate_substrate_d4_pair.get('adsorbate_D4_nonadditive_involving_eV', 0.0):+.15e} eV"
        )

    # ---------------------------------------------------------
    # Phonons
    # ---------------------------------------------------------
    clear_cuda_cache()

    slab_ph = calculate_phonons(
        slab,
        calc,
        displacement=config.phonon_displacement,
        prefix=f"{config.output_prefix}_slab_phonons",
        verbose=config.verbose,
        frequency_cutoff_cm=config.phonon_cutoff_cm,
        symmetrize=config.phonon_symmetrize,
        n_frozen_layers_vibrations=config.n_frozen_layers_vibrations,
        adsorbate_projection_threshold=config.adsorbate_mode_projection_threshold,
        nfree=config.ase_vibration_nfree,
        method=config.phonon_method,
        fail_on_imaginary=config.fail_on_imaginary,
        substrate_symbol=config.substrate_symbol,
    )

    if config.cuda_clear_between_phonons:
        clear_cuda_cache()
        gc.collect()

    adsorbate_slab_ph = calculate_phonons(
        adsorbate_slab,
        calc,
        displacement=config.phonon_displacement,
        prefix=f"{config.output_prefix}_{config.adsorbate_symbol}_slab_phonons",
        verbose=config.verbose,
        frequency_cutoff_cm=config.phonon_cutoff_cm,
        symmetrize=config.phonon_symmetrize,
        n_frozen_layers_vibrations=config.n_frozen_layers_vibrations,
        adsorbate_projection_threshold=config.adsorbate_mode_projection_threshold,
        nfree=config.ase_vibration_nfree,
        method=config.phonon_method,
        fail_on_imaginary=config.fail_on_imaginary,
        substrate_symbol=config.substrate_symbol,
    )

    slab_freq = slab_ph["frequencies_cm"]
    adsorbate_slab_freq = adsorbate_slab_ph["frequencies_cm"]

    print(
        "Thermodynamics backend:",
        "project custom equations"
        if config.thermo_method == "custom"
        else "ASE HarmonicThermo + IdealGasThermo",
    )

    thermo = calculate_adsorption_thermodynamics(
        Eads,
        adsorbate_slab_freq,
        slab_freq,
        config.temperature,
        config.pressure,
        include_zpe=config.include_zpe,
        reference_pressure=config.gas_reference_pressure,
        method=config.thermo_method,
    )

    if config.thermo_method == "ase":
        print_ase_adsorbateslab_thermo_batch(adsorbate_slab_freq, config.temperatures)

    print("\n" + "=" * 70)
    print("ADSORPTION THERMODYNAMICS")
    print("=" * 70)
    print(f"T = {config.temperature:.2f} K")
    print(f"p = {config.pressure:.6g} bar")
    print(f"ΔE_ads          = {thermo['E_ads']:.8f} eV")
    print(f"ΔZPE            = {thermo['ZPE_ads']:.8f} eV")
    print(f"ΔU_vib          = {thermo['delta_U_vib_thermal']:.8f} eV")
    print(f"H_gas thermal   = {thermo['H_gas_thermal']:.8f} eV")
    print(f"ΔH_ads          = {thermo['H_ads']:.8f} eV")
    print(f"S_gas           = {thermo['S_gas']:.8e} eV/K")
    print(f"ΔS_vib          = {thermo['delta_S_vib']:.8e} eV/K")
    print(f"ΔS_ads          = {thermo['S_ads']:.8e} eV/K")
    print(f"ΔF_vib          = {thermo['delta_F_vib']:.8f} eV")
    print(f"μ_gas thermal   = {thermo['mu_gas_thermal']:.8f} eV")
    print(f"ΔG_ads          = {thermo['G_ads']:.8f} eV")
    print("=" * 70)

    results = {
        "E_slab": E_slab,
        "E_adsorbate_slab": E_adsorbate_slab,
        "E_adsorbate": E_adsorbate,
        "E_ads": Eads,
        "E_ads_MACE": component_adsorption.get("MACE", 0.0),
        "E_ads_DFTD4": component_adsorption.get("DFT-D4", 0.0),
        "E_ads_component_sum": component_sum,
        "adsorbate_substrate_D4_pair_diagnostic": adsorbate_substrate_d4_pair,
        "d4_enabled": bool(config.d4_enabled),
        "d4_method": config.d4_method if config.d4_enabled else None,
        "d4_scale": float(config.d4_scale) if config.d4_enabled else None,
        "adsorbate_surface_distance_initial_A": float(adsorbate_surface_distance_before),
        "adsorbate_surface_distance_final_A": float(adsorbate_surface_distance_after),
        "adsorbate_surface_distance_change_A": float(
            adsorbate_surface_distance_after - adsorbate_surface_distance_before
        ),
        "nearest_adsorbate_surface_atom_distance_initial_A": float(
            nearest_surface_distance_before
        ),
        "nearest_adsorbate_surface_atom_distance_final_A": float(
            nearest_surface_distance_after
        ),
        "nearest_adsorbate_surface_atom_distance_change_A": float(
            nearest_surface_distance_after - nearest_surface_distance_before
        ),
        "nearest_surface_atom_index_initial": int(nearest_surface_index_before),
        "nearest_surface_atom_index_final": int(nearest_surface_index_after),
        **thermo,
        "substrate_symbol": config.substrate_symbol,
        "substrate_n_atoms": int(slab.get_chemical_symbols().count(config.substrate_symbol)),
        "vibrational_n_frozen_layers": int(config.n_frozen_layers_vibrations),
        "slab_n_mobile": int(len(slab_ph["phonon_atoms"])),
        "AdsorbateSlab_n_mobile": int(len(adsorbate_slab_ph["phonon_atoms"])),
        "slab_n_modes": int(len(slab_freq)),
        "AdsorbateSlab_n_modes": int(len(adsorbate_slab_freq)),
        "adsorbate_mode_frequencies_cm-1": (
            adsorbate_slab_ph["signed_frequencies_cm"][
                adsorbate_slab_ph["adsorbate_mode_mask"]
            ].tolist()
        ),
    }

    if config.calculate_temperature_scan or config.mode == "scan":
        scan = _thermo_scan(Eads, slab_freq, adsorbate_slab_freq, config)
        results["temperature_scan"] = scan

    if config.calculate_g_ads_zero_crossing:
        root = _find_g_ads_zero(Eads, slab_freq, adsorbate_slab_freq, config)
        results["G_ads_zero_crossing"] = root

        print("\n" + "=" * 70)
        print("ΔG_ads = 0 CROSSING TEMPERATURE")
        print("=" * 70)
        if root["found"]:
            print(f"T_cross        = {root['temperature_K']:.6f} K")
            print(f"ΔG_ads(T_cross)= {root['G_ads_eV']:.6e} eV")
            print(f"Method          = {root['method']}")
            print(
                "Final bracket   = "
                f"[{root['bracket_K'][0]:.8f}, "
                f"{root['bracket_K'][1]:.8f}] K"
            )
            print(f"Iterations      = {root['iterations']}")
        else:
            print("No ΔG_ads = 0 crossing found.")
            print(
                f"Search interval = "
                f"[{root['search_min_K']:.2f}, "
                f"{root['search_max_K']:.2f}] K"
            )
            print(
                f"ΔG_ads at limits = "
                f"{root['G_ads_at_min_eV']:.8f}, "
                f"{root['G_ads_at_max_eV']:.8f} eV"
            )
            print(root["message"])

    print("=" * 70)
    _save_json(config.output_prefix + "_results.json", results)
    print("\nWorkflow completed.")
    return results
