#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main executable for configurable single-atom adsorption."""

from __future__ import annotations

import gc

from calculator import (
    clear_cuda_cache,
    get_calculator,
    print_version,
)
from cli import parse_arguments
from config import Config
from workflow import run_workflow


def main():
    args = parse_arguments()

    config = Config()
    config.load(args.config)

    if args.model is not None:
        config.model_path = args.model

    if args.mode is not None:
        config.mode = args.mode

    if args.temperature is not None:
        config.temperature = args.temperature

    if args.pressure is not None:
        config.pressure = args.pressure

    if args.d4_enabled is not None:
        config.d4_enabled = args.d4_enabled

    if args.d4_method is not None:
        config.d4_method = args.d4_method

    if args.d4_scale is not None:
        config.d4_scale = args.d4_scale

    if config.mode == "scan":
        config.calculate_temperature_scan = True

    config.validate()

    clear_cuda_cache()
    print_version()

    calc = get_calculator(config)

    try:
        results = run_workflow(config, calc)
    finally:
        del calc
        gc.collect()
        clear_cuda_cache()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    # Explicit units are included in every scalar line so that the final
    # output is self-describing.  Energies are per adsorbate atom unless
    # otherwise noted; entropy values are per adsorbate atom.
    scalar_units = {
        "E_slab": "eV",
        "E_adsorbate_slab": "eV",
        "E_adsorbate": "eV",
        "E_ads": "eV",
        "E_ads_MACE": "eV",
        "E_ads_DFTD4": "eV",
        "E_ads_component_sum": "eV",
        "ZPE_ads": "eV",
        "delta_U_vib_thermal": "eV",
        "delta_U_vib_total": "eV",
        "H_gas_thermal": "eV",
        "H_ads": "eV",
        "S_gas": "eV/K",
        "delta_S_vib": "eV/K",
        "S_ads": "eV/K",
        "delta_F_vib": "eV",
        "mu_gas_thermal": "eV",
        "G_ads": "eV",
        "adsorbate_surface_distance_initial_A": "Å",
        "adsorbate_surface_distance_final_A": "Å",
        "adsorbate_surface_distance_change_A": "Å",
        "nearest_adsorbate_surface_atom_distance_initial_A": "Å",
        "nearest_adsorbate_surface_atom_distance_final_A": "Å",
        "nearest_adsorbate_surface_atom_distance_change_A": "Å",
        "substrate_n_atoms": "atoms",
        "slab_n_mobile": "atoms",
        "adsorbate_slab_n_mobile": "atoms",
    }

    scalar_keys = list(scalar_units)

    for key in scalar_keys:
        if key in results:
            unit = scalar_units[key]
            if unit == "atoms":
                print(f"{key:<24}: {results[key]: .0f} {unit}")
            else:
                print(f"{key:<24}: {results[key]: .10f} {unit}")

    root = results.get("G_ads_zero_crossing")
    if root is not None:
        print("\n" + "-" * 70)
        print("TEMPERATURE WHERE ΔG_ads = 0")
        print("-" * 70)

        if root.get("found", False):
            print(
                f"T_cross             : "
                f"{root['temperature_K']:.6f} K"
            )
            print(
                f"ΔG_ads(T_cross)     : "
                f"{root['G_ads_eV']:.6e} eV"
            )
            print(
                f"Root method         : "
                f"{root['method']}"
            )
            if root.get("bracket_K") is not None:
                print(
                    f"Final bracket       : "
                    f"[{root['bracket_K'][0]:.8f}, "
                    f"{root['bracket_K'][1]:.8f}] K"
                )
        else:
            print("No ΔG_ads = 0 crossing found.")
            print(
                f"Search interval     : "
                f"[{root['search_min_K']:.2f}, "
                f"{root['search_max_K']:.2f}] K"
            )
            print(
                f"ΔG_ads at limits    : "
                f"{root['G_ads_at_min_eV']:.8f}, "
                f"{root['G_ads_at_max_eV']:.8f} eV"
            )

    print("=" * 70)
    print("\nRun finished successfully.")


if __name__ == "__main__":
    main()
