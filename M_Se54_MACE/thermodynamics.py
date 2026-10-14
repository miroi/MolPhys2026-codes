#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Thermodynamics for adsorbate(g) -> adsorbate* on the supplied slab.

Two selectable backends are supported:

    custom
        The project's original explicit harmonic/ideal-gas equations.

    ase
        ASE's HarmonicThermo for the two surface states and ASE's
        IdealGasThermo for monatomic adsorbate(g).  Electronic adsorption energy is
        kept separate from thermochemistry so that the reaction convention
        remains exactly

            ΔG_ads = E_ads + ΔF_vib - μ_adsorbate,thermal.

The backend is selected with [thermodynamics] method = custom|ase.

All energies are eV per adsorbed adsorbate atom and entropies are eV/K per atom.
"""

from __future__ import annotations

import math
import numpy as np

from constants import (
    BAR_TO_PA,
    EV_J,
    H_PLANCK,
    KB_EV,
    KB_J,
    HC_EV_CM,
)


ADSORBATE_SYMBOL = "Hg"  # Set from Config at workflow startup.

def _adsorbate_mass_kg():
    from ase.data import atomic_masses, atomic_numbers
    if ADSORBATE_SYMBOL not in atomic_numbers:
        raise ValueError(f"Unknown adsorbate element symbol: {ADSORBATE_SYMBOL}")
    return float(atomic_masses[atomic_numbers[ADSORBATE_SYMBOL]] * 1.0e-3 / 6.02214076e23)


def _positive_frequencies(frequencies_cm):
    f = np.asarray(frequencies_cm, dtype=float).ravel()
    return f[f > 0.0]


def zero_point_energy(frequencies_cm):
    """Zero-point vibrational energy in eV."""
    f = _positive_frequencies(frequencies_cm)
    if f.size == 0:
        return 0.0
    return float(0.5 * np.sum(HC_EV_CM * f))


def zpe(frequencies_cm):
    """Backward-compatible alias for zero_point_energy()."""
    return zero_point_energy(frequencies_cm)


def thermal_vibrational_energy(frequencies_cm, temperature):
    """Thermal vibrational internal energy, excluding ZPE, in eV."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    f = _positive_frequencies(frequencies_cm)
    if f.size == 0:
        return 0.0

    hv = HC_EV_CM * f
    x = hv / (KB_EV * temperature)
    thermal = np.zeros_like(x)
    moderate = x < 700.0
    thermal[moderate] = hv[moderate] / np.expm1(x[moderate])
    return float(np.sum(thermal))


def vibrational_internal_energy(frequencies_cm, temperature, include_zpe=True):
    """Total vibrational internal energy in eV."""
    zpe_term = zero_point_energy(frequencies_cm) if include_zpe else 0.0
    return float(zpe_term + thermal_vibrational_energy(frequencies_cm, temperature))


def vibrational_entropy(frequencies_cm, temperature):
    """Harmonic vibrational entropy in eV/K."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    f = _positive_frequencies(frequencies_cm)
    if f.size == 0:
        return 0.0

    hv = HC_EV_CM * f
    x = hv / (KB_EV * temperature)
    entropy = np.zeros_like(x)

    small = x < 1.0e-6
    entropy[small] = KB_EV * (1.0 - np.log(x[small]))

    regular = ~small
    xr = x[regular]
    entropy[regular] = KB_EV * (
        xr / np.expm1(xr) - np.log1p(-np.exp(-xr))
    )
    return float(np.sum(entropy))


def thermal_vibrational_free_energy(frequencies_cm, temperature):
    """Thermal vibrational free-energy contribution excluding ZPE, in eV."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    f = _positive_frequencies(frequencies_cm)
    if f.size == 0:
        return 0.0

    hv = HC_EV_CM * f
    x = hv / (KB_EV * temperature)
    thermal_term = np.zeros_like(x)
    moderate = x < 700.0
    thermal_term[moderate] = (
        KB_EV * temperature * np.log1p(-np.exp(-x[moderate]))
    )
    return float(np.sum(thermal_term))


def vibrational_free_energy(frequencies_cm, temperature, include_zpe=True):
    """Total harmonic vibrational Helmholtz free energy in eV."""
    zpe_term = zero_point_energy(frequencies_cm) if include_zpe else 0.0
    return float(zpe_term + thermal_vibrational_free_energy(frequencies_cm, temperature))


def adsorbate_gas_entropy(temperature, pressure_bar=1.0, reference_pressure_bar=1.0):
    """Ideal-gas translational entropy of one adsorbate atom in eV/K."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if pressure_bar <= 0 or reference_pressure_bar <= 0:
        raise ValueError("pressures must be > 0")

    p0_pa = reference_pressure_bar * BAR_TO_PA
    thermal_wavelength = H_PLANCK / math.sqrt(
        2.0 * math.pi * _adsorbate_mass_kg() * KB_J * temperature
    )
    s0_j_per_k = KB_J * (
        math.log(KB_J * temperature / (p0_pa * thermal_wavelength**3)) + 2.5
    )
    s_j_per_k = s0_j_per_k - KB_J * math.log(
        pressure_bar / reference_pressure_bar
    )
    return float(s_j_per_k / EV_J)


def adsorbate_gas_entropy_standard(temperature, reference_pressure_bar=1.0):
    """Ideal-gas adsorbate entropy at the configured standard/reference pressure."""
    return adsorbate_gas_entropy(
        temperature,
        pressure_bar=reference_pressure_bar,
        reference_pressure_bar=reference_pressure_bar,
    )


def adsorbate_gas_enthalpy_thermal(temperature):
    """Ideal monatomic-gas thermal enthalpy relative to E_adsorbate."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return float(2.5 * KB_EV * temperature)


def adsorbate_gas_chemical_potential_thermal(
    temperature, pressure_bar=1.0, reference_pressure_bar=1.0
):
    """Ideal-gas μ thermal correction relative to isolated-atom electronic E."""
    s_gas = adsorbate_gas_entropy(
        temperature,
        pressure_bar=pressure_bar,
        reference_pressure_bar=reference_pressure_bar,
    )
    h_gas = adsorbate_gas_enthalpy_thermal(temperature)
    return float(h_gas - temperature * s_gas)


def _ase_thermochemistry_components(
    frequencies_cm, temperature, potentialenergy=0.0, reference_pressure_bar=1.0
):
    """Return ASE HarmonicThermo U/S/F components for positive frequencies."""
    try:
        from ase.thermochemistry import HarmonicThermo
    except ImportError as exc:
        raise ImportError(
            "The ASE thermodynamics backend requires ASE. Install ASE in the "
            "same environment used to run the workflow."
        ) from exc

    f = _positive_frequencies(frequencies_cm)

    # ASE 3.29 HarmonicThermo does not robustly handle an empty vibrational
    # spectrum.  This occurs legitimately for the clean slab when every substrate
    # layer is frozen for vibrational/thermochemical calculations.  Return
    # exact zero vibrational contributions instead of constructing an empty
    # HarmonicThermo object.
    if f.size == 0:
        return {
            "thermo": None,
            "zpe": 0.0,
            "u_total": float(potentialenergy),
            "u_thermal": 0.0,
            "entropy": 0.0,
            "free_energy": float(potentialenergy),
        }

    vib_energies = (HC_EV_CM * f).tolist()

    thermo = HarmonicThermo(
        vib_energies=vib_energies,
        potentialenergy=float(potentialenergy),
        ignore_imag_modes=False,
    )

    zpe = float(thermo.get_ZPE_correction())
    u_total = float(thermo.get_internal_energy(temperature, verbose=False))
    entropy = float(thermo.get_entropy(temperature, verbose=False))
    free_energy = float(thermo.get_helmholtz_energy(temperature, verbose=False))

    return {
        "thermo": thermo,
        "zpe": zpe,
        "u_total": u_total,
        "u_thermal": u_total - float(potentialenergy) - zpe,
        "entropy": entropy,
        "free_energy": free_energy,
    }



def print_ase_adsorbateslab_thermo_batch(frequencies_cm, temperatures):
    """Print ASE HarmonicThermo properties for adsorbate@slab only.

    Output columns:
        Temp(K)    ZPE(eV)    F(eV)    U(eV)    S(eV/K)

    This is a diagnostic report for the adsorbed adsorbate@slab vibrational
    system only. It does not subtract the clean substrate slab contribution.
    """
    try:
        from ase.thermochemistry import HarmonicThermo
    except ImportError as exc:
        raise ImportError(
            "ASE HarmonicThermo reporting requires ASE."
        ) from exc

    f = _positive_frequencies(frequencies_cm)
    if f.size == 0:
        print(f"\n{ADSORBATE_SYMBOL}@slab ASE HarmonicThermo: no vibrational modes")
        return

    vib_energies = (HC_EV_CM * f).tolist()
    thermo = HarmonicThermo(
        vib_energies=vib_energies,
        potentialenergy=0.0,
        ignore_imag_modes=False,
    )

    zpe = float(thermo.get_ZPE_correction())

    # Keep the columns visually separated.  Explicit two-space gaps avoid
    # ambiguous concatenation of adjacent scientific/fixed-point values.
    width = 86
    print("\n" + "=" * width)
    print(f"{ADSORBATE_SYMBOL}@slab vibrational thermodynamics (ASE HarmonicThermo)")
    print("=" * width)
    print(
        f"{'Temp(K)':>10s}  "
        f"{'ZPE(eV)':>16s}  "
        f"{'F(eV)':>16s}  "
        f"{'U(eV)':>16s}  "
        f"{'S(eV/K)':>16s}"
    )
    print("-" * width)

    for T in temperatures:
        T = float(T)
        F = float(thermo.get_helmholtz_energy(T, verbose=False))
        U = float(thermo.get_internal_energy(T, verbose=False))
        S = float((U - F) / T)
        print(
            f"{T:10.2f}  "
            f"{zpe:16.10f}  "
            f"{F:16.10f}  "
            f"{U:16.10f}  "
            f"{S:16.10e}"
        )

    print("=" * width)

def _ase_adsorbate_gas_components(temperature, pressure_bar, reference_pressure_bar):
    """Return ASE IdealGasThermo components for monatomic adsorbate(g)."""
    try:
        from ase import Atoms
        from ase.thermochemistry import IdealGasThermo
    except ImportError as exc:
        raise ImportError(
            "The ASE thermodynamics backend requires ASE. Install ASE in the "
            "same environment used to run the workflow."
        ) from exc

    # ASE's ideal-gas pressure is in Pa.  The project configuration uses bar.
    pressure_pa = float(pressure_bar * BAR_TO_PA)
    reference_pressure_pa = float(reference_pressure_bar * BAR_TO_PA)

    # ASE 3.29 still supports the vib_energies interface, although its newer
    # thermochemistry framework is moving toward mode objects.  For a adsorbate atom
    # there are no vibrational or rotational degrees of freedom.
    # ASE requires an Atoms object for entropy/free-energy calculations so
    # that the translational mass can be obtained.  For monatomic adsorbate this is
    # simply a one-atom Atoms object; no molecular vibrations or rotations are
    # present.
    adsorbate_atoms = Atoms(ADSORBATE_SYMBOL)
    gas = IdealGasThermo(
        vib_energies=[],
        geometry="monatomic",
        potentialenergy=0.0,
        atoms=adsorbate_atoms,
        natoms=1,
        symmetrynumber=1,
        spin=0,
        ignore_imag_modes=False,
    )

    # ASE documents 1 bar as the default reference pressure.  Explicitly set
    # it so the project setting is honored even when it differs from 1 bar.
    if hasattr(gas, "referencepressure"):
        gas.referencepressure = reference_pressure_pa

    h = float(gas.get_enthalpy(temperature, verbose=False))
    s = float(gas.get_entropy(temperature, pressure_pa, verbose=False))
    g = float(gas.get_gibbs_energy(temperature, pressure_pa, verbose=False))

    return {
        "thermo": gas,
        "H": h,
        "S": s,
        "G": g,
        "mu_thermal": g,
        "reference_pressure_pa": reference_pressure_pa,
    }


def _calculate_adsorption_thermodynamics_custom(
    E_ads,
    f_ads,
    f_surface,
    temperature,
    pressure,
    include_zpe,
    reference_pressure,
):
    """Original project thermodynamics implementation."""
    zpe_ads_system = zero_point_energy(f_ads)
    zpe_surface = zero_point_energy(f_surface)
    delta_zpe_raw = zpe_ads_system - zpe_surface
    delta_zpe = delta_zpe_raw if include_zpe else 0.0

    u_ads_thermal = thermal_vibrational_energy(f_ads, temperature)
    u_surface_thermal = thermal_vibrational_energy(f_surface, temperature)
    delta_u_vib_thermal = u_ads_thermal - u_surface_thermal

    u_ads_total = vibrational_internal_energy(
        f_ads, temperature, include_zpe=include_zpe
    )
    u_surface_total = vibrational_internal_energy(
        f_surface, temperature, include_zpe=include_zpe
    )
    delta_u_vib_total = u_ads_total - u_surface_total

    s_ads_vib = vibrational_entropy(f_ads, temperature)
    s_surface_vib = vibrational_entropy(f_surface, temperature)
    delta_s_vib = s_ads_vib - s_surface_vib

    f_ads_vib = vibrational_free_energy(
        f_ads, temperature, include_zpe=include_zpe
    )
    f_surface_vib = vibrational_free_energy(
        f_surface, temperature, include_zpe=include_zpe
    )
    delta_f_vib = f_ads_vib - f_surface_vib

    gas_s_standard = adsorbate_gas_entropy_standard(temperature, reference_pressure)
    gas_s = adsorbate_gas_entropy(
        temperature, pressure_bar=pressure,
        reference_pressure_bar=reference_pressure,
    )
    gas_h = adsorbate_gas_enthalpy_thermal(temperature)
    gas_mu = adsorbate_gas_chemical_potential_thermal(
        temperature, pressure_bar=pressure,
        reference_pressure_bar=reference_pressure,
    )

    H_ads = float(E_ads + delta_zpe + delta_u_vib_thermal - gas_h)
    S_ads = float(delta_s_vib - gas_s)
    G_ads = float(H_ads - temperature * S_ads)
    G_from_mu = float(E_ads + delta_f_vib - gas_mu)

    if not np.isclose(G_ads, G_from_mu, atol=2e-9, rtol=2e-10):
        raise RuntimeError(
            "Thermodynamic inconsistency: H-TS and F-mu routes disagree "
            f"({G_ads:.12e} vs {G_from_mu:.12e} eV)"
        )

    return {
        "thermodynamics_backend": "custom",
        "E_ads": float(E_ads),
        "ZPE_ads": float(delta_zpe),
        "ZPE_ads_raw": float(delta_zpe_raw),
        "delta_U_vib_thermal": float(delta_u_vib_thermal),
        "delta_U_vib_total": float(delta_u_vib_total),
        "delta_F_vib": float(delta_f_vib),
        "delta_S_vib": float(delta_s_vib),
        "S_gas_standard": float(gas_s_standard),
        "S_gas": float(gas_s),
        "H_gas_thermal": float(gas_h),
        "mu_gas_thermal": float(gas_mu),
        "H_ads": float(H_ads),
        "S_ads": float(S_ads),
        "G_ads": float(G_ads),
        "T_S_ads": float(temperature * S_ads),
        "zpe_surface": float(zpe_surface),
        "zpe_ads_system": float(zpe_ads_system),
        "U_surface_thermal": float(u_surface_thermal),
        "U_ads_system_thermal": float(u_ads_thermal),
        "U_surface_total": float(u_surface_total),
        "U_ads_system_total": float(u_ads_total),
        "S_surface_vib": float(s_surface_vib),
        "S_ads_vib": float(s_ads_vib),
        "F_surface": float(f_surface_vib),
        "F_ads_system": float(f_ads_vib),
        "pressure_bar": float(pressure),
        "reference_pressure_bar": float(reference_pressure),
        "include_zpe": bool(include_zpe),
    }


def _calculate_adsorption_thermodynamics_ase(
    E_ads,
    f_ads,
    f_surface,
    temperature,
    pressure,
    include_zpe,
    reference_pressure,
):
    """ASE HarmonicThermo + IdealGasThermo implementation."""
    ads = _ase_thermochemistry_components(f_ads, temperature)
    surface = _ase_thermochemistry_components(f_surface, temperature)
    gas = _ase_adsorbate_gas_components(temperature, pressure, reference_pressure)

    delta_zpe_raw = ads["zpe"] - surface["zpe"]
    delta_zpe = delta_zpe_raw if include_zpe else 0.0
    delta_u_vib_thermal = ads["u_thermal"] - surface["u_thermal"]
    delta_u_vib_total = (
        delta_zpe + delta_u_vib_thermal if include_zpe
        else delta_u_vib_thermal
    )
    delta_s_vib = ads["entropy"] - surface["entropy"]

    # HarmonicThermo always contains ZPE in its Helmholtz free energy.  Remove
    # ΔZPE when the project option include_zpe=False so the switch retains the
    # same meaning in both backends.
    delta_f_vib_ase_full = ads["free_energy"] - surface["free_energy"]
    delta_f_vib = (
        delta_f_vib_ase_full if include_zpe
        else delta_f_vib_ase_full - delta_zpe_raw
    )

    gas_h = gas["H"]
    gas_s = gas["S"]
    gas_mu = gas["G"]

    H_ads = float(E_ads + delta_zpe + delta_u_vib_thermal - gas_h)
    S_ads = float(delta_s_vib - gas_s)
    G_ads = float(H_ads - temperature * S_ads)
    G_from_mu = float(E_ads + delta_f_vib - gas_mu)

    if not np.isclose(G_ads, G_from_mu, atol=2e-8, rtol=2e-9):
        raise RuntimeError(
            "ASE thermodynamic inconsistency: H-TS and F-mu routes disagree "
            f"({G_ads:.12e} vs {G_from_mu:.12e} eV)"
        )

    return {
        "thermodynamics_backend": "ase",
        "E_ads": float(E_ads),
        "ZPE_ads": float(delta_zpe),
        "ZPE_ads_raw": float(delta_zpe_raw),
        "delta_U_vib_thermal": float(delta_u_vib_thermal),
        "delta_U_vib_total": float(delta_u_vib_total),
        "delta_F_vib": float(delta_f_vib),
        "delta_S_vib": float(delta_s_vib),
        "S_gas_standard": float(
            _ase_adsorbate_gas_components(
                temperature, reference_pressure, reference_pressure
            )["S"]
        ),
        "S_gas": float(gas_s),
        "H_gas_thermal": float(gas_h),
        "mu_gas_thermal": float(gas_mu),
        "H_ads": float(H_ads),
        "S_ads": float(S_ads),
        "G_ads": float(G_ads),
        "T_S_ads": float(temperature * S_ads),
        "zpe_surface": float(surface["zpe"]),
        "zpe_ads_system": float(ads["zpe"]),
        "U_surface_thermal": float(surface["u_thermal"]),
        "U_ads_system_thermal": float(ads["u_thermal"]),
        "U_surface_total": float(
            surface["u_thermal"] + (surface["zpe"] if include_zpe else 0.0)
        ),
        "U_ads_system_total": float(
            ads["u_thermal"] + (ads["zpe"] if include_zpe else 0.0)
        ),
        "S_surface_vib": float(surface["entropy"]),
        "S_ads_vib": float(ads["entropy"]),
        "F_surface": float(
            surface["free_energy"] - (surface["zpe"] if not include_zpe else 0.0)
        ),
        "F_ads_system": float(
            ads["free_energy"] - (ads["zpe"] if not include_zpe else 0.0)
        ),
        "pressure_bar": float(pressure),
        "reference_pressure_bar": float(reference_pressure),
        "include_zpe": bool(include_zpe),
        "ase_reference_pressure_pa": float(reference_pressure * BAR_TO_PA),
    }


def calculate_adsorption_thermodynamics(
    E_ads,
    frequencies_ads,
    frequencies_surface,
    temperature,
    pressure=1.0,
    include_zpe=True,
    reference_pressure=1.0,
    method="custom",
):
    """Calculate ΔH, ΔS and ΔG using the selected thermodynamics backend."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if pressure <= 0:
        raise ValueError("pressure must be > 0")
    if reference_pressure <= 0:
        raise ValueError("reference_pressure must be > 0")

    method = str(method).strip().lower()
    if method not in {"custom", "ase"}:
        raise ValueError("thermodynamics method must be either 'custom' or 'ase'")

    f_ads = _positive_frequencies(frequencies_ads)
    f_surface = _positive_frequencies(frequencies_surface)

    if method == "custom":
        return _calculate_adsorption_thermodynamics_custom(
            E_ads, f_ads, f_surface, temperature, pressure,
            include_zpe, reference_pressure
        )

    return _calculate_adsorption_thermodynamics_ase(
        E_ads, f_ads, f_surface, temperature, pressure,
        include_zpe, reference_pressure
    )


def adsorption_enthalpy(
    E_ads,
    frequencies_ads,
    frequencies_surface,
    temperature,
    pressure=1.0,
    include_zpe=True,
    reference_pressure=1.0,
    method="custom",
):
    """Return only ΔH_ads using the selected thermodynamics backend."""
    return float(
        calculate_adsorption_thermodynamics(
            E_ads,
            frequencies_ads,
            frequencies_surface,
            temperature,
            pressure=pressure,
            include_zpe=include_zpe,
            reference_pressure=reference_pressure,
            method=method,
        )["H_ads"]
    )


def vibrational_corrections(frequencies, temperature):
    """Backward-compatible free-energy helper; includes ZPE."""
    return vibrational_free_energy(frequencies, temperature, include_zpe=True)


def adsorbate_gas_correction(temperature, pressure=1.0, reference_pressure=1.0):
    """Backward-compatible pressure-dependent -T*S gas correction."""
    return -temperature * adsorbate_gas_entropy(
        temperature,
        pressure_bar=pressure,
        reference_pressure_bar=reference_pressure,
    )
