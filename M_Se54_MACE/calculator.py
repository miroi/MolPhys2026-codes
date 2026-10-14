#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Calculator construction and diagnostics for MACE + optional DFT-D4."""

from __future__ import annotations

ADSORBATE_SYMBOL = "Hg"  # Set from Config at workflow startup.

import gc
import platform
from typing import Dict, Tuple

import ase
import mace
import numpy as np
import torch
from ase.calculators.calculator import all_changes
from ase.calculators.mixing import SumCalculator
from mace.calculators import MACECalculator

try:
    from dftd4.ase import DFTD4
except ImportError:
    DFTD4 = None


class ScaledCalculator:
    """Thin ASE Calculator wrapper that scales energy/forces of a child."""

    def __init__(self, base, scale: float):
        from ase.calculators.calculator import Calculator

        class _Scaled(Calculator):
            implemented_properties = ["energy", "free_energy", "forces", "stress"]

            def __init__(self, wrapped, factor):
                super().__init__()
                self.wrapped = wrapped
                self.factor = float(factor)

            def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                # Force a fresh calculation on the wrapped calculator for the
                # exact Atoms object received here. This avoids stale child
                # results when SumCalculator is used repeatedly.
                self.wrapped.calculate(atoms, properties, all_changes)
                self.results["energy"] = self.factor * float(self.wrapped.results["energy"])
                self.results["free_energy"] = self.results["energy"]
                if "forces" in self.wrapped.results:
                    self.results["forces"] = self.factor * np.asarray(
                        self.wrapped.results["forces"], dtype=float
                    )
                if "stress" in self.wrapped.results:
                    self.results["stress"] = self.factor * np.asarray(
                        self.wrapped.results["stress"], dtype=float
                    )

            def get_base_calculator(self):
                return self.wrapped

        self.calculator = _Scaled(base, scale)

    def __getattr__(self, name):
        return getattr(self.calculator, name)


def clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


def print_version() -> None:
    print("\n" + "=" * 70)
    print("VERSION INFORMATION")
    print("=" * 70)
    print("Python :", platform.python_version())
    print("ASE    :", ase.__version__)
    print("MACE   :", mace.__version__)
    print("Torch  :", torch.__version__)
    if DFTD4 is not None:
        try:
            import dftd4
            version = getattr(dftd4, "__version__", None)
            if version is None:
                from dftd4.libdftd4 import get_api_version
                version = get_api_version()
            print("DFT-D4 :", version)
        except Exception:
            print("DFT-D4 : import available (version query failed)")
    else:
        print("DFT-D4 : not installed")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("=" * 70)


def _make_d4(config):
    if DFTD4 is None:
        raise ImportError(
            "DFT-D4 is enabled, but dftd4 is not installed. "
            "Install dftd4==4.2.0 from requirements.txt."
        )

    # Use the same custom damping parameters as the validated reference
    # QE + DFT-D4 calculation.  These are intentionally explicit rather than
    # silently falling back to the default PBE D4 parameterization.
    params_tweaks = {
        "s6": config.d4_s6,
        "s9": config.d4_s9,
        "alp": config.d4_alp,
        "s8": config.d4_s8,
        "a1": config.d4_a1,
        "a2": config.d4_a2,
    }
    # DFT-D4 4.2.x treats method-based and explicit damping parameters as
    # mutually exclusive.  Passing method="PBE" together with s6/s8/...
    # makes DampingParam.load_param() receive the explicit keywords and fail.
    # The complete custom set below is therefore used without a method name.
    d4 = DFTD4(params_tweaks=params_tweaks)
    if config.d4_scale == 1.0:
        return d4
    return ScaledCalculator(d4, config.d4_scale).calculator


def get_calculator(config):
    """Build the production MACE (+ optional DFT-D4) calculator.

    When D4 is enabled, the returned SumCalculator evaluates

        E = E_MACE + scale * E_D4
        F = F_MACE + scale * F_D4

    for every geometry, including optimizations and finite-difference phonons.
    """
    clear_cuda_cache()
    requested_device = str(getattr(config, "device", "auto")).strip().lower()
    if requested_device not in {"cpu", "cuda", "auto"}:
        raise ValueError(
            f"Unsupported runtime.device={requested_device!r}; choose cpu, cuda, or auto"
        )
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "runtime.device is set to 'cuda', but PyTorch reports that CUDA is unavailable. "
            "Set device = cpu or device = auto in adsorption_config.ini."
        )
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if requested_device == "auto" else requested_device

    print(f"Requested MACE device: {requested_device}")
    print(f"Using device: {device}")
    print("MACE model:", config.model_path)

    mace_calc = MACECalculator(
        model_paths=[config.model_path],
        device=device,
        default_dtype="float64",
    )

    if not config.d4_enabled:
        print("DFT-D4 correction: DISABLED")
        return mace_calc

    d4_calc = _make_d4(config)
    print("DFT-D4 correction: ENABLED")
    print("DFT-D4 parameter set:", config.d4_method)
    print("DFT-D4 custom parameters:", {
        "s6": config.d4_s6, "s9": config.d4_s9, "alp": config.d4_alp,
        "s8": config.d4_s8, "a1": config.d4_a1, "a2": config.d4_a2,
    })
    print("DFT-D4 scale:", config.d4_scale)

    # DFT-D4 documents DFTD4.add_calculator() as the convenience interface;
    # it returns a SumCalculator with D4 first. We construct the same official
    # combination explicitly so the child calculators remain inspectable.
    calc = SumCalculator([d4_calc, mace_calc])
    calc._adsorbate_component_names = {id(d4_calc): "DFT-D4", id(mace_calc): "MACE"}
    return calc


def _child_label(child) -> str:
    cls = child.__class__.__name__
    if cls == "DFTD4" or "DFTD4" in cls:
        return "DFT-D4"
    if cls == "_Scaled" or hasattr(child, "get_base_calculator"):
        return "DFT-D4"
    if "MACE" in cls:
        return "MACE"
    return cls


def _component_calculators(calc):
    """Return a stable {label: child-calculator} mapping.

    ASE 3.29 stores SumCalculator children in ``calc.mixer.calcs``.
    Older ASE releases exposed them directly as ``calc.calcs``.  Support
    both layouts so component decomposition never mistakes the whole
    SumCalculator for a single MACE component.
    """
    if hasattr(calc, "mixer") and hasattr(calc.mixer, "calcs"):
        children = calc.mixer.calcs
    elif hasattr(calc, "calcs"):
        children = calc.calcs
    else:
        return {"MACE": calc}
    out = {}
    for child in children:
        label = _child_label(child)
        if label in out:
            raise RuntimeError(f"Duplicate calculator component label: {label}")
        out[label] = child
    return out


def get_calculator_component_energies(atoms, calc) -> Dict[str, float]:
    """Evaluate and return fresh individual component energies.

    The child calculators are explicitly recalculated with ``all_changes``.
    This is intentional: component reporting must not depend on SumCalculator
    or child-calculator cache state.
    """
    components = _component_calculators(calc)
    energies = {}
    for label, child in components.items():
        child.calculate(atoms, properties=["energy"], system_changes=all_changes)
        value = float(child.results["energy"])
        if not np.isfinite(value):
            raise RuntimeError(f"Non-finite {label} energy: {value}")
        energies[label] = value
    return energies


def get_calculator_component_forces(atoms, calc) -> Dict[str, np.ndarray]:
    """Evaluate and return fresh individual component forces."""
    components = _component_calculators(calc)
    forces = {}
    for label, child in components.items():
        child.calculate(atoms, properties=["forces"], system_changes=all_changes)
        value = np.asarray(child.results["forces"], dtype=float).copy()
        if not np.all(np.isfinite(value)):
            raise RuntimeError(f"Non-finite {label} forces")
        forces[label] = value
    return forces


def validate_calculator_decomposition(atoms, calc, label="system", atol_energy=1e-10, atol_force=1e-8):
    """Verify total MACE+D4 energy/forces equal the child sum.

    Returns a diagnostics dictionary. Raises on inconsistency.
    """
    components = _component_calculators(calc)
    child_energy = get_calculator_component_energies(atoms, calc)
    child_forces = get_calculator_component_forces(atoms, calc)

    total_energy = float(atoms.get_potential_energy())
    total_forces = np.asarray(atoms.get_forces(), dtype=float)
    expected_energy = float(sum(child_energy.values()))
    expected_forces = np.sum(list(child_forces.values()), axis=0)

    energy_error = total_energy - expected_energy
    force_error = float(np.max(np.abs(total_forces - expected_forces)))

    if abs(energy_error) > atol_energy:
        raise RuntimeError(
            f"Calculator energy decomposition failed for {label}: "
            f"total={total_energy:.16e}, sum={expected_energy:.16e}, "
            f"difference={energy_error:.3e} eV"
        )
    if force_error > atol_force:
        raise RuntimeError(
            f"Calculator force decomposition failed for {label}: "
            f"max difference={force_error:.3e} eV/Å"
        )

    return {
        "energy_total_eV": total_energy,
        "energy_components_eV": child_energy,
        "energy_reconstruction_error_eV": energy_error,
        "forces_total_max_abs_eVA": float(np.max(np.abs(total_forces))),
        "force_reconstruction_error_eVA": force_error,
    }


def validate_d4_adsorption_component(
    adsorbate_slab,
    slab,
    adsorbate,
    calc,
    tolerance_eV=1e-12,
    fail_if_zero=True,
    adsorbate_symbol=None,
    substrate_symbol="substrate",
):
    """Diagnose the D4 contribution to the adsorbate adsorption reaction.

    The reaction contribution

        ΔE_D4 = E_D4(adsorbate/slab) - E_D4(slab) - E_D4(adsorbate)

    is a physical result. It is therefore not, by itself, a valid test of
    whether the D4 calculator is active. ``fail_if_zero`` is retained as an
    optional strict diagnostic for callers that explicitly want that policy.
    """
    labels = _component_calculators(calc)
    if "DFT-D4" not in labels:
        return {"enabled": False, "E_ads_DFTD4_eV": 0.0}

    e_adsorbates = get_calculator_component_energies(adsorbate_slab, calc)["DFT-D4"]
    e_slab = get_calculator_component_energies(slab, calc)["DFT-D4"]
    e_adsorbate = get_calculator_component_energies(adsorbate, calc)["DFT-D4"]
    e_ads = e_adsorbates - e_slab - e_adsorbate

    if not np.isfinite(e_ads):
        raise RuntimeError("DFT-D4 adsorption contribution is non-finite")

    atom_label = adsorbate_symbol or "adsorbate"
    print("\nDFT-D4 diagnostic")
    print(f"  E_D4({atom_label}/{substrate_symbol} slab) : {e_adsorbates:.12f} eV")
    print(f"  E_D4({substrate_symbol} slab)       : {e_slab:.12f} eV")
    print(f"  E_D4({atom_label})                  : {e_adsorbate:.12f} eV")
    print(f"  ΔE_D4         : {e_ads:+.12f} eV")

    if abs(e_ads) <= tolerance_eV:
        message = (
            "DFT-D4 adsorbate adsorption contribution is numerically zero within "
            f"{tolerance_eV:.1e} eV."
        )
        if fail_if_zero:
            raise RuntimeError(message)
        print(f"  WARNING: {message}")

    return {
        "enabled": True,
        "E_DFTD4_AdsorbateSlab_eV": e_adsorbates,
        "E_DFTD4_Slab_eV": e_slab,
        "E_DFTD4_adsorbate_eV": e_adsorbate,
        "E_ads_DFTD4_eV": e_ads,
    }


def evaluate_adsorbate_substrate_d4_pair(adsorbate_slab, calc, substrate_symbol="Se",
                            method="PBE", scale=1.0, use_mic=True,
                            params_tweaks=None):
    """Evaluate adsorbate--substrate DFT-D4 pair terms from the full optimized cell.

    The diagnostic uses DFT-D4's pair-resolved Python API on the complete
    periodic adsorbate/slab structure, rather than constructing an isolated two-atom
    ASE child calculation.  This preserves the actual periodic geometry and
    D4 environment used for the adsorption calculation.

    Returns the additive pairwise adsorbate--nearest-substrate term, the sum of all
    additive adsorbate--substrate terms, and the adsorbate-involving non-additive terms.
    These are diagnostics and are not substituted for the reaction-level
    adsorption contribution E_D4(adsorbate/slab)-E_D4(slab)-E_D4(adsorbate).
    """
    components = _component_calculators(calc)
    d4 = components.get("DFT-D4")
    if d4 is None:
        return {"enabled": False, "adsorbateX_D4_pair_interaction_eV": 0.0}

    try:
        from ase.units import Bohr, Hartree
        from dftd4.interface import DampingParam, DispersionModel
    except ImportError as exc:
        raise RuntimeError(
            "Pair-resolved DFT-D4 diagnostics require the installed dftd4 "
            "Python interface (dftd4.interface)."
        ) from exc

    symbols = np.asarray(adsorbate_slab.get_chemical_symbols())
    adsorbate_indices = np.where(symbols == ADSORBATE_SYMBOL)[0]
    x_indices = np.where(symbols == substrate_symbol)[0]
    if len(adsorbate_indices) != 1:
        raise ValueError(f"Expected exactly one adsorbate atom, found {len(adsorbate_indices)}")
    if len(x_indices) == 0:
        raise ValueError(f"No {substrate_symbol} atoms found in adsorbate/slab structure")

    adsorbate_index = int(adsorbate_indices[0])
    distances = adsorbate_slab.get_distances(adsorbate_index, x_indices.tolist(), mic=use_mic)
    nearest_local = int(np.argmin(distances))
    x_index = int(x_indices[nearest_local])
    distance = float(distances[nearest_local])

    numbers = np.asarray(adsorbate_slab.get_atomic_numbers(), dtype=np.int32)
    positions_bohr = np.asarray(adsorbate_slab.get_positions(), dtype=float) / Bohr
    pbc = np.asarray(adsorbate_slab.get_pbc(), dtype=bool)
    lattice = (np.asarray(adsorbate_slab.get_cell().array, dtype=float) / Bohr
               if np.any(pbc) else None)
    periodic = pbc if np.any(pbc) else None

    model = DispersionModel(
        numbers=numbers,
        positions=positions_bohr,
        lattice=lattice,
        periodic=periodic,
        model="d4",
    )
    damping_kwargs = dict(params_tweaks or {})
    # Explicit damping parameters must be constructed without method=...;
    # DFT-D4 4.2.x does not allow mixing a named parameter set with explicit
    # s6/s8/s9/a1/a2/alp values.
    if damping_kwargs:
        param = DampingParam(**damping_kwargs)
    else:
        param = DampingParam(method=str(method).lower())
    pairwise = model.get_pairwise_dispersion(param)
    pair2 = np.asarray(pairwise["additive pairwise energy"], dtype=float)
    pair3 = np.asarray(pairwise["non-additive pairwise energy"], dtype=float)

    pair_hartree = float(pair2[adsorbate_index, x_index])
    all_adsorbate_x_hartree = float(np.sum(pair2[adsorbate_index, x_indices]))
    adsorbate_nonadd_hartree = float(np.sum(pair3[adsorbate_index, :]))

    pair_eV = pair_hartree * Hartree * float(scale)
    all_adsorbate_x_eV = all_adsorbate_x_hartree * Hartree * float(scale)
    adsorbate_nonadd_eV = adsorbate_nonadd_hartree * Hartree * float(scale)

    print("\nExplicit adsorbate--substrate DFT-D4 pair-resolved diagnostic")
    print(f"  Evaluation system         : optimized periodic {ADSORBATE_SYMBOL}/{substrate_symbol} slab")
    print(f"  D4 parameter set          : {method}")
    if damping_kwargs:
        print(f"  D4 custom parameters      : {damping_kwargs}")
    print(f"  D4 scale                  : {float(scale):.8f}")
    print(f"  Adsorbate atom index             : {adsorbate_index}")
    print(f"  {substrate_symbol} atom index             : {x_index}")
    print(f"  actual optimized distance : {distance:.12f} Å")
    print(f"  additive adsorbate--{substrate_symbol} pair : {pair_eV:+.15e} eV")
    print(f"  all adsorbate--{substrate_symbol} pair terms : {all_adsorbate_x_eV:+.15e} eV")
    print(f"  Adsorbate-involving non-additive D4 : {adsorbate_nonadd_eV:+.15e} eV")

    return {
        "enabled": True,
        "adsorbate_index": adsorbate_index,
        "substrate_index": x_index,
        "distance_A": distance,
        "adsorbate_substrate_D4_pair_interaction_eV": pair_eV,
        "adsorbateX_D4_pair_interaction_eV": pair_eV,
        "adsorbate_substrate_D4_all_pair_terms_eV": all_adsorbate_x_eV,
        "adsorbate_D4_nonadditive_involving_eV": adsorbate_nonadd_eV,
        "d4_method": str(method),
        "d4_scale": float(scale),
        "pairwise_matrix_additive_Hartree": pair2,
        "pairwise_matrix_nonadditive_Hartree": pair3,
    }

def validate_d4_calculator_active(calc, tolerance_eV=1.0e-12):
    """Verify that the DFT-D4 child itself produces a nonzero interaction.

    This test is deliberately independent of the adsorbate/Se adsorption geometry.
    A zero *adsorption contribution* is not evidence that D4 is inactive:
    in principle the reaction combination E(adsorbate/Se) - E(Se) - E(adsorbate) can be
    numerically zero even when the D4 calculator is working correctly.
    """
    from ase import Atoms

    components = _component_calculators(calc)
    d4 = components.get("DFT-D4")
    if d4 is None:
        return {"enabled": False, "test_energy_eV": 0.0}

    pair = Atoms(
        "ArAr",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]],
        pbc=False,
    )
    d4.calculate(pair, properties=["energy"], system_changes=all_changes)
    energy = float(d4.results["energy"])
    if not np.isfinite(energy):
        raise RuntimeError(f"DFT-D4 active-check returned non-finite energy: {energy}")
    if abs(energy) <= tolerance_eV:
        raise RuntimeError(
            "DFT-D4 is enabled but the independent D4 activity check returned "
            f"{energy:.6e} eV for Ar2 at 4.0 Å. "
            "The D4 child appears inactive or incompatible with the installed API."
        )

    print("\nDFT-D4 activity check")
    print(f"  D4(Ar2, 4.0 Å) : {energy:+.12f} eV")
    print("  DFT-D4 child calculator is active.")
    return {"enabled": True, "test_energy_eV": energy}


def safe_energy(atoms):
    clear_cuda_cache()
    energy = float(atoms.get_potential_energy())
    clear_cuda_cache()
    return energy


def safe_forces(atoms):
    clear_cuda_cache()
    forces = atoms.get_forces()
    clear_cuda_cache()
    return forces
