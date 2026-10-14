#!/usr/bin/env python3
"""
Visualize adsorbate-dominated Gamma-point vibrations from an existing ASE Vibrations cache.

This script does NOT run new force calculations.

It:
1. Reads the optimized full adsorbate/slab structure from an ASE trajectory.
2. Derives the mobile-atom coordinate space directly from the existing ASE cache.
3. Reopens the existing ASE Vibrations cache from --phonon-dir.
4. Reproduces the production dynamical-matrix construction and mode analysis.
5. Identifies adsorbate-dominated positive-frequency modes.
6. Converts mass-weighted eigenvectors to Cartesian displacement vectors.
7. Embeds the displacements into the full structure, keeping frozen atoms fixed.
8. Writes ASE .traj and XYZ animation files suitable for `ase gui`.

The production phonons.py uses:
    Vibrations(full_atoms, indices=mobile_indices, delta=0.005, nfree=2,
               name=prefix)
followed by:
    vib.run()
    vib.read(method="standard", direction="central")
The cache is indexed by the original/full atom indices.

The same force-constant -> mass-weighting -> eigensolver procedure is
reproduced here so that the visualization corresponds to the production
phonon frequencies and eigenvectors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.vibrations import Vibrations


# Conversion used by ASE for sqrt(eV / Angstrom^2 / amu) -> cm^-1.
ASE_TO_CM = 521.47083


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize adsorbate-dominated Gamma-point modes from an ASE Vibrations cache."
    )

    parser.add_argument(
        "--structure",
        default="Pb_Se_opt.traj",
        help="Optimized full adsorbate/slab structure trajectory.",
    )
    parser.add_argument(
        "--phonon-dir",
        default="adsorption_se54_dip_Pb_slab_phonons",
        help="ASE Vibrations cache directory/name.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for generated trajectories and diagnostics (default: <addatom>_vibrations).",
    )
    parser.add_argument(
        "--addatom-symbol",
        default=None,
        help="Adsorbed element symbol; inferred if exactly one non-substrate element is present.",
    )
    parser.add_argument(
        "--substrate-symbol",
        default="Se",
        help="Substrate element excluded during adsorbate inference (default: Se).",
    )
    parser.add_argument(
        "--frozen",
        type=int,
        default=None,
        help=(
            "Number of frozen atoms at the beginning of the full structure. "
            "By default this is derived from the ASE cache filenames."
        ),
    )
    parser.add_argument(
        "--surface-layer-tolerance",
        type=float,
        default=0.50,
        help=(
            "Se atoms within this distance (Angstrom) of the highest Se projection "
            "along the surface normal define the top surface layer (default: 0.50)."
        ),
    )
    parser.add_argument(
        "--perpendicular-fraction", type=float, default=0.50,
        help=(
            "Fallback minimum fraction of the Add-atom's own Cartesian displacement "
            "along the surface normal (default: 0.50). Used when no mode passes --threshold."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Minimum squared mass-weighted Add-atom projection for primary mode selection (default: 0.20).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=1.0,
        help="Positive-frequency cutoff in cm^-1 (default: 1.0).",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.35,
        help="Maximum Cartesian displacement in Angstrom for visualization (default: 0.35).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=21,
        help="Number of frames per animation (default: 21).",
    )
    parser.add_argument(
        "--phase-cycles",
        type=float,
        default=1.0,
        help="Number of complete oscillation cycles in each trajectory (default: 1.0).",
    )
    parser.add_argument(
        "--n-modes",
        type=int,
        default=3,
        help="Number of adsorbate-dominated modes to write automatically (default: 3).",
    )
    parser.add_argument(
        "--all-addatom",
        action="store_true",
        help="Write all stable modes passing the primary projection threshold; if none pass, use perpendicular-motion fallback ranking.",
    )
    parser.add_argument(
        "--mode",
        type=int,
        action="append",
        default=None,
        help=(
            "Explicit zero-based mode index to visualize. Repeat the option for "
            "multiple modes, e.g. --mode 0 --mode 1."
        ),
    )

    return parser.parse_args()


def load_structure(path: str) -> Atoms:
    structure = read(path, index=-1)

    if not isinstance(structure, Atoms):
        raise TypeError(f"Could not read an ASE Atoms object from {path!r}.")

    if len(structure) == 0:
        raise ValueError(f"Structure {path!r} contains no atoms.")

    return structure


def discover_cache_indices(phonon_dir: str) -> tuple[np.ndarray, set[int]]:
    """Derive the vibrational mobile indices directly from an ASE cache.

    ASE Vibrations caches displacement keys using the *original/full-system*
    atom indices, e.g. ``64x+``, ``64x-``, ..., ``96z-``.  This function
    reconstructs those original indices instead of assuming that the first
    mobile atom is index zero.

    Every mobile atom must have all six central-difference displacement files
    (x+/x-/y+/y-/z+/z-).  ``cache.eq.json`` is the equilibrium entry and is
    not itself used to infer the mobile atom list.
    """
    cache_dir = Path(phonon_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(
            f"Phonon cache directory not found: {cache_dir}"
        )

    import re

    pattern = re.compile(r"^cache\.(\d+)([xyz])([+-])\.json$")
    required = {f"{axis}{sign}" for axis in "xyz" for sign in "+-"}
    found: dict[int, set[str]] = {}

    for path in cache_dir.iterdir():
        match = pattern.match(path.name)
        if match is None:
            continue
        atom_index = int(match.group(1))
        component = match.group(2) + match.group(3)
        found.setdefault(atom_index, set()).add(component)

    if not found:
        raise RuntimeError(
            f"No ASE Vibrations displacement files matching "
            f"cache.<atom><x|y|z><+|->.json were found in {cache_dir}."
        )

    incomplete = {
        index: sorted(required - components)
        for index, components in found.items()
        if components != required
    }
    if incomplete:
        details = "; ".join(
            f"{index}: missing {','.join(missing)}"
            for index, missing in sorted(incomplete.items())
        )
        raise RuntimeError(
            "The ASE cache is incomplete; cannot safely infer the mobile "
            f"atom set. {details}"
        )

    mobile_indices = np.array(sorted(found), dtype=int)
    return mobile_indices, set(int(i) for i in mobile_indices)


def make_mobile_atoms(
    full: Atoms, mobile_indices: np.ndarray
) -> tuple[Atoms, np.ndarray, np.ndarray]:
    """Construct the mobile coordinate system from original atom indices."""
    mobile_indices = np.asarray(mobile_indices, dtype=int)
    if mobile_indices.ndim != 1 or mobile_indices.size == 0:
        raise ValueError("The cache-derived mobile index list is empty or invalid.")
    if np.any(mobile_indices < 0) or np.any(mobile_indices >= len(full)):
        raise ValueError(
            "The phonon cache contains atom indices outside the supplied "
            f"structure (0..{len(full)-1}): {mobile_indices.tolist()}"
        )

    mobile = full[mobile_indices].copy()
    mobile.calc = SinglePointCalculator(
        mobile,
        energy=0.0,
        forces=np.zeros((len(mobile), 3), dtype=float),
    )

    frozen_mask = np.ones(len(full), dtype=bool)
    frozen_mask[mobile_indices] = False
    frozen_indices = np.where(frozen_mask)[0]
    return mobile, mobile_indices, frozen_indices


def load_phonon_cache(
    full: Atoms,
    mobile_indices: np.ndarray,
    phonon_dir: str,
    displacement: float = 0.005,
    nfree: int = 2,
) -> Vibrations:
    """Reopen the production ASE Vibrations cache without recalculating forces."""
    vib = Vibrations(
        full,
        indices=mobile_indices.tolist(),
        delta=float(displacement),
        nfree=int(nfree),
        name=phonon_dir,
    )
    vib.read(method="standard", direction="central")
    return vib

def gamma_dynamical_matrix(vib: Vibrations) -> np.ndarray:
    """Reproduce the production mass-weighted Gamma dynamical matrix exactly."""
    H = np.asarray(vib.H, dtype=float)
    n = len(vib.indices)
    expected = 3 * n
    if H.shape != (expected, expected):
        raise ValueError(
            f"Vibrational Hessian has shape {H.shape}, "
            f"expected {(expected, expected)}"
        )

    masses = np.asarray(vib.atoms.get_masses(), dtype=float)[vib.indices]
    if np.any(masses <= 0.0):
        raise ValueError("All vibrational atoms must have positive masses.")

    m_inv_sqrt = np.repeat(masses ** -0.5, 3)
    D = H * np.outer(m_inv_sqrt, m_inv_sqrt)
    return 0.5 * (D + D.T)

def gamma_modes(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize the mass-weighted Gamma dynamical matrix.

    This reproduces the production phonons.py convention:
        sign(lambda) * sqrt(abs(lambda)) * ASE_TO_CM
    for signed frequencies in cm^-1.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(D)
    signed_cm = (
        np.sign(eigenvalues)
        * np.sqrt(np.abs(eigenvalues))
        * ASE_TO_CM
    )
    return signed_cm, eigenvectors


def adsorbate_projections(
    mobile: Atoms,
    eigenvectors: np.ndarray,
    addatom_symbol: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return adsorbate atom indices in the mobile system and adsorbate projection per mode.

    Eigenvectors here are mass-weighted normal-mode eigenvectors. The projection
    is the squared eigenvector amplitude on the adsorbate Cartesian degrees of freedom,
    matching the diagnostic used by production phonons.py.
    """
    addatom_indices = np.array(
        [i for i, atom in enumerate(mobile) if atom.symbol == addatom_symbol],
        dtype=int,
    )

    if len(addatom_indices) == 0:
        raise ValueError(f"No {addatom_symbol} atom was found in the mobile phonon system.")

    addatom_dofs = np.concatenate(
        [3 * addatom_indices + axis for axis in range(3)]
    )

    # eigenvectors has shape (3N, 3N), columns are normal modes.
    projection = np.sum(eigenvectors[addatom_dofs, :] ** 2, axis=0)

    return addatom_indices, projection


def surface_geometry(
    full: Atoms,
    addatom_symbol: str,
    substrate_symbol: str,
    layer_tolerance: float = 0.50,
) -> dict:
    """Measure Add-atom height above the top Se layer and nearest periodic Se distance.

    Surface height is a signed *normal height*: the Add-atom coordinate projected
    onto the slab normal minus the mean projected coordinate of Se atoms in the
    topmost layer. The top layer comprises Se atoms no more than `layer_tolerance`
    Angstrom below the maximum Se projection. This differs from a nearest-neighbour
    distance, which is computed separately with periodic minimum-image distances.
    The first two cell vectors define the surface plane; their cross product defines
    its normal, oriented toward the positive third-cell-vector direction where possible.
    """
    if layer_tolerance < 0:
        raise ValueError("--surface-layer-tolerance must be non-negative.")
    add_indices = np.array([i for i, a in enumerate(full) if a.symbol == addatom_symbol], dtype=int)
    se_indices = np.array([i for i, a in enumerate(full) if a.symbol == substrate_symbol], dtype=int)
    if len(add_indices) == 0:
        raise ValueError(f"No Add-atom ({addatom_symbol}) found in structure.")
    if len(se_indices) == 0:
        raise ValueError(f"No substrate atoms ({substrate_symbol}) found in structure.")

    cell = np.asarray(full.cell, dtype=float)
    normal = np.cross(cell[0], cell[1])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("Cannot define surface normal: first two cell vectors are collinear/zero.")
    normal /= norm
    if np.dot(normal, cell[2]) < 0:
        normal *= -1.0

    projections = np.asarray(full.positions) @ normal
    se_proj = projections[se_indices]
    top_proj = float(np.max(se_proj))
    top_layer_mask = (top_proj - se_proj) <= layer_tolerance
    top_layer_indices = se_indices[top_layer_mask]
    surface_reference = float(np.mean(projections[top_layer_indices]))
    add_heights = projections[add_indices] - surface_reference

    # Explicit pair list plus ASE minimum-image distances respects periodicity.
    distances = full.get_distances(int(add_indices[0]), se_indices, mic=True)
    min_pair_local = int(np.argmin(distances))
    min_distance = float(distances[min_pair_local])
    closest_se_index = int(se_indices[min_pair_local])

    return {
        "surface_distance_definition": (
            "signed height along the normal to cell[0] x cell[1], measured from the "
            "mean normal coordinate of top-layer substrate atoms (within the configured "
            "tolerance of the highest substrate projection) to the Add-atom; not a bond length"
        ),
        "surface_normal_unit_vector": normal.tolist(),
        "surface_layer_tolerance_A": float(layer_tolerance),
        "surface_layer_atom_indices": top_layer_indices.tolist(),
        "surface_reference_mean_projection_A": surface_reference,
        "addatom_indices_full_structure": add_indices.tolist(),
        "addatom_surface_heights_A": add_heights.tolist(),
        "addatom_surface_distance_A": float(np.mean(add_heights)),
        "minimum_periodic_addatom_substrate_distance_A": min_distance,
        "minimum_distance_addatom_index": int(add_indices[0]),
        "minimum_distance_substrate_index": closest_se_index,
    }


def addatom_mode_metrics(
    mobile: Atoms,
    eigenvectors: np.ndarray,
    addatom_indices: np.ndarray,
    surface_normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return Add-atom Cartesian displacement share, normal fraction, and normal weight.

    For each mode, convert the mass-weighted eigenvector to Cartesian displacements.
    `normal_fraction` is |u_add . n|^2 / |u_add|^2, i.e. the perpendicular
    fraction of the Add-atom's *own* displacement. `normal_weight` multiplies this
    by the existing mass-weighted total Add-atom projection, preventing a nearly
    stationary Add-atom with a numerically noisy direction from dominating fallback.
    """
    masses = np.asarray(mobile.get_masses(), dtype=float)
    n_modes = eigenvectors.shape[1]
    cart = eigenvectors.reshape(len(mobile), 3, n_modes) / np.sqrt(masses)[:, None, None]
    add_cart = cart[addatom_indices, :, :]
    # For a single Add atom this is its Cartesian vector; for multiple atoms sum norms.
    total_sq = np.sum(add_cart**2, axis=(0, 1))
    normal_components = np.einsum("ajm,j->am", add_cart, surface_normal)
    normal_sq = np.sum(normal_components**2, axis=0)
    normal_fraction = np.divide(normal_sq, total_sq, out=np.zeros_like(normal_sq), where=total_sq > 1e-30)
    # Eigenvectors are normalized globally; this share is dimensionless and comparable across modes.
    all_cart_sq = np.sum(cart**2, axis=(0, 1))
    addatom_cartesian_share = np.divide(total_sq, all_cart_sq, out=np.zeros_like(total_sq), where=all_cart_sq > 1e-30)
    normal_weight = addatom_cartesian_share * normal_fraction
    return addatom_cartesian_share, normal_fraction, normal_weight


def cartesian_mode_vector(
    mobile: Atoms,
    mass_weighted_vector: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """
    Convert a mass-weighted eigenvector to a Cartesian displacement vector.

    The resulting vector is normalized so that its largest per-atom displacement
    has magnitude `amplitude` Angstrom.
    """
    n = len(mobile)

    masses = np.asarray(mobile.get_masses(), dtype=float)
    sqrt_masses = np.sqrt(masses)

    q = mass_weighted_vector.reshape(n, 3)

    # q is mass-weighted. Convert to Cartesian displacement.
    displacement = q / sqrt_masses[:, None]

    per_atom = np.linalg.norm(displacement, axis=1)
    max_atom = float(np.max(per_atom))

    if not np.isfinite(max_atom) or max_atom <= 0.0:
        raise ValueError("Mode has zero or invalid Cartesian displacement.")

    displacement *= amplitude / max_atom

    return displacement


def build_animation(
    full: Atoms,
    mobile_indices: np.ndarray,
    displacement: np.ndarray,
    frames: int,
    phase_cycles: float,
) -> list[Atoms]:
    """
    Embed a sinusoidal Cartesian mode into the full structure.

    Frozen atoms are left exactly at their equilibrium coordinates.
    """
    if frames < 2:
        raise ValueError("--frames must be at least 2.")

    if phase_cycles <= 0.0:
        raise ValueError("--phase-cycles must be positive.")

    result: list[Atoms] = []

    phases = np.linspace(
        0.0,
        2.0 * np.pi * phase_cycles,
        frames,
        endpoint=True,
    )

    for phase in phases:
        frame = full.copy()
        frame.positions[mobile_indices] += displacement * np.sin(phase)
        result.append(frame)

    return result


def safe_frequency_tag(frequency_cm: float) -> str:
    """
    Create a filename-safe frequency string with three decimal places.
    """
    return f"{frequency_cm:07.3f}"


def write_mode(
    full: Atoms,
    mobile_indices: np.ndarray,
    displacement: np.ndarray,
    mode_index: int,
    frequency_cm: float,
    projection: float,
    output_dir: Path,
    frames: int,
    phase_cycles: float,
    addatom_symbol: str,
) -> dict:
    animation = build_animation(
        full=full,
        mobile_indices=mobile_indices,
        displacement=displacement,
        frames=frames,
        phase_cycles=phase_cycles,
    )

    freq_tag = safe_frequency_tag(frequency_cm)

    traj_path = output_dir / f"{addatom_symbol}_mode_{freq_tag}_cm-1.traj"
    xyz_path = output_dir / f"{addatom_symbol}_mode_{freq_tag}_cm-1.xyz"

    write(traj_path, animation, format="traj")
    write(xyz_path, animation, format="xyz")

    return {
        "mode_index": int(mode_index),
        "frequency_cm-1": float(frequency_cm),
        "addatom_projection": float(projection),
        "trajectory": str(traj_path),
        "xyz": str(xyz_path),
    }


def main() -> None:
    args = parse_args()

    if args.amplitude <= 0.0:
        raise ValueError("--amplitude must be positive.")

    if args.frames < 2:
        raise ValueError("--frames must be at least 2.")

    full = load_structure(args.structure)
    if args.addatom_symbol:
        addatom_symbol = args.addatom_symbol
    else:
        candidates = sorted({a.symbol for a in full if a.symbol != args.substrate_symbol})
        if len(candidates) != 1:
            raise ValueError(
                f"Cannot infer unique adsorbate element from structure; elements other than "
                f"{args.substrate_symbol}: {candidates}. Use --addatom-symbol."
            )
        addatom_symbol = candidates[0]
    if addatom_symbol not in full.get_chemical_symbols():
        raise ValueError(f"Adsorbate element {addatom_symbol!r} is absent from {args.structure!r}.")
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"{addatom_symbol}_vibrations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # The production cache is authoritative for the vibrational coordinate
    # space.  In particular, do not infer the mobile set from a hard-coded
    # frozen-atom count: production ASE Vibrations stores keys using the
    # original/full atom indices.
    cache_mobile_indices, _ = discover_cache_indices(args.phonon_dir)

    if args.frozen is not None:
        # Retain the CLI option as an explicit validation aid, but never use
        # it to construct the coordinate space when the cache is available.
        expected_mobile = np.arange(args.frozen, len(full), dtype=int)
        if not np.array_equal(expected_mobile, cache_mobile_indices):
            raise ValueError(
                f"--frozen={args.frozen} does not match the mobile indices "
                "encoded by the production cache. "
                f"Cache mobile indices: {cache_mobile_indices.tolist()}"
            )

    mobile, mobile_indices, frozen_indices = make_mobile_atoms(
        full,
        mobile_indices=cache_mobile_indices,
    )

    phon = load_phonon_cache(
        full,
        mobile_indices,
        args.phonon_dir,
        displacement=0.005,
        nfree=2,
    )

    # Verify ASE accepted exactly the cache-derived original indices.
    vib_indices = np.asarray(phon.indices, dtype=int)
    if not np.array_equal(vib_indices, mobile_indices):
        raise RuntimeError(
            "ASE Vibrations returned indices different from those inferred "
            "from the production cache: "
            f"{vib_indices.tolist()} vs {mobile_indices.tolist()}"
        )

    D = gamma_dynamical_matrix(phon)
    frequencies_cm, eigenvectors = gamma_modes(D)
    addatom_mobile_indices, addatom_projection = adsorbate_projections(
        mobile, eigenvectors, addatom_symbol
    )
    geometry = surface_geometry(
        full, addatom_symbol, args.substrate_symbol, args.surface_layer_tolerance
    )
    surface_normal = np.asarray(geometry["surface_normal_unit_vector"], dtype=float)
    addatom_cartesian_share, addatom_perpendicular_fraction, addatom_perpendicular_weight = (
        addatom_mode_metrics(mobile, eigenvectors, addatom_mobile_indices, surface_normal)
    )

    positive = frequencies_cm >= args.cutoff
    selected_indices = np.where(positive)[0]

    if len(selected_indices) == 0:
        raise RuntimeError(
            f"No positive phonon modes remain at the cutoff "
            f"{args.cutoff:.3f} cm^-1."
        )

    # Significant imaginary modes are diagnostic errors for this stable
    # visualization workflow. Tiny negative/near-zero numerical modes are
    # allowed below the positive cutoff.
    significant_imaginary = frequencies_cm < -args.cutoff

    if np.any(significant_imaginary):
        bad = np.where(significant_imaginary)[0]
        preview = ", ".join(f"{frequencies_cm[i]:.3f}" for i in bad[:10])
        raise RuntimeError(
            "Significant imaginary modes were found: "
            f"{preview} cm^-1. Check the optimized structure and phonon cache."
        )

    selection_method = "explicit_mode_indices"
    if args.mode is not None:
        mode_indices = list(dict.fromkeys(args.mode))

        for mode_index in mode_indices:
            if mode_index < 0 or mode_index >= len(frequencies_cm):
                raise ValueError(
                    f"Mode index {mode_index} is outside the valid range "
                    f"0..{len(frequencies_cm)-1}."
                )
            if frequencies_cm[mode_index] < args.cutoff:
                raise ValueError(
                    f"Mode {mode_index} has frequency "
                    f"{frequencies_cm[mode_index]:.6f} cm^-1, below the "
                    f"{args.cutoff:.3f} cm^-1 cutoff."
                )

    else:
        addatom_candidates = [
            int(i) for i in selected_indices if addatom_projection[i] >= args.threshold
        ]
        selection_method = "total_addatom_projection_threshold"
        if not addatom_candidates:
            # Fallback: rank modes by the amount of Add-atom motion perpendicular
            # to the surface, weighted by the Add-atom's total participation. This
            # intentionally does not require total projection >= --threshold.
            addatom_candidates = [
                int(i) for i in selected_indices
                if addatom_perpendicular_fraction[i] >= args.perpendicular_fraction
                and addatom_perpendicular_weight[i] > 1e-12
            ]
            selection_method = "fallback_addatom_perpendicular_motion"
            # If no mode reaches the requested directional fraction, still return
            # the best-supported perpendicular modes instead of failing outright.
            if not addatom_candidates:
                addatom_candidates = [
                    int(i) for i in selected_indices
                    if addatom_perpendicular_weight[i] > 1e-12
                ]
                selection_method = "fallback_ranked_perpendicular_weight_no_fraction_passed"
            addatom_candidates.sort(
                key=lambda i: (-float(addatom_perpendicular_weight[i]),
                               -float(addatom_perpendicular_fraction[i]),
                               float(frequencies_cm[i]))
            )
        else:
            addatom_candidates.sort(
                key=lambda i: (-float(addatom_projection[i]), float(frequencies_cm[i]))
            )

        if args.all_addatom:
            mode_indices = addatom_candidates
        else:
            mode_indices = addatom_candidates[: args.n_modes]

    if not mode_indices:
        raise RuntimeError(
            "No modes satisfy the requested selection. "
            f"Try lowering --threshold (currently {args.threshold}) "
            "or use explicit --mode."
        )

    equilibrium_path = output_dir / f"{addatom_symbol}_equilibrium.traj"
    write(equilibrium_path, full, format="traj")

    diagnostics = {
        "structure": str(Path(args.structure).resolve()),
        "phonon_dir": str(Path(args.phonon_dir).resolve()),
        "n_full_atoms": int(len(full)),
        "n_mobile_atoms": int(len(mobile)),
        "frozen_atoms": int(len(frozen_indices)),
        "frozen_indices": frozen_indices.tolist(),
        "mobile_indices": mobile_indices.tolist(),
        "index_source": "ASE Vibrations cache filenames",
        "addatom_symbol": addatom_symbol,
        "substrate_symbol": args.substrate_symbol,
        "addatom_mobile_indices": addatom_mobile_indices.tolist(),
        **geometry,
        "mode_selection_method": selection_method,
        "fallback_perpendicular_fraction_threshold": float(args.perpendicular_fraction),
        "cutoff_cm-1": float(args.cutoff),
        "addatom_projection_threshold": float(args.threshold),
        "visualization_amplitude_A": float(args.amplitude),
        "frames": int(args.frames),
        "phase_cycles": float(args.phase_cycles),
        "n_positive_modes": int(len(selected_indices)),
        "frequency_range_positive_cm-1": [
            float(np.min(frequencies_cm[selected_indices])),
            float(np.max(frequencies_cm[selected_indices])),
        ],
        "selected_modes": [],
    }

    print()
    print(f"{addatom_symbol} vibration visualization")
    print("===========================")
    print(f"Structure       : {args.structure}")
    print(f"Phonon cache    : {args.phonon_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Full atoms      : {len(full)}")
    print(f"Frozen atoms    : {len(frozen_indices)}")
    print(f"Frozen indices  : {frozen_indices.tolist()}")
    print(f"Mobile indices  : {mobile_indices.tolist()}")
    print(f"Mobile atoms    : {len(mobile)}")
    print(f"Adsorbate symbol: {addatom_symbol}")
    print(f"Adsorbate index : {addatom_mobile_indices.tolist()}")
    print(f"Surface normal  : {surface_normal.tolist()}")
    print(f"Surface distance: {geometry['addatom_surface_distance_A']:.4f} Å "
          "(height above mean top-layer plane; not nearest-neighbour distance)")
    print(f"Minimum periodic {addatom_symbol}–{args.substrate_symbol} distance: "
          f"{geometry['minimum_periodic_addatom_substrate_distance_A']:.4f} Å "
          f"(indices {geometry['minimum_distance_addatom_index']}–{geometry['minimum_distance_substrate_index']})")
    print(f"Positive modes  : {len(selected_indices)}")
    print(
        f"Positive range  : "
        f"{np.min(frequencies_cm[selected_indices]):.3f}–"
        f"{np.max(frequencies_cm[selected_indices]):.3f} cm^-1"
    )
    print()

    print("Selected modes:")
    for mode_index in mode_indices:
        print(
            f"  mode {mode_index:3d}: "
            f"{frequencies_cm[mode_index]:10.3f} cm^-1, "
            f"{addatom_symbol} projection = {addatom_projection[mode_index]:.4f}, "
            f"{addatom_symbol} perpendicular fraction = "
            f"{addatom_perpendicular_fraction[mode_index]:.3f}, "
            f"perpendicular weight = {addatom_perpendicular_weight[mode_index]:.3e}"
        )

    print()

    for mode_index in mode_indices:
        displacement = cartesian_mode_vector(
            mobile=mobile,
            mass_weighted_vector=eigenvectors[:, mode_index],
            amplitude=args.amplitude,
        )

        info = write_mode(
            full=full,
            mobile_indices=mobile_indices,
            displacement=displacement,
            mode_index=mode_index,
            frequency_cm=frequencies_cm[mode_index],
            projection=addatom_projection[mode_index],
            output_dir=output_dir,
            frames=args.frames,
            phase_cycles=args.phase_cycles,
            addatom_symbol=addatom_symbol,
        )

        info.update({
            "addatom_cartesian_share": float(addatom_cartesian_share[mode_index]),
            "addatom_perpendicular_fraction": float(addatom_perpendicular_fraction[mode_index]),
            "addatom_perpendicular_weight": float(addatom_perpendicular_weight[mode_index]),
            "selection_method": selection_method,
        })
        diagnostics["selected_modes"].append(info)

        print(f"Wrote: {info['trajectory']}")
        print(f"       {info['xyz']}")

    # Full mode table makes threshold/fallback decisions auditable, including modes
    # not selected for animation.
    mode_table_path = output_dir / "mode_analysis.csv"
    with mode_table_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "mode_index,frequency_cm-1,signed_frequency_cm-1,stable_positive,"
            "addatom_projection,addatom_cartesian_share,"
            "addatom_perpendicular_fraction,addatom_perpendicular_weight,selected\n"
        )
        chosen = set(int(i) for i in mode_indices)
        for i, frequency in enumerate(frequencies_cm):
            handle.write(
                f"{i},{frequency:.10g},{frequency:.10g},{int(frequency >= args.cutoff)},"
                f"{addatom_projection[i]:.10g},{addatom_cartesian_share[i]:.10g},"
                f"{addatom_perpendicular_fraction[i]:.10g},"
                f"{addatom_perpendicular_weight[i]:.10g},{int(i in chosen)}\n"
            )
    diagnostics["mode_analysis_csv"] = str(mode_table_path)

    diagnostics_path = output_dir / "visualization_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Equilibrium: {equilibrium_path}")
    print(f"Diagnostics: {diagnostics_path}")
    print(f"Mode analysis: {mode_table_path}")
    print()
    print("Open an animation with ASE GUI, for example:")
    if diagnostics["selected_modes"]:
        print(
            "  ase gui "
            + diagnostics["selected_modes"][0]["trajectory"]
        )
    print()
    print("No new force calculations were performed.")


if __name__ == "__main__":
    main()
