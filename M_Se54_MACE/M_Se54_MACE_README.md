# MACE Thermodynamic Analysis of Pb Adsorption on Se54 Surface

## Overview

This README documents the machine-learning interatomic potential (MLIP)
workflow for studying Pb adsorption on a Se54 surface using MACE. The
workflow combines:

-   Geometry optimization
-   Adsorption energy evaluation
-   Vibrational analysis
-   Hessian matrix calculation
-   Phonon frequencies
-   Zero-point energy (ZPE)
-   Entropy corrections
-   Enthalpy and Gibbs free-energy analysis

The objective is to determine the thermodynamic stability of Pb
adsorption configurations.

------------------------------------------------------------------------

# 1. MACE Energy and Force Model

MACE predicts atomic energies and forces using an equivariant neural
network representation.

The total energy is written as:

\[ E\_{`\mathrm{total}`{=tex}} = `\sum`{=tex}\_i E_i \]

where each atomic contribution is learned from local atomic
environments.

The forces are obtained from:

\[ F_i = -`\frac{\partial E}{\partial R_i}`{=tex} \]

where (R_i) represents atomic coordinates.

------------------------------------------------------------------------

# 2. Adsorption Energy

The adsorption energy of Pb on Se54 is calculated as:

\[ E\_{`\mathrm{ads}`{=tex}} = E\_{`\mathrm{Pb+Se54}`{=tex}} -
(E\_{`\mathrm{Se54}`{=tex}} + E\_{`\mathrm{Pb}`{=tex}}) \]

where:

-   (E\_{`\mathrm{Pb+Se54}`{=tex}}) is the optimized energy of Pb
    adsorbed on Se54.
-   (E\_{`\mathrm{Se54}`{=tex}}) is the clean Se54 slab energy.
-   (E\_{`\mathrm{Pb}`{=tex}}) is the energy of isolated Pb.

A negative value indicates stable adsorption.

------------------------------------------------------------------------

# 3. Pb-Se Bond Distance

The Pb-Se interaction is analyzed using:

\[ d\_{`\mathrm{Pb-Se}`{=tex}} =
\|`\mathbf{R}`{=tex}*{Pb}-`\mathbf{R}`{=tex}*{Se}\| \]

Shorter distances generally indicate stronger interaction.

------------------------------------------------------------------------

# 4. Hessian Matrix and Vibrational Analysis

The Hessian matrix is defined as:

\[ H\_{ij} = `\frac{\partial^2 E}`{=tex} {`\partial `{=tex}R_i
`\partial `{=tex}R_j} \]

The vibrational frequencies are obtained from the mass-weighted Hessian:

\[ D = M^{-1/2}HM^{-1/2} \]

The eigenvalue equation is:

\[ Dq_k = `\omega`{=tex}\_k\^2 q_k \]

where:

-   (q_k) are vibrational eigenvectors.
-   (`\omega`{=tex}\_k) are angular frequencies.

Positive frequencies indicate a stable minimum.

------------------------------------------------------------------------

# 5. Zero-Point Energy

The zero-point vibrational contribution is:

\[ E\_{ZPE} = `\frac{1}{2}`{=tex} `\sum`{=tex}\_i h`\nu`{=tex}\_i \]

where ( `\nu`{=tex}\_i ) are vibrational frequencies.

------------------------------------------------------------------------

# 6. Entropy Contribution

The vibrational entropy is calculated using:

\[ S = k_B `\sum`{=tex}\_i `\left[
\frac{x_i}{e^{x_i}-1}
-
\ln(1-e^{-x_i})
\right]`{=tex}\]

where:

\[ x_i=`\frac{h\nu_i}{k_BT}`{=tex} \]

------------------------------------------------------------------------

# 7. Enthalpy and Gibbs Free Energy

The temperature-dependent enthalpy correction is:

\[ H(T)=E\_{elec}+E\_{ZPE}+E\_{thermal} \]

The Gibbs free energy is:

\[ G(T)=H(T)-TS(T) \]

A lower Gibbs free energy corresponds to a more favorable adsorption
state.

------------------------------------------------------------------------

# 8. Pb/Se54 Thermodynamic Interpretation

For Pb adsorption on Se54:

-   Geometry optimization identifies the preferred adsorption structure.
-   Adsorption energy determines energetic stability.
-   Pb-Se distances characterize chemical interaction.
-   Hessian analysis confirms whether the structure is a true minimum.
-   Vibrational frequencies provide thermal corrections.
-   Gibbs free energy determines temperature-dependent stability.

A configuration with:

-   negative adsorption energy,
-   no imaginary frequencies,
-   favorable entropy correction,
-   lowest Gibbs free energy,

is considered thermodynamically stable.

------------------------------------------------------------------------

# 9. Computational Workflow

1.  Build Se54 surface model.
2.  Add Pb adsorbate.
3.  Optimize structure using MACE.
4.  Calculate adsorption energy.
5.  Compute Hessian matrix.
6.  Obtain phonon frequencies.
7.  Calculate ZPE and entropy.
8.  Evaluate Gibbs free energy.
9.  Compare adsorption configurations.

------------------------------------------------------------------------

# 10. Software

Recommended environment:

-   ASE
-   MACE MLIP
-   PyTorch
-   Phonon analysis tools

The workflow is designed for reproducible MLIP-based surface
thermochemistry calculations.
