# Single Pb Atom Adsorption on a Se54 Surface
## Structural Optimization, Adsorption Energetics, Vibrational Analysis, and Thermodynamic Stability Using MACE

---

# 1. Overview

This project investigates the adsorption of a single Pb atom on a trigonal selenium (Se54) surface using a machine-learning interatomic potential (MACE).

The computational workflow combines:

- periodic surface modeling,
- adsorption geometry construction,
- machine-learning-based energy and force evaluation,
- atomic structure optimization,
- adsorption energy calculation,
- harmonic vibrational analysis,
- thermochemical corrections,
- Gibbs free-energy evaluation.

The main objectives are:

1. Determine the equilibrium adsorption geometry of Pb on Se54.
2. Calculate the Pb–Se interaction strength.
3. Verify the dynamical stability of the optimized structure.
4. Determine vibrational frequencies and Pb surface modes.
5. Evaluate adsorption thermodynamics as a function of temperature.

The final optimized system represents a stable Pb adsorption configuration.

---

# 2. Physical Background

## 2.1 Atomic adsorption on semiconductor surfaces

Adsorption of isolated metal atoms on semiconductor and chalcogenide surfaces is governed by:

- chemical bonding,
- charge transfer,
- surface reconstruction,
- elastic deformation,
- electrostatic interactions,
- dispersion forces.

A metal atom adsorbed on a surface can form localized bonds with surface atoms, modifying:

- electronic structure,
- surface reactivity,
- catalytic activity,
- defect chemistry.

For a single atom adsorbate:

\[
\mathrm{Pb(g)}+\mathrm{Se_{surface}}
\rightarrow
\mathrm{Pb/Se_{surface}}
\]

the adsorption process is characterized by the adsorption energy.

---

# 3. Adsorption Energy Theory

## 3.1 Electronic adsorption energy

The adsorption energy is defined as:

\[
E_{ads}
=
E_{\mathrm{Pb/Se}}
-
E_{\mathrm{Se}}
-
E_{\mathrm{Pb}}
\]

where:

- \(E_{\mathrm{Pb/Se}}\) is the total energy of the optimized Pb/Se54 system,
- \(E_{\mathrm{Se}}\) is the energy of the clean Se54 slab,
- \(E_{\mathrm{Pb}}\) is the energy of an isolated Pb atom.

The sign convention is:

\[
E_{ads}<0
\]

for favorable adsorption.

A more negative adsorption energy indicates stronger interaction between Pb and the surface.

---

# 4. Surface Model

The substrate consists of:

- 54 selenium atoms,
- six Se layers,
- periodic boundary conditions.

The adsorbate:

- one Pb atom,
- total atoms after adsorption: 55.

The slab atoms are allowed to relax during optimization.

The nearest Pb–Se distance is calculated using the minimum-image convention:

\[
d_{\mathrm{Pb-Se}}
=
\min_i
\left|
r_{\mathrm{Pb}}-r_{\mathrm{Se},i}
\right|
\]

This represents the actual atomic interaction distance and is different from the vertical adsorption height.

---

# 5. Machine-Learning Interatomic Potential

## 5.1 MACE theory

The atomic interactions are described using the MACE machine-learning potential.

MACE (Message Passing Atomic Cluster Expansion) combines:

- equivariant neural networks,
- local atomic environments,
- many-body interactions.

The total energy is expressed as:

\[
E=
\sum_i E_i
\]

where each atomic energy contribution depends on the local environment.

The force acting on atom \(i\) is:

\[
F_i
=
-\frac{\partial E}{\partial r_i}
\]

MACE provides near-density-functional-theory accuracy at a much lower computational cost.

---

# 6. Structural Optimization Theory

The equilibrium structure is obtained by minimizing the potential energy surface:

\[
\nabla E(r)=0
\]

The optimization condition is:

\[
|F_i| < F_{max}
\]

where:

- \(F_i\) is the atomic force,
- \(F_{max}\) is the convergence threshold.

A converged structure corresponds to a local minimum of the potential-energy surface.

---

# 7. Vibrational Frequency Theory

## 7.1 Harmonic approximation

Near equilibrium, the potential energy is expanded as:

\[
E
=
E_0
+
\frac12
\sum_{ij}
H_{ij}q_iq_j
\]

where:

- \(q_i\) are atomic displacements,
- \(H_{ij}\) is the Hessian matrix.

The Hessian is:

\[
H_{ij}
=
\frac{\partial^2E}
{\partial x_i\partial x_j}
\]

It contains the force constants describing atomic interactions.

---

## 7.2 Numerical Hessian calculation

The Hessian is obtained by finite differences:

\[
H_{ij}
=
-\frac{
F_i(x_j+\Delta x)
-
F_i(x_j-\Delta x)
}
{2\Delta x}
\]

where:

- \(F_i\) is the force component,
- \(\Delta x\) is the displacement step.

---

## 7.3 Dynamical matrix

The mass-weighted dynamical matrix is:

\[
D_{ij}
=
\frac{H_{ij}}
{\sqrt{m_i m_j}}
\]

where:

- \(m_i,m_j\) are atomic masses.

Solving:

\[
Dq_k
=
\lambda_k q_k
\]

gives the vibrational eigenvalues.

The frequency is:

\[
\omega_k
=
\sqrt{\lambda_k}
\]

and:

\[
\nu_k
=
\frac{\omega_k}{2\pi}
\]

---

# 8. Frequency Conversion

The vibrational energy of a mode is:

\[
\epsilon_k=h\nu_k
\]

The spectroscopic frequency is:

\[
\tilde{\nu}
=
\frac{\nu}{c}
\]

with units:

\[
cm^{-1}
\]

The conversion between wavenumber and energy is:

\[
E(eV)
=
1.239841984\times10^{-4}
\tilde{\nu}(cm^{-1})
\]

---

# 9. Thermochemical Corrections

## 9.1 Zero-point energy

Each vibrational mode contributes:

\[
E_{ZPE}
=
\frac12
\sum_k h\nu_k
\]

The adsorption zero-point correction is:

\[
\Delta ZPE
=
ZPE_{\mathrm{Pb/Se}}
-
ZPE_{\mathrm{Se}}
-
ZPE_{\mathrm{Pb}}
\]

---

## 9.2 Vibrational internal energy

The thermal vibrational energy is:

\[
U_{vib}
=
\sum_k
\frac{h\nu_k}
{
\exp(h\nu_k/k_BT)-1
}
\]

---

## 9.3 Vibrational entropy

The entropy contribution is:

\[
S_{vib}
=
k_B
\sum_k
\left[
\frac{x_k}
{e^{x_k}-1}
-
\ln(1-e^{-x_k})
\right]
\]

where:

\[
x_k=
\frac{h\nu_k}{k_BT}
\]

---

## 9.4 Enthalpy

The adsorption enthalpy is:

\[
\Delta H_{ads}
=
\Delta E_{ads}
+
\Delta ZPE
+
\Delta U_{thermal}
\]

---

## 9.5 Gibbs free energy

The adsorption Gibbs free energy is:

\[
\Delta G_{ads}
=
\Delta H_{ads}
-
T\Delta S_{ads}
\]

A negative value indicates thermodynamically favorable adsorption.

---

# 10. Computational Results

## 10.1 Optimized adsorption geometry

The optimized Pb adsorption structure gives:

### Vertical adsorption height

\[
h_{\mathrm{Pb-surface}}
=
0.938 \AA
\]

This describes the Pb position relative to the average surface plane.

---

## 10.2 Nearest Pb–Se distance

The physically meaningful Pb–Se interaction distance is:

| Configuration | Pb–Se distance |
|---|---:|
|Initial | 2.7678 Å |
|Optimized | 2.7737 Å |

The change is:

\[
\Delta d
=
+0.0059\AA
\]

The nearest surface atom remains unchanged.

Therefore:

- Pb does not migrate,
- the adsorption site is preserved,
- structural relaxation is small.

The final Pb–Se distance is:

\[
\boxed{
d_{\mathrm{Pb-Se}}
=
2.7737\AA
}
\]

---

# 11. Adsorption Energy Results

The calculated adsorption energy is:

\[
\boxed{
E_{ads}
=
-3.0913\,eV
}
\]

This corresponds to approximately:

\[
-298\,kJ/mol
\]

The adsorption is therefore strongly favorable.

The strong binding originates from Pb–Se interaction rather than dispersion correction.

---

# 12. Vibrational Results

## 12.1 Pb/Se54 system

The vibrational calculation gives:

| Quantity | Result |
|-|-:|
|Number of modes|165|
|Positive modes|162|
|Imaginary modes|None|
|Frequency range|5.718–248.640 cm⁻¹|

The absence of imaginary frequencies confirms:

- a stable local minimum,
- no spontaneous structural distortion,
- mechanically stable adsorption geometry.

---

# 13. Pb Vibrational Modes

Pb participates mainly in low-frequency surface modes.

Important Pb-related modes:

|Mode|Frequency (cm⁻¹)|Character|
|-|-:|-|
|5|11.017|Pb perpendicular motion|
|71|56.291|Pb–surface coupled vibration|
|113|148.875|strong Pb contribution|

The lowest Pb mode represents weak restoring forces perpendicular to the surface.

Higher-frequency modes indicate stronger Pb coupling with Se atoms.

---

# 14. Thermodynamic Stability

At 298 K:

\[
\Delta G_{ads}
=
-2.8094\,eV
\]

The adsorption remains favorable.

Temperature dependence:

|Temperature|ΔG_ads|
|-|-:|
|0 K|-3.085 eV|
|298 K|-2.809 eV|
|600 K|-2.542 eV|

No desorption transition occurs below 600 K.

---

# 15. Scientific Interpretation

The Pb/Se54 system demonstrates:

## Strong chemical interaction

\[
E_{ads}=-3.0913\,eV
\]

indicates strong Pb binding.

## Stable adsorption geometry

The Pb–Se distance remains almost unchanged after relaxation:

\[
\Delta d=0.0059\AA
\]

## Dynamical stability

No imaginary phonon frequencies are present.

## Thermal stability

The Gibbs free energy remains negative over the investigated temperature range.

---

# 16. Conclusions

The single Pb atom forms a stable adsorption structure on the Se54 surface.

The key results are:

|Property|Value|
|-|-:|
|Nearest Pb–Se distance|2.7737 Å|
|Adsorption energy|-3.0913 eV|
|Gibbs adsorption free energy (298 K)|-2.8094 eV|
|Lowest vibrational frequency|5.718 cm⁻¹|
|Highest vibrational frequency|248.640 cm⁻¹|
|Imaginary modes|None|

The optimized Pb/Se54 structure is therefore:

- energetically stable,
- dynamically stable,
- thermodynamically favorable.

---

# 17. Future Extensions

Possible future investigations:

1. Compare adsorption sites:
   - top-Se,
   - bridge,
   - hollow.

2. Include dispersion corrections.

3. Perform charge-transfer analysis.

4. Compare MACE predictions with DFT calculations.

5. Study multiple Pb adsorption and surface coverage effects.

6. Investigate adsorption kinetics and diffusion barriers.

---
