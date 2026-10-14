# Hg and Pb adsorption on Se54: Modular MACE workflow

## Purpose and scope

This README documents the supplied modular code for single-atom **Hg/Se54** and **Pb/Se54** adsorption. The workflow reads pre-built VASP structures, optimizes the clean slab and adsorbate/slab independently, evaluates adsorption energies, calculates constrained Gamma-point harmonic vibrations, evaluates adsorption thermodynamics, scans temperature, and searches for a temperature where $\Delta G_{\mathrm{ads}}=0$.

The project includes optional additive DFT-D4 dispersion on top of MACE. Results depend on the selected configuration, especially whether D4 is enabled and how many Se layers are included in the vibrational coordinate space. The bundled logs include multiple settings and must not be treated as one directly interchangeable data set.

This is a finite-supercell, Gamma-point harmonic model, not a Brillouin-zone-converged phonon calculation.

## 1. Package contents

Main modules:

- `run_adsorption.py`: command-line entry point and workflow launcher.
- `config.py`: reads and validates the INI configuration.
- `surface.py`: reads VASP structures and handles substrate-layer constraints and adsorbate–surface distances.
- `optimization.py`: geometry optimization.
- `calculator.py`: MACE calculator and optional additive DFT-D4 calculator; reports energy components and D4 diagnostics.
- `energies.py`: electronic adsorption-energy calculations.
- `phonons.py`: custom or ASE finite-displacement vibrational calculation, frequency filtering, imaginary-mode checks, and adsorbate-mode projection.
- `thermodynamics.py`: harmonic vibrational thermodynamics and monatomic ideal-gas thermodynamics for the selected adsorbate.
- `workflow.py`: orchestrates optimization, energy, phonon, thermochemistry, temperature scan, and zero-crossing search.
- `visualize_vibrations.py`: analyzes an existing ASE vibration cache and writes mode trajectories; it does not run new force calculations.
- `test_dftd4_ase.py`: standalone smoke test for DFT-D4 and additive energy/force behavior.

Input structures included in the archive:

- `Se54_Dip.vasp`: clean 54-Se slab.
- `Hg_Se54_Dip.vasp`: Hg plus the same Se54 slab.
- `Pb_Se54_Dip.vasp`: Pb plus the same Se54 slab.

The supplied structure files contain 54 Se substrate atoms and one adsorbate atom in each adsorbate/slab system (55 atoms total).

## 2. Environment and execution

The package requirements are listed in `requirements.txt`:

```text
ASE==3.29.0
mace-torch==0.3.16
numpy
dftd4==4.2.0
```

Activate an environment containing these dependencies and point the configuration to an existing MACE model file. The bundled INI files contain a machine-specific model path; update `[model] path` for your installation or pass `--model`.

Run Hg using the Hg configuration:

```bash
python run_adsorption.py --config adsorption_config.ini_HgSe54_MACE-D4
```

Run Pb using the Pb configuration:

```bash
python run_adsorption.py --config adsorption_config.ini_PbSe54_MACE
```

The primary `adsorption_config.ini` in the archive is currently configured for Pb, with six Se layers frozen in the vibrational coordinate space and D4 enabled. Note that the archived file `adsorption_config.ini_HgSe54_MACE-D4` currently has `d4_enabled = false` despite its filename, and `adsorption_config.ini_PbSe54_MACE` also has D4 disabled. To reproduce the logged MACE+D4 runs, explicitly set `d4_enabled = true` in the chosen INI file (or use `--d4`). Always pass the intended config explicitly so that the calculation does not depend on which config happens to be the default.

Useful command-line overrides supported by `run_adsorption.py` include:

```bash
python run_adsorption.py --config adsorption_config.ini_HgSe54_MACE-D4 --model /absolute/path/to/mace-mp-0b3-medium.model
python run_adsorption.py --config adsorption_config.ini_PbSe54_MACE --mode single --temperature 298.15 --pressure 1.0
```

The CLI also supports `--d4`/`--no-d4`, `--d4-method`, and `--d4-scale` overrides. Use `python run_adsorption.py --help` to inspect the exact options available in the checked-out version.

## 3. Configuration: selecting Hg or Pb on Se54

The configuration is element-aware. The `[system]` section must keep the substrate as Se and set the adsorbate file and symbol consistently.

### Hg/Se54

```ini
[system]
adsorbate_file = Hg_Se54_Dip.vasp
slab_file = Se54_Dip.vasp
substrate_symbol = Se
adsorbate_symbol = Hg
n_frozen_layers = 0
n_frozen_layers_vibrations = 0
fmax_slab = 0.01
fmax_adsorbate = 0.005
```

### Pb/Se54

```ini
[system]
adsorbate_file = Pb_Se54_Dip.vasp
slab_file = Se54_Dip.vasp
substrate_symbol = Se
adsorbate_symbol = Pb
n_frozen_layers = 0
n_frozen_layers_vibrations = 0
fmax_slab = 0.01
fmax_adsorbate = 0.005
```

Keep the remaining calculation, phonon, thermodynamics, runtime, dispersion, and model sections consistent with the desired protocol. In particular:

- `n_frozen_layers` controls the bottom Se layers fixed during geometry optimization.
- `n_frozen_layers_vibrations` controls the bottom Se layers excluded from the vibrational coordinate space. It must be at least `n_frozen_layers`.
- `adsorbate_symbol` is used by the workflow to set the adsorbate symbol for the calculator diagnostics, vibrational projections, and gas-phase mass used in thermodynamics.
- `output_prefix` determines the names of the generated results and phonon files. Use distinct prefixes for Hg and Pb and for different calculation settings to avoid overwriting results.
- `pressure` and `gas_reference_pressure` are in bar.
- `phonon_displacement` is in Å, `frequency_cutoff_cm` in cm⁻¹, temperature in K, and energies in eV.
- `method = ase` or `method = custom` can be selected independently under `[phonons]` and `[thermodynamics]`.

### Example calculation settings

The supplied configurations use ASE vibrations and ASE thermochemistry, `ase_vibration_nfree = 2`, `phonon_displacement = 0.005`, `frequency_cutoff_cm = 1.0`, `include_zpe = true`, pressure 1 bar, and a temperature grid containing 0.1, 50, 100, 200, 298.15, 400, 500, and 600 K. The configuration's temperature grid must contain positive temperatures.

For a frozen-slab vibrational calculation, set:

```ini
n_frozen_layers_vibrations = 6
```

The Se54 slab has six Se layers. This setting leaves only the adsorbate mobile in the adsorbate/slab vibrational coordinate space and leaves no mobile vibrational coordinates for the clean slab. For a full-slab calculation, set `n_frozen_layers_vibrations = 0` (provided `n_frozen_layers = 0`). These are different physical approximations and their thermodynamic results should be reported separately.

## 4. Energy and thermodynamics definitions

For an adsorbate $M$, where $M$ is Hg or Pb, the electronic adsorption energy is

$$
\Delta E_{\mathrm{ads}} =
E_{M/\mathrm{Se54}} - E_{\mathrm{Se54}} - E_M.
$$

Negative values indicate energetically favorable adsorption relative to the isolated atom and clean slab, under the selected calculator.

When D4 is enabled, the calculator is additive:

$$
E_{\mathrm{total}} = E_{\mathrm{MACE}} + E_{\mathrm{D4}},
$$

and the output reports `E_ads_MACE`, `E_ads_DFTD4`, and `E_ads_component_sum` as diagnostics. `E_ads` is the adsorption energy from the production calculator. The component sum should agree with `E_ads` within numerical tolerance.

The workflow combines the electronic energy, the vibrational free-energy difference, and the thermal chemical potential of the monatomic gas reference:

$$
\Delta G_{\mathrm{ads}}(T,P)
= \Delta E_{\mathrm{ads}} + \Delta F_{\mathrm{vib}}(T)
- \mu_{M,\mathrm{thermal}}(T,P).
$$

The thermodynamic mass is selected from `adsorbate_symbol`, so Hg and Pb use their own atomic masses in the ideal-gas translational contribution. No gas-phase rotational or molecular-vibrational contribution is used for either monatomic reference. The workflow also evaluates the equivalent $\Delta H_{\mathrm{ads}}-T\Delta S_{\mathrm{ads}}$ route and checks the two results for consistency.

The temperature crossing is only reported if a sign change in $\Delta G_{\mathrm{ads}}$ is found within the configured search interval. “No crossing found” is a valid outcome and must not be interpreted as a calculation failure.


### 4.1 Thermochemical model and state definitions

The reported thermochemistry is a **harmonic-vibration + ideal monatomic-gas reference** model. It combines the electronic adsorption energy from the selected calculator with vibrational corrections for the adsorbate/slab and clean slab, and with the translational entropy and thermal enthalpy of an isolated gas-phase Hg or Pb atom. It is not a liquid-phase, molecular-gas, or fully anharmonic treatment.

Use the following notation:

- $M$: adsorbate atom, Hg or Pb.
- $E_{\mathrm{M/Se54}}$, $E_{\mathrm{Se54}}$, $E_M$: electronic potential energies of the optimized adsorbate/slab, clean slab, and isolated atom, respectively.
- $\{\tilde\nu_i^{\mathrm{M/Se54}}\}$, $\{\tilde\nu_j^{\mathrm{Se54}}\}$: positive harmonic frequencies in cm$^{-1}$ for the adsorbate/slab and clean slab vibrational models.
- $T$: temperature in K; $P$: gas pressure in bar; $P^\circ$: configured reference pressure in bar.
- $k_B$: Boltzmann constant; $h$: Planck constant; $c$: speed of light; $m_M$: mass of the adsorbate atom.
- $\epsilon_i = h c \tilde\nu_i$: quantum energy of mode $i$. When $\tilde\nu_i$ is entered in cm$^{-1}$, use $hc$ in matching energy·cm units, or the code constant `HC_EV_CM` to obtain $\epsilon_i$ in eV.

The code retains positive frequencies only in the thermochemistry routines. Zero and negative/imaginary frequencies are not included in the sums. The phonon stage separately reports mode and imaginary-frequency diagnostics; a frequency-filter setting may further affect which frequencies are supplied to thermochemistry. Check those diagnostics rather than assuming a stationary minimum from the thermochemistry output alone.

### 4.2 Electronic adsorption energy

The electronic adsorption energy is

$$
\boxed{\Delta E_{\mathrm{ads}} =
E_{\mathrm{M/Se54}} - E_{\mathrm{Se54}} - E_M}
$$

with units of eV per adsorbed atom in this model. A negative value means that the combined adsorbate/slab system is lower in electronic energy than the separated clean slab and isolated atom evaluated with the same calculator and compatible settings.

For the additive calculator,

$$
E_X = E_{X,\mathrm{MACE}} + E_{X,\mathrm{D4}},
$$

so the component-resolved adsorption energy is

$$
\Delta E_{\mathrm{ads},k} =
E_{\mathrm{M/Se54},k} - E_{\mathrm{Se54},k} - E_{M,k},
\qquad
\Delta E_{\mathrm{ads}}=\sum_k \Delta E_{\mathrm{ads},k}.
$$

The D4 component is a correction to the energy model, not a separate thermodynamic entropy term. The total and component-sum checks should agree numerically.

### 4.3 Harmonic vibrational thermodynamics

For a harmonic mode with frequency $\tilde\nu_i>0$, define

$$
\epsilon_i = h c \tilde\nu_i,\qquad
x_i = \frac{\epsilon_i}{k_B T}.
$$

All sums below are over the positive frequencies supplied to the thermochemistry calculation.

**Zero-point energy (ZPE)**

$$
E_{\mathrm{ZPE}} = \frac{1}{2}\sum_i \epsilon_i
= \frac{1}{2}\sum_i h c\tilde\nu_i.
$$

The raw ZPE correction for adsorption is

$$
\Delta E_{\mathrm{ZPE}}^{\mathrm{raw}} =
E_{\mathrm{ZPE}}^{\mathrm{M/Se54}} -
E_{\mathrm{ZPE}}^{\mathrm{Se54}}.
$$

With `include_zpe = true`, $\Delta E_{\mathrm{ZPE}}=\Delta E_{\mathrm{ZPE}}^{\mathrm{raw}}$; with `include_zpe = false`, the applied correction is set to zero. The raw value can still be reported diagnostically.

**Thermal vibrational internal energy excluding ZPE**

$$
U_{\mathrm{vib,thermal}}(T)
= \sum_i \frac{\epsilon_i}{\exp(x_i)-1}.
$$

The adsorption difference is

$$
\Delta U_{\mathrm{vib,thermal}} =
U_{\mathrm{vib,thermal}}^{\mathrm{M/Se54}} -
U_{\mathrm{vib,thermal}}^{\mathrm{Se54}}.
$$

The total harmonic vibrational internal energy, when ZPE is included, is

$$
U_{\mathrm{vib}}(T) =
E_{\mathrm{ZPE}} + U_{\mathrm{vib,thermal}}(T).
$$

For the `include_zpe = false` convention, the ZPE term is omitted from the reported total used in adsorption thermochemistry. The output `delta_U_vib_total` is the applied ZPE difference plus the thermal difference when ZPE is enabled, and is just the thermal difference when it is disabled.

**Harmonic vibrational entropy**

$$
S_{\mathrm{vib}}(T) =
k_B \sum_i \left[
\frac{x_i}{\exp(x_i)-1}
-\ln\!\left(1-\exp(-x_i)\right)
\right].
$$

The entropy difference is

$$
\Delta S_{\mathrm{vib}} =
S_{\mathrm{vib}}^{\mathrm{M/Se54}} -
S_{\mathrm{vib}}^{\mathrm{Se54}}.
$$

ZPE has no entropy contribution in the harmonic oscillator model.

**Harmonic vibrational Helmholtz free energy**

With ZPE included,

$$
F_{\mathrm{vib}}(T) =
\sum_i\left[
\frac{\epsilon_i}{2}
+k_B T\ln\!\left(1-\exp(-x_i)\right)
\right].
$$

Equivalently, $F_{\mathrm{vib}}=U_{\mathrm{vib}}-T S_{\mathrm{vib}}$. If ZPE is excluded, the $\epsilon_i/2$ term is omitted:

$$
F_{\mathrm{vib,no\,ZPE}}(T) =
\sum_i k_B T\ln\!\left(1-\exp(-x_i)\right).
$$

The vibrational free-energy correction used for adsorption is

$$
\boxed{\Delta F_{\mathrm{vib}} =
F_{\mathrm{vib}}^{\mathrm{M/Se54}} -
F_{\mathrm{vib}}^{\mathrm{Se54}}}
$$

using the same ZPE convention for both systems. The clean-slab contribution is subtracted; it is not assumed to be zero unless the chosen vibrational model has no mobile slab modes.

### 4.4 Ideal monatomic gas reference

The gas reference is one ideal monatomic atom. It therefore has translational contributions but no molecular rotational or internal vibrational contributions.

The thermal de Broglie wavelength is

$$
\Lambda(T)=\frac{h}{\sqrt{2\pi m_M k_B T}}.
$$

At reference pressure $P^\circ$, the ideal-gas translational entropy per atom is

$$
S_M^\circ(T;P^\circ) =
k_B\left[
\ln\!\left(
\frac{k_B T}{P^\circ \Lambda^3}
\right)+\frac{5}{2}
\right].
$$

At pressure $P$, the entropy becomes

$$
\boxed{S_M(T,P) =
S_M^\circ(T;P^\circ) -
k_B\ln\!\left(\frac{P}{P^\circ}\right)}
$$

or, equivalently,

$$
S_M(T,P) =
k_B\left[
\ln\!\left(\frac{k_B T}{P\Lambda^3}\right)+\frac{5}{2}
\right].
$$

Use consistent SI units in this expression: $P$ and $P^\circ$ in Pa, $\Lambda$ in m, and $k_B$ in J K$^{-1}$. The code configuration takes pressure in bar and converts it to Pa internally. The atom mass is selected from the configured adsorbate element, so Hg and Pb have different translational entropies.

For an ideal monatomic gas, the thermal enthalpy relative to the isolated-atom electronic energy is

$$
\boxed{H_{M,\mathrm{thermal}}(T)=\frac{5}{2}k_B T}.
$$

This comprises $3k_BT/2$ translational internal energy and $k_BT$ from $PV$ per atom. It is independent of pressure in the ideal-gas model.

The thermal gas chemical-potential correction relative to the isolated atom's electronic reference is evaluated as

$$
\boxed{\mu_{M,\mathrm{thermal}}(T,P) =
H_{M,\mathrm{thermal}}(T)-T S_M(T,P)}.
$$

It is a thermal correction; it is not the full absolute chemical potential including $E_M$. The electronic isolated-atom reference is already included in $\Delta E_{\mathrm{ads}}$, so adding it again would double count that energy.

### 4.5 Adsorption enthalpy, entropy, and Gibbs free energy

The code's adsorption enthalpy correction is

$$
\boxed{\Delta H_{\mathrm{ads}}(T,P) =
\Delta E_{\mathrm{ads}}+
\Delta E_{\mathrm{ZPE}}+
\Delta U_{\mathrm{vib,thermal}}(T)-
H_{M,\mathrm{thermal}}(T)}
$$

where $\Delta E_{\mathrm{ZPE}}$ is zero if `include_zpe = false`. The ideal monatomic-gas thermal enthalpy is subtracted because the reference state contains a gas-phase atom.

The adsorption entropy is

$$
\boxed{\Delta S_{\mathrm{ads}}(T,P) =
\Delta S_{\mathrm{vib}}(T)-S_M(T,P)}.
$$

The gas translational entropy is lost on adsorption in this model; the vibrational entropy change of the surface-bound system is added. Entropy is reported in eV K$^{-1}$ per adsorbed atom (the gas entropy routine converts from J K$^{-1}$ to eV K$^{-1}$).

The principal reported Gibbs free-energy change is

$$
\boxed{\Delta G_{\mathrm{ads}}(T,P) =
\Delta H_{\mathrm{ads}}(T,P)-T\Delta S_{\mathrm{ads}}(T,P)}.
$$

The equivalent Helmholtz/chemical-potential expression is

$$
\boxed{\Delta G_{\mathrm{ads}}(T,P) =
\Delta E_{\mathrm{ads}}+\Delta F_{\mathrm{vib}}(T)
-\mu_{M,\mathrm{thermal}}(T,P)}.
$$

The implementation calculates both routes and raises an error if they disagree beyond the backend-specific numerical tolerance. This identity is also a useful consistency check on units, ZPE conventions, and pressure handling. At fixed $T$, increasing the ideal-gas pressure lowers $S_M$, raises $\mu_{M,\mathrm{thermal}}$, and generally makes adsorption more favorable in this reference convention.

**Sign convention:** negative $\Delta E_{\mathrm{ads}}$, $\Delta H_{\mathrm{ads}}$, or $\Delta G_{\mathrm{ads}}$ means adsorption is favorable relative to the specified separated-state reference at the stated conditions. A zero crossing of $\Delta G_{\mathrm{ads}}(T,P)$ is only found if the function changes sign within the configured temperature interval.

### 4.6 Units, constants, and implementation details

- Electronic energies, ZPE, $U$, $F$, $H$, $G$, and $\mu_{\mathrm{thermal}}$: eV per adsorbate.
- Entropies: eV K$^{-1}$ per adsorbate in the thermochemistry result dictionaries; ideal-gas expressions are evaluated in SI units first and converted to eV K$^{-1}$.
- Frequencies: cm$^{-1}$; temperature: K; configured pressures: bar.
- The code uses $k_B$ in both J K$^{-1}$ and eV K$^{-1}$, $h$ in J s, the eV-to-joule conversion, and the $hc$ conversion appropriate for wavenumbers in cm$^{-1}$.
- At very large $x_i=\epsilon_i/(k_BT)$, the thermal occupation and thermal logarithm tend to zero; the implementation guards these terms against numerical overflow. At very small $x_i$, the entropy expression uses a small-$x$ limit for numerical stability.
- The `custom` and `ase` thermochemistry backends are alternative implementations. Both follow the same intended harmonic/ideal-gas decomposition, but small numerical differences can occur due to backend implementation and constants. Record the backend in any report.
- `print_ase_adsorbateslab_thermo_batch` is a diagnostic of the adsorbate/slab vibrational system alone. Its table is not the adsorption free energy because it does not subtract the clean slab or include the gas reference.
- For a fully frozen clean slab, there are no clean-slab mobile modes, so its vibrational $U$, $S$, and $F$ contributions are zero by construction. In a full-slab calculation, the clean slab's included positive modes contribute to the subtracted reference. Thus full-slab and frozen-slab thermochemistry are different models and should not be mixed.

### 4.7 Temperature scans and root finding

For each configured temperature $T_k$, the workflow evaluates the same equations at the selected pressure and writes values including `E_ads`, `ZPE_ads`, `delta_U_vib_thermal`, `delta_U_vib_total`, `delta_F_vib`, `S_gas`, `S_ads_vib`, `S_surface_vib`, `S_ads`, `H_gas_thermal`, `H_ads`, and `G_ads` to `<prefix>_thermodynamics_scan.csv`.

The root finder first evaluates the configured temperature grid, looks for an exact zero or a sign change between adjacent temperatures, and then uses bisection on the configured minimum-to-maximum temperature interval. It does not run new electronic-structure or phonon calculations during the scan. If there is no sign change in the interval, it reports no crossing; this does not imply that $\Delta G$ could not cross zero outside that interval.

## 5. Outputs

The exact filenames are controlled by `output_prefix`. Main outputs include:

- `<prefix>_results.json`: final scalar results and thermodynamic information.
- `<prefix>_thermodynamics_scan.csv`: thermodynamic values over the configured temperature grid when scanning is enabled.
- `<prefix>_slab_phonons...`: clean Se54 vibrational output/cache.
- `<prefix>_<adsorbate>_slab_phonons...`: adsorbate/Se54 vibrational output/cache.
- ASE optimization trajectories when `save_trajectories = true`.
- Frequency, signed-frequency, eigenvalue, metadata, and mode-diagnostic files produced by the selected phonon backend.

The visualization utility can be used with the optimized structure and matching phonon cache, for example:

```bash
python visualize_vibrations.py \
  --structure Hg_Se_opt.traj \
  --phonon-dir adsorption_se54_dip_Hg_slab_phonons \
  --addatom-symbol Hg \
  --substrate-symbol Se
```

For Pb, change the structure, phonon-cache path, and `--addatom-symbol Pb`. Paths must match the files generated by the run being analyzed. The visualization script writes mode `.traj` and `.xyz` files and diagnostics into its output directory.

## 6. Results present in the bundled output logs

The archive includes several historical runs with different vibrational constraints and dispersion settings. The following values are copied from those logs and are **representative of those specific configurations**, not universal predictions.

### Hg/Se54, MACE + DFT-D4, full-slab vibrations

From `outputs/Hg_on_Se64/logfile_HgSe54_cpu_full_asevibr_1bar`:

| Quantity | Value |
|---|---:|
| $E_{\mathrm{slab}}$ | -201.0272064335 eV |
| $E_{\mathrm{Hg/Se54}}$ | -201.4115970950 eV |
| $E_{\mathrm{Hg}}$ | -0.1245467400 eV |
| $\Delta E_{\mathrm{ads}}$ | -0.2598439216 eV |
| MACE adsorption component | +0.0298611410 eV |
| DFT-D4 adsorption component | -0.2897050626 eV |
| $\Delta F_{\mathrm{vib}}$ at reported final temperature | -0.1975541241 eV |
| $G_{\mathrm{ads}}$ at reported final temperature | +0.0190499359 eV |
| $T_{\mathrm{cross}}$ | 276.839421 K |
| Final adsorbate height/distance diagnostic | 2.3543235873 Å |

The corresponding visualization log reports 162 positive modes for the 55-atom full-mobile system and identifies Hg-dominated modes at 9.836, 11.877, and 31.010 cm⁻¹. The 31.010 cm⁻¹ mode is marked in that log as predominantly perpendicular to the surface.

### Hg/Se54, MACE + DFT-D4, frozen-slab vibrations

From `outputs/Hg_on_Se64/logfile_HgSe54_cpu_frozenslab_asevibr_1bar`:

| Quantity | Value |
|---|---:|
| $\Delta E_{\mathrm{ads}}$ | -0.2598439216 eV |
| ZPE adsorption correction | +0.0034984073 eV |
| $\Delta F_{\mathrm{vib}}$ | -0.1911623917 eV |
| $G_{\mathrm{ads}}$ at reported final temperature | +0.0254416683 eV |
| $T_{\mathrm{cross}}$ | 270.363676 K |
| Mobile atoms in clean slab phonons | 0 |
| Mobile atoms in Hg/slab phonons | 1 |

The visualization log gives the three Hg-only modes as 11.974, 14.587, and 29.871 cm⁻¹; the 29.871 cm⁻¹ mode has a reported perpendicular-motion fraction of 0.9775. The difference from the full-slab values is expected because the coordinate space is different.

### Pb/Se54, MACE + DFT-D4, full-slab vibrations

From `outputs/Pb_on_Se64/logfile_full_ase_vibr_PbSe54`:

| Quantity | Value |
|---|---:|
| $E_{\mathrm{slab}}$ | -201.0272064335 eV |
| $E_{\mathrm{Pb/Se54}}$ | -205.2829070388 eV |
| $E_{\mathrm{Pb}}$ | -0.7764493200 eV |
| $\Delta E_{\mathrm{ads}}$ | -3.4792512853 eV |
| MACE adsorption component | -3.0721059220 eV |
| DFT-D4 adsorption component | -0.4071453634 eV |
| $\Delta F_{\mathrm{vib}}$ | -0.1191048345 eV |
| $G_{\mathrm{ads}}$ at reported final temperature | -3.1206590355 eV |
| $\Delta G_{\mathrm{ads}}$ zero crossing | None found from 0.1 to 600 K |
| Final adsorbate height diagnostic | 1.0698370536 Å |
| Minimum periodic Pb–Se distance | 2.7736867176 Å |

The matching Pb visualization log reports 162 positive modes from 5.718 to 248.640 cm⁻¹. Its selected modes include 148.875, 11.017, and 56.291 cm⁻¹; none passes the default 0.20 total adsorbate-projection threshold in the reported full-slab analysis.

### Pb/Se54, MACE-only, full-slab vibrations

The logs `logfile_MACEonly_full_ase_vibr_PbSe54` and `_02` report a separate MACE-only calculation:

| Quantity | Value |
|---|---:|
| $E_{\mathrm{slab}}$ | -186.2491159133 eV |
| $\Delta E_{\mathrm{ads}}$ | -3.0912773959 eV |
| MACE adsorption component | -3.0912773959 eV |
| DFT-D4 adsorption component | 0.0000000000 eV |
| $G_{\mathrm{ads}}$ at reported final temperature | -2.8094139118 eV |
| $\Delta G_{\mathrm{ads}}$ zero crossing | None found from 0.1 to 600 K |
| Final adsorbate height diagnostic | 0.9381958329 Å |
| Minimum periodic Pb–Se distance | 2.7736867176 Å |

Do not compare the absolute slab energies from the MACE-only Pb run directly with the MACE+D4 logs as if they were the same calculation: they use different calculator setups. The MACE-only and MACE+D4 adsorption energies are useful as separate protocol results.

## 7. Vibrational constraints and mode interpretation

The phonon calculation uses finite displacements and a Gamma-point Hessian. With `n_frozen_layers_vibrations = 0`, all 55 atoms are mobile and there are 165 Cartesian coordinates before any frequency filtering. With all six Se layers vibrationally frozen, the clean slab has no mobile coordinates and the adsorbate/slab system has three coordinates (the adsorbate atom).

The code keeps the full physical structure in force evaluations while selecting only mobile coordinates for displacement and Hessian diagonalization. Significant imaginary modes are not silently converted into positive thermodynamic frequencies; with `fail_on_imaginary = true`, an unstable mode can stop the calculation.

Mode projection is a diagnostic. It helps identify vibrations involving Hg or Pb, but thermochemistry uses the retained positive modes of the configured mobile-coordinate system, not only adsorbate-dominated modes.

## 8. Validation and reproducibility checklist

Before interpreting the numbers as quantitative adsorption thermodynamics:

1. Confirm the correct adsorbate file and `adsorbate_symbol` (Hg or Pb) in the INI file.
2. Record whether DFT-D4 is enabled and the D4 reference method, scale, and damping parameters.
3. Record `n_frozen_layers` and `n_frozen_layers_vibrations`; do not mix full-slab and frozen-slab thermochemistry.
4. Use a unique `output_prefix` for each element and protocol.
5. Check the signed vibrational spectrum and investigate significant imaginary modes rather than ignoring them.
6. Test finite-displacement sensitivity, e.g. 0.003, 0.005, and 0.008 Å.
7. Compare custom and ASE thermochemistry backends for the same frequencies and settings.
8. Check convergence with slab size, slab thickness, adsorption geometry, and constraints.
9. For D4 runs, verify that `E_ads_component_sum` agrees with `E_ads`; retain the MACE and D4 components in reports.
10. Archive the exact INI file, model path/version, package versions, and output logs with the reported results.

## 9. Scope and limitations

- The vibrational model is Gamma-point only for the supplied finite slab.
- Harmonic thermochemistry omits anharmonicity and temperature-dependent structural effects.
- Frozen-layer settings change the vibrational coordinate space and therefore change the reported free-energy correction.
- The outputs are model- and protocol-dependent; the bundled numerical results should be labeled by element, calculator setup, pressure, and vibrational constraints.
- The current package runs one adsorbate element per configuration. To switch from Hg to Pb, edit/select the corresponding INI file rather than trying to include both adsorbates in one run.
