#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Physical constants used by the single-atom adsorption workflow."""

EV_TO_KJ = 96.48533212
EV_TO_KCAL = 23.060548
KB_EV = 8.617333262e-5
KB_J = 1.380649e-23
H_PLANCK = 6.62607015e-34
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
C_LIGHT = 299792458.0
BAR_TO_PA = 1.0e5

# Conversion for sqrt(eV / Angstrom^2 / amu) -> cm^-1.
ASE_TO_CM = (
    (EV_J / (1.0e-20 * AMU_KG)) ** 0.5
    / (2.0 * 3.141592653589793 * C_LIGHT * 100.0)
)

HC_EV_CM = 1.2398419843320026e-4
