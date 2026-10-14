#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuration management for single-atom adsorption on an input slab."""

from __future__ import annotations

import configparser
import os


class Config:
    def __init__(self):
        self.adsorbate_file = "Adsorbate_Se54_Dip.vasp"
        self.slab_file = "Se54_Dip.vasp"
        self.substrate_symbol = "Se"
        self.adsorbate_symbol = "Hg"  # Example default; set any valid element in the INI file.

        self.n_frozen_layers = 2
        self.n_frozen_layers_vibrations = 6
        self.fmax_slab = 0.01
        self.fmax_adsorbate = 0.005
        self.optimizer = "FIRE"

        self.phonon_displacement = 0.005
        self.phonon_method = "ase"
        self.phonon_cutoff_cm = 1.0
        self.phonon_symmetrize = True
        self.ase_vibration_nfree = 2
        self.fail_on_imaginary = True
        self.adsorbate_mode_projection_threshold = 0.20

        self.temperature = 298.15
        self.temperatures = [50, 100, 200, 298.15, 400, 500, 600]
        self.pressure = 1e-3
        self.gas_reference_pressure = 1.0

        self.include_zpe = True
        self.thermo_method = "ase"
        self.calculate_temperature_scan = False
        self.calculate_g_ads_zero_crossing = True

        self.mode = "scan"
        self.output_prefix = "adsorption_se54_dip"
        self.verbose = True
        self.save_trajectories = True
        self.cuda_clear_between_phonons = True
        # Device used by MACE: "cpu", "cuda", or "auto".
        self.device = "cpu"

        self.model_path = None

        # Optional additive DFT-D4 dispersion correction. D4 parameters are
        # associated with an underlying DFT functional, not with MACE itself.
        # PBE is the explicit default reference for MACE-MP-0b3, whose
        # training data are based on PBE(+U).
        self.d4_enabled = True
        self.d4_method = "PBE"
        self.d4_scale = 1.0
        # Custom DFT-D4 damping parameters used by the validated reference
        # QE + DFT-D4 calculation.
        self.d4_s6 = 1.0
        self.d4_s9 = 1.0
        self.d4_alp = 16.0
        self.d4_s8 = 0.95948085
        self.d4_a1 = 0.38574991
        self.d4_a2 = 4.80688534
        self.d4_zero_tolerance_eV = 1.0e-12
        self.config_file = "adsorption_config.ini"

    @staticmethod
    def _as_bool(value, default=False):
        if value is None:
            return default
        return value.strip().lower() in {"true", "yes", "1", "on"}

    def load(self, filename):
        self.config_file = filename
        if not os.path.exists(filename):
            print(f"WARNING: {filename} not found. Using defaults.")
            return False

        cfg = configparser.ConfigParser()
        cfg.read(filename)

        config_dir = os.path.dirname(os.path.abspath(filename))
        if "system" in cfg:
            s = cfg["system"]
            self.adsorbate_file = s.get("adsorbate_file", self.adsorbate_file).strip()
            self.slab_file = s.get("slab_file", self.slab_file).strip()
            if not os.path.isabs(self.adsorbate_file):
                self.adsorbate_file = os.path.join(config_dir, self.adsorbate_file)
            if not os.path.isabs(self.slab_file):
                self.slab_file = os.path.join(config_dir, self.slab_file)
            self.substrate_symbol = s.get("substrate_symbol", self.substrate_symbol).strip()
            self.adsorbate_symbol = s.get("adsorbate_symbol", self.adsorbate_symbol).strip()
            self.n_frozen_layers = s.getint("n_frozen_layers", self.n_frozen_layers)
            self.n_frozen_layers_vibrations = s.getint(
                "n_frozen_layers_vibrations", self.n_frozen_layers_vibrations
            )
            self.fmax_slab = s.getfloat("fmax_slab", self.fmax_slab)
            self.fmax_adsorbate = s.getfloat("fmax_adsorbate", self.fmax_adsorbate)

        if "calculation" in cfg:
            c = cfg["calculation"]
            self.mode = c.get("mode", self.mode).strip().lower()
            self.temperature = c.getfloat("single_temperature", self.temperature)
            temps = c.get("temperatures", "").strip()
            if temps:
                self.temperatures = [float(x.strip()) for x in temps.split(",") if x.strip()]
            legacy_zero_crossing = c.get("calculate_g_ads_zero_temperature", fallback=None)
            self.calculate_g_ads_zero_crossing = self._as_bool(
                c.get("calculate_g_ads_zero_crossing", fallback=legacy_zero_crossing),
                self.calculate_g_ads_zero_crossing,
            )
            self.pressure = c.getfloat("pressure", self.pressure)
            self.output_prefix = c.get("output_prefix", self.output_prefix).strip()
            self.verbose = self._as_bool(c.get("verbose"), self.verbose)
            self.save_trajectories = self._as_bool(
                c.get("save_trajectories"), self.save_trajectories
            )
            self.optimizer = c.get("optimizer", self.optimizer).strip()

        if "phonons" in cfg:
            p = cfg["phonons"]
            self.phonon_method = p.get("method", self.phonon_method).strip().lower()
            self.phonon_displacement = p.getfloat(
                "phonon_displacement", self.phonon_displacement
            )
            self.phonon_cutoff_cm = p.getfloat(
                "frequency_cutoff_cm", self.phonon_cutoff_cm
            )
            self.phonon_symmetrize = self._as_bool(
                p.get("symmetrize"), self.phonon_symmetrize
            )
            self.ase_vibration_nfree = p.getint(
                "ase_vibration_nfree", self.ase_vibration_nfree
            )
            self.fail_on_imaginary = self._as_bool(
                p.get("fail_on_imaginary"), self.fail_on_imaginary
            )
            self.adsorbate_mode_projection_threshold = p.getfloat(
                "adsorbate_mode_projection_threshold", self.adsorbate_mode_projection_threshold
            )

        if "thermodynamics" in cfg:
            t = cfg["thermodynamics"]
            self.thermo_method = t.get("method", self.thermo_method).strip().lower()
            self.include_zpe = self._as_bool(t.get("include_zpe"), self.include_zpe)
            self.gas_reference_pressure = t.getfloat(
                "gas_reference_pressure", self.gas_reference_pressure
            )
            self.calculate_temperature_scan = self._as_bool(
                t.get("calculate_temperature_scan"), self.calculate_temperature_scan
            )

        if "runtime" in cfg:
            runtime = cfg["runtime"]
            self.cuda_clear_between_phonons = self._as_bool(
                runtime.get("cuda_clear_between_phonons"),
                self.cuda_clear_between_phonons,
            )
            self.device = runtime.get("device", self.device).strip().lower()

        if "dispersion" in cfg:
            d = cfg["dispersion"]
            self.d4_enabled = self._as_bool(
                d.get("d4_enabled"), self.d4_enabled
            )
            self.d4_method = d.get("d4_method", self.d4_method).strip()
            self.d4_scale = d.getfloat("d4_scale", self.d4_scale)
            self.d4_s6 = d.getfloat("d4_s6", self.d4_s6)
            self.d4_s9 = d.getfloat("d4_s9", self.d4_s9)
            self.d4_alp = d.getfloat("d4_alp", self.d4_alp)
            self.d4_s8 = d.getfloat("d4_s8", self.d4_s8)
            self.d4_a1 = d.getfloat("d4_a1", self.d4_a1)
            self.d4_a2 = d.getfloat("d4_a2", self.d4_a2)
            self.d4_zero_tolerance_eV = d.getfloat(
                "d4_zero_tolerance_eV", self.d4_zero_tolerance_eV
            )

        if "model" in cfg:
            self.model_path = cfg["model"].get("path", self.model_path).strip()
            if self.model_path and not os.path.isabs(self.model_path):
                self.model_path = os.path.join(config_dir, self.model_path)

        return True

    def validate(self):
        if not self.adsorbate_file or not os.path.exists(self.adsorbate_file):
            raise FileNotFoundError(self.adsorbate_file)
        if not self.slab_file or not os.path.exists(self.slab_file):
            raise FileNotFoundError(self.slab_file)
        if self.n_frozen_layers < 0:
            raise ValueError("n_frozen_layers must be >= 0")
        if self.n_frozen_layers_vibrations < self.n_frozen_layers:
            raise ValueError("n_frozen_layers_vibrations must be >= n_frozen_layers")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("runtime.device must be one of: cpu, cuda, auto")
        if self.phonon_method not in {"custom", "ase"}:
            raise ValueError("phonon method must be either custom or ase")
        if self.phonon_displacement <= 0:
            raise ValueError("phonon_displacement must be > 0")
        if self.phonon_cutoff_cm < 0:
            raise ValueError("phonon frequency cutoff must be >= 0")
        if not (0.0 < self.adsorbate_mode_projection_threshold <= 1.0):
            raise ValueError("adsorbate_mode_projection_threshold must be in (0, 1]")
        if self.thermo_method not in {"custom", "ase"}:
            raise ValueError("thermodynamics method must be either custom or ase")
        if self.temperature <= 0 or any(float(t) <= 0 for t in self.temperatures):
            raise ValueError("All temperatures must be > 0")
        if self.pressure <= 0 or self.gas_reference_pressure <= 0:
            raise ValueError("Pressures must be > 0")
        if self.d4_scale <= 0:
            raise ValueError("d4_scale must be > 0")
        if self.d4_zero_tolerance_eV < 0:
            raise ValueError("d4_zero_tolerance_eV must be >= 0")
        if not self.d4_method:
            raise ValueError("d4_method must not be empty")
        if not self.model_path:
            raise RuntimeError("No MACE model specified")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(self.model_path)
        return True
