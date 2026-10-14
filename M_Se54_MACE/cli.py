#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Command-line interface for single-atom adsorption on a supplied slab."""

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Single-atom adsorption thermodynamics on a supplied slab"
    )
    parser.add_argument(
        "-c", "--config", default="adsorption_config.ini",
        help="INI configuration file",
    )
    parser.add_argument("--model", help="Override MACE model path")
    parser.add_argument(
        "--mode", choices=["single", "scan"],
        help="Override calculation mode",
    )
    parser.add_argument(
        "-t", "--temperature", type=float,
        help="Override temperature in K",
    )
    parser.add_argument(
        "-p", "--pressure", type=float,
        help="Override adsorbate gas pressure in bar",
    )
    d4_group = parser.add_mutually_exclusive_group()
    d4_group.add_argument(
        "--d4", dest="d4_enabled", action="store_true",
        help="Enable the additive DFT-D4 correction",
    )
    d4_group.add_argument(
        "--no-d4", dest="d4_enabled", action="store_false",
        help="Disable the additive DFT-D4 correction",
    )
    parser.set_defaults(d4_enabled=None)
    parser.add_argument(
        "--d4-method",
        help="DFT-D4 reference functional/parameter set (default: PBE)",
    )
    parser.add_argument(
        "--d4-scale", type=float,
        help="Scale the DFT-D4 contribution (default: 1.0)",
    )
    return parser.parse_args()
