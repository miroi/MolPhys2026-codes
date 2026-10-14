#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ZIP_NAME="M_on_Se54_modular.zip"
ZIP_PATH="$SCRIPT_DIR/$ZIP_NAME"

FILES=(
    "calculator.py"
    "cli.py"
    "config.py"
    "constants.py"
    "energies.py"
    "optimization.py"
    "phonons.py"
    "run_adsorption.py"
    "surface.py"
    "test_dftd4_ase.py"
    "thermodynamics.py"
    "visualize_vibrations.py"
    "workflow.py"
    "requirements.txt"
    "adsorption_config.ini"
    "Hg_Se54_Dip.vasp"
    "Pb_Se54_Dip.vasp"
    "Se54_Dip.vasp"
    "README.md"
    "outputs/Hg_on_Se64/logfile_HgSe54_cpu_full_asevibr_1bar"
    "outputs/Hg_on_Se64/logfile_HgSe54_full_customvibr_1bar"
    "outputs/Hg_on_Se64/visualize_vibrations_HgSe54_fullslab_logfile"
    "outputs/Pb_on_Se64/logfile_full_ase_vibr_PbSe54"
    "outputs/Pb_on_Se64/logfile_MACEonly_full_ase_vibr_PbSe54"
    "outputs/Pb_on_Se64/vibrations/logfile_vibrations_MACE_fullslabASE_PbSe54"
)

echo "Verifying required files..."
for file in "${FILES[@]}"; do
    if [[ ! -f "$SCRIPT_DIR/$file" ]]; then
        echo "Error: Required file '$file' is missing."
        exit 1
    fi
done

if ! command -v zip >/dev/null 2>&1; then
    echo "Error: 'zip' command is not installed."
    exit 1
fi

rm -f "$ZIP_PATH"
(
    cd "$SCRIPT_DIR"
    zip "$ZIP_PATH" "${FILES[@]}"
)

echo "Success! $ZIP_PATH has been created."
