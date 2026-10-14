#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ZIP_NAME="M_on_Se54_modular.zip"
ZIP_PATH="$SCRIPT_DIR/$ZIP_NAME"
SCRIPT_NAME="$(basename -- "${BASH_SOURCE[0]}")"

if ! command -v zip >/dev/null 2>&1; then
    echo "Error: 'zip' command is not installed."
    exit 1
fi

rm -f "$ZIP_PATH"
(
    cd "$SCRIPT_DIR"
    shopt -s dotglob nullglob
    entries=()
    for entry in *; do
        # Skip the zip file itself and any other zip files
        [[ "$entry" == "$ZIP_NAME" ]] && continue
        [[ "$entry" == *.zip ]] && continue

        # Skip this script
        [[ "$entry" == "$SCRIPT_NAME" ]] && continue

        # Skip unwanted directories/files
        [[ "$entry" == "__pycache__" ]] && continue
        [[ "$entry" == ".git" ]] && continue
        [[ "$entry" == ".venv" ]] && continue
        [[ "$entry" == "venv" ]] && continue
        [[ "$entry" == "env" ]] && continue
        [[ "$entry" == ".mypy_cache" ]] && continue
        [[ "$entry" == ".pytest_cache" ]] && continue
        [[ "$entry" == ".ruff_cache" ]] && continue
        [[ "$entry" == ".idea" ]] && continue
        [[ "$entry" == ".vscode" ]] && continue
        [[ "$entry" == *.pyc ]] && continue
        [[ "$entry" == *.pyo ]] && continue
        [[ "$entry" == *.egg-info ]] && continue
        [[ "$entry" == ".DS_Store" ]] && continue
        [[ "$entry" == "node_modules" ]] && continue

        entries+=("$entry")
    done

    if [[ ${#entries[@]} -eq 0 ]]; then
        echo "Error: No files to zip."
        exit 1
    fi

    zip -r "$ZIP_PATH" "${entries[@]}" \
        -x "*/__pycache__/*" \
        -x "*/.git/*" \
        -x "*/.venv/*" \
        -x "*/venv/*" \
        -x "*/env/*" \
        -x "*/.mypy_cache/*" \
        -x "*/.pytest_cache/*" \
        -x "*/.ruff_cache/*" \
        -x "*/.idea/*" \
        -x "*/.vscode/*" \
        -x "*.pyc" \
        -x "*.pyo" \
        -x "*.egg-info/*" \
        -x "*.DS_Store" \
        -x "*/node_modules/*"
)

echo
echo "=========================================="
echo "Files included in $ZIP_NAME:"
echo "=========================================="
unzip -Z1 "$ZIP_PATH" | sort
echo "=========================================="
echo "Total entries: $(unzip -Z1 "$ZIP_PATH" | wc -l)"
echo "=========================================="
echo
echo "Success! $ZIP_PATH has been created."
