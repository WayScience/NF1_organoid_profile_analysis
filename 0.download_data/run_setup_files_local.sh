#!/bin/bash

set -euo pipefail

OVERWRITE=false

git_root=$(git rev-parse --show-toplevel)
if [ -z "$git_root" ]; then
    echo "Error: Could not find the git root directory."
    exit 1
fi

jupyter nbconvert --to=script --FilesWriter.build_directory="$git_root"/0.download_data/scripts/ "$git_root"/0.download_data/notebooks/*.ipynb


if [ "$OVERWRITE" = true ]; then
    uv run python "$git_root"/0.download_data/scripts/setup_files.py --overwrite
else
    uv run python "$git_root"/0.download_data/scripts/setup_files.py
fi

echo "Data setup complete."
