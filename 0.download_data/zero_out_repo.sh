#!/bin/bash
set -euo pipefail

git_root=$(git rev-parse --show-toplevel) || { echo "Not inside a git repo" >&2; exit 1; }

DEL_ALL_RESULTS_AND_FIGURES=${1:-false}

ARRAY_OF_DIRS_TO_DELETE=("results" "figures")

# Build: -name results -o -name figures
name_args=()
for d in "${ARRAY_OF_DIRS_TO_DELETE[@]}"; do
    [ ${#name_args[@]} -gt 0 ] && name_args+=( -o )
    name_args+=( -name "$d" )
done
# Paths to skip entirely
exclude_args=( \( -path "$git_root/.git" \
               -o -path "$git_root/.venv" \
               -o -path "$git_root/data" \
               -o -path "$git_root/.uvr" \) -prune -o )

# Top-level dirs to keep even if their name matches, since they hold source
# files (e.g. figures/drugs_figure/scripts) alongside generated output.
top_level_keep_args=()
for d in "${ARRAY_OF_DIRS_TO_DELETE[@]}"; do
    top_level_keep_args+=( ! -path "$git_root/$d" )
done

if [ "$DEL_ALL_RESULTS_AND_FIGURES" = "true" ]; then
    find "$git_root" "${exclude_args[@]}" \
        -type d \( "${name_args[@]}" \) "${top_level_keep_args[@]}" -prune -exec rm -rf {} +
else
    find "$git_root" "${exclude_args[@]}" \
        -type d \( "${name_args[@]}" \) "${top_level_keep_args[@]}" -prune -print
fi
