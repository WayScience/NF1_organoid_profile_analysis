# 1. Exploratory Data Analysis (EDA)

This module contains exploratory analyses of image-based morphological profiles from NF1 organoids imaged in 2D (max projection, middle slice, and middle-N slice) and 3D.
Analyses are run across organoid-level and single-cell-level profiles, each with normal, feature-selected (fs), aggregated (agg), and consensus data, and across multiple 3D normalization variants (organoid, nucleocentric, SAM-med, and combinations thereof).

Notebooks are numbered in the order they should be run; `run_all_eda_local.sh` converts each notebook to a script under `scripts/` and executes them in sequence.

## Notebooks

| Notebook | Purpose |
|---|---|
| `0.generate_umap.ipynb` | Generates individual UMAP embeddings for organoid and single-cell profiles. |
| `1.plot_umap.ipynb` | Plots UMAPs for organoid and single-cell datasets across 2D (max projection, middle slice, middle-N slice) profiles, faceted/colored by patient and treatment. |
| `2.generate_pca.ipynb` | Generates PCA embeddings and explained-variance tables for every profile type. |
| `3.plot_pca.ipynb` | Plots PCA embeddings for every profile type in `results/pca`, for both 2D and 3D pipelines. |
| `4.calculate_correlation_matrix.ipynb` | Computes correlation matrices across patients/treatments for each 2D and 3D profile type (aggregated and consensus). |
| `5.plot_correlation_heatmaps.ipynb` | Plots each correlation matrix as a heatmap, grouped into per-dimension PDFs. |
| `7.generate_cell_counts.ipynb` | Computes cell counts per organoid and identifies single cells without a parent organoid. |
| `8.plot_single_cell_counts.ipynb` | Plots cell count distributions (by patient, by treatment, and 2D vs. 3D comparisons). |

## Outputs

- `results/pca/` — PCA embeddings and explained variance per profile type.
- `results/correlation/` — Correlation matrices per profile type.
- `results/cell_counts/` — Cell count and organoid count tables.
- `figures/correlation_heatmaps/` — Correlation heatmaps, grouped into a 2D and a 3D PDF.
- `figures/cell_counts/` — Cell/organoid count plots (bar, box, and density plots).

## Running

```bash
./run_all_eda_local.sh
```

This converts all notebooks to scripts and runs them in order, using the `GFF_analysis` conda environment for Python steps and the `gff_figure_env` conda environment for R steps.
