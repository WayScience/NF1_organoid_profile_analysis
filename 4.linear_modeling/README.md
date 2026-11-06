# 4. Linear modeling

This module fits one ordinary least squares (OLS) model per **(patient, treatment/dose, feature)** on the 3D NF1 organoid profiles and uses the saved fit statistics to ask which morphology changes come from treatment rather than from counts or technical (plate / position) covariates.
Each model compares one treatment with DMSO within a single patient:

```text
feature ~ treatment + cell_count + organoid_count + cell_per_organoid_count                      (2.linear_modeling)
        + manhattan_distance_from_center + cell_x_position + cell_y_position
        + cell_z_position + cell_z_depth                                                          (3.linear_modeling_technical_vars)
```

For every model and term the results store the coefficient, the p-value, the Benjamini-Hochberg FDR (`pvalue_fdr`, corrected separately per term across all features, treatments and patients), R2 / adjusted R2, and the variance shares (`term_pct_of_total_var` from type II sums of squares, `residual_pct`).
The steps after 3 only read these saved statistics; no model is refit.

## Pipeline

Every step is a notebook in `notebooks/`; `run_all_linear_modeling_local.sh` regenerates the mirrored `scripts/` (same base names) from them with `uv run jupyter nbconvert` before running.
Python steps run in the `uv` environment and R (ggplot2) steps run in the `uvr` environment (see the [root README](../README.md#computational-environment)).

| Step | Language | What it does | Main outputs |
|------|----------|--------------|--------------|
| `0.aggregating_non_fs_profiles` | Python | Aggregates the normalized organoid and single-cell profiles to one row per well (median) | `data/organoid_norm_aggregated_profile.parquet`, `data/sc_norm_aggregated_profile.parquet` |
| `1.calculate_well_manhattan_distance` | Python | Manhattan distance of every well from the plate center | `results/well_manhattan_distance/` (parquet + platemap pdf) |
| `2.linear_modeling` | Python | Treatment + count covariate models for four profiles (organoid, single cell, and their well-aggregated versions) | `results/linear_modeling/{organoid_norm,sc_norm,organoid_agg,single_cell_agg}.parquet` |
| `3.linear_modeling_technical_vars` | Python | Same four profiles refit with the plate-position and cell-position covariates | `results/linear_modeling/<profile>_technical_model.parquet` |
| `4.variance_decomposition` | Python | Where the variance goes, and why so much of it is residual | `figures/variance_decomposition/variance_decomposition.pdf` (all figures in one pdf), `results/decomposed_variance/`, `results/residual_diagnostics/`, `results/variate_vs_residual/`, `results/top_models/` |
| `5.calculate_variate_importance` | Python | Venn / UpSet tables of which features are hits for which term (`sc_norm` fits, original and technical models) | `results/variate_importance/` |
| `6.plot_variate_importance` | R | Venn / UpSet plots, treatment-only co-occurrence heatmaps, title-free multiresult-figure subpanels | `figures/variate_importance/` (subpanels in `multiresult_figure_subpanels/`) |
| `7.calculate_variate_class_upsets_and_clustermap` | Python | Variate-class membership, patient UpSets, patient x treatment clustermaps, treatment-only features | `results/variate_class_plots/<tag>/` |
| `8.plot_variate_class_upsets_and_clustermap` | R | Plots for step 7: titled patient UpSets (one pdf per class), clustermaps, treatment-only and shared-feature plots | `figures/variate_class_plots/<figure_name>/` |
| `9.explore_linear_model_haystacks` | Python | Exhaustive exploration of the technical models: effect sizes, hit landscape, reproducibility, dose response, the "needles" | `results/explore_linear_models/` |
| `10.plot_explore_linear_model_haystacks` | R | Plots for step 9 | `figures/explore_linear_models/explore_linear_models.pdf` |
| `11.assembled_multiresult_figure` | R | Multi-panel summary figure from the outputs of 6 and 9 | `figures/headline_results_figure/multiresult_figure.png` |

Step 7 is configurable from the command line (`python scripts/7.calculate_variate_class_upsets_and_clustermap.py --help`); its run tag (for example `sc_T-O-C-N-M-X-Y-Z-D`) names its results folder.
Step 8 plots one run: set `tag` in its parameter cell (the figures go to `figure_name`, `sc_technical_model_results` for the default tag).
Step 8 reads the top treatment-only features from step 7's tables, so it always matches step 7's `--top-features`.

## Feature importance definition

Each step uses the hit definition that suits its question, so the counts are not interchangeable:

* `5.calculate_variate_importance`: `pvalue_fdr < 0.05` and `coefficient > 0.1` (increases only).
* `7.calculate_variate_class_upsets_and_clustermap`: `pvalue_fdr < --fdr-max` and standardized effect `> --effect-min` (set on the command line), with count coefficients standardized by the covariate SD.
* `9.explore_linear_model_haystacks`: `pvalue_fdr < 0.05`, `R2 > 0.5`, `adj. R2 > 0` and `|coefficient| > 0.01`.

## Plotting code

The R steps (6, 8, 10, 11) share code in `utils/`:

* `utils/r_plot_themes.r`: every palette (for example `linear_modeling_term_palette`, `profile_palette`, `patient_palette`, `treatment_only_palette`), `variate_letters`, and the themes `theme_manuscript` and `theme_manuscript_void`. All plots use these themes.
* `utils/r_plot_funcs.r`: the plotting functions (`plot_venn`, `plot_upset`, `plot_patient_upset`, `cooccurrence_heatmap`, `plot_clustermap`, plus the shared heatmap and pdf helpers). Dense point layers are rasterized with `ggrastr`; the title-free panels that step 11 embeds are called multiresult-figure subpanels.

The Python figure step (4) writes one titled pdf with rasterized boxplot outliers.

## Running

```bash
source uv_setup.sh                                   # once: uv + uvr environments and Jupyter kernels
bash 4.linear_modeling/run_all_linear_modeling_local.sh
```

`run_all_linear_modeling_local.sh` regenerates `scripts/` from `notebooks/` with `uv run jupyter nbconvert`, then runs every step in order.
