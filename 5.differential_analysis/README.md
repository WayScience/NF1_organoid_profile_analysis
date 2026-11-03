# 5. Differential analysis

This module compares every treatment to that patient's DMSO control, for both morphology (organoid and single-cell profiles) and viability, then uses those contrasts to find plate-position artifacts, DMSO consistency, and MEK inhibitor (MEKi) signatures.

## A note on "log2 fold change" naming

The column and file names throughout this module say `fc` / `log2fc` / "fold change" for both morphology and viability, but only one of those is a real log2 ratio:

* **Viability** is a true log2 fold change: `log2(treatment viability / DMSO viability)`. Viability is a strictly positive percentage, so the ratio is well-defined.
* **Morphology** (organoid and single-cell profile features) is **not** a log2 fold change. The normalized profiles are already normalized and centered per plate's DMSO (so many feature values are negative), and a log2 ratio is not defined for negative values. The morphology "fold change" is actually `treatment - DMSO reference`, a plain difference on the normalized feature scale. The `fc`/`log2fc` naming is kept only so these files and columns match the existing viability naming and so downstream plotting notebooks (`1`, `9`) can read them without a rename.

Keep this in mind when reading any `*_fc.parquet` / `*_log2fc` column: check whether it is a viability column (true log2FC) or a morphology column (normalized-scale difference) before interpreting its magnitude as a fold change.

## Pipeline

Every step is a notebook in `notebooks/`; `run_all_differential_analyses.sh` regenerates the mirrored `scripts/` (same base names) from them with `jupyter nbconvert` before running.
Python steps run in the `uv` environment and R (ggplot2) steps run in the `uvr` environment (see the [root README](../README.md#computational-environment)).

| Step | Language | What it does | Main outputs |
|------|----------|--------------|--------------|
| `0.calculate_log2_fold_change` | Python | Per-patient consensus morphology difference and viability log2FC vs. DMSO | `results/organoid_fc.parquet`, `results/single_cell_fc.parquet` |
| `0b.calculate_well_fold_changes` | Python | Same comparison at well-level grain (one row per patient x treatment x dose x well), so a single MEK drug x dose has real within-patient replicates | `results/organoid_well_fc.parquet`, `results/single_cell_well_fc.parquet` |
| `1.plot_log2_fold_change` | R | Viability and morphology heatmaps, and morphology-vs-viability comparisons, from step 0 | figures from step 0's data |
| `2.calculate_viability_by_tumor_type` | Python | Descriptive summary (mean, SD, median, n) of per-patient viability per tumor type x treatment | `results/viability_by_tumor_type/` |
| `3.plot_viability_by_tumor_type` | R | Per-patient viability heatmap, grouped by tumor type | figures from step 2's data |
| `4.calculate_plate_position_effects` | Python | Tests whether well position (inner vs. outer) on the plate affects morphology, plus a DMSO column sanity check | `results/plate_position_effects/` |
| `5.plot_plate_position_effects` | R | Plots for step 4 | figures from step 4's data |
| `6.calculate_well_dmso_correlation` | Python | Pearson correlation of every well's morphology to its plate's DMSO reference | `results/well_dmso_correlation/` |
| `7.plot_well_dmso_correlation` | R | Plate maps and per-treatment correlation plots for step 6 | figures from step 6's data |
| `8.calculate_mek_signatures` | Python | MEKi vs. DMSO/Staurosporine/Digoxin contrasts and significance at four levels (within patient, across patients, within tumor type, across tumor types) | `results/mek_signatures/` |
| `9.plot_mek_signatures` | R | Significant feature counts, feature categories, and recurrence plots for step 8 | figures from step 8's data |

## Running

```bash
source uv_setup.sh                                   # once: uv + uvr environments and Jupyter kernels
bash 5.differential_analysis/run_all_differential_analyses.sh
```

`run_all_differential_analyses.sh` regenerates `scripts/` from `notebooks/` with `jupyter nbconvert`, then runs every step in order.
