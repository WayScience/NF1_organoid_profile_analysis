# 2. 2D versus 3D profile analysis

This module compares image-based morphological profiles of NF1 organoids and single cells derived from 2D projections and 3D images.
The projections, representations, and profile stages included depend on the analysis step.

Notebooks are numbered in the order they should be run.

## Notebooks

| Notebook | Purpose |
|---|---|
| `1.generate_patient_correlation.ipynb` | Computes 2D–3D Pearson correlations across matched patient/tumor–well aggregates, per sample and pooled, for every profile-type pair. |
| `2.plot_patient_correlation_heatmaps.ipynb` | Plots each correlation matrix as a clustered heatmap, combined into a single PDF. |
| `3.calculate_inter_intra_patient_distances.ipynb` | Computes inter- and intra-patient distances for each profile type to measure within- vs. across-patient similarity. |
| `4.calculate_mAP.ipynb` | Computes mean average precision (mAP) for intra- and inter-patient retrieval, for every profile type. |
| `5.plot_metrics_and_mAP.ipynb` | Plots mAP and distance metrics for every profile type and 2D-vs-3D pair, combined into PDFs. |
| `6.calculate_shannon_entropy.ipynb` | Computes per-feature Shannon entropy for 2D and 3D profiles. See [Shannon entropy](#shannon-entropy) below. |
| `7.plot_shannon_entropy.ipynb` | Plots Shannon entropy distributions comparing 2D vs. 3D profiles. See [Shannon entropy](#shannon-entropy) below. |
| `8.kbet_analysis.ipynb` | Runs kBET to test whether patient identity or treatment drives local structure in feature space. |
| `9.sparse_cca_analysis.ipynb` | Runs sparse canonical correlation analysis (CCA) between matched 2D and 3D profiles, for every profile-type pair. |
| `10.plot_sparse_cca_results.ipynb` | Plots canonical score scatterplots for each 2D-vs-3D profile-type pair computed in notebook 9. |

## Outputs

- `results/correlation/` — Correlation matrices per profile-type pair, per patient and pooled.
- `results/distance_metrics/` — Inter- and intra-patient distance tables.
- `results/mAP/` — mAP tables per profile type.
- `results/entropy/` — Per-feature Shannon entropy tables.
- `results/kbet/` — kBET results.
- `results/sparse_cca/` — Sparse CCA canonical scores and feature loadings.
- `figures/correlation/` — Correlation heatmaps, combined into a single PDF (gitignored, regenerate locally).
- `figures/mAP_and_distance_metrics/` — mAP and distance metric plots, combined into PDFs.
- `figures/entropy/` — Shannon entropy violin + boxplots.
- `figures/kbet/` — kBET result plots.
- `figures/sparse_cca/` — Sparse CCA canonical score scatterplots.

## Running

Run notebooks (or their paired scripts under `scripts/`) in numeric order.
Python notebooks use the `GFF_analysis` conda environment; R notebooks use `gff_figure_env`, except `9.sparse_cca_analysis.ipynb`, which needs the `PMA` package and runs under `sparse_cca_env` instead.

## Caching

Every step skips work it has already done. An output is reused only if it exists and was produced from the same code and inputs. The fingerprint covers:

- the step's script, so any edit to the code or its constants triggers a recompute,
- each input file's path, size, and modification time (contents are not hashed, because the 3D profiles are tens of gigabytes).

Each step records the fingerprint of each output it writes in `results/_cache_manifests/<step>.parquet`. Python steps use `utils/src/notebook_init_utils/result_cache.py`. R steps use `utils/r_result_cache.r`.

- Set `RESULTS_CACHE_FORCE=1` to ignore all caches and recompute everything.
- Delete a step's manifest to force that step to recompute.
- Mirrored scripts are regenerated from the notebooks by `run_all_2d_vs_3d_local.sh`, so a notebook edit that is followed by a run also invalidates the cache.
- The first run after adding caching has no manifests, so it recomputes everything once. Old mAP checkpoints in `results/mAP/cache/` are no longer read; they can be deleted.
- Figures that are cached are not redrawn or displayed. Notebooks therefore show no plot for a cached figure.

## Shannon entropy

### Why compute it?
Not all morphological features carry meaningful variation as some are effectively constant across the dataset, even after feature selection.
Computing per-feature entropy helps identify which features are informative and supports downstream feature selection.
It also lets us compare the overall information content of 2D vs. 3D profiles.

### What is Shannon entropy?
Shannon entropy quantifies the uncertainty or unpredictability in a distribution.
It captures both how spread out values are and how evenly they occur across different outcomes.
A distribution where all outcomes are equally likely has maximum entropy, while one concentrated on a single outcome has zero entropy.
Entropy is computed in bits (base-2 log).

For morphological features, this means: a feature with high entropy varies widely across organoids and is likely more informative, while a feature with low entropy is nearly constant across samples and unlikely to help distinguish conditions.

### How is it computed?
Each feature's values are binned into a histogram with 50 bins.
The count in each bin is divided by the total number of samples to create a probability distribution.
Shannon entropy is then computed from this distribution using `scipy.stats.entropy`, which sums `-p * log2(p)` for each bin with nonzero probability.
If values are spread evenly across all bins, entropy is high (maximum ~5.6 bits for 50 bins).
If values are concentrated in just a few bins, entropy is low (near 0 bits).

### Why 50 bins?
There is no single correct method for choosing the number of bins for entropy estimation — it depends on the data.
Common heuristics like the square root rule (`sqrt(n)`) provide a rough starting point but were designed for histogram visualization, not entropy computation.
For ~3000 single-cell profiles, the square root rule gives ~55 bins, which is close to our choice of 50.
50 bins is a practical, round number that provides enough resolution to capture distribution shape in both 2D and 3D profiles, which have different numbers of features.
With 3000 samples and 50 bins, each bin averages ~60 samples, which is sufficient for stable probability estimates.
The number of bins can be adjusted if needed, but the relative entropy rankings across features are expected to remain stable across reasonable bin counts.

### Graphs
Each plot is a violin + boxplot comparing the entropy distributions of 2D and 3D features.
The violin shows the full shape of the distribution, while the boxplot inside shows the median, quartiles, and whiskers.
Individual dots are outlier features with unusually high or low entropy.
Each dot represents a single feature, not a treatment or patient.
