---
title: NF1 3D Organoid Analysis
emoji: 📉
colorFrom: yellow
colorTo: red
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# NF1 organoid profile data viewer

Interactive Streamlit app for exploring the NF1 organoid profiling analysis:
drug/plate layout, EDA (UMAP, PCA, correlation, cell counts, area & volume,
neighbors, intensity), and linear-modeling results (volcano, effect sizes,
model fit, UpSet, variate significance).

```bash
pip install -r requirements.txt
bash run_app.sh [port]   # default port 8501
# or: streamlit run app.py
```

`streamlit_app.py` is an alias entry point for hosts that look for that
exact filename (e.g. Streamlit Community Cloud); it re-execs `app.py`,
which holds the real logic.

## Layout
- One tab per module: `0.Overview`, `1.EDA`, `4.linear_modeling`.
  (`3.viability_prediction_models` is implemented in `sections.py` but hidden
  from the tab bar for now -- see the comment in `app.py`.)
- One section per analysis type (pick it with the radio buttons; only the
  selected section loads its data).
- Every plot lets you choose plot type, X, Y, **color**, **facet** and
  (for scatter) **shape** by any column, and **subset** rows by any metadata
  column (keep or exclude values).
- The sidebar holds shared subset filters (patient, treatment, dose, class,
  target, therapeutic category, well, image mode, modality) that apply to
  every plot whose table has that column.
- Metadata column names are harmonized (`Metadata_Experiment_Treatment` and
  `Metadata_treatment` both become `treatment`; see `data_io.CANONICAL_COLUMNS`).
- "Prepare PNG (600 dpi)" saves the current figure as a PNG.

## Data
This repo ships **code only** -- the precomputed results live in an HF
Bucket, not in git (`data/` is `.gitignore`'d on purpose). `sync_data.py`
moves data between the main analysis repo, `data/`, and that bucket:

```bash
python sync_data.py upload    # main repo -> data/ -> bucket (or: just upload_data_to_hf_bucket)
python sync_data.py download  # bucket -> data/            (or: just download_data_from_hf_bucket)
```

`download` is also what the app does automatically the first time it starts
with no local `data/` -- `data_io.py` calls the same `sync_bucket()` into
`DATA_DIR` (default: `data/`), cached for the life of the process so it
only happens once. Set `HF_BUCKET` (`namespace/bucket-name`) to point at a
different bucket; a private one also needs `HF_TOKEN`. On an HF Space that
mounts the bucket as a storage volume instead (Space Settings -> Storage
Buckets), point `DATA_DIR` at that mount path and the download is skipped
entirely, since the files are already there.

| Module | Read from |
|---|---|
| 0.Overview | `data/platemaps/*` |
| 1.EDA | `data/eda/{umap,pca,correlation,cell_counts,area_vs_volume,neighbors,intensity}` |
| 3.viability_prediction_models | `data/viability_models/combined_*.parquet` |
| 4.linear_modeling | `data/linear_modeling/{models,variate_importance}` |

A section whose results aren't present shows which script, in the main
repo, produces them -- it never crashes the app.

## Files
- `run_app.sh`: local launcher.
- `requirements.txt`: runtime deps.
- `sync_data.py`, `justfile`: moves the trimmed data above between the main
  repo, `data/`, and the HF Bucket (both directions).
- `streamlit_app.py`: alias entry point; re-execs `app.py`.
- `app.py`: page, sidebar filters, tabs.
- `sections.py`: one function per analysis type.
- `plots.py`: generic plot explorer (subset / color / facet / shape controls,
  heatmaps, PNG export).
- `data_io.py`: paths, dataset registry, metadata harmonization.
- `palettes.py`: colors/orders ported from the R plotting theme, and
  `humanize_label()` for dropdown display names.
- `background.py`: the "About" text shown in each tab/section.
