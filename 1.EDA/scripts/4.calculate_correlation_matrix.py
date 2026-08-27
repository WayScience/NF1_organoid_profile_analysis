#!/usr/bin/env python
# coding: utf-8

# # Calculate correlation matrices
# 
# Computes two groups of sample-by-sample Pearson correlation matrices, each
# written as a small, consolidated set of *tidy* files rather than one file
# per matrix:
# 
# - **Group A -- `sc_agg`/`sc_consensus`, replicate/treatment-level**:
#   aggregate and consensus profiles (derived by summarizing single-cell
#   measurements up to one row per replicate or per treatment -- not
#   per-cell), for 2D (all 3 slice strategies) and 3D (4 normalization
#   variants -- see note below). Correlations are stored as a long-format
#   "pairs" table (one row per unique sample pair, upper triangle only --
#   correlation is symmetric, so the lower triangle is redundant) plus a
#   small "samples" table carrying per-sample metadata (patient/treatment/
#   dose) for annotation at plot time. **4 files total** (2D pairs+samples,
#   3D pairs+samples -- down from one file per slice-strategy or
#   normalization-variant/profile-type combination).
# - **Group B -- 3D sc_fs, single-cell (per-cell), per patient**: single-cell
#   feature-selected profiles -- one row per *cell*, unlike Group A's
#   aggregated rows -- split per patient, for the normalization variants that
#   are genuinely single-cell (see note below). Patients above
#   `MAX_CELLS_PER_PATIENT` cells are stratified-subsampled by treatment down
#   to that cap (some patients otherwise have 8,000-15,000+ cells, which made
#   correlating/clustering/rendering their matrices slow and memory-heavy).
#   Each (variant, patient) matrix is stored as **one row**, with the
#   correlation values and per-cell treatment labels as nested list columns.
#   **1 file total** (down from one file per patient).
# 
# Preprocessing (`NF0037_T1_CQ1` exclusion, `_Texture_` feature drop,
# `pycytominer.feature_select` outlier drop) matches `0.generate_umap.ipynb` /
# `2.generate_pca.ipynb`, via a shared helper used by both groups.
# 
# **A note on "organoid" normalization variants**: of the 6 3D normalization
# variants, `organoid_norm` and `sammed_organoid_norm` are excluded
# everywhere in this notebook. At the feature-selected (per-cell) stage used
# by Group B, they're genuinely **organoid-level** data (feature columns are
# `Organoid_*`, and there's no `Metadata_Object_ParentOrganoid` column
# linking rows to a parent organoid) despite living in the same directory as
# the genuinely single-cell variants. At the aggregated/consensus stage used
# by Group A, all 6 variants converge to the same replicate-level row count
# regardless of source resolution -- but `organoid_norm`/`sammed_organoid_norm`
# are still excluded there too, for consistency. Group A's 3D section
# therefore uses 4 variants: `nucleocentric_morphem_norm`,
# `sammed_nucleocentric_norm`, `sammed_sc_norm`, `sc_norm`. Group B uses 3 of
# those 4 -- `sammed_sc_norm` (6,976 features) is deferred separately at the
# per-cell resolution Group B works at, since it's far slower to correlate
# per-patient than the others there; it's not a concern at Group A's small,
# already-aggregated row counts, so it's included there.
# 
# Plotting happens separately in `5.plot_correlation_heatmaps.ipynb`.
# 
# **Output**: `1.EDA/results/correlation/`

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from pycytominer import feature_select

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# ## Parameters

# In[2]:


# Feature-selection parameters (matches 2.generate_pca.ipynb / 0.generate_umap.ipynb)
OUTLIER_CUTOFF = 100

correlation_dir = pathlib.Path(f"{root_dir}/1.EDA/results/correlation/").resolve()
correlation_dir.mkdir(parents=True, exist_ok=True)


# ## Shared preprocessing

# In[3]:


def clean_profile(df, patient_col):
    # Exclude NF0037_T1_CQ1: this is a separate analysis and should not be
    # included anywhere in this EDA (matches 0.generate_umap.ipynb /
    # 2.generate_pca.ipynb).
    if patient_col in df.columns:
        df = df[df[patient_col] != "NF0037_T1_CQ1"]

    metadata_columns = [col for col in df.columns if "Metadata_" in col]
    feature_columns = [col for col in df.columns if col not in metadata_columns]

    # Drop Texture features: prone to extreme outlier values (matches
    # 0.generate_umap.ipynb / 2.generate_pca.ipynb).
    feature_columns = [c for c in feature_columns if "_Texture_" not in c]

    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    # Drop any other features with extreme/blown-up values using
    # pycytominer's feature_select, with its magnitude-based outlier
    # operation (matches 2.generate_pca.ipynb).
    df = feature_select(
        df,
        features=feature_columns,
        operation=["drop_outliers"],
        outlier_cutoff=OUTLIER_CUTOFF,
    )
    feature_columns = [c for c in feature_columns if c in df.columns]

    df = df.dropna(subset=feature_columns, axis=0).reset_index(drop=True)
    return df, metadata_columns, feature_columns


def correlate_samples(df, feature_columns):
    # Sample-by-sample (row-by-row) Pearson correlation matrix over the
    # given feature columns.
    features_df = df[feature_columns].reset_index(drop=True)
    return features_df.T.corr().to_numpy()


# ## Group A -- 2D sc_agg / sc_consensus, replicate/treatment-level
# 
# 3 slice strategies x {agg, consensus} = 6 matrices, consolidated into a
# long-format pairs table + a samples metadata table.

# In[4]:


slice_strategies = ["max_projection", "middle_slice", "middle_n_slice"]
profile_types = ["agg", "consensus"]

pairs_rows = []
samples_rows = []

for slice_strategy in tqdm.tqdm(slice_strategies):
    for profile_type in profile_types:
        input_path = pathlib.Path(
            f"{root_dir}/data/profiles_2D/all_patients/{slice_strategy}/sc_{profile_type}_profiles.parquet"
        ).resolve(strict=True)
        df = pd.read_parquet(input_path)

        # Harmonize 2D column names to the 3D metadata convention (matches
        # 3.plot_pca.ipynb).
        df = df.rename(
            columns={
                "Metadata_treatment": "Metadata_Experiment_Treatment",
                "Metadata_patient_tumor": "Metadata_Biology_PatientTumor",
                "Metadata_dose": "Metadata_Experiment_Dose",
            }
        )

        df, metadata_columns, feature_columns = clean_profile(
            df, patient_col="Metadata_Biology_PatientTumor"
        )
        corr_mat = correlate_samples(df, feature_columns)

        n = corr_mat.shape[0]
        iu = np.triu_indices(n)
        pairs_rows.append(
            pd.DataFrame(
                {
                    "slice_strategy": slice_strategy,
                    "profile_type": profile_type,
                    "sample_i": iu[0],
                    "sample_j": iu[1],
                    "correlation": corr_mat[iu],
                }
            )
        )

        sample_metadata = df[metadata_columns].reset_index(drop=True)
        sample_metadata.insert(0, "sample_index", np.arange(n))
        sample_metadata.insert(0, "profile_type", profile_type)
        sample_metadata.insert(0, "slice_strategy", slice_strategy)
        samples_rows.append(sample_metadata)

pairs_df = pd.concat(pairs_rows, ignore_index=True)
samples_df = pd.concat(samples_rows, ignore_index=True)

pairs_path = correlation_dir / "2D_sc_correlation_pairs.parquet"
samples_path = correlation_dir / "2D_sc_correlation_samples.parquet"
pairs_df.to_parquet(pairs_path, index=False)
samples_df.to_parquet(samples_path, index=False)

print(f"Saved {pairs_path.name}: {len(pairs_df)} rows")
print(f"Saved {samples_path.name}: {len(samples_df)} rows")


# ## Group A -- 3D sc_agg / sc_consensus, replicate/treatment-level
# 
# 4 normalization variants x {agg, consensus} = 8 matrices (excludes
# `organoid_norm` / `sammed_organoid_norm`, see note above), consolidated
# into a long-format pairs table + a samples metadata table.

# In[5]:


GROUP_A_3D_VARIANTS = ["nucleocentric_morphem_norm", "sammed_nucleocentric_norm", "sammed_sc_norm", "sc_norm"]
profile_dirs_3d = {"agg": "2.aggregated_profiles", "consensus": "3.consensus_profiles"}
patient_col_3d = "Metadata_Biology_PatientTumor"

pairs_rows_3d = []
samples_rows_3d = []

for variant in tqdm.tqdm(GROUP_A_3D_VARIANTS):
    for profile_type in profile_types:
        input_path = pathlib.Path(
            f"{root_dir}/data/profiles_3D/all_patients/{profile_dirs_3d[profile_type]}/{variant}_sc_{profile_type}_profiles.parquet"
        ).resolve(strict=True)
        df = pd.read_parquet(input_path)

        df, metadata_columns, feature_columns = clean_profile(df, patient_col=patient_col_3d)
        corr_mat = correlate_samples(df, feature_columns)

        n = corr_mat.shape[0]
        iu = np.triu_indices(n)
        pairs_rows_3d.append(
            pd.DataFrame(
                {
                    "normalization_variant": variant,
                    "profile_type": profile_type,
                    "sample_i": iu[0],
                    "sample_j": iu[1],
                    "correlation": corr_mat[iu],
                }
            )
        )

        sample_metadata = df[metadata_columns].reset_index(drop=True)
        sample_metadata.insert(0, "sample_index", np.arange(n))
        sample_metadata.insert(0, "profile_type", profile_type)
        sample_metadata.insert(0, "normalization_variant", variant)
        samples_rows_3d.append(sample_metadata)

pairs_df_3d = pd.concat(pairs_rows_3d, ignore_index=True)
samples_df_3d = pd.concat(samples_rows_3d, ignore_index=True)

pairs_path_3d = correlation_dir / "3D_sc_correlation_pairs.parquet"
samples_path_3d = correlation_dir / "3D_sc_correlation_samples.parquet"
pairs_df_3d.to_parquet(pairs_path_3d, index=False)
samples_df_3d.to_parquet(samples_path_3d, index=False)

print(f"Saved {pairs_path_3d.name}: {len(pairs_df_3d)} rows")
print(f"Saved {samples_path_3d.name}: {len(samples_df_3d)} rows")


# ## Group B -- 3D sc_fs, single-cell, per patient
# 
# Single-cell normalization variants only (see note above); one row per
# (variant, patient), with the correlation matrix and per-cell treatment
# labels stored as nested list columns. Patients with more than
# `MAX_CELLS_PER_PATIENT` cells are stratified-subsampled by treatment (so
# each treatment stays proportionally represented) down to that cap --
# several patients have 8,000-15,000+ cells, and correlating/clustering/
# rendering a matrix that large is what was driving the memory and runtime
# problems worked through in `5.plot_correlation_heatmaps.ipynb`.

# In[6]:


import pyarrow as pa
import pyarrow.parquet as pq

SC_FS_VARIANTS = ["sc_norm", "nucleocentric_morphem_norm", "sammed_nucleocentric_norm"]
# sammed_sc_norm (76,445 cells x 6,976 features) is deferred separately --
# far slower to correlate than the other variants.
# organoid_norm / sammed_organoid_norm are excluded entirely: despite living
# in this same directory, they are organoid-level (Organoid_* features, no
# Metadata_Object_ParentOrganoid), not single-cell.

MAX_CELLS_PER_PATIENT = 2000
RANDOM_SEED = 0


def subsample_balanced(df, group_col, max_total, seed):
    # Stratified sample down to max_total rows, drawing proportionally from
    # each group_col value so no single group (e.g. treatment) is over- or
    # under-represented relative to its original share.
    frac = max_total / len(df)
    parts = [
        group_df.sample(frac=frac, random_state=seed)
        for _, group_df in df.groupby(group_col)
    ]
    return pd.concat(parts, ignore_index=True)


sc_fs_dir = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/1.feature_selected_profiles/"
).resolve(strict=True)
patient_col = "Metadata_Biology_PatientTumor"

# Written incrementally (one patient at a time) via ParquetWriter rather than
# accumulated into one big Python list first: even with the MAX_CELLS_PER_PATIENT
# cap keeping each individual matrix small, holding many of them in memory
# simultaneously before writing anything adds up unnecessarily.
matrices_path = correlation_dir / "3D_sc_fs_per_patient_correlation_matrices.parquet"
writer = None
n_rows_written = 0
summary_rows = []

for variant in tqdm.tqdm(SC_FS_VARIANTS):
    input_path = sc_fs_dir / f"{variant}_fs_profiles.parquet"
    df = pd.read_parquet(input_path)
    df, metadata_columns, feature_columns = clean_profile(df, patient_col=patient_col)

    for patient_id, patient_df in df.groupby(patient_col):
        original_n_cells = len(patient_df)
        if original_n_cells > MAX_CELLS_PER_PATIENT:
            patient_df = subsample_balanced(
                patient_df, "Metadata_Experiment_Treatment", MAX_CELLS_PER_PATIENT, RANDOM_SEED
            )

        corr_mat = correlate_samples(patient_df, feature_columns)
        n = corr_mat.shape[0]

        row_table = pa.table(
            {
                "variant": [variant],
                "patient": [patient_id],
                "n_samples": [n],
                # Row-major (C-order) flatten; reconstruct in R with
                # matrix(x, nrow = n_samples, byrow = TRUE).
                "correlation": [corr_mat.flatten()],
                "treatment": [patient_df["Metadata_Experiment_Treatment"].tolist()],
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(matrices_path, row_table.schema)
        writer.write_table(row_table)
        n_rows_written += 1
        summary_rows.append(
            {
                "variant": variant,
                "patient": patient_id,
                "original_n_cells": original_n_cells,
                "n_samples": n,
            }
        )

        del corr_mat, row_table

if writer is not None:
    writer.close()

print(f"Saved {matrices_path.name}: {n_rows_written} rows (variant x patient combinations)")
print(pd.DataFrame(summary_rows).to_string(index=False))

