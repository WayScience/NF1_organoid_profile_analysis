#!/usr/bin/env python
# coding: utf-8

# # Calculate correlation matrices
# 
# This notebook computes sample-by-sample Pearson correlation matrices for every aggregated/consensus profile (2D and 3D, all slice strategies and normalization variants).
# 
# Preprocessing matches `0.generate_umap.ipynb` / `2.generate_pca.ipynb` so the same samples/features feed every EDA analysis:
# 
# - `NF0037_T1_CQ1` is excluded (separate analysis, not part of this EDA).
# - `_Texture_` features are dropped (prone to extreme outlier values).
# - Remaining features go through `pycytominer.feature_select`'s magnitude-based `drop_outliers` operation (`outlier_cutoff=100`).
# 
# Single-cell/organoid-level (non-aggregated) profiles are excluded: with 20k-77k rows each, a row-by-row correlation matrix would blow up to tens of GB per file.
# 
# Plotting happens separately in `5.plot_correlation_heatmaps.ipynb`.
# 
# **Output**: `1.EDA/results/correlation/` — one correlation-matrix parquet
# file per profile.

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
OVERWRITE_EXISTING = True


# ## Discover input profiles

# In[3]:


all_patients_2D_dir = pathlib.Path(
    f"{root_dir}/data/profiles_2D/all_patients/"
).resolve(strict=True)
all_patients_3D_dir = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/"
).resolve(strict=True)

all_patient_2D_files = [x for x in all_patients_2D_dir.iterdir() if x.is_dir()]
all_patient_3D_files = [x for x in all_patients_3D_dir.iterdir() if x.is_dir()]
all_patient_2D_files = [y for x in all_patient_2D_files for y in x.glob("*.parquet")]
all_patient_3D_files = [y for x in all_patient_3D_files for y in x.glob("*.parquet")]
all_patients_2D_and_3D_files = all_patient_2D_files + all_patient_3D_files

# Only aggregated / consensus profiles: single-cell / organoid-level files
# (20k-77k rows each) would blow a row-by-row correlation matrix up to tens
# of GB per file.
all_patients_2D_and_3D_files = [
    x
    for x in all_patients_2D_and_3D_files
    if x.stem.endswith("agg_profiles") or x.stem.endswith("consensus_profiles")
]

data_dict = {
    "profile_type": [],
    "profile_strategy": [],
    "dimension": [],
    "input_path": [],
    "save_path": [],
}
for profile_path in all_patients_2D_and_3D_files:
    profile_type = profile_path.stem
    profile_strategy = profile_path.parent.name
    dimension = profile_path.parent.parent.parent.name.replace("profiles_", "")
    data_dict["save_path"].append(
        pathlib.Path(
            f"{root_dir}/1.EDA/results/correlation/{dimension}_{profile_strategy}_{profile_type}.parquet"
        ).resolve()
    )
    data_dict["profile_type"].append(profile_type)
    data_dict["profile_strategy"].append(profile_strategy)
    data_dict["dimension"].append(dimension)
    data_dict["input_path"].append(profile_path)

len(data_dict["input_path"])


# ## Calculate correlation matrices

# In[4]:


feature_drop_summary = []

for profile_type, profile_strategy, dimension, input_path, output_file_path in tqdm.tqdm(
    zip(
        data_dict["profile_type"],
        data_dict["profile_strategy"],
        data_dict["dimension"],
        data_dict["input_path"],
        data_dict["save_path"],
    ),
    total=len(data_dict["input_path"]),
):
    if output_file_path.exists() and not OVERWRITE_EXISTING:
        continue
    df = pd.read_parquet(input_path)

    # Exclude NF0037_T1_CQ1: this is a separate analysis and should not be
    # included anywhere in this EDA (matches 0.generate_umap.ipynb /
    # 2.generate_pca.ipynb).
    patient_tumor_column = (
        "Metadata_Biology_PatientTumor" if dimension == "3D" else "Metadata_patient_tumor"
    )
    if patient_tumor_column in df.columns:
        df = df[df[patient_tumor_column] != "NF0037_T1_CQ1"]

    # Separate metadata columns from feature columns.
    metadata_columns = [col for col in df.columns if "Metadata_" in col]
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    n_features_total = len(feature_columns)

    # Drop Texture features: prone to extreme outlier values (matches
    # 0.generate_umap.ipynb / 2.generate_pca.ipynb).
    texture_columns = [c for c in feature_columns if "_Texture_" in c]
    feature_columns = [c for c in feature_columns if "_Texture_" not in c]

    # Coerce all feature columns to numeric, treating infs as missing
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
    selected_feature_columns = [c for c in feature_columns if c in df.columns]
    n_outlier_dropped = len(feature_columns) - len(selected_feature_columns)
    feature_columns = selected_feature_columns

    # Remove rows with NaN values in the remaining feature columns
    df = df.dropna(subset=feature_columns, axis=0)

    feature_drop_summary.append(
        {
            "dimension": dimension,
            "profile_strategy": profile_strategy,
            "profile_type": profile_type,
            "total": n_features_total,
            "texture_dropped": len(texture_columns),
            "outlier_dropped": n_outlier_dropped,
            "remaining": len(feature_columns),
            "rows_remaining": len(df),
        }
    )

    metadata_df = df[metadata_columns].reset_index(drop=True)
    features_df = df[feature_columns].reset_index(drop=True)

    # Sample-by-sample (row-by-row) Pearson correlation matrix over the
    # cleaned/selected feature columns.
    corr_df = features_df.T.corr()
    corr_df.columns = [f"Sample_{i}" for i in range(len(corr_df.columns))]
    corr_df = pd.concat([metadata_df, corr_df.reset_index(drop=True)], axis=1)

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_parquet(output_file_path, index=False)

feature_drop_summary_df = pd.DataFrame(feature_drop_summary)
print(feature_drop_summary_df.to_string(index=False))

