#!/usr/bin/env python
# coding: utf-8

# # Generate PCA embeddings
# 
# This notebook computes PCA embeddings for every pooled (all-patient) profile file under `data/profiles_2D/all_patients/` and `data/profiles_3D/all_patients/` whose filename contains `fs`, `agg`, or `consensus` — covering single-cell, organoid-level, aggregated, and consensus profiles across all 2D slice strategies and 3D normalization variants. 
# 
# All of these profile levels were feature-selected (`fs`) file at the same entity/normalization level — `agg`
# and `consensus` are aggregated/replicate-summarized *from* the feature-selected single-cell (or organoid-level) data, not from a separate unselected feature set.
# 
# `NF0037_T1_CQ1` rows are excluded (matching `0.generate_umap.ipynb`) — this is a separate analysis and should not be included anywhere in this EDA.
# 
# The notebook saves PCA coordinates (`PC0`...`PC{N_COMPONENTS-1}`) plus per-component explained variance ratios as parquet files under `1.EDA/results/pca/`.

# In[1]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from pycytominer import feature_select
from sklearn.decomposition import PCA

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


# PCA parameters
N_COMPONENTS = 50
OVERWRITE_EXISTING = True
OUTLIER_CUTOFF = 100


# In[3]:


all_patients_2D_dir = pathlib.Path(
    f"{root_dir}/data/profiles_2D/all_patients/"
).resolve(strict=True)
all_patients_3D_dir = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/"
).resolve(strict=True)

all_patient_2D_dirs = [x for x in all_patients_2D_dir.iterdir() if x.is_dir()]
all_patient_3D_dirs = [x for x in all_patients_3D_dir.iterdir() if x.is_dir()]
all_patient_2D_dirs = [y for x in all_patient_2D_dirs for y in x.glob("*.parquet")]
all_patient_3D_dirs = [y for x in all_patient_3D_dirs for y in x.glob("*.parquet")]
all_patients_2D_and_3D_dirs = all_patient_2D_dirs + all_patient_3D_dirs

# filter for only fs, agg, consensus
all_patients_2D_and_3D_dirs = [
    x
    for x in all_patients_2D_and_3D_dirs
    if any(substr in x.name for substr in ["fs", "agg", "consensus"])
]


# In[4]:


# generate a dict for each path
data_dict = {
    "profile_type": [],
    "profile_strategy": [],
    "dimension": [],
    "input_path": [],
    "save_path": [],
}
for profile_path in all_patients_2D_and_3D_dirs:
    profile_type = profile_path.stem
    profile_strategy = profile_path.parent.name
    dimension = profile_path.parent.parent.parent.name.replace("profiles_", "")
    data_dict["save_path"].append(
        pathlib.Path(
            f"{root_dir}/1.EDA/results/pca/{dimension}_{profile_strategy}_{profile_type}_embeddings.parquet"
        ).resolve()
    )
    data_dict["profile_type"].append(profile_type)
    data_dict["profile_strategy"].append(profile_strategy)
    data_dict["dimension"].append(dimension)
    data_dict["input_path"].append(profile_path)

feature_drop_summary = []

for i, (
    profile_type,
    profile_strategy,
    dimension,
    input_path,
    output_file_path,
) in tqdm.tqdm(
    enumerate(
        zip(
            data_dict["profile_type"],
            data_dict["profile_strategy"],
            data_dict["dimension"],
            data_dict["input_path"],
            data_dict["save_path"],
        )
    ),
    total=len(data_dict["input_path"]),
):
    explained_variance_output_file_path = (
        output_file_path.parent
        / f"{dimension}_{profile_strategy}_{profile_type}_explained_variance.parquet"
    )

    if (
        output_file_path.exists()
        and explained_variance_output_file_path.exists()
        and not OVERWRITE_EXISTING
    ):
        continue
    df = pd.read_parquet(input_path)

    # Exclude NF0037_T1_CQ1: this is a separate analysis and should not be
    # included anywhere in this EDA (matches 0.generate_umap.ipynb).
    patient_tumor_column = (
        "Metadata_Biology_PatientTumor" if dimension == "3D" else "Metadata_patient_tumor"
    )
    if patient_tumor_column in df.columns:
        df = df[df[patient_tumor_column] != "NF0037_T1_CQ1"]

    # Separate metadata columns from feature columns. Feature naming varies
    # across profile levels in this dataset (Cells/Nuclei/Cytoplasm for
    # single-cell, Organoid for organoid-level), so a plain "Metadata_"
    # prefix check is used instead of pycytominer's compartment-based
    # inference, which would fail outright on non-CellProfiler-style
    # prefixes like "Organoid".
    metadata_columns = [col for col in df.columns if "Metadata_" in col]
    feature_columns = [col for col in df.columns if col not in metadata_columns]
    n_features_total = len(feature_columns)

    # Drop Texture features: this feature family is prone to extreme
    # outlier values (see 0.generate_umap.ipynb for the same issue and its
    # root cause: near-zero-variance StandardScaler denominators upstream).
    texture_columns = [c for c in feature_columns if "_Texture_" in c]
    feature_columns = [c for c in feature_columns if "_Texture_" not in c]

    # Coerce all feature columns to numeric, treating infs as missing
    df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)

    # Drop any other features with extreme/blown-up values using
    # pycytominer's feature_select, with its magnitude-based outlier
    # operation.
    df = feature_select(
        df,
        features=feature_columns,
        operation=["drop_outliers"],
        outlier_cutoff=OUTLIER_CUTOFF,
    )
    selected_feature_columns = [c for c in feature_columns if c in df.columns]
    n_outlier_dropped = len(feature_columns) - len(selected_feature_columns)
    feature_columns = selected_feature_columns

    # Remove rows with NaN values in the feature columns
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

    # Skip files with too few rows/features to fit the requested number of components
    n_components = min(N_COMPONENTS, features_df.shape[0], features_df.shape[1])
    if n_components < 1:
        print(f"Skipping {input_path}: not enough valid data after cleaning.")
        continue

    # Initialize and fit PCA model. randomized (rather than full) SVD is
    # used since we only keep the top n_components out of up to ~2000
    # features — full SVD would compute the entire decomposition before
    # truncating, which is far more expensive for no benefit here.
    pca_model = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    features_array = features_df.to_numpy(dtype=np.float64, copy=False)
    pca_embeddings = pca_model.fit_transform(features_array)

    # Create DataFrame with PCA coordinates
    pca_columns = [f"PC{i}" for i in range(n_components)]
    pca_df = pd.DataFrame(pca_embeddings, columns=pca_columns)

    # Combine metadata with PCA coordinates
    pca_df = pd.concat([metadata_df, pca_df], axis=1)

    # Save results to parquet file
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    pca_df.to_parquet(
        output_file_path,
        index=False,
    )

    # define a new df for explained variance ratio
    explained_variance_df = pd.DataFrame(
        pca_model.explained_variance_ratio_.reshape(1, -1),
        columns=[f"PC{i}_explained_variance" for i in range(n_components)],
    )
    explained_variance_df.to_parquet(
        explained_variance_output_file_path,
        index=False,
    )

# Summarize feature/row dropping across all processed files
feature_drop_summary_df = pd.DataFrame(feature_drop_summary)
print(feature_drop_summary_df.to_string(index=False))

