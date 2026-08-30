#!/usr/bin/env python
# coding: utf-8

# This notebooks looks at and finds the inter and intra patient distances for each patient profile.
# The goal of this is to find how similar or different profiles are within and across patients.
#

# In[1]:


import pathlib
from typing import Union

import numpy as np
import pandas as pd
import scipy
import scipy.spatial.distance
import sklearn.metrics.pairwise

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")


# ## Load Data / Set Paths

# In[ ]:


path_2d_organoid = pathlib.Path(
    f"{root_dir}/data/profiles_2D/all_patients/max_projection/organoid_agg_profiles.parquet"
).resolve(strict=True)
path_2d_sc = pathlib.Path(
    f"{root_dir}/data/profiles_2D/all_patients/max_projection/sc_agg_profiles.parquet"
).resolve(strict=True)
dir_3d_agg = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/2.aggregated_profiles"
).resolve(strict=True)

# (label, slug, path) for every profile type in the 2D + 3D combinatorics
profile_types = [
    ("2D Organoid", "2D_organoid", path_2d_organoid),
    ("2D Single-cell", "2D_sc", path_2d_sc),
    (
        "3D Organoid - Handcrafted",
        "3D_organoid_handcrafted",
        dir_3d_agg / "organoid_norm_sc_agg_profiles.parquet",
    ),
    (
        "3D Organoid - DL (SAM-Med3D)",
        "3D_organoid_sammed",
        dir_3d_agg / "sammed_organoid_norm_sc_agg_profiles.parquet",
    ),
    (
        "3D Single-cell - Handcrafted",
        "3D_sc_handcrafted",
        dir_3d_agg / "sc_norm_sc_agg_profiles.parquet",
    ),
    (
        "3D Single-cell - DL (SAM-Med3D)",
        "3D_sc_sammed",
        dir_3d_agg / "sammed_sc_norm_sc_agg_profiles.parquet",
    ),
    (
        "3D Single-cell - Nucleocentric DL (SAM-Med3D)",
        "3D_sc_sammed_nucleocentric",
        dir_3d_agg / "sammed_nucleocentric_norm_sc_agg_profiles.parquet",
    ),
    (
        "3D Single-cell - Nucleocentric DL (MorphEM)",
        "3D_sc_nucleocentric_morphem",
        dir_3d_agg / "nucleocentric_morphem_norm_sc_agg_profiles.parquet",
    ),
]

dist_results_dir = pathlib.Path(
    f"{root_dir}/2.2d_vs_3d_analysis/results/distance_metrics"
).resolve()
dist_results_dir.mkdir(parents=True, exist_ok=True)


def load_and_harmonize(path: pathlib.Path) -> pd.DataFrame:
    """Load a profile parquet and harmonize 2D/3D metadata column names."""
    df = pd.read_parquet(path)
    if "Metadata_Biology_PatientTumor" in df.columns:
        df = df.rename(
            columns={
                "Metadata_Biology_PatientTumor": "Metadata_patient_tumor",
                "Metadata_Experiment_Well": "Metadata_Well",
                "Metadata_Experiment_Treatment": "Metadata_treatment",
                "Metadata_Experiment_Dose": "Metadata_dose",
            }
        )
    # Keep the full patient_tumor value: some patients (e.g. NF0014) have multiple
    # tumor samples, so stripping to bare patient would merge distinct tumors together.
    df["Metadata_patient"] = df["Metadata_patient_tumor"]
    df["Metadata_treatment_dose"] = (
        df["Metadata_treatment"]
        + "_"
        + df["Metadata_dose"].fillna(0).astype(float).astype(int).astype(str)
    )
    # Some upstream Texture features contain corrupted inf values; treat like NaN
    # so the existing NaN-handling in the distance functions covers them too.
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df


# ## Define the functions

# In[3]:


def calculate_intra_patient_distance_metrics(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_treatment",
    reference_group: str = "DMSO",
    output_path: pathlib.Path = None,
) -> Union[None, pd.DataFrame]:
    """
    Calculate intra-patient cosine distance metrics between each treatment
    and the reference group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing profiles and metadata.
    metadata_columns : list
        List of metadata columns to exclude from distance calculations.
    col_for_reference : str
        Column name for treatment grouping.
    reference_group : str
        Reference treatment label.
    output_path : pathlib.Path, optional
        Path to save results as parquet.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with distance metrics per patient per treatment.
    """
    results = []
    for patient in df["Metadata_patient"].unique():
        patient_df = df[df["Metadata_patient"] == patient]
        for drug in patient_df[col_for_reference].unique():
            if drug == reference_group:
                continue

            drug_df = patient_df.loc[patient_df[col_for_reference] == drug].copy()
            dmso_df = patient_df.loc[
                patient_df[col_for_reference] == reference_group
            ].copy()

            drug_features = drug_df.drop(columns=metadata_columns)
            dmso_features = dmso_df.drop(columns=metadata_columns)

            valid_cols = drug_features.columns[
                ~drug_features.isna().all() & ~dmso_features.isna().all()
            ]
            drug_features = drug_features[valid_cols].fillna(0)
            dmso_features = dmso_features[valid_cols].fillna(0)

            if drug_features.shape[0] == 0 or dmso_features.shape[0] == 0:
                continue

            cosine_dist = sklearn.metrics.pairwise.cosine_distances(
                dmso_features.values, drug_features.values
            ).reshape(-1)

            results.append(
                {
                    "Metadata_patient": patient,
                    col_for_reference: drug,
                    "cosine_distance_mean": cosine_dist.mean(),
                    "cosine_distance_std": cosine_dist.std(),
                }
            )

    output_df = pd.DataFrame(results)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
        return None
    return output_df


def calculate_inter_patient_distance_metrics(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_treatment",
    reference_group: str = "DMSO",
    output_path: pathlib.Path = None,
) -> Union[None, pd.DataFrame]:
    """
    Calculate inter-patient cosine distance metrics between each treatment
    and the reference group across all patients.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing profiles and metadata.
    metadata_columns : list
        List of metadata columns to exclude from distance calculations.
    col_for_reference : str
        Column name for treatment grouping.
    reference_group : str
        Reference treatment label.
    output_path : pathlib.Path, optional
        Path to save results as parquet.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with distance metrics per treatment across all patients.
    """
    results = []
    for drug in df[col_for_reference].unique():
        if drug == reference_group:
            continue

        drug_df = df.loc[df[col_for_reference] == drug].copy()
        dmso_df = df.loc[df[col_for_reference] == reference_group].copy()

        drug_features = drug_df.drop(columns=metadata_columns)
        dmso_features = dmso_df.drop(columns=metadata_columns)

        valid_cols = drug_features.columns[
            ~drug_features.isna().all() & ~dmso_features.isna().all()
        ]
        drug_features = drug_features[valid_cols].fillna(0)
        dmso_features = dmso_features[valid_cols].fillna(0)

        if drug_features.shape[0] == 0 or dmso_features.shape[0] == 0:
            continue

        cosine_dist = sklearn.metrics.pairwise.cosine_distances(
            dmso_features.values, drug_features.values
        ).reshape(-1)

        results.append(
            {
                col_for_reference: drug,
                "cosine_distance_mean": cosine_dist.mean(),
                "cosine_distance_std": cosine_dist.std(),
            }
        )

    output_df = pd.DataFrame(results)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
        return None
    return output_df


# ## Run the functions

# In[ ]:


# Inter and intra patient distances for every profile type, saved with a slug prefix
for label, slug, path in profile_types:
    print("===", label, "===")
    df = load_and_harmonize(path)
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]

    calculate_intra_patient_distance_metrics(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=dist_results_dir / f"{slug}_intra_patient_cosine_distance.parquet",
    )
    calculate_inter_patient_distance_metrics(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=dist_results_dir / f"{slug}_inter_patient_cosine_distance.parquet",
    )
