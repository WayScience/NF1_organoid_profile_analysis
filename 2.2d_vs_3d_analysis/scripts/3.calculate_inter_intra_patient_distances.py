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
from notebook_init_utils.notebook_init_utils import init_notebook
from notebook_init_utils.result_cache import is_fresh, mark_fresh

root_dir, in_notebook = init_notebook()


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

# Each 3D aggregated profile is derived two ways -- aggregated from the
# feature-selected single cells (``from_fs``) and from the normalized single
# cells (``from_norm``) -- so both variants are scored for every profile type.
MODES = ["from_fs", "from_norm"]
MODE_LABELS = {"from_fs": "from FS", "from_norm": "from norm"}

# (label, slug, filename stem) base specs for every 3D profile type; the
# ``_from_fs`` / ``_from_norm`` suffix is appended per mode below.
base_3d_profile_types = [
    ("3D Organoid - Handcrafted", "3D_organoid_handcrafted", "organoid_norm_agg"),
    ("3D Organoid - DL (SAM-Med3D)", "3D_organoid_sammed", "sammed_organoid_norm_agg"),
    ("3D Single-cell - Handcrafted", "3D_sc_handcrafted", "sc_norm_agg"),
    ("3D Single-cell - DL (SAM-Med3D)", "3D_sc_sammed", "sammed_sc_norm_agg"),
    (
        "3D Single-cell - Nucleocentric DL (SAM-Med3D)",
        "3D_sc_sammed_nucleocentric",
        "sammed_nucleocentric_norm_agg",
    ),
    (
        "3D Single-cell - Nucleocentric DL (MorphEM)",
        "3D_sc_nucleocentric_morphem",
        "nucleocentric_morphem_norm_agg",
    ),
]

# (label, slug, path) for every profile type in the 2D + 3D combinatorics
profile_types = [
    ("2D Organoid", "2D_organoid", path_2d_organoid),
    ("2D Single-cell", "2D_sc", path_2d_sc),
]
for _label, _slug, _stem in base_3d_profile_types:
    for _mode in MODES:
        profile_types.append(
            (
                f"{_label} ({MODE_LABELS[_mode]})",
                f"{_slug}_{_mode}",
                dir_3d_agg / f"{_stem}_{_mode}.parquet",
            )
        )

# Caching: a profile type is skipped when its outputs were built from unchanged inputs and code
CACHE_STEP = "3.calculate_inter_intra_patient_distances"
CODE_PATH = pathlib.Path(
    f"{root_dir}/2.2d_vs_3d_analysis/scripts/3.calculate_inter_intra_patient_distances.py"
).resolve(strict=True)
results_root = pathlib.Path(f"{root_dir}/2.2d_vs_3d_analysis/results").resolve()

dist_results_dir = pathlib.Path(
    f"{root_dir}/2.2d_vs_3d_analysis/results/distance_metrics"
).resolve()
dist_results_dir.mkdir(parents=True, exist_ok=True)


def load_profile(path: pathlib.Path) -> pd.DataFrame:
    """Load a profile parquet."""
    return pd.read_parquet(path)


def harmonize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonize 2D/3D metadata column names and derive analysis columns.

    The 3D pipeline's naming convention (Metadata_Biology_PatientTumor,
    Metadata_Experiment_{Well,Treatment,Dose}) is the canonical one; 2D data
    is renamed up to match it here rather than the other way around.
    """
    if "Metadata_patient_tumor" in df.columns:
        df = df.rename(
            columns={
                "Metadata_patient_tumor": "Metadata_Biology_PatientTumor",
                "Metadata_Well": "Metadata_Experiment_Well",
                "Metadata_treatment": "Metadata_Experiment_Treatment",
                "Metadata_dose": "Metadata_Experiment_Dose",
            }
        )
    # Keep the full patient_tumor value: some patients (e.g. NF0014) have multiple
    # tumor samples, so stripping to bare patient would merge distinct tumors together.
    df["Metadata_patient"] = df["Metadata_Biology_PatientTumor"]
    df["Metadata_treatment_dose"] = (
        df["Metadata_Experiment_Treatment"]
        + "_"
        + df["Metadata_Experiment_Dose"].fillna(0).astype(float).astype(int).astype(str)
    )

    return df


def load_and_harmonize(path: pathlib.Path) -> pd.DataFrame:
    """Load a profile parquet and harmonize its metadata columns."""
    return harmonize_metadata(load_profile(path))


# ## Define the functions

# In[3]:


def calculate_intra_patient_distance_metrics(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_Experiment_Treatment",
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
    col_for_reference: str = "Metadata_Experiment_Treatment",
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

# In[4]:


# Inter and intra patient distances for every profile type, saved with a slug prefix
for label, slug, path in profile_types:
    intra_output_path = (
        dist_results_dir / f"{slug}_intra_patient_cosine_distance.parquet"
    )
    inter_output_path = (
        dist_results_dir / f"{slug}_inter_patient_cosine_distance.parquet"
    )
    outputs = [intra_output_path, inter_output_path]
    inputs = [path]
    if is_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs):
        print("===", label, "(cached) ===")
        continue

    print("===", label, "===")
    df = load_and_harmonize(path)

    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]

    calculate_intra_patient_distance_metrics(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=intra_output_path,
    )
    calculate_inter_patient_distance_metrics(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=inter_output_path,
    )
    mark_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs)
