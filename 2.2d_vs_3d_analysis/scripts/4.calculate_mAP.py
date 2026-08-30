#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
from typing import Union

import numpy as np
import pandas as pd
from copairs import map
from copairs.matching import assign_reference_index
from notebook_init_utils.notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()


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

results_dir = pathlib.Path(f"{root_dir}/2.2d_vs_3d_analysis/results/mAP").resolve()
results_dir.mkdir(parents=True, exist_ok=True)


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
    # so the existing NaN-handling in calculate_mAP covers them too.
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df


# In[ ]:


# TEMPORARY - remove once ZedProfiler's Texture-feature computation is fixed
# upstream. Aggregation is already fixed (all types are 1 row/well now), but
# ZedProfiler (organoid_handcrafted, sc_handcrafted) still has astronomically
# large corrupted values in a subset of Texture features. This masks those so
# calculate_mAP's existing "drop any column with a NaN" logic excludes them;
# for every other profile type this is a no-op.
def temporary_clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]

    df[feature_cols] = df[feature_cols].mask(df[feature_cols].abs() > 1e10)

    # No-op if already 1 row per well; kept as a safety net.
    group_keys = ["Metadata_patient_tumor", "Metadata_Well"]
    agg_dict = {c: "first" for c in meta_cols if c not in group_keys}
    agg_dict.update({c: "mean" for c in feature_cols})
    return df.groupby(group_keys, as_index=False).agg(agg_dict)


# In[4]:


def calculate_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_treatment",
    reference_group: str = "DMSO",
) -> Union[None, pd.DataFrame]:
    """
    Calculate mean Average Precision (mAP) for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    metadata_columns : list
        List of metadata columns to be used for grouping.
    col_for_reference : str
        Column name to be used for reference profiles.
    reference_group : str
        The value in col_for_reference that indicates reference profiles.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with mAP results, or None if calculation fails.
    """
    reference_col = "reference_index"

    df = assign_reference_index(
        df,
        f"{col_for_reference} == '{reference_group}'",
        reference_col=reference_col,
        default_value=-1,
    )
    feature_cols = [
        col
        for col in df.columns
        if not col.startswith("Metadata_") and col != reference_col
    ]
    df = df.drop(columns=[col for col in feature_cols if df[col].isna().any()])
    metadata_columns_with_ref = metadata_columns + [reference_col]
    metadata = df[metadata_columns_with_ref]
    profiles = df.drop(columns=metadata_columns_with_ref).values

    pos_sameby = [col_for_reference, reference_col]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = [col_for_reference, reference_col]

    try:
        df_ap = map.average_precision(
            metadata, profiles, pos_sameby, pos_diffby, neg_sameby, neg_diffby
        )
        df_ap = df_ap.query(f"{col_for_reference} != '{reference_group}'")
        activity_map = map.mean_average_precision(
            df_ap, pos_sameby, null_size=1000000, threshold=0.05, seed=0
        )
        activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(
            np.log10
        )
        return activity_map
    except Exception as e:
        print(f"Error calculating mAP: {e}")
        return None


def calculate_intra_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
) -> Union[None, pd.DataFrame]:
    """
    Calculate intra-patient mAP for each treatment per patient.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    metadata_columns : list
        List of metadata columns to exclude from profiles.
    col_for_reference : str
        Column name for treatment grouping.
    reference_group : str
        Reference treatment label (e.g., "DMSO").
    output_path : Union[None, pathlib.Path]
        Path to save results as parquet.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with mAP results per patient per treatment.
    """
    list_of_dfs = []
    for patient in df["Metadata_patient"].unique():
        patient_df = df.loc[df["Metadata_patient"] == patient, :].copy()
        for drug in patient_df[col_for_reference].unique():
            if drug == reference_group:
                continue
            drug_df = patient_df.loc[patient_df[col_for_reference] == drug, :].copy()
            dmso_df = patient_df.loc[
                patient_df[col_for_reference] == reference_group, :
            ].copy()
            drug_df = pd.concat([drug_df, dmso_df], ignore_index=True)

            mAP = calculate_mAP(
                drug_df,
                metadata_columns=metadata_columns,
                col_for_reference=col_for_reference,
                reference_group=reference_group,
            )
            if mAP is not None:
                mAP["Metadata_patient"] = patient
                mAP["Metadata_treatment"] = drug
                list_of_dfs.append(mAP)

    if len(list_of_dfs) == 0:
        print("No mAP results computed.")
        return None

    output_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
    return output_df


def calculate_inter_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
) -> Union[None, pd.DataFrame]:
    """
    Calculate inter-patient mAP across all patients for each treatment.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    metadata_columns : list
        List of metadata columns to exclude from profiles.
    col_for_reference : str
        Column name for treatment grouping.
    reference_group : str
        Reference treatment label (e.g., "DMSO").
    output_path : Union[None, pathlib.Path]
        Path to save results as parquet.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with mAP results per treatment across all patients.
    """
    list_of_dfs = []
    for drug in df[col_for_reference].unique():
        if drug == reference_group:
            continue
        drug_df = df.loc[df[col_for_reference] == drug, :].copy()
        dmso_df = df.loc[df[col_for_reference] == reference_group, :].copy()
        drug_df = pd.concat([drug_df, dmso_df], ignore_index=True)

        mAP = calculate_mAP(
            drug_df,
            metadata_columns=metadata_columns,
            col_for_reference=col_for_reference,
            reference_group=reference_group,
        )
        if mAP is not None:
            mAP["Metadata_treatment"] = drug
            mAP["Metadata_patient"] = "all_patients"
            list_of_dfs.append(mAP)

    if len(list_of_dfs) == 0:
        print("No mAP results computed.")
        return None

    output_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
    return output_df


# In[ ]:


# Intra and inter patient mAP by treatment and dose, for every profile type
for label, slug, path in profile_types:
    print("===", label, "===")
    df = load_and_harmonize(path)
    df = temporary_clean_and_aggregate(
        df
    )  # TEMPORARY - remove once upstream data is fixed
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]

    calculate_intra_patient_mAP(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=results_dir / f"{slug}_intra_patient_mAP_by_dose.parquet",
    )
    calculate_inter_patient_mAP(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=results_dir / f"{slug}_inter_patient_mAP_by_dose.parquet",
    )
