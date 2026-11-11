"""Shared mean Average Precision (mAP) calculation utilities, built on copairs."""

import pathlib
from typing import Union

import numpy as np
import pandas as pd
from copairs import map
from copairs.matching import assign_reference_index

REFERENCE_COL = "reference_index"
PATIENT_COL = "Metadata_patient"


def calculate_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_Experiment_Treatment",
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
    df = assign_reference_index(
        df,
        f"{col_for_reference} == '{reference_group}'",
        reference_col=REFERENCE_COL,
        default_value=-1,
    )
    feature_cols = [
        col
        for col in df.columns
        if not col.startswith("Metadata_") and col != REFERENCE_COL
    ]
    df = df.drop(columns=[col for col in feature_cols if df[col].isna().any()])
    metadata_columns_with_ref = metadata_columns + [REFERENCE_COL]
    metadata = df[metadata_columns_with_ref]
    profiles = df.drop(columns=metadata_columns_with_ref).values

    pos_sameby = [col_for_reference, REFERENCE_COL]
    pos_diffby = []
    neg_sameby = []
    neg_diffby = [col_for_reference, REFERENCE_COL]

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


def calculate_inter_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_Experiment_Treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
) -> Union[None, pd.DataFrame]:
    """
    Calculate inter-patient mAP across all patients pooled, for every
    treatment-dose against a single reference group.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    metadata_columns : list
        List of metadata columns to exclude from profiles.
    col_for_reference : str
        Column name for treatment grouping.
    reference_group : str
        Reference treatment-dose label (e.g., "DMSO_1").
    output_path : Union[None, pathlib.Path]
        Path to save results as parquet.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with mAP results per treatment-dose across all patients.
    """
    mAP = calculate_mAP(
        df,
        metadata_columns=metadata_columns,
        col_for_reference=col_for_reference,
        reference_group=reference_group,
    )
    if mAP is None:
        print("No mAP results computed.")
        return None

    mAP[PATIENT_COL] = "all_patients"
    if output_path is not None:
        mAP.to_parquet(output_path, index=False)
    return mAP


def calculate_intra_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_Experiment_Treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
) -> Union[None, pd.DataFrame]:
    """
    Calculate per-patient mAP for every treatment-dose against a single
    reference group, with each patient's hypotheses corrected separately.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the profiles and metadata.
    metadata_columns : list
        List of metadata columns to exclude from profiles.
    col_for_reference : str
        Column name for treatment grouping.
    reference_group : str
        Reference treatment-dose label (e.g., "DMSO_1").
    output_path : Union[None, pathlib.Path]
        Path to save results as parquet.

    Returns
    ----------
    Union[None, pd.DataFrame]
        DataFrame with mAP results per treatment-dose per patient.
    """
    list_of_dfs = []
    for patient in df[PATIENT_COL].unique():
        patient_df = df.loc[df[PATIENT_COL] == patient].copy()
        if reference_group not in patient_df[col_for_reference].unique():
            continue
        mAP = calculate_mAP(
            patient_df,
            metadata_columns=metadata_columns,
            col_for_reference=col_for_reference,
            reference_group=reference_group,
        )
        if mAP is not None:
            mAP[PATIENT_COL] = patient
            list_of_dfs.append(mAP)

    if len(list_of_dfs) == 0:
        print("No mAP results computed.")
        return None

    output_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
    return output_df
