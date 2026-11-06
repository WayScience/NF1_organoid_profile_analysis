#!/usr/bin/env python
# coding: utf-8

# In[1]:


import logging
import pathlib
from typing import Union

import numpy as np
import pandas as pd
from copairs import map
from copairs.matching import assign_reference_index
from notebook_init_utils.notebook_init_utils import init_notebook

logging.getLogger("copairs").setLevel(logging.WARNING)
root_dir, in_notebook = init_notebook()


# In[2]:


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


# In[3]:


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
    -------
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
            metadata,
            profiles,
            pos_sameby,
            pos_diffby,
            neg_sameby,
            neg_diffby,
            progress_bar=False,
        )
        df_ap = df_ap.query(f"{col_for_reference} != '{reference_group}'")
        activity_map = map.mean_average_precision(
            df_ap,
            pos_sameby,
            null_size=1000000,
            threshold=0.05,
            seed=0,
            progress_bar=False,
        )
        activity_map["-log10(p-value)"] = -activity_map["corrected_p_value"].apply(
            np.log10
        )
        return activity_map
    except Exception as e:
        group_desc = f"{col_for_reference}={df[col_for_reference].unique().tolist()}"
        if "Metadata_patient" in df.columns:
            group_desc += f", patient={df['Metadata_patient'].unique().tolist()}"
        print(f"Error calculating mAP for group ({group_desc}): {e}")
        return None


def _unit_cache_path(cache_dir: pathlib.Path, *key_parts: str) -> pathlib.Path:
    """Build a checkpoint path for one unit of mAP work (e.g. one patient/drug pair)."""
    safe_parts = [str(part).replace("/", "_") for part in key_parts]
    return cache_dir / ("__".join(safe_parts) + ".parquet")


def _load_or_compute_unit(
    cache_path: Union[None, pathlib.Path], compute_fn
) -> Union[None, pd.DataFrame]:
    """Return a cached unit result if present, else compute and checkpoint it.

    Checkpointing each unit (rather than only the final concatenated output)
    lets a run resume from where it left off if the process is pre-empted
    (e.g. killed by a cluster scheduler) partway through a profile type,
    instead of redoing every already-finished patient/drug pair.
    """
    if cache_path is not None and cache_path.exists():
        return pd.read_parquet(cache_path)
    result = compute_fn()
    if cache_path is not None and result is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
    return result


def calculate_intra_patient_mAP(
    df: pd.DataFrame,
    metadata_columns: list,
    col_for_reference: str = "Metadata_Experiment_Treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
    cache_dir: Union[None, pathlib.Path] = None,
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
        Path to save results as parquet. If this file already exists, it is
        returned as-is without recomputing anything.
    cache_dir : Union[None, pathlib.Path]
        Directory used to checkpoint each patient/drug pair's result, so a
        pre-empted run can resume without recomputing finished pairs.

    Returns
    -------
    Union[None, pd.DataFrame]
        DataFrame with mAP results per patient per treatment.
    """
    if output_path is not None and output_path.exists():
        print(f"Using cached result: {output_path.name}")
        return pd.read_parquet(output_path)

    list_of_dfs = []
    for patient in df["Metadata_patient"].unique():
        patient_df = df.loc[df["Metadata_patient"] == patient, :].copy()
        for drug in patient_df[col_for_reference].unique():
            if drug == reference_group:
                continue

            def _compute(
                patient=patient, drug=drug, patient_df=patient_df
            ) -> Union[None, pd.DataFrame]:
                drug_df = patient_df.loc[
                    patient_df[col_for_reference] == drug, :
                ].copy()
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
                return mAP

            unit_cache_path = (
                _unit_cache_path(cache_dir, patient, drug)
                if cache_dir is not None
                else None
            )
            mAP = _load_or_compute_unit(unit_cache_path, _compute)
            if mAP is not None:
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
    col_for_reference: str = "Metadata_Experiment_Treatment",
    reference_group: str = "DMSO",
    output_path: Union[None, pathlib.Path] = None,
    cache_dir: Union[None, pathlib.Path] = None,
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
        Path to save results as parquet. If this file already exists, it is
        returned as-is without recomputing anything.
    cache_dir : Union[None, pathlib.Path]
        Directory used to checkpoint each drug's result, so a pre-empted run
        can resume without recomputing finished drugs.

    Returns
    -------
    Union[None, pd.DataFrame]
        DataFrame with mAP results per treatment across all patients.
    """
    if output_path is not None and output_path.exists():
        print(f"Using cached result: {output_path.name}")
        return pd.read_parquet(output_path)

    list_of_dfs = []
    for drug in df[col_for_reference].unique():
        if drug == reference_group:
            continue

        def _compute(drug=drug) -> Union[None, pd.DataFrame]:
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
            return mAP

        unit_cache_path = (
            _unit_cache_path(cache_dir, drug) if cache_dir is not None else None
        )
        mAP = _load_or_compute_unit(unit_cache_path, _compute)
        if mAP is not None:
            list_of_dfs.append(mAP)

    if len(list_of_dfs) == 0:
        print("No mAP results computed.")
        return None

    output_df = pd.concat(list_of_dfs, ignore_index=True)
    if output_path is not None:
        output_df.to_parquet(output_path, index=False)
    return output_df


# ## Calculate mAP for every profile type
#
# This takes a while to run, so it is not executed within the notebook.

# In[4]:


# Intra and inter patient mAP by treatment and dose, for every profile type
for label, slug, path in profile_types:
    intra_output_path = results_dir / f"{slug}_intra_patient_mAP_by_dose.parquet"
    inter_output_path = results_dir / f"{slug}_inter_patient_mAP_by_dose.parquet"

    # Skip profile types that already finished on a prior (possibly pre-empted) run.
    if intra_output_path.exists() and inter_output_path.exists():
        print("===", label, "(cached) ===")
        continue

    print("===", label, "===")
    df = load_and_harmonize(path)
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]
    profile_cache_dir = results_dir / "cache" / slug

    calculate_intra_patient_mAP(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=intra_output_path,
        cache_dir=profile_cache_dir / "intra",
    )
    calculate_inter_patient_mAP(
        df,
        metadata_columns=metadata_cols,
        col_for_reference="Metadata_treatment_dose",
        reference_group="DMSO_1",
        output_path=inter_output_path,
        cache_dir=profile_cache_dir / "inter",
    )
