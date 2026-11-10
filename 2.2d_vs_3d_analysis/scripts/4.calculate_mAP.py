#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pathlib
import warnings

import numpy as np
import pandas as pd
from copairs import map
from copairs.matching import assign_reference_index
from eda_helper_utils.mAP_utils import (
    PATIENT_COL,
    REFERENCE_COL,
    calculate_inter_patient_mAP,
    calculate_intra_patient_mAP,
    calculate_mAP,
)
from notebook_init_utils.notebook_init_utils import init_notebook
from tqdm.std import TqdmExperimentalWarning

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

root_dir, in_notebook = init_notebook()
if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[5]:


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
        dir_3d_agg / "organoid_norm_agg_from_norm.parquet",
    ),
    (
        "3D Organoid - DL (SAM-Med3D)",
        "3D_organoid_sammed",
        dir_3d_agg / "sammed_organoid_norm_agg_from_norm.parquet",
    ),
    (
        "3D Single-cell - Handcrafted",
        "3D_sc_handcrafted",
        dir_3d_agg / "sc_norm_agg_from_norm.parquet",
    ),
    (
        "3D Single-cell - DL (SAM-Med3D)",
        "3D_sc_sammed",
        dir_3d_agg / "sammed_sc_norm_agg_from_norm.parquet",
    ),
    (
        "3D Single-cell - Nucleocentric DL (SAM-Med3D)",
        "3D_sc_sammed_nucleocentric",
        dir_3d_agg / "sammed_nucleocentric_norm_agg_from_norm.parquet",
    ),
    (
        "3D Single-cell - Nucleocentric DL (MorphEM)",
        "3D_sc_nucleocentric_morphem",
        dir_3d_agg / "nucleocentric_morphem_norm_agg_from_norm.parquet",
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
    # Some upstream Texture features contain corrupted inf values; treat like NaN
    # so the existing NaN-handling in calculate_mAP covers them too.
    feature_cols = [c for c in df.columns if not c.startswith("Metadata_")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return df


def load_and_harmonize(path: pathlib.Path) -> pd.DataFrame:
    """Load a profile parquet and harmonize its metadata columns."""
    return harmonize_metadata(load_profile(path))


# In[6]:


# Intra and inter patient mAP by treatment and dose, for every profile type
for label, slug, path in tqdm.tqdm(
    profile_types, total=len(profile_types), desc="Profile Types"
):
    inter_patient_output_file = (
        results_dir / f"{slug}_inter_patient_mAP_by_dose.parquet"
    )
    intra_patient_output_file = (
        results_dir / f"{slug}_intra_patient_mAP_by_dose.parquet"
    )
    if inter_patient_output_file.exists() and intra_patient_output_file.exists():
        continue
    print("===", label, "===")
    df = load_and_harmonize(path)
    metadata_cols = [c for c in df.columns if c.startswith("Metadata_")]

    if not inter_patient_output_file.exists():
        calculate_inter_patient_mAP(
            df=df,
            metadata_columns=metadata_cols,
            col_for_reference="Metadata_treatment_dose",
            reference_group="DMSO_1",
            output_path=inter_patient_output_file,
        )
    if not intra_patient_output_file.exists():
        calculate_intra_patient_mAP(
            df=df,
            metadata_columns=metadata_cols,
            col_for_reference="Metadata_treatment_dose",
            reference_group="DMSO_1",
            output_path=intra_patient_output_file,
        )
