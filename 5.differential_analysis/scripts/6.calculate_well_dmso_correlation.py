#!/usr/bin/env python
# coding: utf-8

# # Calculate well correlation to DMSO per plate
# For every well on every plate, the Pearson correlation of its morphology profile to that plate's DMSO profile.
#
# The normalized profiles are centered on each plate's DMSO, so the DMSO profile is ~0 there and correlations to it are unstable.
# Instead:
# 1. Load the raw QC profiles, drop QC-flagged objects (same flags as the normalization step), and take the median per well.
# 2. Standardize each feature per plate across all wells (mean 0, SD 1), so every feature contributes equally.
# 3. DMSO reference = median of the plate's DMSO wells; each DMSO well is compared to the median of the *other* DMSO wells (leave-one-out).
# 4. Pearson correlation of each well to its DMSO reference.

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


profiles_root = pathlib.Path(root_dir, "data/profiles_3D")
patient_ids_path = pathlib.Path(root_dir, "data/patient_IDs.txt")
results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")
results_dir.mkdir(parents=True, exist_ok=True)
output_path = pathlib.Path(results_dir, "well_dmso_correlation.parquet")

compartment_files = {
    "organoid": "organoid_flagged_outliers.parquet",
    "single_cell": "sc_flagged_outliers.parquet",
}
CONTROL = "DMSO"
OUTER_ROWS = ["B", "G"]
OUTER_COLUMNS = [2, 11]
# Missing parent organoid is kept, the same as in the normalization step.
kept_qc_flags = ["Metadata_cqc_missing_parent_organoid"]
well_cols = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Biology_TumorType",
    "Metadata_Experiment_Well",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Unit",
]

patient_ids = patient_ids_path.read_text().split()


# In[3]:


def load_well_medians(patient: str, input_file_name: str) -> tuple[pd.DataFrame, list]:
    """Raw QC profiles for one patient, QC-flagged objects removed, median per well."""
    df = pd.read_parquet(
        pathlib.Path(profiles_root, patient, "4.qc_profiles", input_file_name)
    )
    qc_flags = [
        col
        for col in df.columns
        if col.startswith("Metadata_cqc_") and col not in kept_qc_flags
    ]
    df = df.loc[~df[qc_flags].any(axis=1)].dropna(
        subset=["Metadata_Experiment_Well", "Metadata_Experiment_Treatment"]
    )
    feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]
    # Some features are nullable Float64; cast so numpy sees plain floats.
    wells = (
        df.groupby(well_cols, dropna=False)[feature_cols]
        .median()
        .astype("float64")
        .reset_index()
    )
    return wells, feature_cols


def correlation_to_dmso(wells: pd.DataFrame, feature_cols: list) -> pd.Series:
    """Pearson correlation of each well to the plate DMSO median (leave-one-out for DMSO)."""
    values = wells[feature_cols]
    values = values.loc[:, values.notna().all() & (values.std() > 0)]
    values = ((values - values.mean()) / values.std()).to_numpy()
    is_control = (wells["Metadata_Experiment_Treatment"] == CONTROL).to_numpy()
    control_idx = np.flatnonzero(is_control)
    reference = np.median(values[is_control], axis=0)
    correlations = np.empty(len(wells))
    for i in range(len(wells)):
        if is_control[i]:
            others = control_idx[control_idx != i]
            well_reference = np.median(values[others], axis=0)
        else:
            well_reference = reference
        correlations[i] = np.corrcoef(values[i], well_reference)[0, 1]
    return pd.Series(correlations, index=wells.index)


# In[4]:


correlation_list = []
for compartment, input_file_name in compartment_files.items():
    for patient in tqdm(patient_ids, desc=compartment):
        wells, feature_cols = load_well_medians(patient, input_file_name)
        wells = wells[well_cols].assign(
            compartment=compartment,
            correlation_to_dmso=correlation_to_dmso(wells, feature_cols),
        )
        correlation_list.append(wells)

correlation_df = pd.concat(correlation_list, ignore_index=True)
correlation_df["Metadata_Plate_Row"] = correlation_df["Metadata_Experiment_Well"].str[0]
correlation_df["Metadata_Plate_Column"] = (
    correlation_df["Metadata_Experiment_Well"].str[1:].astype(int)
)
correlation_df["Metadata_Plate_Position"] = np.where(
    correlation_df["Metadata_Plate_Row"].isin(OUTER_ROWS)
    | correlation_df["Metadata_Plate_Column"].isin(OUTER_COLUMNS),
    "outer",
    "inner",
)
correlation_df.to_parquet(output_path, index=False)
print(
    correlation_df.groupby(
        ["compartment", correlation_df["Metadata_Experiment_Treatment"] == CONTROL]
    )["correlation_to_dmso"]
    .describe()[["count", "mean", "25%", "50%", "75%"]]
    .rename_axis(["compartment", "is_dmso"])
    .round(3)
    .to_string()
)
