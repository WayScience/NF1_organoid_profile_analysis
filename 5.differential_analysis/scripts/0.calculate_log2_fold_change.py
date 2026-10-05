#!/usr/bin/env python
# coding: utf-8

# # Calculate fold changes vs DMSO per patient
# For each patient, calculate the morphology difference from DMSO and the viability log2
# fold change for every treatment relative to that patient's DMSO control.
#
# **Morphology**
# 1. Load the consensus profiles from
# `data/profiles_3D/all_patients/3.consensus_profiles`, one row per patient x treatment.
# Objects flagged by QC are removed upstream, in the normalization step.
# 2. Keep the patients listed in `data/patient_IDs.txt`.
# 3. Reference = median of that patient's DMSO rows, per feature.
# 4. Difference = `treatment - DMSO reference` per feature, for every treatment of each
# patient.
#
# The consensus profiles are normalized and centered: many of the values are negative, so
# a log2 ratio is not defined for them. The morphology values are therefore differences
# on the normalized scale, not log2 fold changes. The `log2fc` file and column names are
# kept so that `1.plot_log2_fold_change` still reads them.
#
# The consensus profiles have no plate or tumor type, so those come from
# `0.normalized_profiles`. Each patient is imaged on exactly one plate, so the DMSO
# reference always comes from the same plate as the treatment; this is checked below.
#
# **Viability**
#
# log2 fold change = `log2(treatment viability / DMSO viability)` per patient.

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

root_dir, in_notebook = init_notebook()
progress_bar = tqdm_notebook if in_notebook else tqdm


# In[2]:


profiles_root = pathlib.Path(root_dir, "data/profiles_3D/all_patients")
patient_ids_path = pathlib.Path(root_dir, "data/patient_IDs.txt")
viability_path = pathlib.Path(root_dir, "data/viabilities/combined_platemaps.parquet")
results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")
results_dir.mkdir(parents=True, exist_ok=True)

compartment_dict = {
    "organoid": {
        "profile_path": pathlib.Path(
            profiles_root,
            "3.consensus_profiles/organoid_norm_sc_consensus_profiles.parquet",
        ),
        "normalized_profile_path": pathlib.Path(
            profiles_root, "0.normalized_profiles/organoid_norm_norm_profile.parquet"
        ),
        "output_profile_path": pathlib.Path(results_dir, "organoid_log2fc.parquet"),
    },
    "single_cell": {
        "profile_path": pathlib.Path(
            profiles_root, "3.consensus_profiles/sc_norm_sc_consensus_profiles.parquet"
        ),
        "normalized_profile_path": pathlib.Path(
            profiles_root, "0.normalized_profiles/sc_norm_norm_profile.parquet"
        ),
        "output_profile_path": pathlib.Path(results_dir, "single_cell_log2fc.parquet"),
    },
}
viability_output_path = pathlib.Path(results_dir, "viability_log2fc.parquet")
summary_output_path = pathlib.Path(results_dir, "log2fc_summary.parquet")

CONTROL = "DMSO"
treatment_cols = [
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Unit",
]

patient_ids = patient_ids_path.read_text().split()
print(f"{len(patient_ids)} patients")


# ## Morphology

# In[3]:


def load_patient_plates(normalized_profile_path: pathlib.Path) -> pd.DataFrame:
    """One row per patient with its tumor type and plate."""
    plate_df = pd.read_parquet(
        normalized_profile_path,
        columns=[
            "Metadata_Biology_PatientTumor",
            "Metadata_Biology_TumorType",
            "Metadata_Experiment_PlateID",
        ],
    )
    # The DMSO reference must come from the same plate as the treatment.
    n_plates = plate_df.groupby("Metadata_Biology_PatientTumor")[
        "Metadata_Experiment_PlateID"
    ].nunique()
    if (n_plates > 1).any():
        raise ValueError("A patient is spread over more than one plate.")
    # Some rows have a missing tumor type, so take the first value that is present.
    return plate_df.groupby("Metadata_Biology_PatientTumor").first()


def calculate_difference(patient_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Each treatment minus the median of that patient's DMSO rows."""
    is_control = patient_df["Metadata_Experiment_Treatment"] == CONTROL
    if not is_control.any():
        raise ValueError(f"No {CONTROL} rows found for this patient.")
    control = patient_df.loc[is_control, feature_cols].median()
    treated = patient_df.loc[~is_control]
    difference = treated[feature_cols] - control
    return pd.concat(
        [treated.drop(columns=feature_cols), difference], axis=1
    ).reset_index(drop=True)


# In[4]:


morphology_delta_dict = {}
for compartment, compartment_info in compartment_dict.items():
    consensus_df = pd.read_parquet(compartment_info["profile_path"])
    consensus_df = consensus_df.loc[
        consensus_df["Metadata_Biology_PatientTumor"].isin(patient_ids)
    ].dropna(subset=["Metadata_Experiment_Treatment"])
    plate_df = load_patient_plates(compartment_info["normalized_profile_path"])
    feature_cols = [
        col for col in consensus_df.columns if not col.startswith("Metadata_")
    ]
    print(
        f"{compartment}: {len(feature_cols)} features, "
        f"{consensus_df.shape[0]} patient x treatment rows"
    )

    delta_list = []
    for patient in progress_bar(patient_ids, desc=f"Calculating {compartment}"):
        patient_df = consensus_df.loc[
            consensus_df["Metadata_Biology_PatientTumor"] == patient
        ]
        patient_delta_df = calculate_difference(patient_df, feature_cols)
        patient_delta_df.insert(
            0,
            "Metadata_Biology_TumorType",
            plate_df.loc[patient, "Metadata_Biology_TumorType"],
        )
        patient_delta_df.insert(
            0,
            "Metadata_Experiment_PlateID",
            plate_df.loc[patient, "Metadata_Experiment_PlateID"],
        )
        delta_list.append(patient_delta_df)
    delta_df = pd.concat(delta_list, ignore_index=True)
    delta_df.to_parquet(compartment_info["output_profile_path"], index=False)
    morphology_delta_dict[compartment] = delta_df
    print(f"{compartment}: {delta_df.shape[0]} patient x treatment rows")
morphology_delta_dict["single_cell"].head()


# ## Viability

# In[5]:


viability_df = pd.read_parquet(viability_path).rename(
    columns={
        "Treatment": "Metadata_Experiment_Treatment",
        "Dose": "Metadata_Experiment_Dose",
        "Unit": "Metadata_Experiment_Unit",
        "Metadata_Viability_percentage": "Metadata_Viability_Percentage",
    }
)
viability_df["Metadata_Experiment_Dose"] = viability_df[
    "Metadata_Experiment_Dose"
].astype(float)
# The platemap repeats each patient x treatment value on every well row.
viability_df = viability_df.drop_duplicates(
    ["Metadata_Biology_PatientTumor", *treatment_cols]
)[["Metadata_Biology_PatientTumor", *treatment_cols, "Metadata_Viability_Percentage"]]

control_viability = (
    viability_df.loc[viability_df["Metadata_Experiment_Treatment"] == CONTROL]
    .set_index("Metadata_Biology_PatientTumor")["Metadata_Viability_Percentage"]
    .rename("Metadata_DMSO_Viability_Percentage")
)
if control_viability.index.duplicated().any():
    raise ValueError(f"A patient has more than one {CONTROL} viability value.")
viability_log2fc_df = viability_df.loc[
    viability_df["Metadata_Experiment_Treatment"] != CONTROL
].join(control_viability, on="Metadata_Biology_PatientTumor")
viability_log2fc_df["viability_log2fc"] = np.log2(
    viability_log2fc_df["Metadata_Viability_Percentage"]
    / viability_log2fc_df["Metadata_DMSO_Viability_Percentage"]
)
viability_log2fc_df = viability_log2fc_df.reset_index(drop=True)
viability_log2fc_df.to_parquet(viability_output_path, index=False)
print(
    f"viability: {viability_log2fc_df['Metadata_Biology_PatientTumor'].nunique()} "
    f"patients x {viability_log2fc_df.groupby(treatment_cols).ngroups} treatments"
)
viability_log2fc_df.head()


# ## Combined summary
# One row per patient x treatment, with the mean absolute morphology difference
# from DMSO across features per compartment and the viability log2 fold change.
# The absolute value is used because signed differences in different directions cancel.
# The `*_mean_abs_log2fc` column names are kept for the plotting notebook; the
# morphology values are differences on the normalized scale, not log2 fold changes.

# In[6]:


key_cols = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Biology_TumorType",
    *treatment_cols,
]
summary_df = None
for compartment, delta_df in morphology_delta_dict.items():
    feature_cols = [col for col in delta_df.columns if not col.startswith("Metadata_")]
    compartment_summary = delta_df[key_cols].copy()
    compartment_summary[f"{compartment}_mean_abs_log2fc"] = (
        delta_df[feature_cols].abs().mean(axis=1)
    )
    summary_df = (
        compartment_summary
        if summary_df is None
        else summary_df.merge(compartment_summary, on=key_cols, how="outer")
    )
summary_df = summary_df.merge(
    viability_log2fc_df[
        ["Metadata_Biology_PatientTumor", *treatment_cols, "viability_log2fc"]
    ],
    on=["Metadata_Biology_PatientTumor", *treatment_cols],
    how="left",
)
summary_df.to_parquet(summary_output_path, index=False)
print(
    f"{summary_df.shape[0]} patient x treatment rows; "
    f"{summary_df['viability_log2fc'].notna().sum()} with viability"
)
summary_df.head()
