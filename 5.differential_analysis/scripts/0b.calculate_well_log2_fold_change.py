#!/usr/bin/env python
# coding: utf-8

# # Calculate well-level fold changes vs DMSO per patient
# Companion to `0.calculate_log2_fold_change`, which collapses every well and organoid/cell into a single consensus value per patient x treatment x dose. That consensus grain gives `5.calculate_mek_signatures` only one data point per patient for a single MEK drug x dose, so within-patient significance there is undefined (n=1).
#
# This notebook keeps one row per **well** instead: for each patient, treated wells are aggregated (median across their organoids/cells) and compared against that same patient's DMSO reference (median across all of that patient's DMSO rows, identical reference to `0.calculate_log2_fold_change`). Real replicate wells exist per patient x treatment x dose (2 for the MEK drugs, 8 for DMSO, 2-4 for the comparators), so a single MEK drug x dose now has n=2 real within-patient replicates instead of 1.
#
# As in `0.calculate_log2_fold_change`, the consensus/normalized profiles are already normalized and centered (many values negative), so these are differences on the normalized scale, not log2 fold changes; the `log2fc` file and column names are kept only so the naming matches the existing per-patient files.

# In[ ]:


import pathlib

import pandas as pd
from notebook_init_utils import init_notebook
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

root_dir, in_notebook = init_notebook()
progress_bar = tqdm_notebook if in_notebook else tqdm


# In[ ]:


profiles_root = pathlib.Path(root_dir, "data/profiles_3D/all_patients")
patient_ids_path = pathlib.Path(root_dir, "data/patient_IDs.txt")
results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")

compartment_dict = {
    "organoid": {
        "normalized_profile_path": pathlib.Path(
            profiles_root, "0.normalized_profiles/organoid_norm_norm_profile.parquet"
        ),
        "log2fc_path": pathlib.Path(results_dir, "organoid_log2fc.parquet"),
        "output_path": pathlib.Path(results_dir, "organoid_well_log2fc.parquet"),
    },
    "single_cell": {
        "normalized_profile_path": pathlib.Path(
            profiles_root, "0.normalized_profiles/sc_norm_norm_profile.parquet"
        ),
        "log2fc_path": pathlib.Path(results_dir, "single_cell_log2fc.parquet"),
        "output_path": pathlib.Path(results_dir, "single_cell_well_log2fc.parquet"),
    },
}

CONTROL = "DMSO"
well_group_cols = [
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Unit",
    "Metadata_Experiment_Well",
]

patient_ids = patient_ids_path.read_text().split()
print(f"{len(patient_ids)} patients")


# In[ ]:


def load_patient_plates(normalized_profile_path: pathlib.Path) -> pd.DataFrame:
    """
    One row per patient with its tumor type and plate.

    Parameters
    ----------
    normalized_profile_path : pathlib.Path
        Path to the per-object normalized profile parquet file.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by patient, with tumor type and plate ID columns.
    """
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


def well_difference(patient_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Each well's median feature value minus the median of that patient's DMSO rows.

    Parameters
    ----------
    patient_df : pd.DataFrame
        Per-object normalized profile rows for a single patient.
    feature_cols : list
        List of feature columns to aggregate and difference.

    Returns
    -------
    pd.DataFrame
        One row per treated well, with its median feature values minus the
        patient's DMSO reference.
    """
    is_control = patient_df["Metadata_Experiment_Treatment"] == CONTROL
    if not is_control.any():
        raise ValueError(f"No {CONTROL} rows found for this patient.")
    control = patient_df.loc[is_control, feature_cols].median()
    treated = patient_df.loc[~is_control]
    well_median = treated.groupby(well_group_cols, dropna=False)[feature_cols].median()
    return (well_median - control).reset_index()


# In[ ]:


for compartment, compartment_info in compartment_dict.items():
    # Use the same feature columns as the patient-level log2fc file: the
    # normalized profiles carry extra features that were dropped before the
    # consensus step the rest of 5.differential_analysis relies on.
    feature_cols = [
        col
        for col in pd.read_parquet(compartment_info["log2fc_path"]).columns
        if not col.startswith("Metadata_")
    ]
    normalized_df = pd.read_parquet(compartment_info["normalized_profile_path"])
    normalized_df = normalized_df.loc[
        normalized_df["Metadata_Biology_PatientTumor"].isin(patient_ids)
    ].dropna(subset=["Metadata_Experiment_Treatment"])
    plate_df = load_patient_plates(compartment_info["normalized_profile_path"])
    print(
        f"{compartment}: {len(feature_cols)} features, "
        f"{normalized_df.shape[0]} object rows"
    )

    well_list = []
    for patient in progress_bar(patient_ids, desc=f"Calculating {compartment}"):
        patient_df = normalized_df.loc[
            normalized_df["Metadata_Biology_PatientTumor"] == patient
        ]
        patient_well_df = well_difference(patient_df, feature_cols)
        patient_well_df.insert(0, "Metadata_Biology_PatientTumor", patient)
        patient_well_df.insert(
            1,
            "Metadata_Biology_TumorType",
            plate_df.loc[patient, "Metadata_Biology_TumorType"],
        )
        patient_well_df.insert(
            2,
            "Metadata_Experiment_PlateID",
            plate_df.loc[patient, "Metadata_Experiment_PlateID"],
        )
        well_list.append(patient_well_df)
    well_df = pd.concat(well_list, ignore_index=True)
    well_df.to_parquet(compartment_info["output_path"], index=False)

    wells_per_condition = well_df.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Dose",
        ]
    ).size()
    print(
        f"{compartment}: {well_df.shape[0]} patient x treatment x well rows; "
        f"wells per patient x treatment x dose: min={wells_per_condition.min()}, "
        f"median={wells_per_condition.median()}, max={wells_per_condition.max()}"
    )
well_df.head()
