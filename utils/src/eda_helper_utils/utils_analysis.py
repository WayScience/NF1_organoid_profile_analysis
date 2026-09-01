"""Shared utilities for 4.analysis: patient discovery, metadata harmonization,
and CellProfiler-style feature-name parsing for 2D and 3D profiles.
"""

import re
from typing import Optional, Tuple

import pandas as pd

PROJECTIONS = ["max_projection", "middle_slice", "middle_n_slice"]
PROJECTION_FILE_PREFIX = {
    "max_projection": "max_projected",
    "middle_slice": "middle_slice",
    "middle_n_slice": "middle_n_slice",
}

MISSING_VIABILITY_PATIENTS = {"NF0016_T1", "NF0030_T1", "NF0040_T1", "NF0037_T1_CQ1"}

HARMONIZE_2D_MAP = {
    "Metadata_treatment": "Metadata_Experiment_Treatment",
    "Metadata_dose": "Metadata_Experiment_Dose",
    "Metadata_dose_unit": "Metadata_Experiment_Unit",
    "Metadata_target": "Metadata_Experiment_Target",
    "Metadata_class": "Metadata_Experiment_Class",
    "Metadata_therapeutic_categories": "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Well": "Metadata_Experiment_Well",
}


CHANNELS = {"AGP", "Brightfield", "ER", "Hoechst", "Mito", "DNA", "NoChannel"}
CATEGORIES_2D = {
    "AreaShape",
    "Intensity",
    "Texture",
    "Granularity",
    "Correlation",
    "RadialDistribution",
    "Neighbors",
    "Location",
}
CATEGORIES_3D = {
    "AreaSizeShape",
    "Intensity",
    "Texture",
    "Granularity",
    "Correlation",
    "RadialDistribution",
    "Neighbors",
    "Location",
}


EXCLUDED_PATIENTS = {"NF0037_T1_CQ1"}


def list_patient_dirs(profiles_root) -> list:
    """Return sorted patient directory names under a profiles_2D/profiles_3D root, excluding all_patients and EXCLUDED_PATIENTS."""
    return sorted(
        p.name
        for p in profiles_root.iterdir()
        if p.is_dir() and p.name != "all_patients" and p.name not in EXCLUDED_PATIENTS
    )


def patient_timepoint(patient_dir_name: str) -> Tuple[str, str]:
    """Split a directory name like 'NF0014_T1' or 'NF0037_T1_CQ1' into (patient, timepoint)."""
    m = re.match(r"^(.*?)_T(\d.*)$", patient_dir_name)
    if not m:
        return patient_dir_name, ""
    return m.group(1), "T" + m.group(2)


def harmonize_metadata(
    df: pd.DataFrame, modality: str, patient_dir_name: str
) -> pd.DataFrame:
    """Rename 2D/3D metadata columns to a common schema and add patient/timepoint/modality columns.

    Common schema columns added: Metadata_patient, Metadata_timepoint,
    Metadata_patient_tumor (original directory name), Metadata_modality.
    """
    df = df.copy()
    rename_map = HARMONIZE_2D_MAP if modality == "2D" else HARMONIZE_3D_MAP
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    patient, timepoint = patient_timepoint(patient_dir_name)
    df["Metadata_patient"] = patient
    df["Metadata_timepoint"] = timepoint
    df["Metadata_patient_tumor"] = patient_dir_name
    df["Metadata_modality"] = modality
    return df


def column_rename_report(modality: str) -> pd.DataFrame:
    """Return a dataframe documenting the rename map for a modality, for results/metadata_column_map.csv."""
    rename_map = HARMONIZE_2D_MAP if modality == "2D" else HARMONIZE_3D_MAP
    return pd.DataFrame(
        [
            {"modality": modality, "original_column": k, "harmonized_column": v}
            for k, v in rename_map.items()
        ]
    )


def parse_feature_2d(col: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """Parse a 2D CellProfiler feature name: <Object>_<Category>_<Measurement>[_<Channel>].

    Returns (compartment/object, category, channel, measurement).
    """
    parts = col.split("_")
    obj = parts[0]
    category = None
    for i in range(1, len(parts)):
        if parts[i] in CATEGORIES_2D:
            category = parts[i]
            rest = parts[i + 1 :]
            break
    else:
        return obj, None, None, "_".join(parts[1:])
    if rest and rest[-1] in CHANNELS:
        channel = rest[-1]
        measurement = "_".join(rest[:-1])
    else:
        channel = None
        measurement = "_".join(rest)
    return obj, category, channel, measurement


def parse_feature_3d(col: str) -> Tuple[str, Optional[str], Optional[str], str]:
    """Parse a 3D CellProfiler feature name: <Object>_<Channel>_<Category>_<Measurement...>.

    Returns (compartment/object, category, channel, measurement).
    """
    parts = col.split("_")
    if len(parts) < 3:
        return parts[0], None, None, "_".join(parts[1:])
    obj = parts[0]
    channel = parts[1]
    category = None
    for i in range(2, len(parts)):
        if parts[i] in CATEGORIES_3D:
            category = parts[i]
            measurement = "_".join(parts[i + 1 :])
            break
    else:
        return obj, None, channel, "_".join(parts[2:])
    return obj, category, channel, measurement


COMPARTMENT_LABELS = {
    "Organoid": "whole_organoid",
    "Nuclei": "nucleus",
    "Cytoplasm": "cytoplasm",
    "Cells": "cell",
    "Cell": "cell",
}
