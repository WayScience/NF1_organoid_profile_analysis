#!/usr/bin/env python
# coding: utf-8

# Section 7: per-object Mean/MedianIntensity values across channels and compartments, long-format, for 16.plot_intensity_facets. Uses organoid-level tables for the whole_organoid compartment and sc-level tables for the cell/nucleus compartments. `value` is z-scored PER PATIENT (within patient x compartment x channel x stat, across all treatments) so every panel in the downstream plot is comparable on a common scale.

# In[1]:


import re

import pandas as pd
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

profiles_dir = root_dir / "data" / "profiles_3D"
results_dir = root_dir / "1.EDA" / "results" / "intensity"
results_dir.mkdir(parents=True, exist_ok=True)

KEEP_STATS = ["MeanIntensity", "MedianIntensity"]
COMPARTMENT_LABELS = {"Organoid": "whole_organoid", "Cell": "cell", "Nuclei": "nucleus"}


# In[2]:


def patient_id(patient_dir_name: str) -> str:
    """Strip the '_T<n>[...]' timepoint/acquisition suffix from a patient
    directory name, e.g. 'NF0037_T1_CQ1' -> 'NF0037'."""
    m = re.match(r"^(.*?)_T\d", patient_dir_name)
    return m.group(1) if m else patient_dir_name


def melt_intensity(df: pd.DataFrame, patient: str) -> pd.DataFrame:
    """Melt a wide organoid/sc profile table to long format, one row per
    object x compartment x channel x stat, for Mean/MedianIntensity only."""
    value_cols = []
    parsed = {}
    for col in df.columns:
        if col.startswith("Metadata_"):
            continue
        parts = col.split("_", 3)
        if len(parts) != 4 or parts[2] != "Intensity" or parts[3] not in KEEP_STATS:
            continue
        compartment = COMPARTMENT_LABELS.get(parts[0])
        if compartment is None:
            continue
        value_cols.append(col)
        parsed[col] = (compartment, parts[1], parts[3])
    if not value_cols:
        return pd.DataFrame()
    long = df.melt(
        id_vars=["Metadata_Experiment_Treatment"],
        value_vars=value_cols,
        var_name="feature",
        value_name="value",
    )
    compartment, channel, stat = zip(*long["feature"].map(parsed))
    long["compartment"] = compartment
    long["channel"] = channel
    long["stat"] = stat
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long["Metadata_patient"] = patient
    long = long.rename(columns={"Metadata_Experiment_Treatment": "Metadata_treatment"})
    return long.drop(columns=["feature"]).dropna(subset=["value"])


# In[3]:


rows = []
patient_dirs = sorted(
    p.name for p in profiles_dir.iterdir() if p.is_dir() and p.name != "all_patients"
)
for patient_dir in patient_dirs:
    patient = patient_id(patient_dir)
    normalized_dir = profiles_dir / patient_dir / "5.normalized_profiles"
    for fname in ["organoid_norm.parquet", "sc_norm.parquet"]:
        f = normalized_dir / fname
        if not f.exists():
            continue
        df = pd.read_parquet(f)
        rows.append(melt_intensity(df, patient))

intensity_values_3d = pd.concat(rows, ignore_index=True)

# z-score within patient x compartment x channel x stat, across all
# treatments/objects, so each panel of the downstream plot is on a common,
# patient-normalized scale.
group_cols = ["Metadata_patient", "compartment", "channel", "stat"]
grouped = intensity_values_3d.groupby(group_cols)["value"]
intensity_values_3d["value"] = (
    intensity_values_3d["value"] - grouped.transform("mean")
) / grouped.transform("std")
intensity_values_3d = intensity_values_3d.dropna(subset=["value"])

out_path = results_dir / "intensity_values_3D.parquet"
intensity_values_3d.to_parquet(out_path, index=False)
