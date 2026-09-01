#!/usr/bin/env python
# coding: utf-8

# Section 3 (cont'd): join derived per-organoid cell counts (script 4's
# organoid_cell_counts.parquet, 3D only) with viability via
# combined_platemaps.parquet. Reports and logs dropped patients (those in
# profiles but not in the platemap, and vice versa).

# In[1]:


import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# In[2]:


from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

results_dir = root_dir / "1.EDA" / "results"
cell_counts = pd.read_parquet(
    results_dir / "cell_counts" / "organoid_cell_counts.parquet"
)
platemaps = pd.read_parquet(
    root_dir / "data" / "viabilities" / "combined_platemaps.parquet"
)

# Mean cells per organoid, per patient x treatment x dose (3D only; across all wells)
mean_counts = (
    cell_counts.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Dose",
        ]
    )["mean_cells_per_organoid"]
    .mean()
    .reset_index()
    .rename(
        columns={
            "Metadata_Biology_PatientTumor": "Metadata_patient_tumor",
            "Metadata_Experiment_Treatment": "Metadata_treatment",
            "Metadata_Experiment_Dose": "Metadata_dose",
            "mean_cells_per_organoid": "mean_cell_count",
        }
    )
)

joined = mean_counts.merge(
    platemaps,
    left_on=["Metadata_patient_tumor", "Metadata_treatment", "Metadata_dose"],
    right_on=["patient_id", "Treatment", "Dose"],
    how="inner",
)
joined.to_parquet(results_dir / "count_viability_joined.parquet", index=False)
print(f"Wrote {results_dir / 'count_viability_joined.parquet'} ({len(joined)} rows)")

profile_patients = set(mean_counts["Metadata_patient_tumor"].unique())
platemap_patients = set(platemaps["patient_id"].unique())
dropped_from_profiles = sorted(profile_patients - platemap_patients)
dropped_from_platemaps = sorted(platemap_patients - profile_patients)

dropped_patients = pd.DataFrame(
    [
        {"patient_id": p, "reason": "missing_from_platemap"}
        for p in dropped_from_profiles
    ]
    + [
        {"patient_id": p, "reason": "missing_from_profiles"}
        for p in dropped_from_platemaps
    ],
    columns=["patient_id", "reason"],
)
dropped_patients.to_parquet(
    results_dir / "count_viability_dropped_patients.parquet", index=False
)

print("Dropped from profiles (no viability/platemap coverage):", dropped_from_profiles)
print("In platemaps but not in profiles:", dropped_from_platemaps)
print(f"Wrote {results_dir / 'count_viability_dropped_patients.parquet'}")


# In[3]:


# --- Additional metrics: total cells and total organoids per treatment,
# normalized by FOV count, for 3D and all 2D projection methods, from the
# general (not organoid-specific) cell_counts.parquet ---
raw_counts = pd.read_parquet(results_dir / "cell_counts" / "cell_counts.parquet")

# Canonical (non-sammed/non-nucleocentric) profile type per modality/projection,
# matching the sc_norm.parquet / *_sc.parquet and organoid_norm.parquet /
# *_organoid.parquet files used throughout the rest of the EDA pipeline.
# 3D has no projection method, so it gets the placeholder "none" (not None -
# groupby/pivot_table silently drop NaN-valued index levels).
CANONICAL_PROFILE_TYPES = {
    "organoid_norm_norm_profile_3D": ("organoid", "3D", "none"),
    "sc_norm_norm_profile_3D": ("cell", "3D", "none"),
    "organoid_profiles_2D_max_projection": ("organoid", "2D", "max_projection"),
    "sc_profiles_2D_max_projection": ("cell", "2D", "max_projection"),
    "organoid_profiles_2D_middle_slice": ("organoid", "2D", "middle_slice"),
    "sc_profiles_2D_middle_slice": ("cell", "2D", "middle_slice"),
    "organoid_profiles_2D_middle_n_slice": ("organoid", "2D", "middle_n_slice"),
    "sc_profiles_2D_middle_n_slice": ("cell", "2D", "middle_n_slice"),
}

raw_counts = raw_counts[
    raw_counts["Metadata_profile_type"].isin(CANONICAL_PROFILE_TYPES)
].copy()
kind_mod_proj = raw_counts["Metadata_profile_type"].map(CANONICAL_PROFILE_TYPES)
raw_counts["kind"] = kind_mod_proj.apply(lambda x: x[0])
raw_counts["modality"] = kind_mod_proj.apply(lambda x: x[1])
raw_counts["projection"] = kind_mod_proj.apply(lambda x: x[2])

# Total cells / total FOVs per treatment (not a mean of per-row ratios, which
# would be skewed by the exact-duplicate rows notebook 7's FOV-lookup merge
# introduces for 2D profile types).
total_norm_counts = (
    raw_counts.groupby(
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Dose",
            "modality",
            "projection",
            "kind",
        ]
    )
    .agg(
        total_cells=("Metadata_n_cells", "sum"),
        total_fovs=("Metadata_n_fovs", "sum"),
    )
    .reset_index()
)
total_norm_counts["total_count_norm"] = (
    total_norm_counts["total_cells"] / total_norm_counts["total_fovs"]
)

# pivot "kind" (cell vs. organoid) into separate metric columns
pivoted = total_norm_counts.pivot_table(
    index=[
        "Metadata_Biology_PatientTumor",
        "Metadata_Experiment_Treatment",
        "Metadata_Experiment_Dose",
        "modality",
        "projection",
    ],
    columns="kind",
    values="total_count_norm",
).reset_index()
pivoted.columns.name = None
pivoted = pivoted.rename(
    columns={
        "Metadata_Biology_PatientTumor": "Metadata_patient_tumor",
        "Metadata_Experiment_Treatment": "Metadata_treatment",
        "Metadata_Experiment_Dose": "Metadata_dose",
        "cell": "total_cell_count_norm",
        "organoid": "total_organoid_count_norm",
    }
)

joined_norm = pivoted.merge(
    platemaps,
    left_on=["Metadata_patient_tumor", "Metadata_treatment", "Metadata_dose"],
    right_on=["patient_id", "Treatment", "Dose"],
    how="inner",
)
joined_norm.to_parquet(results_dir / "count_norm_viability_joined.parquet", index=False)
print(
    f"Wrote {results_dir / 'count_norm_viability_joined.parquet'} ({len(joined_norm)} rows)"
)
print(joined_norm["modality"].value_counts().to_dict())
