#!/usr/bin/env python
# coding: utf-8

# Raw (pre-normalization) area (2D, max_projection only) vs. volume (3D), at
# both organoid and single-cell level. Reads directly from each modality's
# pre-normalization stage (2D: 4.annotated, 3D: 4.qc_profiles) rather than
# the z-scored 5.normalized stage, since area and volume come from separate
# pipelines/segmentations with different normalization behavior (area was
# z-scored per patient upstream, volume was not) -- plotting raw values
# sidesteps that comparability problem entirely. Writes raw per-record
# area/volume tables (patient_tumor x treatment x dose, no aggregation);
# script 18 does the plotting.

# In[1]:


import warnings

import pandas as pd

warnings.filterwarnings("ignore")


# In[2]:


from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

results_dir = root_dir / "1.EDA" / "results" / "area_vs_volume"
results_dir.mkdir(parents=True, exist_ok=True)

COL_2D = {"organoid": "Organoid_AreaShape_Area", "cell": "Cells_AreaShape_Area"}
COL_3D = {
    "organoid": "Organoid_NoChannel_AreaSizeShape_Volume",
    "cell": "Cell_NoChannel_AreaSizeShape_Volume",
}
FILE_2D = {
    "organoid": "max_projected_organoid.parquet",
    "cell": "max_projected_sc.parquet",
}
FILE_3D = {
    "organoid": "organoid_flagged_outliers.parquet",
    "cell": "sc_flagged_outliers.parquet",
}


def list_patient_dirs(base_dir):
    return sorted(
        p.name for p in base_dir.iterdir() if p.is_dir() and p.name != "all_patients"
    )


patients_2d = list_patient_dirs(root_dir / "data" / "profiles_2D")
patients_3d = list_patient_dirs(root_dir / "data" / "profiles_3D")

# --- 2D area, raw, max_projection only (raw values not available for the
# other 2 projection methods without extra QC work, and moving away from
# comparing projection methods anyway) ---
for kind, area_col in COL_2D.items():
    rows = []
    for patient in patients_2d:
        f = root_dir / "data" / "profiles_2D" / patient / "4.annotated" / FILE_2D[kind]
        if not f.exists():
            continue
        df = pd.read_parquet(
            f, columns=["Metadata_treatment", "Metadata_dose", area_col]
        )
        g = df.rename(columns={area_col: "area"})
        g["Metadata_patient_tumor"] = patient
        rows.append(g)
    area_df = pd.concat(rows, ignore_index=True)
    area_df.to_parquet(results_dir / f"area_2D_{kind}_raw.parquet", index=False)

# --- 3D volume, raw ---
for kind, volume_col in COL_3D.items():
    rows = []
    for patient in patients_3d:
        f = (
            root_dir
            / "data"
            / "profiles_3D"
            / patient
            / "4.qc_profiles"
            / FILE_3D[kind]
        )
        if not f.exists():
            continue
        df = pd.read_parquet(
            f,
            columns=[
                "Metadata_Biology_PatientTumor",
                "Metadata_Experiment_Treatment",
                "Metadata_Experiment_Dose",
                volume_col,
            ],
        )
        g = df.rename(
            columns={
                volume_col: "volume",
                "Metadata_Biology_PatientTumor": "Metadata_patient_tumor",
                "Metadata_Experiment_Treatment": "Metadata_treatment",
                "Metadata_Experiment_Dose": "Metadata_dose",
            }
        )
        rows.append(g)
    volume_df = pd.concat(rows, ignore_index=True)
    volume_df.to_parquet(results_dir / f"volume_3D_{kind}_raw.parquet", index=False)

# --- coverage diagnostics (organoid-level patient_tumor keys only in one modality) ---
area_organoid = pd.read_parquet(results_dir / "area_2D_organoid_raw.parquet")
volume_organoid = pd.read_parquet(results_dir / "volume_3D_organoid_raw.parquet")
dropped_2d_only = sorted(
    set(area_organoid["Metadata_patient_tumor"])
    - set(volume_organoid["Metadata_patient_tumor"])
)
dropped_3d_only = sorted(
    set(volume_organoid["Metadata_patient_tumor"])
    - set(area_organoid["Metadata_patient_tumor"])
)
dropped_patients = pd.DataFrame(
    [{"patient": p, "present_in": "2D_only"} for p in dropped_2d_only]
    + [{"patient": p, "present_in": "3D_only"} for p in dropped_3d_only],
    columns=["patient", "present_in"],
)
dropped_patients.to_parquet(
    results_dir / "area_vs_volume_dropped_patients.parquet", index=False
)
print(f"2D-only patients: {dropped_2d_only}")
print(f"3D-only patients: {dropped_3d_only}")
