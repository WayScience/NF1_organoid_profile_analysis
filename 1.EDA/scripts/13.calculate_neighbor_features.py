#!/usr/bin/env python
# coding: utf-8

# Section 5: space between nuclei.
#
# 2D: nuclei-level Nuclei_Neighbors_* (FirstClosestDistance, SecondClosestDistance,
# NumberOfNeighbors, PercentTouching, AngleBetweenNeighbors) from sc profiles, and
# organoid-level Organoid_Neighbors_NumberOfNeighbors_Adjacent from organoid profiles.
#
# 3D: DEVIATION from the roadmap's literal column names. 3D has no
# Nuclei_Neighbors_FirstClosestDistance-style columns and no
# Organoid_Neighbors_NumberOfNeighbors_Adjacent equivalent. Instead 3D sc
# profiles carry shell/distance-from-center based neighbor metadata:
# Metadata_Neighbors_NeighborsCountAdjacent, Metadata_Neighbors_DistancesFromCenter,
# Metadata_Neighbors_DistancesFromExterior (the latter two are plain scalars per
# nucleus, not per-neighbor arrays). These are genuinely different
# concepts (distance from organoid center/exterior + count of adjacent
# neighbors within a nucleus's local 3D shell, vs. 2D's nearest-neighbor
# distance/angle/percent-touching) and are reported separately, not forced
# into the same columns as 2D. No 3D organoid-level neighbor-density
# equivalent to 2D's Organoid_Neighbors_* was found in the profiles.

# In[1]:


import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# In[2]:


from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()


# In[3]:


from eda_helper_utils.utils_analysis import (
    PROJECTION_FILE_PREFIX,
    PROJECTIONS,
    harmonize_metadata,
    list_patient_dirs,
)

results_dir = root_dir / "1.EDA" / "results" / "neighbors"
results_dir.mkdir(parents=True, exist_ok=True)

NUCLEI_NEIGHBOR_COLS_2D = [
    "Nuclei_Neighbors_FirstClosestDistance_Adjacent",
    "Nuclei_Neighbors_SecondClosestDistance_Adjacent",
    "Nuclei_Neighbors_NumberOfNeighbors_Adjacent",
    "Nuclei_Neighbors_PercentTouching_Adjacent",
    "Nuclei_Neighbors_AngleBetweenNeighbors_Adjacent",
]
ORGANOID_NEIGHBOR_COLS_2D = ["Organoid_Neighbors_NumberOfNeighbors_Adjacent"]


# In[4]:


# --- 2D nuclei-level ---
patients_2d = list_patient_dirs(root_dir / "data" / "profiles_2D")
nuc_rows = []
org_rows = []
for projection in PROJECTIONS:
    prefix = PROJECTION_FILE_PREFIX[projection]
    for patient in patients_2d:
        fs = (
            root_dir
            / "data"
            / "profiles_2D"
            / patient
            / "5.normalized"
            / f"{prefix}_sc.parquet"
        )
        fo = (
            root_dir
            / "data"
            / "profiles_2D"
            / patient
            / "5.normalized"
            / f"{prefix}_organoid.parquet"
        )
        if fs.exists():
            cols = ["Metadata_treatment"] + [c for c in NUCLEI_NEIGHBOR_COLS_2D]
            df = pd.read_parquet(fs)
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
            df = harmonize_metadata(df, "2D", patient)
            df["projection"] = projection
            nuc_rows.append(df)
        if fo.exists():
            df = pd.read_parquet(fo)
            cols = ["Metadata_treatment"] + [
                c for c in ORGANOID_NEIGHBOR_COLS_2D if c in df.columns
            ]
            df = df[cols]
            df = harmonize_metadata(df, "2D", patient)
            df["projection"] = projection
            org_rows.append(df)
        print(f"2D {projection} {patient}: neighbor features extracted")

nuc_2d = pd.concat(nuc_rows, ignore_index=True) if nuc_rows else pd.DataFrame()
org_2d = pd.concat(org_rows, ignore_index=True) if org_rows else pd.DataFrame()
nuc_2d.to_parquet(results_dir / "nuclei_neighbors_2D.parquet", index=False)
org_2d.to_parquet(results_dir / "organoid_neighbors_2D.parquet", index=False)
print(f"Wrote {results_dir / 'nuclei_neighbors_2D.parquet'} ({len(nuc_2d)} rows)")
print(f"Wrote {results_dir / 'organoid_neighbors_2D.parquet'} ({len(org_2d)} rows)")


# In[5]:


# --- 3D sc-level shell/distance neighbor metadata ---
patients_3d = list_patient_dirs(root_dir / "data" / "profiles_3D")
sc3_rows = []
SHELL_COLS_3D = [
    "Metadata_Neighbors_NeighborsCountAdjacent",
    "Metadata_Neighbors_DistancesFromCenter",
    "Metadata_Neighbors_DistancesFromExterior",
]
for patient in patients_3d:
    f = (
        root_dir
        / "data"
        / "profiles_3D"
        / patient
        / "5.normalized_profiles"
        / "sc_norm.parquet"
    )
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    cols = ["Metadata_Experiment_Treatment"] + [
        c for c in SHELL_COLS_3D if c in df.columns
    ]
    df = df[cols]
    df = harmonize_metadata(df, "3D", patient)
    # DistancesFromCenter/FromExterior are already a scalar per nucleus (not a
    # per-neighbor array), so no aggregation is needed - just coerce to float.
    for col in [
        "Metadata_Neighbors_DistancesFromCenter",
        "Metadata_Neighbors_DistancesFromExterior",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: (
                    np.mean(v)
                    if isinstance(v, (list, np.ndarray))
                    else pd.to_numeric(v, errors="coerce")
                )
            )
    sc3_rows.append(df)
    print(f"3D {patient}: shell/distance neighbor features extracted")

sc_3d = pd.concat(sc3_rows, ignore_index=True) if sc3_rows else pd.DataFrame()
sc_3d.to_parquet(results_dir / "nuclei_neighbors_3D.parquet", index=False)
print(f"Wrote {results_dir / 'nuclei_neighbors_3D.parquet'} ({len(sc_3d)} rows)")
print(
    "No 3D organoid-level neighbor-density equivalent found; skipping organoid_neighbors_3D output."
)
