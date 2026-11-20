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
# Organoid_Neighbors_NumberOfNeighbors_Adjacent equivalent in the profiles
# themselves. Instead 3D sc profiles carry shell/distance-from-center based
# neighbor metadata: Metadata_Neighbors_NucleiNoChannelNeighborsNeighborsCountAdjacent,
# Metadata_Neighbors_NucleiNoChannelNeighborsDistancesFromCenter, Metadata_Neighbors_NucleiNoChannelNeighborsDistancesFromExterior
# (the latter two are plain scalars per nucleus, not per-neighbor arrays), plus
# nucleus volume. These are genuinely different concepts (distance from organoid
# center/exterior + count of adjacent neighbors within a nucleus's local 3D
# shell, vs. 2D's nearest-neighbor distance/angle/percent-touching) and are
# reported separately, not forced into the same columns as 2D.
#
# 3D organoid-level neighbor counts also have no ready-made column, so they're
# derived here from each organoid's location/bounding-box metadata: two
# organoids are "adjacent" if their bounding spheres (centroid + mean
# half-extent radius, both converted to um) touch, computed within each
# organoid's own image (Metadata_Experiment_WellFOV). This is an approximation
# (no true segmentation-mask touching test, since only bounding boxes are
# available) but gives a physically-grounded organoid-level crowding metric
# that pairs with the single-cell one, instead of falling back to a 2D proxy.

# In[1]:


import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

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

nuc_2d = pd.concat(nuc_rows, ignore_index=True) if nuc_rows else pd.DataFrame()
org_2d = pd.concat(org_rows, ignore_index=True) if org_rows else pd.DataFrame()
nuc_2d.to_parquet(results_dir / "nuclei_neighbors_2D.parquet", index=False)
org_2d.to_parquet(results_dir / "organoid_neighbors_2D.parquet", index=False)


# In[5]:


# --- 3D sc-level shell/distance neighbor metadata ---
patients_3d = list_patient_dirs(root_dir / "data" / "profiles_3D")
sc3_rows = []
SHELL_COLS_3D = [
    "Metadata_Neighbors_NucleiNoChannelNeighborsNeighborsCountAdjacent",
    "Metadata_Neighbors_NucleiNoChannelNeighborsDistancesFromCenter",
    "Metadata_Neighbors_NucleiNoChannelNeighborsDistancesFromExterior",
    "Metadata_Neighbors_NucleiNoChannelNeighborsNormalizedDistancesFromCenter",
]
# Nucleus volume, joined here (rather than kept only in the organoid-level
# volume table) so single-cell volume can be related to single-cell adjacent
# neighbor count directly, without needing an object ID to merge on.
NUCLEUS_SIZE_COLS_3D = ["Nuclei_NoChannel_VolumeSizeShape_Volume"]
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
    cols = ["Metadata_Experiment_Treatment", "Metadata_Biology_PatientTumor"] + [
        c for c in SHELL_COLS_3D + NUCLEUS_SIZE_COLS_3D if c in df.columns
    ]
    df = df[cols]
    # DistancesFromCenter/FromExterior/NormalizedDistancesFromCenter are already
    # a scalar per nucleus (not a per-neighbor array), so no aggregation is
    # needed - just coerce to float.
    for col in [
        "Metadata_Neighbors_NucleiNoChannelNeighborsDistancesFromCenter",
        "Metadata_Neighbors_NucleiNoChannelNeighborsDistancesFromExterior",
        "Metadata_Neighbors_NucleiNoChannelNeighborsNormalizedDistancesFromCenter",
    ]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda v: (
                    np.mean(v)
                    if isinstance(v, (list, np.ndarray))
                    else pd.to_numeric(v, errors="coerce")
                )
            )
    df = harmonize_metadata(df, "3D", patient)
    sc3_rows.append(df)

sc_3d = pd.concat(sc3_rows, ignore_index=True) if sc3_rows else pd.DataFrame()
sc_3d.to_parquet(results_dir / "nuclei_neighbors_3D.parquet", index=False)


# In[6]:


# --- 3D organoid-level volume (organoid-size story, kept separate from
# nucleus position-in-organoid so the two aren't conflated downstream) ---
org3_rows = []
for patient in patients_3d:
    f = (
        root_dir
        / "data"
        / "profiles_3D"
        / patient
        / "5.normalized_profiles"
        / "organoid_norm.parquet"
    )
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    cols = [
        "Metadata_Experiment_Treatment",
        "Organoid_NoChannel_VolumeSizeShape_Volume",
    ]
    cols = [c for c in cols if c in df.columns]
    df = df[cols]
    df = harmonize_metadata(df, "3D", patient)
    org3_rows.append(df)

org_3d = pd.concat(org3_rows, ignore_index=True) if org3_rows else pd.DataFrame()
org_3d.to_parquet(results_dir / "organoid_volume_3D.parquet", index=False)


# In[7]:


# --- 3D organoid-level adjacent neighbor count ---
# No Organoid_Neighbors_* columns exist in the 3D organoid profiles (unlike
# 2D), so this is derived rather than read off the shelf: two organoids are
# called "adjacent" if their bounding spheres touch, i.e. the physical
# (um) distance between their centroids is <= the sum of their approximate
# radii (mean of the half-extents of their XYZ bounding box, converted from
# pixels/z-slices to um via the per-image resolution metadata). Distances
# are computed only within an organoid's own Metadata_Experiment_WellFOV
# (its single image/stack), matching how the 2D Organoid_Neighbors_* columns
# are scoped to one image.
LOCATION_COLS_3D = [
    "Metadata_Location_Organoid_CenterX",
    "Metadata_Location_Organoid_CenterY",
    "Metadata_Location_Organoid_CenterZ",
    "Metadata_Location_Organoid_MinX",
    "Metadata_Location_Organoid_MaxX",
    "Metadata_Location_Organoid_MinY",
    "Metadata_Location_Organoid_MaxY",
    "Metadata_Location_Organoid_MinZ",
    "Metadata_Location_Organoid_MaxZ",
    "Metadata_Microscopy_XResolutionUm",
    "Metadata_Microscopy_YResolutionUm",
    "Metadata_Microscopy_ZResolutionUm",
    "Metadata_Experiment_WellFOV",
]


def compute_organoid_adjacent_counts(df: pd.DataFrame) -> pd.Series:
    """Count, per organoid, how many other organoids in the same image touch it (bounding-sphere overlap)."""
    x_um = (
        df["Metadata_Location_Organoid_CenterX"]
        * df["Metadata_Microscopy_XResolutionUm"]
    )
    y_um = (
        df["Metadata_Location_Organoid_CenterY"]
        * df["Metadata_Microscopy_YResolutionUm"]
    )
    z_um = (
        df["Metadata_Location_Organoid_CenterZ"]
        * df["Metadata_Microscopy_ZResolutionUm"]
    )
    rx = (
        (df["Metadata_Location_Organoid_MaxX"] - df["Metadata_Location_Organoid_MinX"])
        * df["Metadata_Microscopy_XResolutionUm"]
        / 2
    )
    ry = (
        (df["Metadata_Location_Organoid_MaxY"] - df["Metadata_Location_Organoid_MinY"])
        * df["Metadata_Microscopy_YResolutionUm"]
        / 2
    )
    rz = (
        (df["Metadata_Location_Organoid_MaxZ"] - df["Metadata_Location_Organoid_MinZ"])
        * df["Metadata_Microscopy_ZResolutionUm"]
        / 2
    )
    radius_um = ((rx + ry + rz) / 3).to_numpy()
    coords_um = np.column_stack([x_um.to_numpy(), y_um.to_numpy(), z_um.to_numpy()])

    counts = np.zeros(len(df), dtype=int)
    for _, group_index in df.groupby("Metadata_Experiment_WellFOV").groups.items():
        pos = df.index.get_indexer(group_index)
        if len(pos) < 2:
            continue
        dist_matrix = squareform(pdist(coords_um[pos]))
        touch_threshold = radius_um[pos][:, None] + radius_um[pos][None, :]
        adjacent = dist_matrix <= touch_threshold
        np.fill_diagonal(adjacent, False)
        counts[pos] = adjacent.sum(axis=1)
    return pd.Series(counts, index=df.index)


org_nbr3_rows = []
for patient in patients_3d:
    f = (
        root_dir
        / "data"
        / "profiles_3D"
        / patient
        / "5.normalized_profiles"
        / "organoid_norm.parquet"
    )
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    cols = ["Metadata_Experiment_Treatment"] + LOCATION_COLS_3D
    df = df[cols].reset_index(drop=True)
    df["Organoid_Neighbors_NumberOfNeighbors_Adjacent"] = (
        compute_organoid_adjacent_counts(df)
    )
    df = df[
        [
            "Metadata_Experiment_Treatment",
            "Organoid_Neighbors_NumberOfNeighbors_Adjacent",
        ]
    ]
    df = harmonize_metadata(df, "3D", patient)
    org_nbr3_rows.append(df)

org_nbr_3d = (
    pd.concat(org_nbr3_rows, ignore_index=True) if org_nbr3_rows else pd.DataFrame()
)
org_nbr_3d.to_parquet(results_dir / "organoid_neighbors_3D.parquet", index=False)


# In[8]:


# --- 3D organoid-level cell density (cells per unit volume) ---
# Density = Metadata_Object_OrganoidSingleCellCount / organoid volume. Both
# need to be in real (non-normalized, non-z-scored) units for this ratio to
# mean anything, so this reads from 4.qc_profiles (pre-normalization) rather
# than the 5.normalized_profiles used elsewhere in this notebook. Organoids
# flagged by QC (NaN features, too-small outliers) are excluded, to
# match what the normalized profiles keep. Volume is converted from voxels to
# um^3 using the per-image resolution metadata (constant across patients:
# 0.101 x 0.101 x 1.0 um/voxel).
DENSITY_COLS_3D = [
    "Metadata_Object_OrganoidSingleCellCount",
    "Organoid_NoChannel_VolumeSizeShape_Volume",
    "Metadata_Microscopy_XResolutionUm",
    "Metadata_Microscopy_YResolutionUm",
    "Metadata_Microscopy_ZResolutionUm",
    "Metadata_cqc_nan_detected",
    "Metadata_cqc_small_organoid_outlier",
]
org_density_rows = []
for patient in patients_3d:
    f = (
        root_dir
        / "data"
        / "profiles_3D"
        / patient
        / "4.qc_profiles"
        / "organoid_flagged_outliers.parquet"
    )
    if not f.exists():
        continue
    df = pd.read_parquet(f)
    cols = ["Metadata_Experiment_Treatment"] + DENSITY_COLS_3D
    df = df[cols]
    df = df[
        ~(df["Metadata_cqc_nan_detected"] | df["Metadata_cqc_small_organoid_outlier"])
    ].drop(
        columns=[
            "Metadata_cqc_nan_detected",
            "Metadata_cqc_small_organoid_outlier",
        ]
    )
    voxel_volume_um3 = (
        df["Metadata_Microscopy_XResolutionUm"]
        * df["Metadata_Microscopy_YResolutionUm"]
        * df["Metadata_Microscopy_ZResolutionUm"]
    )
    volume_um3 = df["Organoid_NoChannel_VolumeSizeShape_Volume"] * voxel_volume_um3
    df["Organoid_CellDensity_CellsPerUm3"] = (
        df["Metadata_Object_OrganoidSingleCellCount"] / volume_um3
    )
    df = df[["Metadata_Experiment_Treatment", "Organoid_CellDensity_CellsPerUm3"]]
    df = harmonize_metadata(df, "3D", patient)
    org_density_rows.append(df)

org_density_3d = (
    pd.concat(org_density_rows, ignore_index=True)
    if org_density_rows
    else pd.DataFrame()
)
org_density_3d.to_parquet(results_dir / "organoid_density_3D.parquet", index=False)
