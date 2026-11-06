#!/usr/bin/env python
# coding: utf-8

# # Pearson correlations across profile types
#
# Correlate features across matched well aggregates, per patient/tumor and pooled.

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()


# In[2]:


# Define the comparisons to run
PROJECTIONS = ("max_projection",)

# profile type: (2D handcrafted resolution, 3D file prefix, plot label)
# MorphEM uses projected 2D crops but is stored under profiles_3D.
PROFILE_TYPES = {
    "organoid_handcrafted": ("organoid", "organoid", "Organoid / ZedProfiler"),
    "organoid_sammed": ("organoid", "sammed_organoid", "Organoid / SAM-Med3D"),
    "sc_handcrafted": ("sc", "sc", "Single-cell / ZedProfiler"),
    "sc_sammed": ("sc", "sammed_sc", "Single-cell / SAM-Med3D"),
    "sc_sammed_nucleocentric": (
        "sc",
        "sammed_nucleocentric",
        "Single-cell / Nucleocentric SAM-Med3D",
    ),
    "sc_nucleocentric_morphem": (
        "sc",
        "nucleocentric_morphem",
        "Single-cell / Nucleocentric MorphEM",
    ),
}

PATIENT_TUMORS = None  # None = every shared patient/tumor; or a list of IDs.
INCLUDE_ALL_PATIENTS = True  # Pools every shared, non-excluded sample.
EXCLUDED_PATIENT_TUMORS = ("NF0037_T1_CQ1",)
OUTLIER_CUTOFF = 100

# Wells are matched on these keys, and these annotations travel with them.
# 2D and 3D profiles use different metadata naming conventions; 3D's is the
# canonical one here, so 2D's columns are renamed up to match it.
MATCH_KEYS = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"]
ANNOTATION_COLUMNS = [
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Unit",
]

METADATA_2D_RENAME = {
    "Metadata_patient_tumor": "Metadata_Biology_PatientTumor",
    "Metadata_Well": "Metadata_Experiment_Well",
    "Metadata_treatment": "Metadata_Experiment_Treatment",
    "Metadata_dose": "Metadata_Experiment_Dose",
    "Metadata_dose_unit": "Metadata_Experiment_Unit",
}


# In[3]:


# Define input paths
# 2D profiles (one sub-directory per projection)
input_2d_dir = pathlib.Path(f"{root_dir}/data/profiles_2D/all_patients").resolve(
    strict=True
)

# 3D profiles
input_3d_dir = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/2.aggregated_profiles"
).resolve(strict=True)

# Define output paths
results_dir = pathlib.Path(
    f"{root_dir}/2.2d_vs_3d_analysis/results/correlation"
).resolve()
results_dir.mkdir(parents=True, exist_ok=True)

manifest_path = results_dir / "correlation_manifest.parquet"
matching_summary_path = results_dir / "matching_summary.parquet"
feature_summary_path = results_dir / "feature_cleanup_summary.parquet"


# ## Clean profiles and match wells

# In[ ]:


def load_profile(path: pathlib.Path, dimension: str) -> tuple[pd.DataFrame, dict]:
    """
    Load a well-aggregated profile and apply the same feature filters used in
    the EDA, then report what was dropped.

    Parameters
    ----------
    path : pathlib.Path
        Path to the aggregated profile parquet file.
    dimension : str
        Either "2D" or "3D". 2D metadata columns are renamed to the 3D
        convention so both profiles can be matched on the same keys.

    Returns
    ----------
    tuple[pd.DataFrame, dict]
        The cleaned profile (metadata columns followed by feature columns) and
        a summary of the feature cleanup.
    """
    df = pd.read_parquet(path)
    if dimension == "2D":
        df = df.rename(columns=METADATA_2D_RENAME)

    # Drop excluded samples before counting anything
    n_input_wells = len(df)
    df = df.loc[
        ~df["Metadata_Biology_PatientTumor"].isin(EXCLUDED_PATIENT_TUMORS)
    ].copy()

    features = [col for col in df if not col.startswith("Metadata_")]

    values = df[[col for col in features]].apply(pd.to_numeric, errors="coerce")

    # Same magnitude rule as pycytominer's drop_outliers operation in the EDA
    extreme_features = values.columns[values.abs().gt(OUTLIER_CUTOFF).any()]
    values = values.drop(columns=extreme_features)

    # A feature with one value (or none) cannot correlate with anything
    uninformative_features = values.columns[values.nunique(dropna=True).le(1)]
    values = values.drop(columns=uninformative_features)

    metadata = df[[col for col in df if col.startswith("Metadata_")]]

    summary = {
        "input_path": str(path.relative_to(root_dir)),
        "dimension": dimension,
        "n_input_wells": n_input_wells,
        "n_excluded_wells": n_input_wells - len(df),
        "n_features_input": len(features),
        "n_texture_dropped": 0,
        "n_outlier_dropped": len(extreme_features),
        "n_constant_or_empty_dropped": len(uninformative_features),
        "n_features_retained": values.shape[1],
    }

    return pd.concat([metadata, values], axis=1).reset_index(drop=True), summary


def match_profiles(
    df_2d: pd.DataFrame, df_3d: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Align 2D and 3D well aggregates one-to-one on patient/tumor and well, and
    audit which wells matched.

    Parameters
    ----------
    df_2d : pd.DataFrame
        Cleaned 2D profile data with MATCH_KEYS and ANNOTATION_COLUMNS.
    df_3d : pd.DataFrame
        Cleaned 3D profile data with the same metadata structure.

    Returns
    ----------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        The 2D and 3D profiles restricted to shared wells and in the same row
        order, plus the full outer-join audit of matched and unmatched wells.
    """
    # match_status records whether each well is in both, 2D only, or 3D only
    audit = df_2d[MATCH_KEYS + ANNOTATION_COLUMNS].merge(
        df_3d[MATCH_KEYS + ANNOTATION_COLUMNS],
        on=MATCH_KEYS,
        how="outer",
        suffixes=("_2d", "_3d"),
        indicator="match_status",
        validate="one_to_one",
        sort=True,
    )

    shared = audit.loc[audit["match_status"].eq("both")]
    if len(shared) == 0:
        raise ValueError("No shared patient/tumor-well keys between 2D and 3D.")

    # Index both profiles by the shared keys so the rows line up
    keys = pd.MultiIndex.from_frame(shared[MATCH_KEYS])
    aligned_2d = df_2d.set_index(MATCH_KEYS).loc[keys].reset_index()
    aligned_3d = df_3d.set_index(MATCH_KEYS).loc[keys].reset_index()

    assert list(aligned_2d[MATCH_KEYS].apply(tuple, axis=1)) == list(
        aligned_3d[MATCH_KEYS].apply(tuple, axis=1)
    ), "Wells are not aligned after matching"

    return aligned_2d, aligned_3d, audit


# ## Calculate Pearson correlations

# In[5]:


def standardize(arr: np.ndarray) -> np.ndarray:
    """
    Perform standard scalar normalization on the input array.
    This is NA aware!

    Parameters
    ----------
    arr : np.ndarray
        The input array to be normalized in the following dimensionality: (nxm)

    Returns
    ----------
    np.ndarray
        The standard scalar normalized array of the same dimensionality (nxm)
    """
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=0)
    std = np.where(std == 0, np.nan, std)
    return (arr - mean) / std


def compute_and_save_correlation(
    features_2d: pd.DataFrame, features_3d: pd.DataFrame, output_path: pathlib.Path
) -> dict:
    """
    Compute pairwise Pearson correlations between 2D and 3D features and save
    the results as a long-form parquet file.

    Parameters
    ----------
    features_2d : pd.DataFrame
        2D feature columns, one row per matched well.
    features_3d : pd.DataFrame
        3D feature columns, with the same rows in the same order.
    output_path : pathlib.Path
        Path to save the long-form parquet file to.

    Returns
    ----------
    dict
        A summary of the saved matrix, for the manifest.
    """
    features_2d_cols = list(features_2d.columns)
    features_3d_cols = list(features_3d.columns)

    # Profiles were normalized at the single-cell level before aggregation, so
    # re-standardize within this cohort's matched wells. This does not change
    # Pearson r (it is shift/scale invariant); it only puts the 2D and 3D
    # features on a common scale before the combined .corr() call.
    arr_2d_z = standardize(features_2d.values)
    arr_3d_z = standardize(features_3d.values)

    # Compute correlation via a single .corr() call, then slice out the
    # 2D-vs-3D cross block (pandas handles missing values pairwise already)
    df_2d_std = pd.DataFrame(arr_2d_z, columns=features_2d_cols)
    df_3d_std = pd.DataFrame(arr_3d_z, columns=features_3d_cols)
    combined = pd.concat([df_2d_std, df_3d_std], axis=1)
    full_corr = combined.corr(method="pearson")
    corr_matrix = full_corr.loc[features_2d_cols, features_3d_cols].values

    # Convert to long-form and save
    corr_df = pd.DataFrame(
        corr_matrix, index=features_2d_cols, columns=features_3d_cols
    )
    corr_long = (
        corr_df.reset_index()
        .melt(id_vars="index", var_name="feature_3d", value_name="pearson_r")
        .rename(columns={"index": "feature_2d"})
    )
    corr_long.to_parquet(output_path, index=False)

    return {
        "n_features_2d": len(features_2d_cols),
        "n_features_3d": len(features_3d_cols),
        "n_defined_correlations": int(corr_long["pearson_r"].notna().sum()),
    }


# ## Save correlations and matching summaries

# In[6]:


# Profiles are loaded fresh for each comparison; each file's cleanup summary
# is recorded once, the first time it is loaded
summarized_paths = set()
feature_summaries = []
matching_summaries = []
manifest = []

for projection in PROJECTIONS:
    for profile_type, (resolution, prefix, label) in PROFILE_TYPES.items():
        path_2d = input_2d_dir / projection / f"{resolution}_agg_profiles.parquet"
        path_3d = input_3d_dir / f"{prefix}_norm_sc_agg_profiles.parquet"

        profiles = {}
        for path, dimension in ((path_2d, "2D"), (path_3d, "3D")):
            profiles[path], summary = load_profile(path, dimension)
            if path not in summarized_paths:
                summarized_paths.add(path)
                feature_summaries.append(summary)

        aligned_2d, aligned_3d, audit = match_profiles(
            profiles[path_2d], profiles[path_3d]
        )

        # Save the well-matching audit next to this comparison's correlations
        pair_dir = results_dir / projection / profile_type
        pair_dir.mkdir(parents=True, exist_ok=True)
        audit.to_parquet(pair_dir / "matched_wells.parquet", index=False)

        for patient_tumor, group in audit.groupby("Metadata_Biology_PatientTumor"):
            counts = group["match_status"].value_counts()
            matching_summaries.append(
                {
                    "projection": projection,
                    "profile_type": profile_type,
                    "patient_tumor": patient_tumor,
                    "n_matched_wells": int(counts.get("both", 0)),
                    "n_2d_only": int(counts.get("left_only", 0)),
                    "n_3d_only": int(counts.get("right_only", 0)),
                }
            )

        # Correlate each patient/tumor on its own, and optionally all pooled
        shared_patient_tumors = sorted(
            aligned_2d["Metadata_Biology_PatientTumor"].unique()
        )
        if PATIENT_TUMORS is not None:
            missing = set(PATIENT_TUMORS) - set(shared_patient_tumors)
            if missing:
                raise ValueError(
                    f"{projection}/{profile_type}: unmatched IDs {missing}."
                )
            shared_patient_tumors = [
                patient_tumor
                for patient_tumor in shared_patient_tumors
                if patient_tumor in PATIENT_TUMORS
            ]

        cohorts = ["all_patients"] if INCLUDE_ALL_PATIENTS else []
        cohorts = cohorts + shared_patient_tumors

        features_2d = [col for col in aligned_2d if not col.startswith("Metadata_")]
        features_3d = [col for col in aligned_3d if not col.startswith("Metadata_")]

        for cohort in cohorts:
            if cohort == "all_patients":
                selected = np.ones(len(aligned_2d), dtype=bool)
            else:
                selected = (
                    aligned_2d["Metadata_Biology_PatientTumor"].eq(cohort).to_numpy()
                )

            output_path = pair_dir / f"{cohort}_correlation.parquet"
            summary = compute_and_save_correlation(
                aligned_2d.loc[selected, features_2d],
                aligned_3d.loc[selected, features_3d],
                output_path,
            )

            manifest.append(
                {
                    "projection": projection,
                    "profile_type": profile_type,
                    "profile_label": label,
                    "cohort": cohort,
                    "analysis_unit": "patient_tumor_well_aggregate",
                    "n_matched_wells": int(selected.sum()),
                    "correlation_file": str(output_path.relative_to(results_dir)),
                    **summary,
                }
            )

            print(
                f"{projection} / {profile_type} / {cohort}: "
                f"{selected.sum()} matched wells; "
                f"{summary['n_defined_correlations']:,} defined correlations",
                flush=True,
            )


# In[7]:


# Save the manifest notebook 2 plots from, plus the cleanup and matching audits
correlation_manifest = pd.DataFrame(manifest)

pd.DataFrame(feature_summaries).to_parquet(feature_summary_path, index=False)
pd.DataFrame(matching_summaries).to_parquet(matching_summary_path, index=False)
correlation_manifest.to_parquet(manifest_path, index=False)

correlation_manifest
