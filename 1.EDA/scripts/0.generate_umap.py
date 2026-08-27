#!/usr/bin/env python
# coding: utf-8

# # Generate UMAP embeddings
# 
# This notebook computes 2D UMAP embeddings of single-cell morphology profiles for the NF1 organoid dataset.
# 
# Two sets of embeddings are generated:
# 1. **Pooled (all-patient) profiles** — one embedding per projection (2D max-projection, 2D middle-slice, 3D) and profile variant (feature-selected and consensus), computed across all patients combined.
# 2. **Per-patient profiles** — one independent embedding per patient, projection, and profile variant (feature-selected, and aggregated), but consolidated into a single file per projection/variant combo (patient identity is kept as a metadata column, e.g. `Metadata_Biology_PatientTumor`). UMAP1/UMAP2 are only comparable *within* a patient, never across patients in the same file, since each patient's embedding is its own independent fit.
# 
# The notebook saves results (`UMAP1` coordinate, `UMAP2` coordinate, plus the original metadata columns) as parquet files under `1.EDA/results/umap/`.

# In[1]:


import pathlib
import sys

import pandas as pd
import umap
from pycytominer import feature_select
from pycytominer.cyto_utils import infer_cp_features

cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd
else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break
sys.path.append(str(root_dir / "utils"))
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

profile_base_dir = bandicoot_check(
    pathlib.Path("/home/lippincm/mnt/bandicoot").resolve(), root_dir
)


# In[2]:


# Data Paths

# Create a comprehensive dictionary for all dimension and projection combinations
pooled_profile_paths = {
    "2D_max_projection": {
        "sc_fs": {
            "input": pathlib.Path(
                f"{root_dir}/data/profiles_2D/all_patients/max_projection/sc_fs_profiles.parquet"
            ).resolve(strict=True),
            "output": pathlib.Path(
                f"{root_dir}/1.EDA/results/umap/2D_maxproj_scfs_umap.parquet"
            ).resolve(),
        },
        "sc_consensus": {
            "input": pathlib.Path(
                f"{root_dir}/data/profiles_2D/all_patients/max_projection/sc_consensus_profiles.parquet"
            ).resolve(strict=True),
            "output": pathlib.Path(
                f"{root_dir}/1.EDA/results/umap/2D_maxproj_scconsensus_umap.parquet"
            ).resolve(),
        },
    },
    "2D_middle_slice": {
        "sc_fs": {
            "input": pathlib.Path(
                f"{root_dir}/data/profiles_2D/all_patients/middle_slice/sc_fs_profiles.parquet"
            ).resolve(strict=True),
            "output": pathlib.Path(
                f"{root_dir}/1.EDA/results/umap/2D_midslice_scfs_umap.parquet"
            ).resolve(),
        },
        "sc_consensus": {
            "input": pathlib.Path(
                f"{root_dir}/data/profiles_2D/all_patients/middle_slice/sc_consensus_profiles.parquet"
            ).resolve(strict=True),
            "output": pathlib.Path(
                f"{root_dir}/1.EDA/results/umap/2D_midslice_scconsensus_umap.parquet"
            ).resolve(),
        },
    },
    "3D": {
        "sc_fs": {
            "input": pathlib.Path(
                f"{root_dir}/data/profiles_3D/all_patients/1.feature_selected_profiles/sc_norm_fs_profiles.parquet"
            ).resolve(strict=True),
            "output": pathlib.Path(
                f"{root_dir}/1.EDA/results/umap/3D_scfs_umap.parquet"
            ).resolve(),
        },
        "sc_consensus": {
            "input": pathlib.Path(
                f"{root_dir}/data/profiles_3D/all_patients/3.consensus_profiles/sc_norm_sc_consensus_profiles.parquet"
            ).resolve(strict=True),
            "output": pathlib.Path(
                f"{root_dir}/1.EDA/results/umap/3D_scconsensus_umap.parquet"
            ).resolve(),
        },
    },
}

# Create output directory
pathlib.Path(f"{root_dir}/1.EDA/results/umap").mkdir(parents=True, exist_ok=True)


# In[3]:


OUTLIER_CUTOFF = 100

for projection_key in pooled_profile_paths:
    compartments = (
        ["Cell", "Nuclei", "Cytoplasm"]
        if projection_key == "3D"
        else ["Cells", "Nuclei", "Cytoplasm"]
    )
    patient_tumor_column = (
        "Metadata_Biology_PatientTumor" if projection_key == "3D" else "Metadata_patient_tumor"
    )
    for profile_variant, profile_paths in pooled_profile_paths[projection_key].items():
        # Load the data
        profile_df = pd.read_parquet(profile_paths["input"])

        # Exclude NF0037_T1_CQ1: this is a separate analysis and should not be
        # included anywhere in this EDA (pooled or per-patient).
        profile_df = profile_df[profile_df[patient_tumor_column] != "NF0037_T1_CQ1"]

        metadata_columns = infer_cp_features(profile_df, metadata=True)
        feature_columns = infer_cp_features(profile_df, compartments=compartments)
        n_features_total = len(feature_columns)

        # Drop Texture features: some single cells have extreme outlier
        # Texture values that overflow to inf when cast to float32 by UMAP.
        # TODO: revisit once v3 profiles fix the underlying Texture computation.
        texture_columns = [c for c in feature_columns if "_Texture_" in c]
        feature_columns = [c for c in feature_columns if "_Texture_" not in c]

        # Drop any other features with extreme/blown-up values (e.g. from
        # near-zero-variance StandardScaler denominators upstream) using
        # pycytominer's feature_select, with its magnitude-based outlier
        # operation.
        profile_df = feature_select(
            profile_df,
            features=feature_columns,
            operation=["drop_outliers"],
            outlier_cutoff=OUTLIER_CUTOFF,
        )
        selected_feature_columns = [c for c in feature_columns if c in profile_df.columns]
        n_outlier_dropped = len(feature_columns) - len(selected_feature_columns)
        feature_columns = selected_feature_columns

        print(
            f"{projection_key}/{profile_variant}: {n_features_total} features -> "
            f"-{len(texture_columns)} texture -> -{n_outlier_dropped} outlier "
            f"(|value| > {OUTLIER_CUTOFF}) -> {len(feature_columns)} remaining"
        )

        # Remove rows with NaN values in the feature columns
        profile_df = profile_df.dropna(subset=feature_columns, axis=0, how="any")
        metadata_df = profile_df[metadata_columns]
        features_df = profile_df[feature_columns]

        # Extract features and apply UMAP
        umap_reducer = umap.UMAP(
            n_neighbors=50,
            min_dist=0.0,
            metric="cosine",
            repulsion_strength=2,
            random_state=0,
        )
        umap_embedding = umap_reducer.fit_transform(features_df)

        # Create a DataFrame with UMAP results
        umap_results_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
        umap_results_df = pd.concat(
            [metadata_df.reset_index(drop=True), umap_results_df], axis=1
        )

        # Save the UMAP results
        umap_results_df.to_parquet(profile_paths["output"], index=False)


# ## Individual umaps

# In[4]:


patients = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    names=["patient"],
)["patient"].to_list()


# In[5]:


# Individual patients, 2D and 3D, feature-selected and aggregated
patient_profile_paths = {}

patient_output_dir = pathlib.Path(f"{root_dir}/1.EDA/results/umap/patient_specific")
patient_output_dir.mkdir(parents=True, exist_ok=True)

# NF0037_T1 has no 2D per-patient data (the 2D pipeline never produced output
# for this plate); skip only its 2D entries, not 3D (which is fully populated).
patients_with_2D_data = [p for p in patients if p != "NF0037_T1"]

for patient in patients:
    patient_profile_paths[patient] = {
        "3D": {
            "sc_fs": {
                "input": pathlib.Path(
                    f"{root_dir}/data/profiles_3D/{patient}/6.feature_selected_profiles/sc_fs.parquet"
                ).resolve(strict=True),
            },
            "sc_agg": {
                "input": pathlib.Path(
                    f"{root_dir}/data/profiles_3D/{patient}/7.aggregated_profiles/sc_agg_well_level.parquet"
                ).resolve(strict=True),
            },
        },
    }

    if patient in patients_with_2D_data:
        patient_profile_paths[patient]["2D_max_projection"] = {
            "sc_fs": {
                "input": pathlib.Path(
                    f"{root_dir}/data/profiles_2D/{patient}/6.feature_selected/max_projected_sc.parquet"
                ).resolve(strict=True),
            },
            "sc_agg": {
                "input": pathlib.Path(
                    f"{root_dir}/data/profiles_2D/{patient}/7.aggregated/max_projected_sc.parquet"
                ).resolve(strict=True),
            },
        }
        patient_profile_paths[patient]["2D_middle_slice"] = {
            "sc_fs": {
                "input": pathlib.Path(
                    f"{root_dir}/data/profiles_2D/{patient}/6.feature_selected/middle_slice_sc.parquet"
                ).resolve(strict=True),
            },
            "sc_agg": {
                "input": pathlib.Path(
                    f"{root_dir}/data/profiles_2D/{patient}/7.aggregated/middle_slice_sc.parquet"
                ).resolve(strict=True),
            },
        }

# Per-patient results are computed from independent per-patient UMAP fits
# (each patient's embedding is its own model, not a shared one), but are
# consolidated into a single file per projection/variant combo, with patient
# identity kept as a metadata column (e.g. Metadata_Biology_PatientTumor) so
# rows stay distinguishable. UMAP1/UMAP2 are only comparable *within* a
# patient, never across patients in the same file.
patient_specific_output_paths = {
    "3D": {
        "sc_fs": patient_output_dir / "patient_specific_3D_scfs_umap.parquet",
        "sc_agg": patient_output_dir / "patient_specific_3D_scagg_umap.parquet",
    },
    "2D_max_projection": {
        "sc_fs": patient_output_dir / "patient_specific_2D_maxproj_scfs_umap.parquet",
        "sc_agg": patient_output_dir / "patient_specific_2D_maxproj_scagg_umap.parquet",
    },
    "2D_middle_slice": {
        "sc_fs": patient_output_dir / "patient_specific_2D_midslice_scfs_umap.parquet",
        "sc_agg": patient_output_dir / "patient_specific_2D_midslice_scagg_umap.parquet",
    },
}


# In[6]:


feature_drop_summary = []
patient_specific_results = {
    projection_key: {variant: [] for variant in variants}
    for projection_key, variants in patient_specific_output_paths.items()
}

for patient in patient_profile_paths:
    for projection_key in patient_profile_paths[patient]:
        compartments = (
            ["Cell", "Nuclei", "Cytoplasm"]
            if projection_key == "3D"
            else ["Cells", "Nuclei", "Cytoplasm"]
        )
        for profile_variant, profile_paths in patient_profile_paths[patient][
            projection_key
        ].items():
            # Load the data
            profile_df = pd.read_parquet(profile_paths["input"])
            metadata_columns = infer_cp_features(profile_df, metadata=True)
            feature_columns = infer_cp_features(profile_df, compartments=compartments)
            n_features_total = len(feature_columns)

            # Drop Texture features: some single cells have extreme outlier
            # Texture values that overflow to inf when cast to float32 by UMAP.
            # TODO: revisit once v3 profiles fix the underlying Texture computation.
            texture_columns = [c for c in feature_columns if "_Texture_" in c]
            feature_columns = [c for c in feature_columns if "_Texture_" not in c]

            # Drop any other features with extreme/blown-up values (e.g. from
            # near-zero-variance StandardScaler denominators upstream) using
            # pycytominer's feature_select, with its magnitude-based outlier
            # operation.
            profile_df = feature_select(
                profile_df,
                features=feature_columns,
                operation=["drop_outliers"],
                outlier_cutoff=OUTLIER_CUTOFF,
            )
            selected_feature_columns = [c for c in feature_columns if c in profile_df.columns]
            n_outlier_dropped = len(feature_columns) - len(selected_feature_columns)
            feature_columns = selected_feature_columns

            feature_drop_summary.append(
                {
                    "projection": projection_key,
                    "variant": profile_variant,
                    "total": n_features_total,
                    "texture_dropped": len(texture_columns),
                    "outlier_dropped": n_outlier_dropped,
                    "remaining": len(feature_columns),
                }
            )

            # Remove rows with NaN values in the feature columns
            profile_df = profile_df.dropna(subset=feature_columns, axis=0, how="any")
            metadata_df = profile_df[metadata_columns]
            features_df = profile_df[feature_columns]

            # Extract features and apply UMAP. Each patient gets its own
            # independent UMAP fit (not a shared embedding across patients).
            umap_reducer = umap.UMAP(
                n_neighbors=50,
                min_dist=0.0,
                metric="cosine",
                repulsion_strength=2,
                random_state=0,
            )
            umap_embedding = umap_reducer.fit_transform(features_df)

            # Create a DataFrame with UMAP results
            umap_results_df = pd.DataFrame(umap_embedding, columns=["UMAP1", "UMAP2"])
            umap_results_df = pd.concat(
                [metadata_df.reset_index(drop=True), umap_results_df], axis=1
            )

            # Stamp patient identity explicitly: the 2D per-patient source
            # files carry no patient-identifying metadata column at all
            # (identity was only ever encoded via the file path/name), unlike
            # 3D which already has Metadata_Biology_PatientTumor natively.
            # Adding this unconditionally keeps the consolidated files'
            # schema uniform and guarantees rows stay distinguishable by
            # patient regardless of projection.
            umap_results_df["Metadata_Biology_PatientTumor"] = patient

            # Collect this patient's results
            patient_specific_results[projection_key][profile_variant].append(
                umap_results_df
            )

# Concatenate all patients' results into one file per projection/variant combo
for projection_key, variants in patient_specific_results.items():
    for profile_variant, patient_dfs in variants.items():
        combined_df = pd.concat(patient_dfs, ignore_index=True)
        combined_df.to_parquet(
            patient_specific_output_paths[projection_key][profile_variant],
            index=False,
        )

# Summarize feature dropping across all per-patient datasets, grouped by
# projection/variant (feature counts are the same across patients within a
# projection, but outlier drops can differ since each patient's profiles
# were normalized independently).
feature_drop_summary_df = pd.DataFrame(feature_drop_summary)
print(
    feature_drop_summary_df.groupby(["projection", "variant"]).agg(
        total=("total", "first"),
        texture_dropped=("texture_dropped", "first"),
        outlier_dropped_min=("outlier_dropped", "min"),
        outlier_dropped_max=("outlier_dropped", "max"),
        remaining_min=("remaining", "min"),
        remaining_max=("remaining", "max"),
    )
)

