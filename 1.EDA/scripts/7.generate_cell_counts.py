#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib

import numpy as np
import pandas as pd
import pyarrow as pq
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


normalized_3D_profiles = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/0.normalized_profiles/"
).resolve()
normalized_profiles = list(normalized_3D_profiles.glob("*.parquet"))
normalized_2D_profiles = pathlib.Path(
    f"{root_dir}/data/profiles_2D/all_patients/"
).resolve()
normalized_2D_profile_paths = [
    x
    for y in normalized_2D_profiles.glob("*")
    if y.is_dir()
    for x in y.glob("*")
    if not any(s in x.name for s in ["fs", "agg", "consensus"])
]
normalized_profiles = normalized_profiles + normalized_2D_profile_paths


# In[3]:


results_dir = pathlib.Path(f"{root_dir}/1.EDA/results/cell_counts/").resolve()
results_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


# --- Pass 1: build counts_df for every profile type, split by 2D/3D ---
dict_of_count_dfs = {}
fov_lookup_frames = []

for path in tqdm.tqdm(normalized_profiles):
    profile_type = f"{path.stem}_{path.parent.parent.parent.name.split('_')[1]}"
    df = pd.read_parquet(path)
    metadata_cols = [x for x in df.columns if "Metadata" in x]
    df = df[metadata_cols]

    is_2d = "2D" in str(path)
    if is_2d:
        profile_type = f"{profile_type}_{path.parent.name}"
        df = df.rename(
            columns={
                "Metadata_patient_tumor": "Metadata_Biology_PatientTumor",
                "Metadata_treatment": "Metadata_Experiment_Treatment",
                "Metadata_dose": "Metadata_Experiment_Dose",
                "Metadata_Well": "Metadata_Experiment_Well",
            }
        )

    group_cols = [
        "Metadata_Biology_PatientTumor",
        "Metadata_Experiment_Treatment",
        "Metadata_Experiment_Dose",
    ]

    has_wellfov = "Metadata_Experiment_WellFOV" in df.columns

    agg_dict = {
        "Metadata_n_wells": pd.NamedAgg(
            column="Metadata_Experiment_Well", aggfunc="nunique"
        ),
    }
    if has_wellfov:
        agg_dict["Metadata_n_cells"] = pd.NamedAgg(
            column="Metadata_Experiment_WellFOV", aggfunc="count"
        )
        agg_dict["Metadata_n_fovs"] = pd.NamedAgg(
            column="Metadata_Experiment_WellFOV", aggfunc="nunique"
        )
    else:
        agg_dict["Metadata_n_cells"] = pd.NamedAgg(
            column="Metadata_Experiment_Well", aggfunc="count"
        )

    counts_df = df.groupby(group_cols).agg(**agg_dict).reset_index()

    if not has_wellfov:
        counts_df["Metadata_n_fovs"] = pd.NA

    counts_df["Metadata_profile_name"] = profile_type
    dict_of_count_dfs[profile_type] = {"df": counts_df, "is_2d": is_2d}

    # build the well-level FOV lookup from 3D data only
    if has_wellfov:
        well_fov_df = (
            df.groupby(group_cols)[
                ["Metadata_Experiment_Well", "Metadata_Experiment_WellFOV"]
            ]
            .nunique()
            .reset_index()
            .rename(
                columns={
                    "Metadata_Experiment_Well": "Metadata_n_wells_lookup",
                    "Metadata_Experiment_WellFOV": "Metadata_n_fovs_lookup",
                }
            )
        )
        fov_lookup_frames.append(well_fov_df)

# combine all 3D-derived fov lookups
fov_lookup = pd.concat(fov_lookup_frames)

fov_lookup.reset_index(drop=True, inplace=True)


# In[5]:


for profile_type, entry in dict_of_count_dfs.items():
    df = entry["df"]
    if entry["is_2d"]:
        df = (
            df.drop(columns=["Metadata_n_fovs"])
            .merge(
                fov_lookup,
                on=[
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Experiment_Treatment",
                    "Metadata_Experiment_Dose",
                ],
                how="left",
            )
            .rename(columns={"Metadata_n_fovs_lookup": "Metadata_n_fovs"})
        )

    df["Metadata_profile_type"] = profile_type
    dict_of_count_dfs[profile_type]["df"] = df

# merge all the profile_types into a single counts_df
merged_df = pd.DataFrame()
for profile_type, entry in dict_of_count_dfs.items():
    if merged_df.empty:
        merged_df = entry["df"]
    else:
        merged_df = pd.concat([merged_df, entry["df"]], ignore_index=True)


# In[6]:


# --- Pass 3: normalize the n_cells column by the number of FOVs, to get a per-FOV cell count ---
# annotate unique treatments

merged_df["Metadata_n_cells_norm_by_well_fov"] = (
    merged_df["Metadata_n_cells"] / merged_df["Metadata_n_fovs"]
)
merged_df["Metadata_unique_treatment"] = (
    merged_df["Metadata_Experiment_Treatment"]
    + "_"
    + merged_df["Metadata_Experiment_Dose"].astype(str)
)
merged_df.to_parquet(results_dir / "cell_counts.parquet", index=False)


# In[7]:


merged_df.head()


# ## From the organoid profiles get the number of cells per organoid
# ## From the single-cell profiles get the number cells without a parent organoid

# In[8]:


organoid_path = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/0.normalized_profiles/organoid_norm_norm_profile.parquet"
)
sc_path = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/0.normalized_profiles/sc_norm_norm_profile.parquet"
)
organoid_counts_output_path = pathlib.Path(
    f"{root_dir}/1.EDA/results/cell_counts/organoid_cell_counts.parquet"
)
organoid_counts_output_path.parent.mkdir(parents=True, exist_ok=True)
organoid_df = pd.read_parquet(organoid_path)
sc_df = pd.read_parquet(sc_path)
metadata_cols_to_group_on = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Well",
]
organoid_cell_count_col = ["Metadata_Object_OrganoidSingleCellCount"]
single_cell_parent_organoid_col = ["Metadata_Object_ParentOrganoid"]


# In[9]:


organoid_cell_counts = organoid_df.groupby(metadata_cols_to_group_on).agg(
    total_cells=pd.NamedAgg(column=organoid_cell_count_col[0], aggfunc="sum"),
    mean_cells_per_organoid=pd.NamedAgg(
        column=organoid_cell_count_col[0], aggfunc="mean"
    ),
    total_number_of_FOVs=pd.NamedAgg(
        column="Metadata_Experiment_WellFOV", aggfunc="nunique"
    ),
)
organoid_cell_counts.reset_index(inplace=True)
organoid_cell_counts["total_cells_normalized_per_FOV"] = (
    organoid_cell_counts["total_cells"] / organoid_cell_counts["total_number_of_FOVs"]
)


# In[10]:


# count the number of single cells that do and do not have a parent organoid
single_cell_counts = sc_df.groupby(metadata_cols_to_group_on).agg(
    n_cells_with_parent_organoid=pd.NamedAgg(
        column=single_cell_parent_organoid_col[0],
        aggfunc=lambda x: sum(1 for val in x if val > 0),
    ),
    n_cells_without_parent_organoid=pd.NamedAgg(
        column=single_cell_parent_organoid_col[0],
        aggfunc=lambda x: sum(1 for val in x if val < 0),
    ),
    total_number_of_FOVs=pd.NamedAgg(
        column="Metadata_Experiment_WellFOV", aggfunc="nunique"
    ),
)
single_cell_counts.reset_index(inplace=True)
single_cell_counts["n_cells_with_parent_organoid_normalized_per_FOV"] = (
    single_cell_counts["n_cells_with_parent_organoid"]
    / single_cell_counts["total_number_of_FOVs"]
)
single_cell_counts["n_cells_without_parent_organoid_normalized_per_FOV"] = (
    single_cell_counts["n_cells_without_parent_organoid"]
    / single_cell_counts["total_number_of_FOVs"]
)
single_cell_counts[
    "proportion_of_cells_with_parent_organoid_compared_to_n_cells_without_parent_organoid"
] = single_cell_counts["n_cells_with_parent_organoid"] / (
    single_cell_counts["n_cells_with_parent_organoid"]
    + single_cell_counts["n_cells_without_parent_organoid"]
)
single_cell_counts[
    "proportion_of_cells_without_parent_organoid_compared_to_n_cells_without_parent_organoid"
] = single_cell_counts["n_cells_without_parent_organoid"] / (
    single_cell_counts["n_cells_with_parent_organoid"]
    + single_cell_counts["n_cells_without_parent_organoid"]
)


# In[11]:


merged_organoid_count_information = pd.merge(
    organoid_cell_counts,
    single_cell_counts,
    on=metadata_cols_to_group_on + ["total_number_of_FOVs"],
    how="outer",
)
merged_organoid_count_information.to_parquet(organoid_counts_output_path, index=False)
