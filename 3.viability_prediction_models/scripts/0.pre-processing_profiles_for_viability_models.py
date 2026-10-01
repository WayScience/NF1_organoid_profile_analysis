#!/usr/bin/env python
# coding: utf-8

# ## Combine the profiles, viabilities, and platemap information
# this information is the new profiles will be already annotated in ad thus we will not need to do this step.

# In[1]:


import pathlib

import pandas as pd
from notebook_init_utils import bandicoot_check, init_notebook
from pycytominer import aggregate
from sklearn.preprocessing import minmax_scale

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


patient_ids = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    sep="\t",
    names=["patient_id"],
)["patient_id"].to_list()


# ## Get all of the morphology profiles to work with

# In[3]:


# Consensus strata: one row per (patient, treatment) combination
consensus_strata_3D = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Class",
    "Metadata_Experiment_Target",
    "Metadata_Experiment_TherapeuticCategories",
    "Metadata_Experiment_Unit",
    "Metadata_Experiment_ViabilityPercentage",
]
consensus_strata_2D = [
    "Metadata_patient_tumor",
    "Metadata_treatment",
    "Metadata_dose",
    "Metadata_class",
    "Metadata_target",
    "Metadata_therapeutic_categories",
    "Metadata_dose_unit",
    "Metadata_Experiment_ViabilityPercentage",
]


# In[4]:


consensus_profiles_3D_paths = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/0.normalized_profiles/"
).resolve(strict=True)
consensus_profiles_3D_paths = list(consensus_profiles_3D_paths.glob("*.parquet"))
consensus_profiles_2D_paths = [
    pathlib.Path(
        f"{root_dir}/data/profiles_2D/all_patients/max_projection/organoid_profiles.parquet"
    ).resolve(strict=True),
    pathlib.Path(
        f"{root_dir}/data/profiles_2D/all_patients/max_projection/sc_profiles.parquet"
    ).resolve(strict=True),
]
paths_dict = {
    "3D": consensus_profiles_3D_paths,
    "2D": consensus_profiles_2D_paths,
}
paths_dict


# In[5]:


# get the well_fov, patient, and viability to merge into 2D
df = pd.read_parquet(consensus_profiles_3D_paths[0])
patient_treatment_viabilty = df[
    [
        "Metadata_Biology_PatientTumor",
        "Metadata_Experiment_Treatment",
        "Metadata_Experiment_ViabilityPercentage",
    ]
].drop_duplicates()


# In[6]:


for dimension in paths_dict.keys():
    for profile_path in paths_dict[dimension]:
        print(f"Processing {profile_path.name} ({dimension})...")

        consensus_output_path = pathlib.Path(
            f"{root_dir}/3.viability_prediction_models/data/processed_profiles_{dimension}/{profile_path.stem.replace('_norm', '')}_consensus.parquet"
        ).resolve()
        consensus_output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.read_parquet(profile_path)
        if dimension == "2D":
            print([x for x in df.columns if "metadata" in x.lower()])
            df = df.merge(
                patient_treatment_viabilty,
                left_on=["Metadata_patient_tumor", "Metadata_treatment"],
                right_on=[
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Experiment_Treatment",
                ],
                how="left",
            )
        df["min_max_viability"] = minmax_scale(
            df[["Metadata_Experiment_ViabilityPercentage"]]
        )
        features_columns = [
            col for col in df.columns if not col.startswith("Metadata_")
        ]
        consensus_df = aggregate(
            population_df=df,
            strata=consensus_strata_3D if dimension == "3D" else consensus_strata_2D,
            features=features_columns,
            operation="median",
            output_file=consensus_output_path,
            output_type="parquet",
        )
        consensus_df = pd.read_parquet(consensus_output_path)
        # add viability to the consensus profiles
        consensus_df.to_parquet(consensus_output_path, index=False)
        print(consensus_df.shape)
