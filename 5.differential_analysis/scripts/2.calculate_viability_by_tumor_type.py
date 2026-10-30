#!/usr/bin/env python
# coding: utf-8

# # Calculate viability per tumor type
# Summarize the per-patient viability (% of DMSO, and its log2 fold change vs DMSO) (from `0.calculate_log2_fold_change`) for each tumor type x treatment.
# Each patient is one observation; the summary is descriptive (mean, SD, median, n patients).
# Patients without a viability measurement (`viability_log2fc` is missing) are kept, so they still appear in the plot; their values are NaN and they are not counted in `n_patients`.

# In[1]:


import pathlib

import pandas as pd
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()


# In[2]:


results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")
summary_path = pathlib.Path(results_dir, "log2fc_summary.parquet")
viability_path = pathlib.Path(results_dir, "viability_log2fc.parquet")
patient_viability_output_path = pathlib.Path(
    results_dir, "viability_log2fc_by_patient_tumor_type.parquet"
)
tumor_type_viability_output_path = pathlib.Path(
    results_dir, "viability_log2fc_by_tumor_type.parquet"
)

treatment_cols = [
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
    "Metadata_Experiment_Unit",
]


# In[3]:


patient_viability_df = (
    pd.read_parquet(summary_path)[
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Biology_TumorType",
            *treatment_cols,
            "viability_log2fc",
        ]
    ]
    .merge(
        pd.read_parquet(viability_path)[
            [
                "Metadata_Biology_PatientTumor",
                *treatment_cols,
                "Metadata_Viability_Percentage",
            ]
        ],
        on=["Metadata_Biology_PatientTumor", *treatment_cols],
        how="left",
        validate="one_to_one",
    )
    .reset_index(drop=True)
)
patient_viability_df.to_parquet(patient_viability_output_path, index=False)
print(
    patient_viability_df.groupby("Metadata_Biology_TumorType")[
        "Metadata_Biology_PatientTumor"
    ]
    .nunique()
    .rename("n_patients")
    .to_string()
)


# In[4]:


tumor_type_viability_df = (
    patient_viability_df.groupby(
        ["Metadata_Biology_TumorType", *treatment_cols], dropna=False
    )
    .agg(
        n_patients=("viability_log2fc", "count"),
        mean_viability_percent=("Metadata_Viability_Percentage", "mean"),
        sd_viability_percent=("Metadata_Viability_Percentage", "std"),
        median_viability_percent=("Metadata_Viability_Percentage", "median"),
        mean_viability_log2fc=("viability_log2fc", "mean"),
        sd_viability_log2fc=("viability_log2fc", "std"),
        median_viability_log2fc=("viability_log2fc", "median"),
    )
    .reset_index()
)
tumor_type_viability_df["mean_viability_percent_of_dmso"] = (
    100 * 2 ** (tumor_type_viability_df["mean_viability_log2fc"])
)
tumor_type_viability_df.to_parquet(tumor_type_viability_output_path, index=False)
tumor_type_viability_df.sort_values(
    ["Metadata_Experiment_Treatment", "Metadata_Biology_TumorType"]
).head(12)
