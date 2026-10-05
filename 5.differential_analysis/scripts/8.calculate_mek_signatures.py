#!/usr/bin/env python
# coding: utf-8

# # Calculate MEK inhibitor signatures
# Which morphology features drive the MEK inhibitor (MEKi) profile away from DMSO, Staurosporine and Digoxin, and are those features shared across patients and tumor types?
#
# **Data.** Per-patient log2 fold change vs DMSO per feature, from `0.calculate_log2_fold_change` (organoid and single-cell compartments).
#
# **MEKi response per patient** = mean log2 fold change over all MEK inhibitor conditions on that plate (Binimetinib, Mirdametinib, Trametinib, Selumetinib at 1 and 10 uM).
#
# **Contrasts per patient and feature**
# - MEKi vs DMSO: MEKi response (already relative to DMSO)
# - MEKi vs Staurosporine: MEKi response - Staurosporine log2 fold change
# - MEKi vs Digoxin: MEKi response - Digoxin log2 fold change
#
# **Across patients, per feature:** mean contrast, one-sample t-test vs 0 (patients are the replicates), Benjamini-Hochberg q per compartment x contrast,
# the fraction of patients whose contrast has the same sign as the mean, and the mean within each tumor type.
# Features are ranked by |t| (large and consistent across patients).
#
# **MEK drug consistency:** the mean log2 fold change of each MEK drug x dose separately, to check that the top MEKi features are shared across drugs.
#
# **Significance at four levels, per contrast and feature** (Benjamini-Hochberg q per compartment x contrast x level x group):
# - Within patient: one-sample t-test vs 0 over that patient's MEK conditions (4 drugs x 2 doses); for the comparator contrasts each MEK condition minus the patient's comparator.
# - Across patients: one-sample t-test vs 0 over the per-patient MEKi responses (patients are the replicates).
# - Within tumor type: one-sample t-test vs 0 over the patient x MEK condition values of every patient in that tumor type, pooled (MPNST and Other only have 2 patients, so patients alone are too few replicates; conditions within a patient are not independent, so these p-values are optimistic).
# - Across tumor types (shared): one-sample t-test vs 0 over the tumor type means of the per-patient responses (is the effect shared by the tumor types?).
# - Across tumor types (differ): one-way ANOVA of the per-patient responses between tumor types (does the effect depend on tumor type?).

# In[1]:


import pathlib

import numpy as np
import pandas as pd
from eda_helper_utils.utils_analysis import parse_feature_3d
from notebook_init_utils import init_notebook
from scipy import stats

root_dir, in_notebook = init_notebook()


# In[2]:


results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")
log2fc_path_dict = {
    "organoid": pathlib.Path(results_dir, "organoid_log2fc.parquet"),
    "single_cell": pathlib.Path(results_dir, "single_cell_log2fc.parquet"),
}
feature_test_output_path = pathlib.Path(
    results_dir, "mek_contrast_feature_tests.parquet"
)
patient_value_output_path = pathlib.Path(
    results_dir, "mek_contrast_patient_values.parquet"
)
mek_condition_output_path = pathlib.Path(
    results_dir, "mek_condition_feature_means.parquet"
)
level_test_output_path = pathlib.Path(results_dir, "mek_contrast_level_tests.parquet")

MEK_DRUGS = ["Binimetinib", "Mirdametinib", "Trametinib", "Selumetinib"]
COMPARATORS = {"Staurosporine": "MEKi vs Staurosporine", "Digoxin": "MEKi vs Digoxin"}
DMSO_CONTRAST = "MEKi vs DMSO"
TUMOR_TYPES = ["cNF", "pNF", "MPNST", "Other"]


# In[3]:


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (NaN kept as NaN)."""
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid = ~np.isnan(p_values)
    ordered = p_values[valid]
    order = np.argsort(ordered)
    ranked = ordered[order] * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    valid_q = np.empty_like(adjusted)
    valid_q[order] = np.minimum(adjusted, 1)
    q_values[valid] = valid_q
    return q_values


def patient_contrasts(log2fc_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Long table of per-patient contrast values (MEKi vs DMSO / comparator)."""
    contrast_list = []
    for patient, patient_df in log2fc_df.groupby("Metadata_Biology_PatientTumor"):
        treatment = patient_df["Metadata_Experiment_Treatment"]
        mek = patient_df.loc[treatment.isin(MEK_DRUGS), feature_cols].mean()
        contrasts = {DMSO_CONTRAST: mek}
        for comparator, contrast_name in COMPARATORS.items():
            comparator_rows = patient_df.loc[treatment == comparator, feature_cols]
            if len(comparator_rows) == 1:
                contrasts[contrast_name] = mek - comparator_rows.iloc[0]
        for contrast_name, values in contrasts.items():
            contrast_list.append(
                pd.DataFrame(
                    {
                        "contrast": contrast_name,
                        "Metadata_Biology_PatientTumor": patient,
                        "Metadata_Biology_TumorType": patient_df[
                            "Metadata_Biology_TumorType"
                        ].iloc[0],
                        "feature": feature_cols,
                        "value": values.to_numpy(),
                    }
                )
            )
    return pd.concat(contrast_list, ignore_index=True)


def feature_tests(contrast_df: pd.DataFrame) -> pd.DataFrame:
    """Per contrast x feature: mean, t-test across patients, sign agreement, tumor type means."""
    rows = []
    for (contrast, feature), block in contrast_df.dropna(subset=["value"]).groupby(
        ["contrast", "feature"]
    ):
        values = block["value"].to_numpy()
        mean = values.mean()
        sd = values.std(ddof=1)
        t = mean / (sd / np.sqrt(len(values))) if sd > 0 else np.nan
        row = {
            "contrast": contrast,
            "feature": feature,
            "n_patients": len(values),
            "mean": mean,
            "sd": sd,
            "ci95": stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)),
            "t": t,
            "p": 2 * stats.t.sf(abs(t), len(values) - 1) if not np.isnan(t) else np.nan,
            "frac_patients_same_sign": (np.sign(values) == np.sign(mean)).mean(),
        }
        tumor_type_means = block.groupby("Metadata_Biology_TumorType")["value"].mean()
        for tumor_type in TUMOR_TYPES:
            row[f"mean_{tumor_type}"] = tumor_type_means.get(tumor_type, np.nan)
        rows.append(row)
    test_df = pd.DataFrame(rows)
    test_df["q"] = test_df.groupby("contrast")["p"].transform(bh_fdr)
    # Features with undefined t (zero variance across patients, or a single patient)
    # keep a missing rank; method="min" gives integer ranks so the column can be Int64.
    test_df["rank"] = (
        test_df.groupby("contrast")["t"]
        .transform(lambda t: t.abs().rank(ascending=False, method="min"))
        .astype("Int64")
    )
    return test_df


def patient_condition_contrasts(
    log2fc_df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """Wide table of per patient x MEK condition contrast values (MEKi vs DMSO / comparator)."""
    contrast_list = []
    for _, patient_df in log2fc_df.groupby("Metadata_Biology_PatientTumor"):
        treatment = patient_df["Metadata_Experiment_Treatment"]
        mek_df = patient_df.loc[treatment.isin(MEK_DRUGS)]
        metadata_df = mek_df[
            ["Metadata_Biology_PatientTumor", "Metadata_Biology_TumorType"]
        ]
        contrasts = {DMSO_CONTRAST: mek_df[feature_cols]}
        for comparator, contrast_name in COMPARATORS.items():
            comparator_rows = patient_df.loc[treatment == comparator, feature_cols]
            if len(comparator_rows) == 1:
                contrasts[contrast_name] = (
                    mek_df[feature_cols] - comparator_rows.iloc[0]
                )
        for contrast_name, values in contrasts.items():
            contrast_list.append(
                pd.concat([metadata_df, values], axis=1).assign(contrast=contrast_name)
            )
    return pd.concat(contrast_list, ignore_index=True)


def one_sample_tests(values: pd.DataFrame) -> pd.DataFrame:
    """Per feature (column): mean and one-sample t-test vs 0 over the rows."""
    result = stats.ttest_1samp(values.to_numpy(), 0, axis=0)
    return pd.DataFrame(
        {
            "feature": values.columns,
            "n": len(values),
            "mean": values.mean().to_numpy(),
            "statistic": result.statistic,
            "p": result.pvalue,
        }
    )


def level_tests(condition_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Per contrast x feature: significance within/across patients and tumor types."""
    patient_df = (
        condition_df.groupby(
            ["contrast", "Metadata_Biology_PatientTumor", "Metadata_Biology_TumorType"]
        )[feature_cols]
        .mean()
        .reset_index()
    )
    level_list = []
    for (contrast, patient), block in condition_df.groupby(
        ["contrast", "Metadata_Biology_PatientTumor"]
    ):
        level_list.append(
            one_sample_tests(block[feature_cols]).assign(
                contrast=contrast,
                level="within_patient",
                group=patient,
                tumor_type=block["Metadata_Biology_TumorType"].iloc[0],
            )
        )
    for (contrast, tumor_type), block in condition_df.groupby(
        ["contrast", "Metadata_Biology_TumorType"]
    ):
        level_list.append(
            one_sample_tests(block[feature_cols]).assign(
                contrast=contrast,
                level="within_tumor_type",
                group=tumor_type,
                tumor_type=tumor_type,
            )
        )
    for contrast, block in patient_df.groupby("contrast"):
        tumor_type_blocks = [
            tumor_type_block[feature_cols].to_numpy()
            for _, tumor_type_block in block.groupby("Metadata_Biology_TumorType")
        ]
        anova = stats.f_oneway(*tumor_type_blocks, axis=0)
        level_list += [
            one_sample_tests(block[feature_cols]).assign(
                contrast=contrast, level="across_patients", group="all"
            ),
            one_sample_tests(
                block.groupby("Metadata_Biology_TumorType")[feature_cols].mean()
            ).assign(contrast=contrast, level="across_tumor_types_shared", group="all"),
            pd.DataFrame(
                {
                    "feature": feature_cols,
                    "n": len(block),
                    "mean": block[feature_cols].mean().to_numpy(),
                    "statistic": anova.statistic,
                    "p": anova.pvalue,
                }
            ).assign(contrast=contrast, level="across_tumor_types_differ", group="all"),
        ]
    level_df = pd.concat(level_list, ignore_index=True)
    level_df["q"] = level_df.groupby(["contrast", "level", "group"])["p"].transform(
        bh_fdr
    )
    return level_df


# ## Contrasts and per-feature tests

# In[4]:


test_list, patient_list, condition_list, level_list = [], [], [], []
for compartment, log2fc_path in log2fc_path_dict.items():
    log2fc_df = pd.read_parquet(log2fc_path)
    feature_cols = [col for col in log2fc_df.columns if not col.startswith("Metadata_")]

    contrast_df = patient_contrasts(log2fc_df, feature_cols).assign(
        compartment=compartment
    )
    test_df = feature_tests(contrast_df).assign(compartment=compartment)
    parsed = pd.DataFrame(
        [parse_feature_3d(feature) for feature in test_df["feature"]],
        columns=[
            "feature_object",
            "feature_category",
            "feature_channel",
            "feature_measurement",
        ],
    )
    test_df = pd.concat([test_df, parsed], axis=1)
    level_df = (
        level_tests(patient_condition_contrasts(log2fc_df, feature_cols), feature_cols)
        .assign(compartment=compartment)
        .merge(
            test_df[["feature", *parsed.columns]].drop_duplicates("feature"),
            on="feature",
        )
    )

    condition_df = (
        log2fc_df.loc[log2fc_df["Metadata_Experiment_Treatment"].isin(MEK_DRUGS)]
        .groupby(["Metadata_Experiment_Treatment", "Metadata_Experiment_Dose"])[
            feature_cols
        ]
        .mean()
        .stack()
        .rename("mean_log2fc")
        .reset_index()
        .rename(columns={"level_2": "feature"})
        .assign(compartment=compartment)
    )
    test_list.append(test_df)
    patient_list.append(contrast_df)
    condition_list.append(condition_df)
    level_list.append(level_df)

feature_test_df = pd.concat(test_list, ignore_index=True)
patient_value_df = pd.concat(patient_list, ignore_index=True)
mek_condition_df = pd.concat(condition_list, ignore_index=True)
level_test_df = pd.concat(level_list, ignore_index=True)
feature_test_df.to_parquet(feature_test_output_path, index=False)
level_test_df.to_parquet(level_test_output_path, index=False)
patient_value_df.to_parquet(patient_value_output_path, index=False)
mek_condition_df.to_parquet(mek_condition_output_path, index=False)

print(
    feature_test_df.assign(significant=lambda x: x["q"] < 0.05)
    .groupby(["compartment", "contrast"])
    .agg(
        n_features=("feature", "size"),
        n_patients=("n_patients", "max"),
        features_q_below_0_05=("significant", "sum"),
    )
    .to_string()
)


# In[5]:


top_columns = [
    "feature",
    "mean",
    "t",
    "q",
    "frac_patients_same_sign",
    "mean_cNF",
    "mean_pNF",
    "mean_MPNST",
    "mean_Other",
]
for (compartment, contrast), block in feature_test_df.groupby(
    ["compartment", "contrast"]
):
    print(f"\n{compartment} | {contrast}: top 10 features by |t|")
    print(block.nsmallest(10, "rank")[top_columns].round(3).to_string(index=False))


# ## Significant features per level

# In[6]:


level_summary_df = (
    level_test_df.assign(significant=lambda x: x["q"] < 0.05)
    .groupby(["compartment", "contrast", "level", "group"])["significant"]
    .sum()
    .groupby(["compartment", "contrast", "level"])
    .agg(n_groups="size", median_per_group="median", max_per_group="max")
)
level_summary_df


# ## Feature categories among significant features

# In[7]:


category_df = (
    feature_test_df.loc[feature_test_df["q"] < 0.05]
    .groupby(
        ["compartment", "contrast", "feature_object", "feature_category"], dropna=False
    )
    .size()
    .rename("n_features")
    .reset_index()
)
category_df.sort_values(
    ["compartment", "contrast", "n_features"], ascending=[True, True, False]
).groupby(["compartment", "contrast"]).head(5)
