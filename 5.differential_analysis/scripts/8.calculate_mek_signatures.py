#!/usr/bin/env python
# coding: utf-8

# # Calculate MEK inhibitor signatures
# Which morphology features drive the MEK inhibitor (MEKi) profile away from DMSO, Staurosporine and Digoxin, and are those features shared across patients and tumor types?
#
# **Data.** Per-patient log2 fold change vs DMSO per feature, from `0.calculate_log2_fold_change` (organoid and single-cell compartments). The within-patient / within-tumor-type levels below instead use the well-level log2 fold change from `0b.calculate_well_log2_fold_change` (one row per patient x treatment x dose x well, instead of one consensus row per patient x treatment x dose), so a single MEK drug x dose has real replicate wells to test within a patient. The across-patient / across-tumor-type levels still use the consensus (patient-level) data, unaffected by this well split.
#
# **MEKi response per patient** = mean log2 fold change over all MEK inhibitor conditions on that plate (Binimetinib, Mirdametinib, Trametinib, Selumetinib at 1 and 10 uM).
#
# **MEKi group.** The analysis below is repeated once for every `mek_group`: `"All MEKi"` (the pooled response above) plus one group per individual drug x dose (e.g. `"Binimetinib 1 uM"`), so drug-level effects can be compared alongside the pooled one. The across-patient and across-tumor-type levels always have one value per patient regardless of `mek_group`. The within-patient and within-tumor-type levels use well-level replicates (2 wells per patient x drug x dose, 16 wells per patient for the pooled "All MEKi" group), so even a single drug x dose group has a real (if low-powered, df=1) within-patient test instead of being undefined. Comparator contrasts (vs Staurosporine / Digoxin) still subtract that patient's single consensus comparator value from every MEKi well — there's no well-to-well pairing between a MEKi well and a comparator well.
#
# **Contrasts per patient and feature**
# - MEKi vs DMSO: MEKi response (already relative to DMSO)
# - MEKi vs Staurosporine: MEKi response - Staurosporine log2 fold change
# - MEKi vs Digoxin: MEKi response - Digoxin log2 fold change
#
# **Across patients, per feature:** mean contrast, one-sample t-test vs 0 (patients are the replicates), Benjamini-Hochberg q per compartment x mek_group x contrast,
# the fraction of patients whose contrast has the same sign as the mean, and the mean within each tumor type.
# Features are ranked by |t| (large and consistent across patients).
#
# **MEK drug consistency:** the mean log2 fold change of each MEK drug x dose separately, to check that the top MEKi features are shared across drugs.
#
# **Significance at four levels, per contrast and feature** (Benjamini-Hochberg q per compartment x mek_group x contrast x level x group):
# - Within patient: one-sample t-test vs 0 over that patient's MEK condition wells (16 well-level values for "All MEKi", 2 for a single drug x dose group); for the comparator contrasts each MEK well minus the patient's comparator.
# - Across patients: one-sample t-test vs 0 over the per-patient MEKi responses (patients are the replicates); identical to `feature_tests`'s numbers.
# - Within tumor type: one-sample t-test vs 0 over the patient x MEK condition well values of every patient in that tumor type, pooled (MPNST and Other only have 2 patients, so patients alone are too few replicates; wells within a patient are not independent, so these p-values are optimistic).
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


# In[ ]:


results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")
log2fc_path_dict = {
    "organoid": pathlib.Path(results_dir, "organoid_log2fc.parquet"),
    "single_cell": pathlib.Path(results_dir, "single_cell_log2fc.parquet"),
}
well_log2fc_path_dict = {
    "organoid": pathlib.Path(results_dir, "organoid_well_log2fc.parquet"),
    "single_cell": pathlib.Path(results_dir, "single_cell_well_log2fc.parquet"),
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
MEK_DOSES = (1.0, 10.0)
MEK_GROUPS = {"All MEKi": [(drug, dose) for drug in MEK_DRUGS for dose in MEK_DOSES]}
MEK_GROUPS.update(
    {f"{drug} {dose:g} uM": [(drug, dose)] for drug in MEK_DRUGS for dose in MEK_DOSES}
)
COMPARATORS = {"Staurosporine": "MEKi vs Staurosporine", "Digoxin": "MEKi vs Digoxin"}
DMSO_CONTRAST = "MEKi vs DMSO"
TUMOR_TYPES = ["cNF", "pNF", "MPNST", "Other"]


# In[ ]:


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (NaN kept as NaN).

    Parameters
    ----------
    p_values : np.ndarray
        Array of p-values.

    Returns
    -------
    np.ndarray
        Benjamini-Hochberg adjusted p-values (q-values). NaNs are preserved.
    """
    p_values = np.asarray(p_values, dtype=float)
    q_values = np.full_like(p_values, np.nan)
    valid = ~np.isnan(p_values)
    q_values[valid] = stats.false_discovery_control(p_values[valid], method="bh")
    return q_values


def mek_condition_mask(patient_df: pd.DataFrame, mek_conditions: list) -> pd.Series:
    """
    Boolean mask selecting the rows of ``patient_df`` that belong to a MEKi group.

    Parameters
    ----------
    patient_df : pd.DataFrame
        DataFrame containing treatment and dose columns for a single patient.
    mek_conditions : list
        List of (treatment, dose) tuples defining the MEKi group.

    Returns
    -------
    pd.Series
        Boolean mask, True for rows matching any (treatment, dose) in mek_conditions.
    """
    treatment = patient_df["Metadata_Experiment_Treatment"]
    dose = patient_df["Metadata_Experiment_Dose"]
    mask = pd.Series(False, index=patient_df.index)
    for drug, drug_dose in mek_conditions:
        mask |= (treatment == drug) & (dose == drug_dose)
    return mask


def patient_contrasts(
    log2fc_df: pd.DataFrame, feature_cols: list, mek_conditions: list
) -> pd.DataFrame:
    """
    Long table of per-patient contrast values (MEKi vs DMSO / comparator).

    Parameters
    ----------
    log2fc_df : pd.DataFrame
        DataFrame containing log2 fold change values for each patient and treatment.
    feature_cols : list
        List of feature columns to consider for contrasts.
    mek_conditions : list
        List of (treatment, dose) tuples defining the MEKi group.

    Returns
    -------
    pd.DataFrame
        Long table of per-patient contrast values (MEKi vs DMSO / comparator).
    """
    contrast_list = []
    for patient, patient_df in log2fc_df.groupby("Metadata_Biology_PatientTumor"):
        treatment = patient_df["Metadata_Experiment_Treatment"]
        mek = patient_df.loc[
            mek_condition_mask(patient_df, mek_conditions), feature_cols
        ].mean()
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


def wilcoxon_signed_rank(values: np.ndarray) -> tuple:
    """
    Wilcoxon signed-rank test vs 0 (NaN when fewer than 2 non-zero differences).

    Parameters
    ----------
    values : np.ndarray
        Array of per-patient contrast values.

    Returns
    -------
    tuple
        (statistic, p-value) from the Wilcoxon signed-rank test, or (nan, nan)
        when there are fewer than 2 non-zero differences.
    """
    if (values != 0).sum() < 2:
        return np.nan, np.nan
    return stats.wilcoxon(values)


def feature_tests(contrast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per contrast x feature: mean, t-test and Wilcoxon signed-rank across patients, sign agreement, tumor type means.

    Parameters
    ----------
    contrast_df : pd.DataFrame
        Long table of per-patient contrast values (MEKi vs DMSO / comparator).

    Returns
    -------
    pd.DataFrame
        DataFrame containing per contrast x feature statistics, including mean, t-test, Wilcoxon signed-rank, sign agreement, and tumor type means.
    """
    rows = []
    for (contrast, feature), block in contrast_df.dropna(subset=["value"]).groupby(
        ["contrast", "feature"]
    ):
        values = block["value"].to_numpy()
        mean = values.mean()
        sd = values.std(ddof=1)
        t, p = stats.ttest_1samp(values, 0)
        wilcoxon_stat, wilcoxon_p = wilcoxon_signed_rank(values)
        row = {
            "contrast": contrast,
            "feature": feature,
            "n_patients": len(values),
            "mean": mean,
            "sd": sd,
            "ci95": stats.t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)),
            "t": t,
            "p": p,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_p": wilcoxon_p,
            "frac_patients_same_sign": (np.sign(values) == np.sign(mean)).mean(),
        }
        tumor_type_means = block.groupby("Metadata_Biology_TumorType")["value"].mean()
        for tumor_type in TUMOR_TYPES:
            row[f"mean_{tumor_type}"] = tumor_type_means.get(tumor_type, np.nan)
        rows.append(row)
    test_df = pd.DataFrame(rows)
    test_df["q"] = test_df.groupby("contrast")["p"].transform(bh_fdr)
    test_df["wilcoxon_q"] = test_df.groupby("contrast")["wilcoxon_p"].transform(bh_fdr)
    # Features with undefined t (zero variance across patients, or a single patient)
    # keep a missing rank; method="min" gives integer ranks so the column can be Int64.
    test_df["rank"] = (
        test_df.groupby("contrast")["t"]
        .transform(lambda t: t.abs().rank(ascending=False, method="min"))
        .astype("Int64")
    )
    return test_df


def patient_condition_contrasts(
    log2fc_df: pd.DataFrame, feature_cols: list, mek_conditions: list
) -> pd.DataFrame:
    """
    Wide table of per patient x MEK condition contrast values (MEKi vs DMSO / comparator).

    One row per patient x MEK condition in ``mek_conditions`` (the pre-split consensus
    grain). Used for the across_patients / across_tumor_types_* levels, so those stay
    identical to ``feature_tests``'s per-patient numbers regardless of ``mek_group``.

    Parameters
    ----------
    log2fc_df : pd.DataFrame
        Long table of per-patient log2 fold change values.
    feature_cols : list
        List of feature columns to consider for contrasts.
    mek_conditions : list
        List of (treatment, dose) tuples defining the MEKi group.

    Returns
    -------
    pd.DataFrame
        Wide table of per patient x MEK condition contrast values (MEKi vs DMSO / comparator).
    """
    contrast_list = []
    for _, patient_df in log2fc_df.groupby("Metadata_Biology_PatientTumor"):
        treatment = patient_df["Metadata_Experiment_Treatment"]
        mek_df = patient_df.loc[mek_condition_mask(patient_df, mek_conditions)]
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


def patient_condition_well_contrasts(
    log2fc_df: pd.DataFrame,
    well_log2fc_df: pd.DataFrame,
    feature_cols: list,
    mek_conditions: list,
) -> pd.DataFrame:
    """
    Wide table of per patient x MEK condition well contrast values (MEKi vs DMSO / comparator).

    Like ``patient_condition_contrasts``, but the MEKi rows come from well-level data
    (one row per patient x treatment x dose x well) instead of the consensus grain, so a
    single drug x dose group has real within-patient replicates. Used only for the
    within_patient / within_tumor_type levels.

    Parameters
    ----------
    log2fc_df : pd.DataFrame
        Long table of per-patient consensus log2 fold change values, used only for the
        comparator (Staurosporine / Digoxin) reference value.
    well_log2fc_df : pd.DataFrame
        Long table of per-patient x well log2 fold change values, used for the MEKi
        condition rows.
    feature_cols : list
        List of feature columns to consider for contrasts.
    mek_conditions : list
        List of (treatment, dose) tuples defining the MEKi group.

    Returns
    -------
    pd.DataFrame
        Wide table of per patient x MEK condition well contrast values (MEKi vs DMSO / comparator).
    """
    contrast_list = []
    for patient, patient_df in log2fc_df.groupby("Metadata_Biology_PatientTumor"):
        treatment = patient_df["Metadata_Experiment_Treatment"]
        well_patient_df = well_log2fc_df.loc[
            well_log2fc_df["Metadata_Biology_PatientTumor"] == patient
        ]
        mek_df = well_patient_df.loc[
            mek_condition_mask(well_patient_df, mek_conditions)
        ]
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
    """
    Per feature (column): mean and one-sample t-test vs 0 over the rows.

    Parameters
    ----------
    values : pd.DataFrame
        DataFrame of feature values to test.

    Returns
    -------
    pd.DataFrame
        DataFrame containing per feature statistics, including mean, t-test statistic, and p-value.
    """
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


def level_tests(
    condition_df: pd.DataFrame, well_condition_df: pd.DataFrame, feature_cols: list
) -> pd.DataFrame:
    """
    Per contrast x feature: significance within/across patients and tumor types.

    within_patient / within_tumor_type come from ``well_condition_df`` (well-level
    replicates); across_patients / across_tumor_types_* come from ``condition_df``
    (one row per patient x MEK condition, consensus grain), matching ``feature_tests``.

    Parameters
    ----------
    condition_df : pd.DataFrame
        DataFrame containing consensus-grain contrast values for each patient and tumor type.
    well_condition_df : pd.DataFrame
        DataFrame containing well-grain contrast values for each patient and tumor type.
    feature_cols : list
        List of feature columns to consider for significance testing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing per contrast x feature significance statistics.
    """
    patient_df = (
        condition_df.groupby(
            ["contrast", "Metadata_Biology_PatientTumor", "Metadata_Biology_TumorType"]
        )[feature_cols]
        .mean()
        .reset_index()
    )
    level_list = []
    for (contrast, patient), block in well_condition_df.groupby(
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
    for (contrast, tumor_type), block in well_condition_df.groupby(
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

# In[ ]:


test_list, patient_list, condition_list, level_list = [], [], [], []
for compartment, log2fc_path in log2fc_path_dict.items():
    log2fc_df = pd.read_parquet(log2fc_path)
    well_log2fc_df = pd.read_parquet(well_log2fc_path_dict[compartment])
    feature_cols = [col for col in log2fc_df.columns if not col.startswith("Metadata_")]

    for mek_group, mek_conditions in MEK_GROUPS.items():
        contrast_df = patient_contrasts(log2fc_df, feature_cols, mek_conditions).assign(
            compartment=compartment, mek_group=mek_group
        )
        test_df = feature_tests(contrast_df).assign(
            compartment=compartment, mek_group=mek_group
        )
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
            level_tests(
                patient_condition_contrasts(log2fc_df, feature_cols, mek_conditions),
                patient_condition_well_contrasts(
                    log2fc_df, well_log2fc_df, feature_cols, mek_conditions
                ),
                feature_cols,
            )
            .assign(compartment=compartment, mek_group=mek_group)
            .merge(
                test_df[["feature", *parsed.columns]].drop_duplicates("feature"),
                on="feature",
            )
        )
        test_list.append(test_df)
        patient_list.append(contrast_df)
        level_list.append(level_df)

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
    condition_list.append(condition_df)

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
    .groupby(["compartment", "mek_group", "contrast"])
    .agg(
        n_features=("feature", "size"),
        n_patients=("n_patients", "max"),
        features_q_below_0_05=("significant", "sum"),
    )
    .to_string()
)


# In[ ]:


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
for (compartment, mek_group, contrast), block in feature_test_df.groupby(
    ["compartment", "mek_group", "contrast"]
):
    print(f"\n{compartment} | {mek_group} | {contrast}: top 10 features by |t|")
    print(block.nsmallest(10, "rank")[top_columns].round(3).to_string(index=False))


# ## Significant features per level

# In[ ]:


level_summary_df = (
    level_test_df.assign(significant=lambda x: x["q"] < 0.05)
    .groupby(["compartment", "mek_group", "contrast", "level", "group"])["significant"]
    .sum()
    .groupby(["compartment", "mek_group", "contrast", "level"])
    .agg(n_groups="size", median_per_group="median", max_per_group="max")
)
level_summary_df


# ## Feature categories among significant features

# In[ ]:


category_df = (
    feature_test_df.loc[feature_test_df["q"] < 0.05]
    .groupby(
        ["compartment", "mek_group", "contrast", "feature_object", "feature_category"],
        dropna=False,
    )
    .size()
    .rename("n_features")
    .reset_index()
)
category_df.sort_values(
    ["compartment", "mek_group", "contrast", "n_features"],
    ascending=[True, True, True, False],
).groupby(["compartment", "mek_group", "contrast"]).head(5)
