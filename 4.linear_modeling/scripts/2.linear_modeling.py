#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from notebook_init_utils import init_notebook
from statsmodels.stats.multitest import multipletests

root_dir, in_notebook = init_notebook()
# number of models fit concurrently (-1 = all cores)
N_JOBS = -1
warnings.filterwarnings("ignore")  # Ignore all warnings
warnings.simplefilter("ignore")  # Additional suppression method

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


profile_dict = {
    "organoid_norm": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "data/profiles_3D/all_patients/0.normalized_profiles/organoid_norm_norm_profile.parquet",
        ),
        "output_profile_path": pathlib.Path(
            root_dir, "4.linear_modeling/results/linear_modeling/organoid_norm.parquet"
        ),
    },
    "single_cell_norm": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "data/profiles_3D/all_patients/0.normalized_profiles/sc_norm_norm_profile.parquet",
        ),
        "output_profile_path": pathlib.Path(
            root_dir, "4.linear_modeling/results/linear_modeling/sc_norm.parquet"
        ),
    },
    "organoid_agg": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/data/organoid_norm_aggregated_profile.parquet",
        ),
        "output_profile_path": pathlib.Path(
            root_dir, "4.linear_modeling/results/linear_modeling/organoid_agg.parquet"
        ),
    },
    "single_cell_agg": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/data/sc_norm_aggregated_profile.parquet",
        ),
        "output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/single_cell_agg.parquet",
        ),
    },
}


# ## Linear Modeling
#
# The goal here is not prediction but **inference**: for each patient and morphology feature, we
# fit one linear model per treatment/dose combination (vs. DMSO), within that patient only, and
# ask which terms (treatment, count covariates) are significantly associated with that feature,
# after accounting for the others.
#
# This produces many separate model fits -- one per (feature x drug/dose combo x patient) --
# each contributing a p-value per term. Because a p-value's meaning depends on what hypothesis
# family it belongs to, multiple-testing correction (FDR, Benjamini-Hochberg) is run
# **separately per term** (e.g. all "treatment" p-values together) but pooled **across all
# features, drug/dose combos, and patients** within that term. This keeps each term's inference
# robust (correcting over its full family of tests) without over-correcting by mixing unrelated
# hypotheses (e.g. treatment effects vs. count covariate effects) into a single correction.
#
# **General form:**
#
# $$y = \beta_0 + x_1\beta_1 + x_2\beta_2 + \dots + x_n\beta_n + \epsilon$$
#
# **Model specification:**
#
# $$y \sim \text{txt} + \text{cell count} + \text{organoid count} + \text{cell/organoid count}$$
#
# $$y = \beta_0 + x_1\beta_1 + x_2\beta_2 + x_3\beta_3 + x_4\beta_4 + \epsilon$$
#
# **Where:**
#
# | $x$ | $\beta$ | Description |
# |-----|---------|--------------|
# | — | $\beta_0$ | Intercept |
# | $x_1$ | $\beta_1$ | Treatment (e.g., control, drug + dosage), fit within a single patient |
# | $x_2$ | $\beta_2$ | Cell count |
# | $x_3$ | $\beta_3$ | Organoid count |
# | $x_4$ | $\beta_4$ | Cell/organoid count |
#
# $y$ = feature to predict, $\epsilon$ = error term
#
# **For each model (feature), we compute the following statistics:**
#
# - **R-squared** / **adjusted R-squared**: Proportion of variance explained by the model.
# - **F-statistic**: Overall significance of the model.
# - **Coefficients**: Effect size of each predictor.
# - **p-value** / **FDR**: Significance of each term (`pvalue`), Benjamini-Hochberg corrected per term (`pvalue_fdr`).
# - **Variance shares**: each term's type II sum of squares as a % of the total variance (`term_pct_of_total_var`), and the residual share (`residual_pct`).
#
# **Profiles and outputs.** Four profiles are modelled, each saved as long-form parquet (one row per model x term) in `results/linear_modeling/`:
#
# | Profile | Input | Output |
# |---------|-------|--------|
# | organoid | normalized organoid profile | `organoid_norm.parquet` |
# | single cell | normalized single-cell profile (organoid count taken from the organoid profile by patient + well) | `sc_norm.parquet` |
# | organoid, well-aggregated | `data/organoid_norm_aggregated_profile.parquet` | `organoid_agg.parquet` |
# | single cell, well-aggregated | `data/sc_norm_aggregated_profile.parquet` | `single_cell_agg.parquet` |
#
# Positional (`CMI`) columns are dropped, the `NF0037_T1_CQ1` patient is excluded, and the dose and unit are appended to the treatment (for example `Trametinib_1uM`) so each dose is its own model. Models are fit in parallel (`joblib`), and a profile whose output file already exists is skipped.
#

# In[4]:


def fit_combo(
    patient,
    combo,
    df_trt,
    feature_columns,
    count_columns,
    drug_name,
    therapeutic_category,
):
    """Fit one OLS model per feature for a single (patient, treatment combo).

    Runs in a worker process, so it takes and returns plain picklable objects.
    Returns a list of long-form result rows (one per feature x term).
    """
    warnings.filterwarnings("ignore")
    rows = []
    for col in feature_columns:
        # skip features with no observed values in this combo (e.g. a channel
        # that was not imaged/segmented for this patient/treatment) --
        # an all-NaN outcome leaves an empty design matrix after patsy drops
        # the missing rows, which smf.ols cannot fit
        if df_trt[col].notna().sum() == 0:
            continue
        # Prepare the formula for the linear model:
        # y ~ txt + object_count + cell/organoid_count
        lm_equation_terms = " + ".join(
            ["C(Metadata_treatment_full)"] + [f"Q('{c}')" for c in count_columns]
        )
        formula = f"Q('{col}') ~ {lm_equation_terms}"
        results = smf.ols(formula=formula, data=df_trt).fit()
        # total (SST), model-explained (SSE) and residual (SSR) sums of
        # squares, plus a type II ANOVA for each term's share of SST
        sst = results.centered_tss
        sse = results.ess
        anova_table = sm.stats.anova_lm(results, typ=2)
        anova_table["pct_of_total_var"] = anova_table["sum_sq"] / sst * 100

        # (output term name, anova key, coefficient name in the fit)
        # treatment effect within this patient, then the count covariates
        term_specs = [
            (
                "treatment",
                "C(Metadata_treatment_full)",
                f"C(Metadata_treatment_full)[T.{combo[1]}]",
            )
        ] + [
            (covariate, f"Q('{covariate}')", f"Q('{covariate}')")
            for covariate in count_columns
        ]
        for term, anova_key, param_name in term_specs:
            rows.append(
                {
                    "term": term,
                    "patient": patient,
                    "treatment": combo[1],
                    "drug": drug_name,
                    "therapeutic_category": therapeutic_category,
                    "feature": col,
                    "rsquared": results.rsquared,
                    "rsquared_adj": results.rsquared_adj,
                    "fvalue": results.fvalue,
                    # residual of the fit: sum of squares, variance (mean
                    # squared error, df-adjusted) and share of total variance
                    "residual_ss": results.ssr,
                    "residual_variance": results.mse_resid,
                    "residual_pct": results.ssr / sst * 100,
                    "sst": sst,
                    "sse": sse,
                    "explained_pct": sse / sst * 100,
                    "term_sum_sq": anova_table.loc[anova_key, "sum_sq"],
                    "term_pct_of_total_var": anova_table.loc[
                        anova_key, "pct_of_total_var"
                    ],
                    "pvalue": results.pvalues[param_name].item(),
                    "coefficient": results.params[param_name].item(),
                    "intercept": results.params["Intercept"].item(),
                }
            )
    return rows


# In[5]:


for profile in tqdm(profile_dict.keys(), desc="Loading profiles"):
    # set the output dictionary for linear modeling results
    # per profile. Results are stored long-form: one row per
    # (combo, feature, term), where "term" identifies which piece of
    # the model specification the coefficient/pvalue belongs to
    # (treatment, cell_count, organoid_count, cell_per_organoid_count).
    if profile_dict[profile]["output_profile_path"].exists():
        continue  # skip if the output already exists

    df = pd.read_parquet(profile_dict[profile]["input_profile_path"])
    # if the column contains CMI then drop the column since
    # it is a postitional marker and not a feature
    cmi_columns = [col for col in df.columns if "CMI" in col]
    df = df.drop(columns=cmi_columns)
    df = df.rename(
        columns={
            "Metadata_Biology_PatientTumor": "patient",
            "Metadata_Experiment_Treatment": "treatment",
        }
    )
    # drop the NF0037_T1_CQ1 patient
    df = df.loc[df["patient"] != "NF0037_T1_CQ1"]
    # a few rows (e.g. NF0018_T6) are missing the experimental metadata
    # (treatment, dose, tumor type) from an incomplete metadata join;
    # drop them so they do not form a spurious "nan" treatment group
    df = df.loc[df["treatment"].notna()]
    # combine treatment, dose, and unit into a single column so that
    # different doses of the same treatment are modeled as distinct groups
    df["Metadata_treatment_full"] = (
        df["treatment"].astype(str)
        + "_"
        + df["Metadata_Experiment_Dose"].astype(str)
        + df["Metadata_Experiment_Unit"].astype(str)
    )
    # map each combined treatment label back to its raw drug name and
    # therapeutic category (MOA) so the modeling results can be grouped
    # by drug/MOA downstream without re-merging metadata
    treatment_meta = (
        df[
            [
                "Metadata_treatment_full",
                "treatment",
                "Metadata_Experiment_TherapeuticCategories",
            ]
        ]
        .drop_duplicates()
        .set_index("Metadata_treatment_full")
    )

    # build the count covariates used in the model specification:
    # cell count, organoid count, and cell/organoid count
    if profile == "single_cell_norm":
        # the single-cell profile has no organoid count of its own,
        # so pull it in from the organoid-level profile via patient + well
        organoid_counts_df = pd.read_parquet(
            profile_dict["organoid_norm"]["input_profile_path"],
            columns=[
                "Metadata_Biology_PatientTumor",
                "Metadata_Experiment_Well",
                "Metadata_WellOrganoidCount",
            ],
        ).drop_duplicates()
        organoid_counts_df = organoid_counts_df.rename(
            columns={"Metadata_Biology_PatientTumor": "patient"}
        )
        df = df.merge(
            organoid_counts_df, on=["patient", "Metadata_Experiment_Well"], how="left"
        )
        # drop rows whose (patient, well) has no matching organoid-level data
        df = df.dropna(subset=["Metadata_WellOrganoidCount"])
        df["cell_count"] = df["Metadata_Object_WellSingleCellCount"]
        df["organoid_count"] = df["Metadata_WellOrganoidCount"]
    elif profile == "single_cell_agg":
        # the aggregated single-cell profile (one row per patient + well) carries
        # no count columns, so pull the well-level organoid count from the
        # organoid profile and the well-level cell count from the single-cell
        # profile; both are constant within a (patient, well)
        organoid_counts_df = (
            pd.read_parquet(
                profile_dict["organoid_norm"]["input_profile_path"],
                columns=[
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Experiment_Well",
                    "Metadata_WellOrganoidCount",
                ],
            )
            .drop_duplicates()
            .rename(columns={"Metadata_Biology_PatientTumor": "patient"})
        )
        cell_counts_df = (
            pd.read_parquet(
                profile_dict["single_cell_norm"]["input_profile_path"],
                columns=[
                    "Metadata_Biology_PatientTumor",
                    "Metadata_Experiment_Well",
                    "Metadata_Object_WellSingleCellCount",
                ],
            )
            .drop_duplicates()
            .rename(columns={"Metadata_Biology_PatientTumor": "patient"})
        )
        df = df.merge(
            organoid_counts_df, on=["patient", "Metadata_Experiment_Well"], how="left"
        ).merge(cell_counts_df, on=["patient", "Metadata_Experiment_Well"], how="left")
        # drop rows whose (patient, well) has no matching well-level counts
        df = df.dropna(
            subset=["Metadata_WellOrganoidCount", "Metadata_Object_WellSingleCellCount"]
        )
        df["cell_count"] = df["Metadata_Object_WellSingleCellCount"]
        df["organoid_count"] = df["Metadata_WellOrganoidCount"]
    else:
        df["cell_count"] = df["Metadata_Object_OrganoidSingleCellCount"]
        df["organoid_count"] = df["Metadata_WellOrganoidCount"]

    df["cell_per_organoid_count"] = df["cell_count"] / df["organoid_count"]

    count_columns = ["cell_count", "organoid_count", "cell_per_organoid_count"]
    metadata_columns = (
        ["patient", "treatment", "Metadata_Biology_TumorType"]
        + count_columns
        + [col for col in df.columns if col.startswith("Metadata_")]
    )
    # rename feature columns as the "." dod not play nice with the formula
    # the linear model interprets the "." as an operator and not as part of the column name
    # track the sanitized -> original name mapping so the original feature
    # names can be recovered after the results are loaded in any other
    # environment/notebook (the "feature" column in the output only ever
    # contains the sanitized names)
    sanitized_to_original_col_map = {}
    for col in df.columns:
        new_col = col.replace(
            ".", "__"
        )  # Replace . with empty string for compatibility in formula
        sanitized_to_original_col_map[new_col] = col
        df.rename(columns={col: new_col}, inplace=True)
    # redefine the feature columns after renaming
    feature_columns = [col for col in df.columns if col not in metadata_columns]

    # Filter for specific treatment/dose combinations
    # DMSO's combined label is consistent across all patients
    dmso_label = df.loc[df["treatment"] == "DMSO", "Metadata_treatment_full"].unique()[
        0
    ]
    # build one independent task per (patient, treatment combo); each task
    # fits every feature for that combo, and tasks run in parallel
    tasks = []
    for patient in sorted(df["patient"].unique()):
        df_pat = df.loc[df["patient"] == patient]
        # each model is fit within this single patient, so there is nothing
        # to compare against if the patient has no DMSO rows
        if dmso_label not in df_pat["Metadata_treatment_full"].unique():
            continue
        combo_list = [
            (dmso_label, i)
            for i in df_pat["Metadata_treatment_full"].unique()
            if i != dmso_label
        ]
        for combo in combo_list:
            drug_name = treatment_meta.loc[combo[1], "treatment"]
            therapeutic_category = treatment_meta.loc[
                combo[1], "Metadata_Experiment_TherapeuticCategories"
            ]

            df_trt = df_pat.loc[df_pat["Metadata_treatment_full"].isin(combo)]
            # order the treatment column to ensure DMSO is first (reference level)
            df_trt = df_trt.copy()
            df_trt["Metadata_treatment_full"] = pd.Categorical(
                df_trt["Metadata_treatment_full"], categories=[combo[0], combo[1]]
            )
            # zero-center the continuous covariates (per patient, per combo) for
            # numerical stability in the OLS fit -- a linear shift with no
            # rescaling leaves the fit (and all coefficients except the
            # intercept) unchanged, so this is purely a conditioning
            # improvement, not a modeling choice
            df_trt[count_columns] = df_trt[count_columns] - df_trt[count_columns].mean()
            # only ship the columns the models need to the workers
            df_trt = df_trt[
                ["Metadata_treatment_full"] + count_columns + feature_columns
            ]
            tasks.append(
                (
                    patient,
                    combo,
                    df_trt,
                    feature_columns,
                    count_columns,
                    drug_name,
                    therapeutic_category,
                )
            )

    # run the models in parallel across all available cores; results come
    # back in task order so the output is identical to a serial run
    task_results = Parallel(n_jobs=N_JOBS, return_as="generator")(
        delayed(fit_combo)(*task) for task in tasks
    )
    linear_modeling_results_rows = []
    for rows in tqdm(
        task_results, total=len(tasks), desc="Fitting models", unit="combo"
    ):
        linear_modeling_results_rows.extend(rows)
    linear_modeling_results_df = pd.DataFrame(linear_modeling_results_rows)
    # map the sanitized feature names back to their original (pre-".": removal)
    # names so downstream consumers loading this parquet in any other
    # env/notebook can recover the native column name without needing access
    # to the sanitization logic above
    linear_modeling_results_df["feature_original"] = linear_modeling_results_df[
        "feature"
    ].map(sanitized_to_original_col_map)
    # split the feature column into multiple columns
    # feature names follow the pattern: Compartment_Channel_Feature_type_Measurement
    linear_modeling_results_df[
        ["Compartment", "Channel", "Feature_type", "Measurement"]
    ] = linear_modeling_results_df["feature"].str.split("_", n=3, expand=True)

    # if feature type is area shape then make the measurement the channel and
    # set the channel to None
    # this because area size shape features are not channel specific
    linear_modeling_results_df.loc[
        linear_modeling_results_df["Feature_type"] == "AreaSizeShape", "Measurement"
    ] = linear_modeling_results_df["Channel"]
    linear_modeling_results_df.loc[
        linear_modeling_results_df["Feature_type"] == "AreaSizeShape", "Channel"
    ] = None
    # set compartment to None if is adjacent
    # this is because adjacent features are not compartment specific
    linear_modeling_results_df.loc[
        linear_modeling_results_df["Compartment"] == "adjacent", "Compartment"
    ] = None

    # run FDR on the p-values, separately per term since each term is its
    # own hypothesis-testing family (different scale/behavior of p-values)
    linear_modeling_results_df["pvalue_fdr"] = float("nan")
    for term, group_index in linear_modeling_results_df.groupby("term").groups.items():
        # a handful of models are degenerate (e.g. a constant feature for one
        # patient/combo) and produce a NaN p-value; multipletests propagates a
        # single NaN to every p-value in the array, so those rows must be
        # excluded from the correction rather than passed through -- they stay
        # pvalue_fdr=NaN (never significant) rather than corrupting every
        # other row's FDR value in this term
        valid_index = (
            linear_modeling_results_df.loc[group_index].dropna(subset=["pvalue"]).index
        )
        pvals = linear_modeling_results_df.loc[valid_index, "pvalue"].values
        _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
        linear_modeling_results_df.loc[valid_index, "pvalue_fdr"] = pvals_fdr

    # map the features back to their original names inplace such that the
    # original feature names are preserved in the output parquet and can be used
    # for downstream grouping/merging without needing to re-run the sanitization
    linear_modeling_results_df["feature"] = linear_modeling_results_df[
        "feature_original"
    ]
    # Save the updated DataFrame with FDR p-values
    profile_dict[profile]["output_profile_path"].parent.mkdir(
        parents=True, exist_ok=True
    )
    linear_modeling_results_df.to_parquet(
        profile_dict[profile]["output_profile_path"], index=False
    )
