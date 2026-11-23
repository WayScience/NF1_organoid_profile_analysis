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
warnings.filterwarnings("ignore")  # Ignore all warnings
warnings.simplefilter("ignore")  # Additional suppression method

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


profile_dict = {
    "organoid_fs": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "data/profiles_3D/all_patients/0.normalized_profiles/organoid_norm_norm_profile.parquet",
        ).resolve(strict=True),
        "output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/organoid_norm_technical_model.parquet",
        ),
        "mek_output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/organoid_norm_technical_model_mek_groups.parquet",
        ),
    },
    "single_cell_fs": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "data/profiles_3D/all_patients/0.normalized_profiles/sc_norm_norm_profile.parquet",
        ).resolve(strict=True),
        "output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/sc_norm_technical_model.parquet",
        ),
        "mek_output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/sc_norm_technical_model_mek_groups.parquet",
        ),
    },
    "organoid_agg": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/data/organoid_norm_aggregated_profile.parquet",
        ).resolve(strict=True),
        "output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/organoid_agg_technical_model.parquet",
        ),
        "mek_output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/organoid_agg_technical_model_mek_groups.parquet",
        ),
    },
    "single_cell_agg": {
        "input_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/data/sc_norm_aggregated_profile.parquet",
        ).resolve(strict=True),
        "output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/single_cell_agg_technical_model.parquet",
        ),
        "mek_output_profile_path": pathlib.Path(
            root_dir,
            "4.linear_modeling/results/linear_modeling/single_cell_agg_technical_model_mek_groups.parquet",
        ),
    },
}

manhattan_distance_df = pd.read_parquet(
    pathlib.Path(
        root_dir,
        "4.linear_modeling/results/well_manhattan_distance/well_manhattan_distance.parquet",
    )
)

# MEK inhibitor (MEKi) groups, as in 5.differential_analysis/8.calculate_mek_signatures:
# "All MEKi" pools every MEK drug x dose into one treatment level, and each
# drug x dose is its own group
MEK_DRUGS = ["Binimetinib", "Mirdametinib", "Trametinib", "Selumetinib"]
MEK_DOSES = (1.0, 10.0)
POOLED_MEK_GROUP = "All MEKi"
MEK_GROUPS = {
    POOLED_MEK_GROUP: [(drug, dose) for drug in MEK_DRUGS for dose in MEK_DOSES]
}
MEK_GROUPS.update(
    {f"{drug} {dose:g} uM": [(drug, dose)] for drug in MEK_DRUGS for dose in MEK_DOSES}
)


# ## Linear Modeling
#
# The goal here is not prediction but **inference**: for each patient and morphology feature, we
# fit one linear model per treatment/dose combination (vs. DMSO), within that patient only, and
# ask which terms (treatment, technical covariates) are significantly associated with that
# feature, after accounting for the others.
#
# This produces many separate model fits -- one per (feature x drug/dose combo x patient) --
# each contributing a p-value per term. Because a p-value's meaning depends on what hypothesis
# family it belongs to, multiple-testing correction (FDR, Benjamini-Hochberg) is run
# **separately per term** (e.g. all "treatment" p-values together) but pooled **across all
# features, drug/dose combos, and patients** within that term. This keeps each term's inference
# robust (correcting over its full family of tests) without over-correcting by mixing unrelated
# hypotheses (e.g. treatment effects vs. technical covariate effects) into a single correction.
#
# **General form:**
#
# $$y = \beta_0 + x_1\beta_1 + x_2\beta_2 + \dots + x_n\beta_n + \epsilon$$
#
# **Model specification:**
#
# $$y \sim \text{treatment} + \text{cell count} + \text{organoid count} + \text{cell/organoid count} + \text{well location on the plate} + \\ \text{cell x-position} + \text{cell y-position} + \text{cell Z-position} + \text{depth of the cell spanning z}$$
#
# $$y = \beta_0 + x_1\beta_1 + x_2\beta_2 + x_3\beta_3 + x_4\beta_4 + x_5\beta_5 + x_6\beta_6 + x_7\beta_7 + x_8\beta_8 + x_9\beta_9 + \epsilon$$
#
# **Where:**
#
# | $x$ | $\beta$ | dtype | Description |
# |-----|---------|-------|--------------|
# | — | $\beta_0$ | float | Intercept |
# | $x_1$ | $\beta_1$ | binary (DMSO or treatment X) | Treatment (e.g., control, drug + dosage), fit within a single patient |
# | $x_2$ | $\beta_2$ | int | Cell count |
# | $x_3$ | $\beta_3$ | int | Organoid count |
# | $x_4$ | $\beta_4$ | float | Cell/organoid count |
# | $x_5$ | $\beta_5$ | float | Well location on the plate (Manhattan distance from the center of the plate) |
# | $x_6$ | $\beta_6$ | float | Cell x-position (x-center coordinate of the cell) |
# | $x_7$ | $\beta_7$ | float | Cell y-position (y-center coordinate of the cell) |
# | $x_8$ | $\beta_8$ | float | Cell Z-position (Z-center coordinate of the cell) |
# | $x_9$ | $\beta_9$ | float | Depth of the cell spanning z (Bounding box max - bounding box min in z) |
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
# **Profiles and outputs.** The same four profiles as `2.linear_modeling` are refit with the technical covariates and saved to `results/linear_modeling/` as `organoid_norm_technical_model.parquet`, `sc_norm_technical_model.parquet`, `organoid_agg_technical_model.parquet` and `single_cell_agg_technical_model.parquet`. The well Manhattan distance comes from `1.calculate_well_manhattan_distance`. As in step 2, positional (`CMI`) columns and the `NF0037_T1_CQ1` patient are dropped, the dose and unit are appended to the treatment, features are fit in parallel chunks (`joblib`), and a profile whose output files already exist is skipped.
#
# **MEKi groups.** As in `5.differential_analysis/8.calculate_mek_signatures`, the MEK inhibitors (Binimetinib, Mirdametinib, Trametinib, Selumetinib at 1 and 10 uM) are also modeled in two groupings:
#
# - **Pooled** (`mek_group = "All MEKi"`): every MEK drug x dose well of a patient is relabeled as one `All MEKi` treatment level and fit vs DMSO with the same model specification.
# - **Individual** (`mek_group = "<drug> <dose> uM"`): one model per MEK drug x dose vs DMSO, the same fits as in the main output.
#
# Both are saved to `results/linear_modeling/<profile>_technical_model_mek_groups.parquet` with `mek_group`, `grouping` (`pooled` / `individual`) and `Metadata_Biology_TumorType` columns. There, `pvalue_fdr` is rerun per term x `mek_group`, so each MEKi group is its own family of tests. The main `<profile>_technical_model.parquet` does not contain the pooled fits, and its FDR (per term) is unchanged. When only the MEKi group output is missing, only the MEKi group models are fit (the pooled model plus the 8 MEK drug x dose models per patient).
#

# In[3]:


lm_equation_terms = (
    "C(Metadata_Experiment_TreatmentFull) "
    "+ cell_count + organoid_count + cell_per_organoid_count "
    "+ manhattan_distance_from_center "
    "+ cell_x_position + cell_y_position + cell_z_position + cell_z_depth"
)


# In[4]:


# number of processes for the joblib fan-out (-1 = all cores) and how many
# features each task fits; smaller chunks balance load, larger ones cut the
# per-task pickling overhead
N_JOBS = -1
FEATURES_PER_TASK = 25


def fit_feature_chunk(
    df_trt,
    features,
    profile,
    patient,
    combo,
    drug_name,
    therapeutic_category,
    numeric_covariates,
    lm_equation_terms,
):
    """Fit the OLS model for each feature in one (patient, combo) subset.

    Runs in a joblib worker process, so everything it needs is passed in and
    the long-form results come back as a DataFrame.
    """
    linear_modeling_results_dict = {
        "model_name": [],
        "term": [],
        "Metadata_Biology_PatientTumor": [],
        "Metadata_Experiment_Treatment": [],
        "drug": [],
        "therapeutic_category": [],
        "feature": [],
        "rsquared": [],
        "rsquared_adj": [],
        "fvalue": [],
        "residual_ss": [],
        "residual_variance": [],
        "residual_pct": [],
        "total_ss": [],
        "explained_ss": [],
        "explained_ss_pct": [],
        "term_sum_sq": [],
        "term_pct_of_total_var": [],
        "pvalue": [],
        "coefficient": [],
        "intercept": [],
    }
    # columns patsy needs for every term in the formula; a NaN in any of
    # them (outcome or predictor) drops that row from the fit, not just a
    # NaN in the outcome
    required_columns = ["Metadata_Experiment_TreatmentFull"] + numeric_covariates
    for col in features:
        model_name = f"{profile}_{patient}_{drug_name}_{col}"
        # skip features with zero complete-case rows (no NaN in the outcome
        # or any predictor) -- a covariate that is entirely missing for this
        # patient/combo (e.g. a well with no location metadata, or no
        # Manhattan-distance match) drops every row via patsy's NA handling
        # even when the outcome itself is fine, which otherwise leaves
        # smf.ols an empty design matrix. A handful of complete rows is
        # still fit (it may yield a degenerate, NaN-pvalue model -- handled
        # downstream by the per-term FDR step), just not zero rows, and both
        # treatment levels must still be present or the treatment dummy
        # below won't exist.
        complete_mask = df_trt[[col] + required_columns].notna().all(axis=1)
        if (
            complete_mask.sum() == 0
            or df_trt.loc[complete_mask, "Metadata_Experiment_TreatmentFull"].nunique()
            < 2
        ):
            continue
        # Prepare the formula for the linear model:
        # y ~ txt + cell_count + organoid_count + cell/organoid_count + technical covariates
        formula = f"Q('{col}') ~ {lm_equation_terms}"
        model = smf.ols(formula=formula, data=df_trt)
        results = model.fit()
        # total (SST), model-explained (SSE) and residual (SSR) sums of
        # squares, plus a type II ANOVA for each term's share of SST
        sst = results.centered_tss
        sse = results.ess
        anova_table = sm.stats.anova_lm(results, typ=2)
        anova_table["pct_of_total_var"] = anova_table["sum_sq"] / sst * 100

        def add_row(term, anova_key, coefficient, pvalue):
            linear_modeling_results_dict["model_name"].append(model_name)
            linear_modeling_results_dict["term"].append(term)
            linear_modeling_results_dict["Metadata_Biology_PatientTumor"].append(
                patient
            )
            linear_modeling_results_dict["Metadata_Experiment_Treatment"].append(
                combo[1]
            )
            linear_modeling_results_dict["drug"].append(drug_name)
            linear_modeling_results_dict["therapeutic_category"].append(
                therapeutic_category
            )
            linear_modeling_results_dict["feature"].append(col)
            linear_modeling_results_dict["rsquared"].append(results.rsquared)
            linear_modeling_results_dict["rsquared_adj"].append(results.rsquared_adj)
            linear_modeling_results_dict["fvalue"].append(results.fvalue)
            # residual of the fit: sum of squares, variance (mean
            # squared error, df-adjusted) and share of total variance
            linear_modeling_results_dict["residual_ss"].append(results.ssr)
            linear_modeling_results_dict["residual_variance"].append(results.mse_resid)
            linear_modeling_results_dict["residual_pct"].append(results.ssr / sst * 100)
            linear_modeling_results_dict["total_ss"].append(sst)
            linear_modeling_results_dict["explained_ss"].append(sse)
            linear_modeling_results_dict["explained_ss_pct"].append(sse / sst * 100)
            linear_modeling_results_dict["term_sum_sq"].append(
                anova_table.loc[anova_key, "sum_sq"]
            )
            linear_modeling_results_dict["term_pct_of_total_var"].append(
                anova_table.loc[anova_key, "pct_of_total_var"]
            )
            linear_modeling_results_dict["pvalue"].append(pvalue)
            linear_modeling_results_dict["coefficient"].append(coefficient)
            linear_modeling_results_dict["intercept"].append(
                results.params["Intercept"].item()
            )

        # term: treatment effect within this patient
        treatment_term = f"C(Metadata_Experiment_TreatmentFull)[T.{combo[1]}]"
        add_row(
            "Metadata_Experiment_Treatment",
            "C(Metadata_Experiment_TreatmentFull)",
            results.params[treatment_term].item(),
            results.pvalues[treatment_term].item(),
        )

        # terms: the numeric covariates
        for covariate in numeric_covariates:
            add_row(
                covariate,
                covariate,
                results.params[covariate].item(),
                results.pvalues[covariate].item(),
            )
    return pd.DataFrame(linear_modeling_results_dict)


def tidy_model_results(results_df, sanitized_to_original_col_map):
    """Split the feature names into their parts and restore the original names.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form fit results, with the formula-safe (sanitized) feature names.
    sanitized_to_original_col_map : dict
        Sanitized feature name -> original feature name.

    Returns
    -------
    pd.DataFrame
        The results with Compartment / Channel / Feature_type / Measurement
        columns and the original feature names in ``feature``.
    """
    # map the sanitized feature names back to their original (pre-".": removal)
    # names so downstream consumers loading this parquet in any other
    # env/notebook can recover the native column name without needing access
    # to the sanitization logic above
    results_df["feature_original"] = results_df["feature"].map(
        sanitized_to_original_col_map
    )
    # split the feature column into multiple columns
    # feature names follow the pattern: Compartment_Channel_Feature_type_Measurement
    results_df[["Compartment", "Channel", "Feature_type", "Measurement"]] = results_df[
        "feature"
    ].str.split("_", n=3, expand=True)

    # if feature type is area shape then make the measurement the channel and
    # set the channel to None
    # this because area size shape features are not channel specific
    area_shape = results_df["Feature_type"] == "AreaSizeShape"
    results_df.loc[area_shape, "Measurement"] = results_df["Channel"]
    results_df.loc[area_shape, "Channel"] = None
    # set compartment to None if is adjacent
    # this is because adjacent features are not compartment specific
    results_df.loc[results_df["Compartment"] == "adjacent", "Compartment"] = None
    # map the features back to their original names inplace such that the
    # original feature names are preserved in the output parquet and can be used
    # for downstream grouping/merging without needing to re-run the sanitization
    results_df["feature"] = results_df["feature_original"]
    return results_df


def add_term_fdr(results_df, group_columns):
    """Benjamini-Hochberg FDR of ``pvalue``, run separately per group.

    Parameters
    ----------
    results_df : pd.DataFrame
        Long-form fit results with a ``pvalue`` column.
    group_columns : list
        Columns that define one hypothesis-testing family (e.g. ``["term"]``).

    Returns
    -------
    pd.DataFrame
        The results with a ``pvalue_fdr`` column.
    """
    results_df = results_df.reset_index(drop=True)
    results_df["pvalue_fdr"] = float("nan")
    for group_index in results_df.groupby(group_columns).groups.values():
        # a handful of models are degenerate (e.g. a constant feature for one
        # patient/combo) and produce a NaN p-value; multipletests propagates a
        # single NaN to every p-value in the array, so those rows must be
        # excluded from the correction rather than passed through -- they stay
        # pvalue_fdr=NaN (never significant) rather than corrupting every
        # other row's FDR value in this group
        valid_index = results_df.loc[group_index].dropna(subset=["pvalue"]).index
        pvals = results_df.loc[valid_index, "pvalue"].values
        _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")
        results_df.loc[valid_index, "pvalue_fdr"] = pvals_fdr
    return results_df


# patient -> tumor type, for the MEKi group output (the aggregated profiles
# carry no tumor type column of their own)
patient_tumor_type_map = (
    pd.read_parquet(
        profile_dict["organoid_fs"]["input_profile_path"],
        columns=["Metadata_Biology_PatientTumor", "Metadata_Biology_TumorType"],
    )
    .dropna()
    .drop_duplicates("Metadata_Biology_PatientTumor")
    .set_index("Metadata_Biology_PatientTumor")["Metadata_Biology_TumorType"]
)

for profile in tqdm(profile_dict.keys(), desc="Loading profiles"):
    # set the output dictionary for linear modeling results
    # per profile. Results are stored long-form: one row per
    # (combo, feature, term), where "term" identifies which piece of
    # the model specification the coefficient/pvalue belongs to
    # (treatment and each entry in numeric_covariates).
    output_path = profile_dict[profile]["output_profile_path"]
    mek_output_path = profile_dict[profile]["mek_output_profile_path"]
    # every treatment/dose model is only fit when the main output is missing;
    # the MEKi group models (pooled "All MEKi" plus each MEK drug x dose) only
    # when the MEKi group output is missing. The MEKi group output always
    # refits its own individual MEK models so it never mixes in a main output
    # made from an older version of the input profile
    fit_individual = not output_path.exists()
    fit_mek_groups = not mek_output_path.exists()
    if not (fit_individual or fit_mek_groups):
        continue  # skip if both outputs already exist

    df = pd.read_parquet(profile_dict[profile]["input_profile_path"])

    print(df.shape)
    df = df.dropna(subset=["Metadata_Experiment_Treatment"], how="any")
    print(df.shape)

    df = pd.merge(
        left=df,
        right=manhattan_distance_df,
        how="left",
        on=["Metadata_Experiment_Well"],
    )
    # manhattan_distance_df
    # drop the NF0037_T1_CQ1 patient
    df = df.loc[df["Metadata_Biology_PatientTumor"] != "NF0037_T1_CQ1"]
    # a few rows (e.g. NF0018_T6) are missing the experimental metadata
    # (treatment, dose, tumor type) from an incomplete metadata join;
    # drop them so they do not form a spurious "nan" treatment group
    df = df.loc[df["Metadata_Experiment_Treatment"].notna()]
    # combine treatment, dose, and unit into a single column so that
    # different doses of the same treatment are modeled as distinct groups
    df["Metadata_Experiment_TreatmentFull"] = (
        df["Metadata_Experiment_Treatment"].astype(str)
        + "_"
        + df["Metadata_Experiment_Dose"].map("{:g}".format)
        + df["Metadata_Experiment_Unit"].astype(str)
    )
    # map each combined treatment label back to its raw drug name and
    # therapeutic category (MOA) so the modeling results can be grouped
    # by drug/MOA downstream without re-merging metadata
    treatment_meta = (
        df[
            [
                "Metadata_Experiment_TreatmentFull",
                "Metadata_Experiment_Treatment",
                "Metadata_Experiment_TherapeuticCategories",
            ]
        ]
        .drop_duplicates()
        .set_index("Metadata_Experiment_TreatmentFull")
    )
    # combined treatment label -> MEKi group, for every individual MEK drug x
    # dose; the pooled label maps to itself
    mek_label_to_group = {POOLED_MEK_GROUP: POOLED_MEK_GROUP}
    for mek_group, mek_conditions in MEK_GROUPS.items():
        if mek_group == POOLED_MEK_GROUP:
            continue
        for drug, dose in mek_conditions:
            condition_mask = (df["Metadata_Experiment_Treatment"] == drug) & (
                df["Metadata_Experiment_Dose"] == dose
            )
            for label in df.loc[
                condition_mask, "Metadata_Experiment_TreatmentFull"
            ].unique():
                mek_label_to_group[label] = mek_group
    individual_mek_labels = [
        label for label in mek_label_to_group if label != POOLED_MEK_GROUP
    ]
    # the pooled group inherits the MEK drugs' therapeutic category
    treatment_meta.loc[POOLED_MEK_GROUP] = [
        POOLED_MEK_GROUP,
        "/".join(
            sorted(
                treatment_meta.loc[
                    individual_mek_labels, "Metadata_Experiment_TherapeuticCategories"
                ].unique()
            )
        ),
    ]

    # build the count covariates used in the model specification:
    # cell count, organoid count, and cell/organoid count
    if profile == "single_cell_fs":
        # the single-cell profile has no organoid count of its own,
        # so pull it in from the organoid-level profile via patient + well
        organoid_counts_df = pd.read_parquet(
            profile_dict["organoid_fs"]["input_profile_path"],
            columns=[
                "Metadata_Biology_PatientTumor",
                "Metadata_Experiment_Well",
                "Metadata_WellOrganoidCount",
            ],
        ).drop_duplicates()
        organoid_counts_df = organoid_counts_df.rename(
            columns={"Metadata_Biology_PatientTumor": "Metadata_Biology_PatientTumor"}
        )
        df = df.merge(
            organoid_counts_df,
            on=["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"],
            how="left",
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
        well_keys = ["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"]
        organoid_counts_df = pd.read_parquet(
            profile_dict["organoid_fs"]["input_profile_path"],
            columns=well_keys + ["Metadata_WellOrganoidCount"],
        ).drop_duplicates()
        cell_counts_df = pd.read_parquet(
            profile_dict["single_cell_fs"]["input_profile_path"],
            columns=well_keys + ["Metadata_Object_WellSingleCellCount"],
        ).drop_duplicates()
        df = df.merge(organoid_counts_df, on=well_keys, how="left").merge(
            cell_counts_df, on=well_keys, how="left"
        )
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

    numeric_covariates = [
        "cell_count",
        "organoid_count",
        "cell_per_organoid_count",
        "manhattan_distance_from_center",
    ]
    # the aggregated profiles are well-level, so they carry no per-object
    # location metadata; the spatial covariates only exist for the object-level
    # profiles (xy/z center position and z-depth of the cell or organoid)
    if not profile.endswith("_agg"):
        location_object = "Cell" if profile == "single_cell_fs" else "Organoid"
        df["cell_x_position"] = df[f"Metadata_Location_{location_object}_CenterX"]
        df["cell_y_position"] = df[f"Metadata_Location_{location_object}_CenterY"]
        df["cell_z_position"] = df[f"Metadata_Location_{location_object}_CenterZ"]
        df["cell_z_depth"] = (
            df[f"Metadata_Location_{location_object}_MaxZ"]
            - df[f"Metadata_Location_{location_object}_MinZ"]
        )
        numeric_covariates += [
            "cell_x_position",
            "cell_y_position",
            "cell_z_position",
            "cell_z_depth",
        ]
    # model spec for this profile: treatment plus its available covariates
    profile_equation_terms = " + ".join(
        ["C(Metadata_Experiment_TreatmentFull)"] + numeric_covariates
    )
    # cast to plain float64: pandas nullable extension dtypes (e.g. Float64/Int64)
    # carry pd.NA instead of np.nan, which patsy's formula NA-handling can't
    # evaluate (raises "boolean value of NA is ambiguous")
    df[numeric_covariates] = df[numeric_covariates].astype("float64")
    metadata_columns = (
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Biology_TumorType",
        ]
        + numeric_covariates
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
    dmso_label = df.loc[
        df["Metadata_Experiment_Treatment"] == "DMSO",
        "Metadata_Experiment_TreatmentFull",
    ].unique()[0]
    patients = sorted(df["Metadata_Biology_PatientTumor"].unique())
    # every (patient, treatment combo, feature chunk) is an independent set of
    # OLS fits, so build them all up front and fan them out across processes
    model_columns = ["Metadata_Experiment_TreatmentFull"] + numeric_covariates
    fit_tasks = []
    for patient in patients:
        df_pat = df.loc[df["Metadata_Biology_PatientTumor"] == patient]
        # each model is fit within this single patient, so there is nothing
        # to compare against if the patient has no DMSO rows
        if dmso_label not in df_pat["Metadata_Experiment_TreatmentFull"].unique():
            continue
        # (combo, rows to fit it on): the individual grouping fits each
        # treatment/dose vs DMSO on the patient's own rows (only the MEK drug x
        # doses when just the MEKi group output is being made)
        combo_list = [
            ((dmso_label, i), df_pat)
            for i in df_pat["Metadata_Experiment_TreatmentFull"].unique()
            if i != dmso_label and (fit_individual or i in individual_mek_labels)
        ]
        # the MEKi grouping pools every MEK drug x dose well into a single
        # "All MEKi" treatment level and fits it vs DMSO; the per drug x dose
        # MEKi groups are the individual fits above
        is_mek = df_pat["Metadata_Experiment_TreatmentFull"].isin(individual_mek_labels)
        if fit_mek_groups and is_mek.any():
            df_pooled = df_pat.loc[
                is_mek | (df_pat["Metadata_Experiment_TreatmentFull"] == dmso_label)
            ].copy()
            df_pooled.loc[
                df_pooled["Metadata_Experiment_TreatmentFull"].isin(
                    individual_mek_labels
                ),
                "Metadata_Experiment_TreatmentFull",
            ] = POOLED_MEK_GROUP
            combo_list.append(((dmso_label, POOLED_MEK_GROUP), df_pooled))
        for combo, df_source in combo_list:
            drug_name = treatment_meta.loc[combo[1], "Metadata_Experiment_Treatment"]
            therapeutic_category = treatment_meta.loc[
                combo[1], "Metadata_Experiment_TherapeuticCategories"
            ]

            df_trt = df_source.loc[
                df_source["Metadata_Experiment_TreatmentFull"].isin(combo)
            ].copy()
            # order the treatment column to ensure DMSO is first (reference level)
            df_trt["Metadata_Experiment_TreatmentFull"] = pd.Categorical(
                df_trt["Metadata_Experiment_TreatmentFull"],
                categories=[combo[0], combo[1]],
            )
            # zero-center the continuous covariates (per patient, per combo) for
            # numerical stability in the OLS fit -- a linear shift with no
            # rescaling leaves the fit (and all coefficients except the
            # intercept) unchanged, so this is purely a conditioning
            # improvement, not a modeling choice
            df_trt[numeric_covariates] = (
                df_trt[numeric_covariates] - df_trt[numeric_covariates].mean()
            )

            # only ship the columns each worker needs, not the whole profile
            for start in range(0, len(feature_columns), FEATURES_PER_TASK):
                chunk = feature_columns[start : start + FEATURES_PER_TASK]
                fit_tasks.append(
                    delayed(fit_feature_chunk)(
                        df_trt[model_columns + chunk],
                        chunk,
                        profile,
                        patient,
                        combo,
                        drug_name,
                        therapeutic_category,
                        numeric_covariates,
                        profile_equation_terms,
                    )
                )
    # return_as="generator" yields results in submission order (so the row
    # order matches a serial run) while letting tqdm track progress
    chunk_results = Parallel(n_jobs=N_JOBS, return_as="generator")(fit_tasks)
    linear_modeling_results_df = tidy_model_results(
        pd.concat(
            list(
                tqdm(
                    chunk_results,
                    total=len(fit_tasks),
                    desc=f"{profile}: fitting models",
                    unit="task",
                )
            ),
            ignore_index=True,
        ),
        sanitized_to_original_col_map,
    )
    is_pooled = (
        linear_modeling_results_df["Metadata_Experiment_Treatment"] == POOLED_MEK_GROUP
    )

    # individual grouping: one model per treatment/dose vs DMSO. Run FDR on
    # the p-values, separately per term since each term is its own
    # hypothesis-testing family (different scale/behavior of p-values)
    if fit_individual:
        individual_results_df = add_term_fdr(
            linear_modeling_results_df.loc[~is_pooled], ["term"]
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        individual_results_df.to_parquet(output_path, index=False)

    # MEKi groups: the pooled "All MEKi" fits plus the individual MEK drug x
    # dose fits. FDR is rerun per term x mek_group (as in
    # 5.differential_analysis/8.calculate_mek_signatures), so each MEKi group
    # is its own family and the pooled fits do not change the main output
    if fit_mek_groups:
        mek_results_df = linear_modeling_results_df.loc[
            is_pooled
            | linear_modeling_results_df["Metadata_Experiment_Treatment"].isin(
                individual_mek_labels
            )
        ].copy()
        mek_results_df["mek_group"] = mek_results_df[
            "Metadata_Experiment_Treatment"
        ].map(mek_label_to_group)
        mek_results_df["grouping"] = (
            mek_results_df["mek_group"]
            .eq(POOLED_MEK_GROUP)
            .map({True: "pooled", False: "individual"})
        )
        mek_results_df["Metadata_Biology_TumorType"] = mek_results_df[
            "Metadata_Biology_PatientTumor"
        ].map(patient_tumor_type_map)
        mek_results_df = add_term_fdr(mek_results_df, ["term", "mek_group"])
        mek_output_path.parent.mkdir(parents=True, exist_ok=True)
        mek_results_df.to_parquet(mek_output_path, index=False)
