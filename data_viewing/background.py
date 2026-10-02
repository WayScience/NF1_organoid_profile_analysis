"""Background text for every module tab, section, and sub-panel of the viewer.

Each entry says what the data are and how they are derived, and names the
script that produces them. Text is plain markdown shown in a collapsed expander.
"""

import streamlit as st

PROFILE_GLOSSARY = """
**Profile types used throughout**
- **sc** (single-cell): one row per segmented cell/nucleus.
- **organoid**: one row per segmented organoid.
- **fs** (feature-selected): normalized profiles after pycytominer feature selection.
- **agg** / **consensus**: summaries of the feature-selected profiles, one row per
  replicate or per treatment (not per cell).
- **2D** profiles come from a slice strategy (max projection, middle slice);
  **3D** profiles come from the full z-stack.
- `NF0037_T1_CQ1` is excluded from every EDA result (it is a separate analysis).
"""

MODULE_BACKGROUND = {
    "0.Overview": (
        "Static experiment design, read directly from `config/platemaps/` (not "
        "computed): which drugs were plated where, and which patient tumor "
        "samples were screened on each platemap."
    ),
    "1.EDA": (
        "Exploratory analysis of the image-based morphology profiles of NF1 patient "
        "tumor organoids treated with drugs at one or more doses (DMSO is the "
        "control). Every table here is precomputed by a script in "
        "`1.EDA/scripts/` and read from `1.EDA/results/`; nothing is recomputed in "
        "the viewer.\n" + PROFILE_GLOSSARY
    ),
    "3.viability_prediction_models": (
        "Can organoid morphology predict cell viability? An Elastic Net regression "
        "is trained on image-based profiles to predict per-well viability "
        "(`3.viability_prediction_models/scripts/1.viability_prediction.py`). The "
        "target is the **per-patient min-max scaled viability**, computed fold-safe "
        "(the min and max come only from the training rows of each split, so no "
        "information leaks from the test set). Features are standardized inside the "
        "model, and `alpha` / `l1_ratio` are tuned with ElasticNetCV.\n\n"
        "Three split strategies are used:\n"
        "- **LOPO**: leave one patient out (train on the other patients).\n"
        "- **LOTO**: leave one treatment out.\n"
        "- **Random split**: repeated random train/test splits.\n\n"
        "Each model is also fit with **shuffled** labels (`shuffle_status`) as a "
        "null baseline. A useful model beats its shuffled control."
    ),
    "4.linear_modeling": (
        "Per-feature linear models that ask which morphology features change with "
        "treatment. One OLS model is fit per **(patient, treatment/dose, feature)** "
        "on 3D normalized profiles:\n\n"
        "`feature ~ treatment + cell_count + organoid_count + cell_per_organoid_count`\n\n"
        "The **technical** model set adds spatial covariates: well distance from "
        "the plate center (Manhattan distance), cell x/y/z position and z depth. "
        "The **original** model set has no spatial covariates. Models are fit on "
        "organoid and single-cell profiles, and on well-level aggregates "
        "(median, `*_agg`). p-values are FDR-corrected (`pvalue_fdr`). "
        "The `coefficient` is the effect in feature-SD units; for `treatment` it "
        "is the contrast against DMSO. Scripts: `4.linear_modeling/scripts/2-5.*`."
    ),
}

SECTION_BACKGROUND = {
    # ---- 0.Overview ----
    "Platemap": (
        "The well -> treatment/dose layout of each named platemap "
        "(`config/platemaps/platemap*.csv`). Two layouts were used across the "
        "screen; pick one to see its grid and which patient samples ran on it."
    ),
    "Drugs": (
        "Every drug (excluding the DMSO control), with its dose(s) and "
        "mechanism of action, deduplicated across both platemaps."
    ),
    "Patients & tumor manifestations": (
        "Every patient tumor sample screened, which platemap it ran on, and "
        "its tumor manifestation (cNF, pNF, MPNST or Other) "
        "(`config/platemaps/barcode_platemap.csv`)."
    ),
    # ---- 1.EDA ----
    "UMAP": (
        "2D UMAP embeddings of single-cell morphology profiles "
        "(`0.generate_umap.py`). **Pooled** embeddings use all patients together, "
        "per projection (2D max, 2D middle, 3D) and profile variant. **Per-patient** "
        "embeddings are an independent UMAP fit for each patient, so UMAP1/UMAP2 "
        "are comparable only *within* a patient, never across patients. "
    ),
    "PCA": (
        "PCA embeddings of the pooled (all-patient) feature-selected, aggregated, "
        "and consensus profiles for 2D and 3D (`2.generate_pca.py`). Columns "
        "`PC0…PCn` are component scores; the caption and scree plot show each "
        "component's explained variance ratio. Use the X/Y component selectors to "
        "move between components."
    ),
    "Correlation heatmaps": (
        "Sample-by-sample **Pearson correlation** matrices of profiles "
        "(`4.calculate_correlation_matrix.py`). A high value means two samples have "
        "similar morphology profiles."
    ),
    "Cell counts": (
        "Number of cells segmented per organoid, patient, and treatment, counted "
        "from the single-cell profiles for each profile type and split by 2D and 3D "
        "(`7.generate_cell_counts.py`). Use it to check whether a treatment changes "
        "cell number, which is also a covariate in the linear models."
    ),
    "Area & volume": (
        "**Raw** (not z-scored) organoid and single-cell size: area from 2D "
        "(max projection only) and volume from 3D "
        "(`17.calculate_area_volume_by_patient_treatment.py`). Raw values are used "
        "because area and volume come from separate pipelines with different "
        "normalization (area was z-scored per patient upstream, volume was not), "
        "so normalized values would not be comparable. One row per record, "
        "with no aggregation."
    ),
    "Neighbors": (
        "How closely nuclei and organoids pack (`13.calculate_neighbor_features.py`). "
        "**2D**: `Nuclei_Neighbors_*` (nearest and second-nearest distance, number "
        "of neighbors, percent touching, angle between neighbors) and "
        "`Organoid_Neighbors_NumberOfNeighbors_Adjacent`. **3D** has no equivalent "
        "columns, so it reports different measures: neighbors adjacent in the local "
        "shell, distance from the organoid center and exterior, and nucleus volume. "
        "3D organoid neighbor counts are approximated: two organoids in the same "
        "well/FOV are adjacent if their bounding spheres (centroid + mean "
        "half-extent radius, in µm) touch. Do not compare 2D and 3D columns directly."
    ),
    "Intensity": (
        "Mean and median fluorescence intensity per channel and compartment "
        "(`15.calculate_intensity_values.py`). The whole-organoid compartment comes "
        "from organoid tables; cell and nucleus compartments come from "
        "single-cell tables. `value` is **z-scored per patient** (within patient × "
        "compartment × channel × statistic, across all treatments), so panels "
        "share one scale and treatments are read relative to that patient."
    ),
    "Count vs viability": (
        "Mean cells per organoid (3D) joined with measured viability from the "
        "platemap, per patient × treatment × dose "
        "(`10.calculate_count_viability_join.py`). Patients present in only one of "
        "the profiles or the platemap are dropped and logged by the script."
    ),
    # ---- 3.viability_prediction_models ----
    "Model performance": (
        "Elastic Net performance for each profile type, split method, and shuffle "
        "status. **Fold metrics** are one row per train/test fold; **summary "
        "metrics** aggregate over folds. Metrics are R² (variance explained; can "
        "be negative if worse than predicting the mean), MSE, and MAE. Compare each "
        "real model to its shuffled-label control."
    ),
    "Predicted vs actual": (
        "Held-out predictions from the models: each row is a sample with its "
        "`Actual_Viability` (per-patient min-max scaled) and `Predicted_Viability` "
        "from a model that did not see it in training. Points on the diagonal "
        "are perfect predictions. The wide feature columns are not loaded here."
    ),
    "Feature importances": (
        "Elastic Net coefficients per feature, split by profile type, split method, "
        "shuffle status, and image mode. Features are standardized before "
        "fitting, so coefficient size is comparable across features. Zero means "
        "the feature was dropped by the L1 penalty. Sign shows the direction "
        "(a positive coefficient means higher viability)."
    ),
    # ---- 4.linear_modeling ----
    "Volcano (effects vs significance)": (
        "One point per model term: `coefficient` (effect size) against "
        "`-log10(pvalue_fdr)`. A point is *significant* when `pvalue_fdr < 0.05`. "
        "Facet by `term` to separate treatment from the covariates. Pick a "
        "model result file (organoid or sc; original or technical; agg or not) first."
    ),
    "Effect sizes": (
        "Distribution of model `coefficient` values by treatment or therapeutic "
        "category. For the `treatment` term this is the change versus DMSO in "
        "feature-SD units. Covariate terms (counts, position) are per-unit slopes "
        "and are not on the same scale."
    ),
    "Model fit": (
        "How well each model fits: `rsquared` and `rsquared_adj` per "
        "(patient, treatment, feature) model, split by feature type and "
        "compartment. Most variance is usually residual "
        "(`residual_pct = (1 - R²) × 100`), so low values are expected."
    ),
    "UpSet (variate combinations)": (
        "A feature is a **hit** for a model term when `pvalue_fdr < threshold` "
        "and `coefficient > minimum` (increases only). Each feature is assigned "
        "to the set of terms it is a hit for; bars show how many features share "
        "each exact combination (computed live, ported from "
        "`5.calculate_variate_importance.py`). *Scope* sets the grouping "
        "(all models, per patient, per treatment, per tumor type, …). The overlap "
        "matrix below counts features shared between every pair of groups."
    ),
    "Unique variates per model": (
        "A feature × (patient, treatment) yes/no heatmap of where treatment is a "
        "hit on its own versus together with at least one term from the chosen "
        "covariate group (biological or technical). Hit = `pvalue_fdr < threshold "
        "and coefficient > minimum`, as in `5.calculate_variate_importance.py`. The "
        "drill-down table below lists every term of one model and which ones are "
        "hits (its *unique variate signature*)."
    ),
}

SUBPANEL_BACKGROUND = {
    "corr_pairs": (
        "**Replicate/treatment-level (pairs).** Correlations between aggregate or "
        "consensus profiles (one row per replicate or treatment, summarized "
        "from single cells). Stored as a long table with one row per unique sample "
        "pair (upper triangle only, since correlation is symmetric), with a "
        "companion table of patient/treatment/dose metadata for annotation. "
        "Files exist for 2D (3 slice strategies) and 3D (4 normalization variants)."
    ),
    "corr_per_patient": (
        "**Per-patient single-cell.** Correlation matrices of 3D single-cell "
        "feature-selected profiles, computed separately for each patient (one "
        "row per cell before correlating, unlike the pairs view). Choose a "
        "normalization variant and patient. Only variants that are truly "
        "single-cell are included."
    ),
    "pca_scree": (
        "**Scree plot.** Explained variance ratio of each principal component "
        "(saved with the embeddings). A steep drop means a few components "
        "capture most of the variation."
    ),
    "lm_upset_profile": (
        "**Profile** sets the unit that was modeled: `organoid` (one row per "
        "organoid) or `sc` (one row per cell). **Model set**: `original` uses "
        "treatment and count covariates only; `technical` adds spatial covariates "
        "(plate position, cell x/y/z, z depth)."
    ),
    "lm_variates_drilldown": (
        "**Drill-down.** Pick one patient, treatment, and feature to see every "
        "term of that single model: its coefficient, FDR-adjusted p-value, R², "
        "and whether it passes the current hit thresholds."
    ),
}


def module_background(module: str) -> None:
    """Collapsed expander with the module-level background."""
    text = MODULE_BACKGROUND.get(module)
    if text:
        with st.expander(f"About {module}"):
            st.markdown(text)


def section_background(section: str) -> None:
    """Collapsed expander with the section-level background."""
    text = SECTION_BACKGROUND.get(section)
    if text:
        with st.expander(f"About: {section}"):
            st.markdown(text)


def subpanel_background(key: str, label: str = "About this panel") -> None:
    """Collapsed expander for a sub-panel inside a section."""
    text = SUBPANEL_BACKGROUND.get(key)
    if text:
        with st.expander(label):
            st.markdown(text)
