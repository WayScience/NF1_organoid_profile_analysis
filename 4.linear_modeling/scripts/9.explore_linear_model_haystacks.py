#!/usr/bin/env python
# coding: utf-8

# # Finding the needles: an exhaustive look at the linear models
#
# Steps 2 and 3 fit one OLS model per **(patient, treatment/dose, feature)**:
#
# `feature ~ treatment + cell_count + organoid_count + cell_per_organoid_count`
#
# (step 3 adds the spatial "technical" covariates; this notebook uses the step 3
# `*_technical_model.parquet` fits). That is hundreds of thousands
# of models -- the **haystack**. The **needles** are the treatment effects that
# are statistically real, large, reproducible across patients and doses,
# not explained away by counts or plate position, and that agree between
# organoid and single-cell readouts.
#
# Everything here uses the *saved* fit statistics (nothing is refit).
#
# | # | Theme | Question |
# |---|-------|----------|
# | 4 | Effect sizes | How big are the treatment effects? Volcano views. |
# | 5 | Variance partitioning | Does treatment or a covariate explain the variance? |
# | 6 | Hit landscape | Which patients / drugs / MOAs / tumor types hit most? |
# | 7 | Feature space | Which compartments / channels / feature types are enriched? |
# | 8 | Dose response | Do 1 uM and 10 uM agree? Does the effect grow with dose? |
# | 9 | Cross-patient reproducibility | Which effects replicate across patients? |
# | 10 | Treatment similarity | Do drugs of the same MOA look alike? Patient vs drug effect? |
# | 11 | Organoid vs single cell | Do the two scales tell the same story? |
# | 12 | Aggregated vs not | Are the results robust to profile aggregation? |
# | 13 | Technical covariates | Which technical covariate terms explain variance and reach significance? |
# | 14 | Count confounding | Are hits really just changes in organoid/cell number? |
# | 15 | Threshold sensitivity | How fragile is the "hit" definition? |
# | 16 | Tumor type | Are there tumor-type-specific responses? |
# | 17 | Feature modules | Are hit features redundant? |
# | 18 | The needles | Composite ranking, patient-private needles, per-drug top features |
# | 19 | Counts, both ways | Which hits are treatment-driven vs count-linked? |
# | 20 | Tumor-type specificity | Which drug x feature effects differ between tumor types? |
# | 21 | Four kinds of "interesting" | Replicated / tumor-type-specific / dose-dependent / MOA-consistent + readout shortlist |
# | 22 | MOA consistency test | Do drugs of one MOA correlate more than chance? |
# | 23 | 30 questions | A direct, numeric answer to 30 concrete questions the models can answer |
# | 24 | Every technical variate | Each covariate term explored like treatment |

# In[ ]:


import pathlib
import warnings

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import fisher_exact, kruskal, spearmanr
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")
root_dir, in_notebook = init_notebook()

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 200)
RNG = np.random.default_rng(0)


# In[ ]:


# paths (all relative to the git root)
lm_results_path = pathlib.Path(root_dir, "4.linear_modeling/results/linear_modeling")
results_path = pathlib.Path(root_dir, "4.linear_modeling/results/explore_linear_models")
results_path.mkdir(parents=True, exist_ok=True)
plot_data_path = results_path / "plot_data"
plot_data_path.mkdir(parents=True, exist_ok=True)


def save_plot_data(name, **frames):
    """Save the tables a figure is drawn from; the R plot step (10.plot_explore_linear_model_haystacks)
    reads `plot_data/<name>__<key>.parquet`."""
    for key, frame in frames.items():
        unnamed = all(level is None for level in frame.index.names)
        frame.reset_index(drop=unnamed).to_parquet(
            plot_data_path / f"{name}__{key}.parquet", index=False
        )


# profile key -> technical-covariate model file (treatment + every technical covariate)
profile_files = {
    "organoid": "organoid_norm_technical_model.parquet",
    "sc": "sc_norm_technical_model.parquet",
    "organoid_agg": "organoid_agg_technical_model.parquet",
    "sc_agg": "single_cell_agg_technical_model.parquet",
}
profiles = list(profile_files)
# the two "full resolution" profiles carry most of the narrative
main_profiles = ["organoid", "sc"]

# "hit" definition (see the 4.linear_modeling README)
FDR_MAX = 0.05
R2_MIN = 0.5
R2_ADJ_MIN = 0
COEF_MIN = 0.01

model_keys = ["patient", "treatment", "feature"]
keep_cols = [
    "term",
    "patient",
    "treatment",
    "drug",
    "therapeutic_category",
    "feature",
    "Compartment",
    "Channel",
    "Feature_type",
    "Measurement",
    "rsquared",
    "rsquared_adj",
    "fvalue",
    "residual_pct",
    "explained_pct",
    "term_pct_of_total_var",
    "pvalue",
    "pvalue_fdr",
    "coefficient",
]


def load_results(file_name):
    """Load a linear modeling parquet and harmonize base/technical schemas."""
    df = pd.read_parquet(lm_results_path / file_name)
    df = df.rename(
        columns={
            "Metadata_Biology_PatientTumor": "patient",
            "Metadata_Experiment_Treatment": "treatment",
            "total_ss": "sst",
            "explained_ss": "sse",
            "explained_ss_pct": "explained_pct",
        }
    )
    # the technical model names the treatment term after the raw column
    df["term"] = df["term"].replace({"Metadata_Experiment_Treatment": "treatment"})
    df = df[keep_cols].copy()
    df["tumor_type"] = df["patient"].map(tumor_type_dict)
    df["dose"] = df["treatment"].str.rsplit("_", n=1).str[-1]
    df["channel_label"] = df["Channel"].fillna("none")
    df["compartment_label"] = df["Compartment"].fillna("adjacent")
    df["sig"] = df["pvalue_fdr"] < FDR_MAX
    df["good_model"] = (df["rsquared"] > R2_MIN) & (df["rsquared_adj"] > R2_ADJ_MIN)
    df["hit"] = df["sig"] & df["good_model"] & (df["coefficient"].abs() > COEF_MIN)
    return df


def add_covariate_dominance(df):
    """Flag treatment rows where treatment explains more variance than any covariate term."""
    wide = df.pivot(index=model_keys, columns="term", values="term_pct_of_total_var")
    covariate_terms = [t for t in wide.columns if t != "treatment"]
    covariate_max = wide[covariate_terms].max(axis=1).rename("covariate_max_pct")
    trt = df.loc[df["term"] == "treatment"].merge(
        covariate_max.reset_index(), on=model_keys, how="left"
    )
    trt["covariate_safe"] = trt["term_pct_of_total_var"] > trt["covariate_max_pct"]
    return trt, wide


lm = {}  # all terms, technical-covariate model
trt = {}  # treatment term only
wide_pct = {}  # per-model pct of total variance for every term
for name, file_name in profile_files.items():
    lm[name] = load_results(file_name)
    trt[name], wide_pct[name] = add_covariate_dominance(lm[name])
    print(
        f"{name}: {len(trt[name]):,} treatment models; {trt[name]['hit'].sum():,} hits"
    )

# every non-treatment term present in any profile, in the order of the model formula
covariate_terms = list(
    dict.fromkeys(
        t for p in profiles for t in lm[p]["term"].unique() if t != "treatment"
    )
)

# term order (used throughout to keep plots/tables consistent) and the full patient/treatment lists
term_order = ["treatment"] + covariate_terms
all_patients = sorted(trt["organoid"]["patient"].unique())
all_treatments = sorted(trt["organoid"]["treatment"].unique())


# ## 4. Effect sizes
#
# *Volcano plots with significance on the y-axis and effect size on the x-axis, colored by R2.*

# In[ ]:


volcano = pd.concat(
    [
        trt[p].assign(
            neglog_fdr=lambda d: -np.log10(d["pvalue_fdr"].clip(lower=1e-300)),
            profile=p,
        )[["profile", "coefficient", "neglog_fdr", "rsquared"]]
        for p in profiles
    ],
    ignore_index=True,
)
save_plot_data("4_volcano_by_profile", volcano=volcano)


# ## 5. Variance partitioning
#
# *For each model the variance is split into treatment, every covariate term
# (counts and technical) and the residual. Which piece dominates?
#
# The gap between `explained_pct` and the sum of each term's type II SS share
# (collinear covariates -- e.g. `cell_per_organoid_count` is
# `cell_count / organoid_count` -- mean Type II sums don't partition explained
# variance cleanly and can swing arbitrarily negative under suppression) is
# left out of these tables: it isn't a meaningful quantity to plot.*

# In[ ]:


partition_rows = []
partition_long_rows = []
for p in profiles:
    w = wide_pct[p].reindex(columns=term_order)  # terms a profile lacks stay NaN
    meta = trt[p].set_index(model_keys)
    resid = meta["residual_pct"].reindex(w.index)
    row = {t: w[t].mean() for t in term_order}
    row["residual"] = resid.mean()
    row["profile"] = p
    partition_rows.append(row)

    long = w.assign(residual=resid, profile=p)
    partition_long_rows.append(
        long.melt(id_vars="profile", var_name="term", value_name="pct").dropna(
            subset=["pct"]
        )
    )
partition = pd.DataFrame(partition_rows).set_index("profile")
partition.reset_index().to_parquet(
    results_path / "variance_partition_mean.parquet", index=False
)
partition_long = pd.concat(partition_long_rows, ignore_index=True)
# exact number of models fit per profile (one row per model in trt[p]); partition_long
# undercounts this since it drops any model whose treatment-term share is NaN
model_counts = pd.DataFrame(
    {"profile": profiles, "n_models": [len(trt[p]) for p in profiles]}
)
save_plot_data(
    "5_variance_partition_by_profile",
    partition=partition,
    partition_long=partition_long,
    model_counts=model_counts,
)
partition


# In[ ]:


# treatment variance share by patient and drug (organoid & sc)
save_plot_data(
    "5_treatment_variance_share_heatmap",
    share=pd.concat(
        [
            trt[p]
            .groupby(["patient", "treatment"])["term_pct_of_total_var"]
            .median()
            .rename("median_pct_var")
            .reset_index()
            .assign(profile=p)
            for p in main_profiles
        ]
    ),
)


# ## 6. Hit landscape
#
# *Which patients, drugs, MOAs, doses and tumor types produce the most hits?
# The hit rate is hits / models tested so unbalanced designs do not mislead.*

# In[ ]:


def hit_rate(df, by):
    return (
        df.groupby(by)["hit"].agg(hit_rate="mean", hits="sum", n="size").reset_index()
    )


landscape = []
for p in profiles:
    for by in [
        "patient",
        "treatment",
        "drug",
        "therapeutic_category",
        "tumor_type",
        "dose",
    ]:
        r = hit_rate(trt[p], by)
        r["level"] = by
        r["profile"] = p
        landscape.append(r.rename(columns={by: "group"}))
landscape = pd.concat(landscape)
landscape.to_parquet(results_path / "hit_rate_landscape.parquet", index=False)
save_plot_data("6_hit_rate_by_group", landscape=landscape)


# In[ ]:


hit_rate_pt = pd.concat(
    [
        trt[p]
        .groupby(["patient", "treatment"])["hit"]
        .mean()
        .reindex(
            pd.MultiIndex.from_product(
                [all_patients, all_treatments], names=["patient", "treatment"]
            )
        )
        .rename("hit_rate")
        .reset_index()
        .assign(profile=p)
        for p in main_profiles
    ],
    ignore_index=True,
)
save_plot_data("6_hit_rate_patient_by_treatment", hit_rate=hit_rate_pt)


# In[ ]:


# direction of hits: is a drug mostly increasing or decreasing features?
direction_frames = []
for p in main_profiles:
    h = trt[p].loc[trt[p]["hit"]]
    d = (
        h.assign(direction=np.where(h["coefficient"] > 0, "up", "down"))
        .groupby(["treatment", "direction"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=all_treatments, columns=["down", "up"], fill_value=0)
    )
    d["down"] = -d["down"]
    direction_frames.append(d.reset_index().assign(profile=p))
direction = pd.concat(direction_frames, ignore_index=True)
save_plot_data("6_hit_direction_by_treatment", direction=direction)


# ## 7. Feature space enrichment
#
# *Which parts of the morphology feature space respond? Fisher exact test of
# hit vs non-hit for every compartment / channel / feature type.*

# In[ ]:


def enrichment(df, col):
    df = df.assign(_lvl=df[col].astype(str))
    tot, tot_hit = len(df), int(df["hit"].sum())
    rows = []
    for lvl, g in df.groupby("_lvl"):
        a = int(g["hit"].sum())
        b = len(g) - a
        c = tot_hit - a
        d = (tot - tot_hit) - b
        odds, pval = fisher_exact([[a, b], [c, d]])
        rows.append(
            {
                "level": lvl,
                "n": len(g),
                "hits": a,
                "hit_rate": a / len(g),
                "odds_ratio": odds,
                "pvalue": pval,
            }
        )
    out = pd.DataFrame(rows)
    out["fdr"] = multipletests(out["pvalue"], method="fdr_bh")[1]
    out["log2_or"] = np.log2(out["odds_ratio"].replace(0, np.nan))
    out["family"] = col
    return out


enr = pd.concat(
    [
        enrichment(trt[p], col).assign(profile=p)
        for p in main_profiles
        for col in ["compartment_label", "channel_label", "Feature_type", "Measurement"]
    ]
)
enr.to_parquet(results_path / "feature_family_enrichment.parquet", index=False)
save_plot_data("7_feature_family_enrichment", enrichment=enr)


# In[ ]:


channel_frames = []
for p in main_profiles:
    for kind, col in [
        ("feature_type", "Feature_type"),
        ("compartment", "compartment_label"),
    ]:
        m = trt[p].pivot_table(
            index="channel_label", columns=col, values="hit", aggfunc="mean"
        )
        long = m.stack().rename("hit_rate").reset_index()
        long = long.rename(columns={"channel_label": "channel", col: "column"})
        long["profile"] = p
        long["kind"] = kind
        channel_frames.append(long)
channel_heat = pd.concat(channel_frames, ignore_index=True)
save_plot_data("7_channel_by_feature_type_heatmaps", heatmap=channel_heat)


# In[ ]:


# which drugs "own" which channels? MOA x channel hit rate
drug_channel_frames = []
for p in main_profiles:
    m = trt[p].pivot_table(
        index="drug", columns="channel_label", values="hit", aggfunc="mean"
    )
    long = m.stack().rename("hit_rate").reset_index()
    long["profile"] = p
    drug_channel_frames.append(long)
drug_channel = pd.concat(drug_channel_frames, ignore_index=True)
save_plot_data("7_drug_by_channel_hit_rate", heatmap=drug_channel)


# ## 8. Dose response
#
# *For drugs tested at more than one dose: do the doses agree in direction?
# Does effect size grow with dose (a hallmark of a real pharmacological
# effect)?*

# In[ ]:


dose_pairs = []
for p in profiles:
    d = trt[p]
    multi = d.groupby("drug")["dose"].nunique()
    multi = multi[multi > 1].index
    d = d.loc[d["drug"].isin(multi)]
    wide = d.pivot_table(
        index=["patient", "drug", "feature"], columns="dose", values="coefficient"
    )
    for drug, sub in wide.groupby(level="drug"):
        sub = sub.dropna(axis=1, how="all").dropna()
        doses = list(sub.columns)
        if len(doses) < 2:
            continue
        lo, hi = doses[0], doses[-1]
        rho = spearmanr(sub[lo], sub[hi])[0]
        agree = (np.sign(sub[lo]) == np.sign(sub[hi])).mean()
        dose_pairs.append(
            {
                "profile": p,
                "drug": drug,
                "dose_a": lo,
                "dose_b": hi,
                "n": len(sub),
                "spearman": rho,
                "sign_agreement": agree,
            }
        )
dose_pairs = pd.DataFrame(dose_pairs)
dose_pairs.to_parquet(results_path / "dose_concordance.parquet", index=False)
dose_pairs


# In[ ]:


# scatter of low vs high dose for each multi-dose drug (organoid)
p = "organoid"
d = trt[p]
multi = d.groupby("drug")["dose"].nunique()
multi = list(multi[multi > 1].index)
dose_scatter_rows = []
for drug in multi:
    sub = (
        d.loc[d["drug"] == drug]
        .pivot_table(index=["patient", "feature"], columns="dose", values="coefficient")
        .dropna()
    )
    a, b = sub.columns[0], sub.columns[-1]
    dose_scatter_rows.append(
        sub[[a, b]]
        .rename(columns={a: "coef_low", b: "coef_high"})
        .reset_index()
        .assign(drug=drug, dose_low=a, dose_high=b)
    )
dose_scatter = (
    pd.concat(dose_scatter_rows, ignore_index=True)
    if dose_scatter_rows
    else pd.DataFrame(
        columns=[
            "patient",
            "feature",
            "coef_low",
            "coef_high",
            "drug",
            "dose_low",
            "dose_high",
        ]
    )
)
save_plot_data("8_dose_scatter_organoid", scatter=dose_scatter)


# In[ ]:


# does |effect| grow with dose? (units of the smaller dose that are hits)
grow_rows = []
for p in profiles:
    d = trt[p]
    multi = d.groupby("drug")["dose"].nunique()
    for drug in multi[multi > 1].index:
        sub = d.loc[d["drug"] == drug].pivot_table(
            index=["patient", "feature"], columns="dose", values="coefficient"
        )
        hits = d.loc[(d["drug"] == drug) & d["hit"]].set_index(["patient", "feature"])
        # dose ordering by numeric value inside the label (1uM < 10uM)
        cols = sorted(
            sub.columns,
            key=lambda s: float("".join(c for c in s if c.isdigit() or c == ".") or 0),
        )
        if len(cols) < 2:
            continue
        lo, hi = cols[0], cols[-1]
        lo_hit_idx = hits.loc[hits["dose"] == lo].index.unique()
        s = sub.reindex(lo_hit_idx).dropna(subset=[lo, hi])
        grow_rows.append(
            {
                "profile": p,
                "drug": drug,
                "low": lo,
                "high": hi,
                "n_low_dose_hits": len(s),
                "frac_grow_with_dose": (s[hi].abs() > s[lo].abs()).mean()
                if len(s)
                else np.nan,
                "frac_same_sign": (np.sign(s[hi]) == np.sign(s[lo])).mean()
                if len(s)
                else np.nan,
            }
        )
dose_growth = pd.DataFrame(grow_rows)
dose_growth.to_parquet(results_path / "dose_growth.parquet", index=False)
dose_growth


# ## 9. Cross-patient reproducibility
#
# *A needle should be found in more than one patient, in the same direction.
# For every (treatment, feature): in how many patients is it a hit, and do
# those hits agree in sign?*

# In[ ]:


def consensus_table(df):
    d = df.assign(
        pos_hit=df["hit"] & (df["coefficient"] > 0),
        neg_hit=df["hit"] & (df["coefficient"] < 0),
        abs_coef=df["coefficient"].abs(),
    )
    g = d.groupby(
        ["treatment", "drug", "therapeutic_category", "feature"], dropna=False
    )
    out = g.agg(
        n_patients_tested=("patient", "nunique"),
        n_patients_hit=("hit", "sum"),
        n_pos=("pos_hit", "sum"),
        n_neg=("neg_hit", "sum"),
        mean_coef=("coefficient", "mean"),
        mean_abs_coef=("abs_coef", "mean"),
        median_treatment_pct=("term_pct_of_total_var", "median"),
    ).reset_index()
    out["sign_concordance"] = out[["n_pos", "n_neg"]].max(axis=1) / out[
        "n_patients_hit"
    ].replace(0, np.nan)
    return out


consensus = {p: consensus_table(trt[p]) for p in profiles}
patients_per_hit = pd.concat(
    [
        consensus[p]["n_patients_hit"]
        .value_counts(normalize=True)
        .rename("fraction")
        .rename_axis("n_patients_hit")
        .reset_index()
        .assign(profile=p)
        for p in profiles
    ],
    ignore_index=True,
)
concordance = pd.concat(
    [
        consensus[p]
        .loc[consensus[p]["n_patients_hit"] >= 2, ["sign_concordance"]]
        .assign(profile=p)
        for p in profiles
    ],
    ignore_index=True,
)
save_plot_data(
    "9_patients_per_hit_and_concordance",
    counts=patients_per_hit,
    concordance=concordance,
)


# In[ ]:


# consensus hits per treatment: >=3 patients, >=80% same sign
consensus_counts = []
for p in main_profiles:
    c = consensus[p]
    cons = c.loc[(c["n_patients_hit"] >= 3) & (c["sign_concordance"] >= 0.8)]
    cnt = (
        cons.groupby("treatment")
        .size()
        .reindex(all_treatments, fill_value=0)
        .sort_values()
    )
    consensus_counts.append(cnt.rename(p))
pd.concat(consensus_counts, axis=1).reset_index().to_parquet(
    results_path / "consensus_hit_counts.parquet", index=False
)


# In[ ]:


# patient x patient agreement of the full treatment signature
patient_agreement_frames = []
for p in main_profiles:
    mats = []
    for trtm, g in trt[p].groupby("treatment"):
        w = g.pivot(index="feature", columns="patient", values="coefficient").reindex(
            columns=all_patients
        )
        mats.append(w.corr(method="spearman"))
    mean_corr = sum(m.fillna(0) for m in mats) / len(mats)
    patient_agreement_frames.append(
        mean_corr.rename_axis("patient_a")
        .reset_index()
        .melt(id_vars="patient_a", var_name="patient_b", value_name="spearman")
        .assign(profile=p)
    )
patient_agreement = pd.concat(patient_agreement_frames, ignore_index=True)
save_plot_data("9_patient_patient_agreement", agreement=patient_agreement)


# ## 10. Treatment similarity
#
# *Do drugs with the same MOA give similar morphological signatures? And
# which is stronger: the patient effect or the drug effect?*

# In[ ]:


sig_matrix = {}
for p in main_profiles:
    sig_matrix[p] = trt[p].pivot_table(
        index="feature", columns=["patient", "treatment"], values="coefficient"
    )

signature_corr_frames = []
signature_moa_frames = []
for p in main_profiles:
    mean_sig = trt[p].pivot_table(
        index="feature", columns="treatment", values="coefficient", aggfunc="mean"
    )
    corr = mean_sig.corr()
    moa = (
        trt[p]
        .drop_duplicates("treatment")
        .set_index("treatment")["therapeutic_category"]
        .reindex(corr.index)
    )
    signature_corr_frames.append(
        corr.rename_axis("treatment_a")
        .reset_index()
        .melt(id_vars="treatment_a", var_name="treatment_b", value_name="correlation")
        .assign(profile=p)
    )
    signature_moa_frames.append(
        moa.rename("therapeutic_category")
        .rename_axis("treatment")
        .reset_index()
        .assign(profile=p)
    )
signature_corr = pd.concat(signature_corr_frames, ignore_index=True)
signature_moa = pd.concat(signature_moa_frames, ignore_index=True)
save_plot_data(
    "10_treatment_signature_clustermap", correlation=signature_corr, moa=signature_moa
)


# In[ ]:


# patient vs drug effect: correlate every (patient, treatment) signature
pair_rows = []
for p in main_profiles:
    m = sig_matrix[p].dropna(axis=0, how="any")
    if m.shape[0] < 10:
        m = sig_matrix[p].fillna(0)
    corr = m.corr()
    idx = corr.index
    pats = idx.get_level_values("patient")
    trts = idx.get_level_values("treatment")
    drugs = trts.str.rsplit("_", n=1).str[0]
    iu = np.triu_indices(len(idx), 1)
    c = corr.values[iu]
    same_p = pats.values[iu[0]] == pats.values[iu[1]]
    same_d = drugs.values[iu[0]] == drugs.values[iu[1]]
    cat = np.where(
        same_p & ~same_d,
        "same patient,\ndifferent drug",
        np.where(
            ~same_p & same_d,
            "different patient,\nsame drug",
            np.where(
                ~same_p & ~same_d, "different both", "same patient,\nsame drug (doses)"
            ),
        ),
    )
    pair_rows.append(pd.DataFrame({"profile": p, "category": cat, "correlation": c}))
pairs = pd.concat(pair_rows)
pairs.to_parquet(results_path / "signature_pair_correlations.parquet", index=False)
save_plot_data("10_patient_vs_drug_signature", pairs=pairs)


# In[ ]:


# PCA (SVD) of all patient x treatment signatures
for p in main_profiles:
    m = sig_matrix[p].fillna(0).T
    x = m.values - m.values.mean(axis=0)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    pcs = u[:, :2] * s[:2]
    var = (s**2 / (s**2).sum())[:2] * 100
    meta = pd.DataFrame(m.index.tolist(), columns=["patient", "treatment"])
    meta["drug"] = meta["treatment"].str.rsplit("_", n=1).str[0]
    meta["tumor_type"] = meta["patient"].map(tumor_type_dict)
    meta["PC1"], meta["PC2"] = pcs[:, 0], pcs[:, 1]
    meta["PC1_var_pct"], meta["PC2_var_pct"] = var[0], var[1]
    meta.assign(profile=p).to_parquet(
        results_path / f"signature_pca_{p}.parquet", index=False
    )


# ## 11. Organoid vs single cell
#
# *Organoid and single-cell features share (Channel, Feature type,
# Measurement) but not a compartment. Averaging the single-cell coefficient
# over its compartments gives a like-for-like comparison. Do the two scales
# agree on direction, on which patient x treatment responds, and on hit rate?*

# In[ ]:


shared = [
    "patient",
    "treatment",
    "drug",
    "therapeutic_category",
    "channel_label",
    "Feature_type",
    "Measurement",
]
org_c = (
    trt["organoid"]
    .groupby(shared, dropna=False)
    .agg(coef_org=("coefficient", "mean"), hit_org=("hit", "any"))
    .reset_index()
)
sc_c = (
    trt["sc"]
    .groupby(shared, dropna=False)
    .agg(coef_sc=("coefficient", "mean"), hit_sc=("hit", "any"))
    .reset_index()
)
os_merge = org_c.merge(sc_c, on=shared, how="inner")
print(f"{len(os_merge):,} shared (patient, treatment, feature) tuples")

per_trt = (
    os_merge.groupby("treatment")
    .apply(lambda g: spearmanr(g["coef_org"], g["coef_sc"])[0])
    .sort_values()
)
per_trt.rename("spearman").reset_index().to_parquet(
    results_path / "organoid_vs_sc_per_treatment.parquet", index=False
)
save_plot_data(
    "11_organoid_vs_sc_agreement",
    merged=os_merge,
    per_treatment=per_trt.rename("spearman").reset_index(),
)


# In[ ]:


# does the same patient x treatment respond at both scales?
rate = (
    trt["organoid"]
    .groupby(["patient", "treatment"])["hit"]
    .mean()
    .rename("organoid")
    .to_frame()
    .join(
        trt["sc"].groupby(["patient", "treatment"])["hit"].mean().rename("sc"),
        how="inner",
    )
    .reset_index()
)
save_plot_data("11_hit_rate_organoid_vs_sc", rate=rate)


# ## 12. Aggregated vs non-aggregated profiles
#
# *`*_agg` profiles have far fewer rows per model. Do their coefficients and
# hits agree with the full-resolution fits?*

# In[ ]:


agg_pairs = [("organoid", "organoid_agg"), ("sc", "sc_agg")]
agg_rows = []
agg_scatter_frames = []
for a, b in agg_pairs:
    m = trt[a][model_keys + ["coefficient", "hit", "sig", "rsquared"]].merge(
        trt[b][model_keys + ["coefficient", "hit", "sig", "rsquared"]],
        on=model_keys,
        suffixes=("_full", "_agg"),
    )
    rho = spearmanr(m["coefficient_full"], m["coefficient_agg"])[0]
    both = (m["hit_full"] & m["hit_agg"]).sum()
    agg_rows.append(
        {
            "pair": f"{a} vs {b}",
            "n_models": len(m),
            "spearman_coef": rho,
            "hits_full": int(m["hit_full"].sum()),
            "hits_agg": int(m["hit_agg"].sum()),
            "hits_both": int(both),
            "jaccard_hits": both / max((m["hit_full"] | m["hit_agg"]).sum(), 1),
            "sign_agree_when_full_hit": (
                np.sign(m.loc[m["hit_full"], "coefficient_full"])
                == np.sign(m.loc[m["hit_full"], "coefficient_agg"])
            ).mean(),
        }
    )
    agg_scatter_frames.append(m.assign(pair=f"{a} vs {b}"))
agg_concordance = pd.DataFrame(agg_rows)
agg_concordance.to_parquet(
    results_path / "full_vs_agg_concordance.parquet", index=False
)
save_plot_data(
    "12_full_vs_agg_coefficients",
    scatter=pd.concat(agg_scatter_frames, ignore_index=True),
    summary=agg_concordance,
)
agg_concordance


# ## 13. Which technical covariates matter?
#
# *The models include well position / distance-from-center / depth covariates on
# top of the cell and organoid counts. Which covariate terms explain variance and
# reach significance?*

# In[ ]:


technical_covariates_frames = []
for p in main_profiles:
    d = lm[p].loc[lm[p]["term"] != "treatment"]
    fr = (
        d.groupby("term")["sig"]
        .mean()
        .reindex(covariate_terms)
        .rename("frac_sig")
        .reset_index()
    )
    fr["profile"] = p
    technical_covariates_frames.append(fr)
technical_covariates_sig = pd.concat(technical_covariates_frames, ignore_index=True)
technical_covariates_var = pd.concat(
    [
        lm[p]
        .loc[lm[p]["term"].isin(covariate_terms), ["term", "term_pct_of_total_var"]]
        .assign(profile=p)
        for p in main_profiles
    ],
    ignore_index=True,
)
save_plot_data(
    "13_technical_covariates",
    frac_sig=technical_covariates_sig,
    variance_share=technical_covariates_var,
)


# In[ ]:


# mean variance share of every term (treatment + covariates)
tv = pd.concat(
    [lm[p].groupby("term")["term_pct_of_total_var"].mean().rename(p) for p in profiles],
    axis=1,
).reindex(term_order)
tv.reset_index().to_parquet(results_path / "term_variance_share.parquet", index=False)
save_plot_data("13_technical_term_variance", term_variance=tv.reset_index())


# ## 14. Are hits just a change in a covariate?
#
# *If a drug kills organoids/cells the count covariates absorb variance, and
# position covariates can absorb plate effects. A hit where treatment explains
# less variance than any covariate term is suspect.*

# In[ ]:


count_rows = []
for p in profiles:
    h = trt[p].loc[trt[p]["hit"]]
    count_rows.append(
        {
            "profile": p,
            "hits": len(h),
            "frac_hits_treatment_dominant": h["covariate_safe"].mean(),
            "frac_all_treatment_dominant": trt[p]["covariate_safe"].mean(),
        }
    )
count_df = pd.DataFrame(count_rows)
count_df.to_parquet(results_path / "count_confounding.parquet", index=False)
organoid_dominant_by_treatment = (
    trt["organoid"]
    .loc[trt["organoid"]["hit"]]
    .groupby("treatment")["covariate_safe"]
    .mean()
    .sort_values()
    .rename("frac_treatment_dominant")
    .reset_index()
)
save_plot_data(
    "14_count_confounding",
    summary=count_df,
    organoid_by_treatment=organoid_dominant_by_treatment,
)
count_df


# In[ ]:


# how much variance do the covariate terms themselves explain?
covariate_variance = pd.concat(
    [
        lm[p]
        .loc[lm[p]["term"].isin(covariate_terms), ["term", "term_pct_of_total_var"]]
        .assign(profile=p)
        for p in main_profiles
    ],
    ignore_index=True,
)
save_plot_data("14_count_covariate_variance_share", variance=covariate_variance)


# ## 15. Sensitivity of the hit definition
#
# *How many hits do you get as the FDR / R2 / effect thresholds move?*

# In[ ]:


fdrs = [0.001, 0.01, 0.05, 0.1]
r2s = [0.0, 0.3, 0.5, 0.7]
coefs = [0.0, 0.01, 0.1, 0.25, 0.5, 1.0]
sens_rows = []
grid_frames = []
for p in main_profiles:
    d = trt[p]
    grid = pd.DataFrame(
        [
            [
                (
                    (d["pvalue_fdr"] < f)
                    & (d["rsquared"] > r)
                    & (d["rsquared_adj"] > 0)
                    & (d["coefficient"].abs() > COEF_MIN)
                ).sum()
                for r in r2s
            ]
            for f in fdrs
        ],
        index=fdrs,
        columns=r2s,
    )
    grid_frames.append(
        grid.rename_axis("fdr")
        .reset_index()
        .melt(id_vars="fdr", var_name="r2_min", value_name="n_hits")
        .assign(profile=p)
    )
    curve = [
        (
            (d["pvalue_fdr"] < FDR_MAX)
            & (d["rsquared"] > R2_MIN)
            & (d["rsquared_adj"] > 0)
            & (d["coefficient"].abs() > c)
        ).sum()
        for c in coefs
    ]
    for c, n in zip(coefs, curve):
        sens_rows.append({"profile": p, "coef_min": c, "n_hits": int(n)})
sens_df = pd.DataFrame(sens_rows)
sens_df.to_parquet(results_path / "threshold_sensitivity_coef.parquet", index=False)
save_plot_data(
    "15_threshold_sensitivity",
    grid=pd.concat(grid_frames, ignore_index=True),
    coef_curve=sens_df,
)


# ## 16. Tumor-type-specific responses
#
# *Do cNF, pNF, MPNST and "Other" tumors respond to different drugs and
# different parts of feature space?*

# In[ ]:


tumor_type_by_treatment = pd.concat(
    [
        trt[p]
        .pivot_table(
            index="tumor_type", columns="treatment", values="hit", aggfunc="mean"
        )
        .rename_axis("tumor_type")
        .reset_index()
        .melt(id_vars="tumor_type", var_name="treatment", value_name="hit_rate")
        .assign(profile=p)
        for p in main_profiles
    ],
    ignore_index=True,
)
save_plot_data("16_tumor_type_by_treatment", heatmap=tumor_type_by_treatment)


# In[ ]:


# features that discriminate tumor types: hit rate per tumor type, keep the most variable
tt_rows = []
for p in main_profiles:
    m = trt[p].pivot_table(
        index="feature", columns="tumor_type", values="hit", aggfunc="mean"
    )
    m["range"] = m.max(axis=1) - m.min(axis=1)
    top = m.sort_values("range", ascending=False).head(40)
    tt_rows.append(top.assign(profile=p).reset_index())
pd.concat(tt_rows).to_parquet(
    results_path / "tumor_type_discriminating_features.parquet", index=False
)


# ## 17. Feature modules: are the hit features redundant?
#
# *Hundreds of morphology features are highly correlated. Clustering hit
# features by their coefficient profile across patient x treatment groups
# shows how many independent "modules" of biology sit in the hits.*

# In[ ]:


module_tables = []
module_corr_frames = []
for p in main_profiles:
    hit_feats = trt[p].loc[trt[p]["hit"]].groupby("feature").size().nlargest(60).index
    m = (
        trt[p]
        .loc[trt[p]["feature"].isin(hit_feats)]
        .pivot_table(
            index="feature", columns=["patient", "treatment"], values="coefficient"
        )
        .fillna(0)
    )
    corr = m.T.corr()
    link = linkage(m.values, method="average", metric="correlation")
    clusters = fcluster(link, t=0.5, criterion="distance")
    module_tables.append(
        pd.DataFrame({"profile": p, "feature": m.index, "module": clusters})
    )
    module_corr_frames.append(
        corr.rename_axis("feature_a")
        .reset_index()
        .melt(id_vars="feature_a", var_name="feature_b", value_name="correlation")
        .assign(profile=p)
    )
pd.concat(module_tables).to_parquet(
    results_path / "feature_modules.parquet", index=False
)
save_plot_data(
    "17_feature_module_clustermap",
    correlation=pd.concat(module_corr_frames, ignore_index=True),
)


# ## 18. The needles
#
# Each hit row is scored on independent lines of evidence:
#
# * **covariate-safe** -- treatment explains more variance than any covariate term
#   (reported, but *not* in the score: counts may be part of the phenotype, see 19)
# * **agg-supported** -- significant, same sign in the aggregated profile
# * **dose-consistent** -- same sign at another dose of the same drug
#
# Rows are then summarized per (treatment, feature):
#
# `needle_score = n_patients_hit * sign_concordance * mean_robustness * mean_|coef|`
#
# (robustness = mean of agg-supported and dose-consistent). This score
# favours *replicated* effects; sections 20-21 rank the other kinds of interesting.

# In[ ]:


def build_needles(p):
    d = trt[p].copy()
    agg_key = {"organoid": "organoid_agg", "sc": "sc_agg"}.get(p)
    # agg-supported
    if agg_key:
        a = trt[agg_key][model_keys + ["sig", "coefficient"]].rename(
            columns={"sig": "agg_sig", "coefficient": "agg_coef"}
        )
        d = d.merge(a, on=model_keys, how="left")
        supported = d["agg_sig"].fillna(False).astype(bool) & (
            np.sign(d["agg_coef"]) == np.sign(d["coefficient"])
        )
        d["agg_supported"] = supported.astype(float).where(d["agg_sig"].notna())
    else:
        d["agg_supported"] = np.nan
    # dose-consistent
    sgn = (
        d.assign(sgn=np.sign(d["coefficient"]))
        .groupby(["patient", "drug", "feature"])["sgn"]
        .agg(["nunique", "size"])
    )
    sgn["dose_consistent"] = np.where(
        sgn["size"] > 1, (sgn["nunique"] == 1).astype(float), np.nan
    )
    d = d.merge(
        sgn[["dose_consistent"]].reset_index(),
        on=["patient", "drug", "feature"],
        how="left",
    )
    ev = d[["agg_supported", "dose_consistent"]].astype(float)
    d["robustness"] = ev.mean(axis=1)
    hits = d.loc[d["hit"]].copy()
    hits["abs_coef"] = hits["coefficient"].abs()
    return d, hits


needle_rows, all_needles = {}, {}
for p in main_profiles:
    d, hits = build_needles(p)
    needle_rows[p] = hits
    cons = consensus[p].copy()
    rob = (
        hits.groupby(["treatment", "feature"])["robustness"]
        .mean()
        .rename("mean_robustness")
        .reset_index()
    )
    cons = cons.merge(rob, on=["treatment", "feature"], how="left")
    cons["needle_score"] = (
        cons["n_patients_hit"]
        * cons["sign_concordance"].fillna(0)
        * cons["mean_robustness"].fillna(0)
        * cons["mean_abs_coef"]
    )
    cons = cons.sort_values("needle_score", ascending=False)
    cons["profile"] = p
    all_needles[p] = cons
    cons.to_parquet(results_path / f"needles_{p}.parquet", index=False)
    hits.to_parquet(results_path / f"hit_rows_with_evidence_{p}.parquet", index=False)
    print(f"{p}: {(cons['needle_score'] > 0).sum():,} scored needles")
all_needles["organoid"].head(25)


# In[ ]:


# evidence overview: what fraction of hits pass each line of evidence?
ev_rows = []
for p in main_profiles:
    h = needle_rows[p]
    ev_rows.append(
        {
            "profile": p,
            "covariate_safe": h["covariate_safe"].astype(float).mean(),
            "agg_supported": h["agg_supported"].astype(float).mean(),
            "dose_consistent": h["dose_consistent"].astype(float).mean(),
            "all_available_pass": (h["robustness"] == 1).mean(),
        }
    )
ev = pd.DataFrame(ev_rows).set_index("profile")
ev.reset_index().to_parquet(results_path / "evidence_pass_rates.parquet", index=False)
save_plot_data("18_evidence_pass_rates", rates=ev.reset_index())
ev


# In[ ]:


# top needles overall
top_needles_frames = []
for p in main_profiles:
    top = all_needles[p].head(30).copy()
    top["label"] = (
        top["drug"]
        + " | "
        + top["treatment"].str.rsplit("_", n=1).str[-1]
        + " | "
        + top["feature"]
    )
    top["profile"] = p
    top_needles_frames.append(top)
save_plot_data(
    "18_top_needles", needles=pd.concat(top_needles_frames, ignore_index=True)
)


# In[ ]:


# the top needles across every patient: is the effect visible everywhere?
needles_across_patients_frames = []
for p in main_profiles:
    top = all_needles[p].head(40)
    keys = list(zip(top["treatment"], top["feature"]))
    d = trt[p].set_index(["treatment", "feature"])
    d = d.loc[d.index.isin(keys)].reset_index()
    d["row"] = (
        d["drug"]
        + " | "
        + d["treatment"].str.rsplit("_", n=1).str[-1]
        + " | "
        + d["feature"]
    )
    needles_across_patients_frames.append(
        d[["row", "patient", "coefficient"]].assign(profile=p)
    )
save_plot_data(
    "18_top_needles_across_patients",
    coefficients=pd.concat(needles_across_patients_frames, ignore_index=True),
)


# In[ ]:


# the best feature for every treatment
best_tables = []
for p in main_profiles:
    best = (
        all_needles[p]
        .sort_values("needle_score", ascending=False)
        .drop_duplicates("treatment")
        .copy()
    )
    best_tables.append(best)
pd.concat(best_tables).to_parquet(
    results_path / "best_needle_per_treatment.parquet", index=False
)


# In[ ]:


# MOA-level needles: features hit by several drugs of the same MOA
moa_rows = []
for p in main_profiles:
    h = needle_rows[p]
    g = (
        h.groupby(["therapeutic_category", "feature"])
        .agg(
            n_drugs=("drug", "nunique"),
            n_patients=("patient", "nunique"),
            mean_coef=("coefficient", "mean"),
            n_hits=("hit", "size"),
        )
        .reset_index()
    )
    tot_drugs = (
        trt[p].groupby("therapeutic_category")["drug"].nunique().rename("moa_drugs")
    )
    g = g.merge(tot_drugs.reset_index(), on="therapeutic_category")
    g["frac_moa_drugs_hit"] = g["n_drugs"] / g["moa_drugs"]
    g["profile"] = p
    moa_rows.append(g)
moa_needles = pd.concat(moa_rows)
moa_needles.to_parquet(results_path / "moa_needles.parquet", index=False)
moa_needles.loc[(moa_needles["moa_drugs"] > 1)].sort_values(
    ["frac_moa_drugs_hit", "n_patients"], ascending=False
).head(25)


# In[ ]:


# patient-private needles: strong, robust hits that occur in exactly one patient.
# these may be patient-specific biology (or noise -- inspect before believing).
private_rows = []
for p in main_profiles:
    c = all_needles[p]
    private = c.loc[(c["n_patients_hit"] == 1) & (c["mean_robustness"] >= 0.75)].copy()
    h = needle_rows[p][["treatment", "feature", "patient", "rsquared", "abs_coef"]]
    private = private.merge(h, on=["treatment", "feature"], how="left")
    private["profile"] = p
    private_rows.append(private.sort_values("abs_coef", ascending=False).head(50))
private_needles = pd.concat(private_rows)
private_needles.to_parquet(
    results_path / "patient_private_needles.parquet", index=False
)
private_needles.groupby("profile").head(10)[
    [
        "profile",
        "patient",
        "treatment",
        "feature",
        "mean_coef",
        "rsquared",
        "mean_robustness",
    ]
]


# In[ ]:


# summary: a compact story of the haystack
summary = []
for p in main_profiles:
    c = all_needles[p]
    summary.append(
        {
            "profile": p,
            "models": len(trt[p]),
            "hits": int(trt[p]["hit"].sum()),
            "(treatment,feature) pairs with >=1 hit": int(
                (c["n_patients_hit"] >= 1).sum()
            ),
            "... hit in >=2 patients": int((c["n_patients_hit"] >= 2).sum()),
            "... >=3 patients & >=80% concordant": int(
                ((c["n_patients_hit"] >= 3) & (c["sign_concordance"] >= 0.8)).sum()
            ),
            "... and mean robustness >= 0.75": int(
                (
                    (c["n_patients_hit"] >= 3)
                    & (c["sign_concordance"] >= 0.8)
                    & (c["mean_robustness"] >= 0.75)
                ).sum()
            ),
        }
    )
summary = pd.DataFrame(summary)
summary.to_parquet(results_path / "haystack_summary.parquet", index=False)
summary.T


# ## 19. Counts, both ways
#
# Whether a count change is a nuisance or a phenotype is undecided, so both
# views are shown. **Adjusted view**: hits where treatment beats every covariate
# term (`covariate_safe`). **Count-linked view**: hits where a covariate term
# explains as much or more variance. A treatment that kills organoids will
# show up in the second group and may be biologically the most interesting.
# (The effect of treatment *on* the counts is not in the saved results; that
# needs a separate model of well-level counts.)

# In[ ]:


count_view = []
for p in main_profiles:
    d = trt[p]
    h = d.loc[d["hit"]]
    tab = pd.DataFrame(
        {
            "hits": h.groupby("treatment").size(),
            "hits_treatment_dominant": h.groupby("treatment")["covariate_safe"].sum(),
            "median_treatment_share_pct": d.groupby("treatment")[
                "term_pct_of_total_var"
            ].median(),
            "median_covariate_share_pct": d.groupby("treatment")[
                "covariate_max_pct"
            ].median(),
        }
    ).reindex(all_treatments)
    tab[["hits", "hits_treatment_dominant"]] = tab[
        ["hits", "hits_treatment_dominant"]
    ].fillna(0)
    tab["hits_count_linked"] = tab["hits"] - tab["hits_treatment_dominant"]
    tab["profile"] = p
    count_view.append(tab.reset_index())
count_view = pd.concat(count_view)
count_view.to_parquet(results_path / "count_view_by_treatment.parquet", index=False)
save_plot_data("19_hits_treatment_vs_count_linked", count_view=count_view)


# In[ ]:


# treatment vs count share of variance, per (patient, treatment) -- points above
# the diagonal are treatments whose count effect exceeds their morphology effect
treatment_vs_covariate_share = pd.concat(
    [
        trt[p]
        .groupby(["treatment", "drug"])
        .agg(
            treatment_share=("term_pct_of_total_var", "median"),
            covariate_share=("covariate_max_pct", "median"),
        )
        .reset_index()
        .assign(profile=p)
        for p in main_profiles
    ],
    ignore_index=True,
)
save_plot_data("19_treatment_vs_covariate_share", share=treatment_vs_covariate_share)


# In[ ]:


# features whose hits are mostly count-linked (candidate "object number" readouts)
feat_count = []
for p in main_profiles:
    h = trt[p].loc[trt[p]["hit"]]
    fc = (
        h.groupby("feature")["covariate_safe"]
        .agg(n_hits="size", frac_treatment_driven="mean")
        .reset_index()
    )
    feat_count.append(fc.assign(profile=p))
feat_count = pd.concat(feat_count)
feat_count.to_parquet(results_path / "feature_count_linkage.parquet", index=False)
feat_count.loc[feat_count["n_hits"] >= 3].sort_values("frac_treatment_driven").head(15)


# ## 20. Tumor-type specificity
#
# *For each (treatment, feature): does the coefficient differ between
# tumor types (Kruskal-Wallis on per-patient coefficients)? Which tumor type
# is the outlier, and is the hit confined to it?* With 2-5 patients per tumor
# type the tests are underpowered, so the effect (`delta`) and the hit-based
# specificity flag matter more than the p-value.

# In[ ]:


def tumor_type_specificity(df):
    rows = []
    for (trtm, feat), g in df.groupby(["treatment", "feature"]):
        tt = g["tumor_type"].to_numpy(dtype=object)
        coef = g["coefficient"].to_numpy()
        hit = g["hit"].to_numpy()
        types = np.unique(tt)
        if len(types) < 2:
            continue
        try:
            pval = kruskal(*[coef[tt == t] for t in types])[1]
        except ValueError:  # every coefficient identical
            pval = np.nan
        best, best_delta = None, 0.0
        for t in types:
            delta = coef[tt == t].mean() - coef[tt != t].mean()
            if abs(delta) >= abs(best_delta):
                best, best_delta = t, delta
        inside, outside = tt == best, tt != best
        rows.append(
            {
                "treatment": trtm,
                "feature": feat,
                "drug": g["drug"].iloc[0],
                "therapeutic_category": g["therapeutic_category"].iloc[0],
                "best_tumor_type": best,
                "n_patients_in_type": int(inside.sum()),
                "delta": best_delta,
                "mean_coef_in_type": coef[inside].mean(),
                "hit_frac_in_type": hit[inside].mean(),
                "hit_frac_elsewhere": hit[outside].mean(),
                "kruskal_p": pval,
            }
        )
    out = pd.DataFrame(rows)
    ok = out["kruskal_p"].notna()
    out["kruskal_fdr"] = np.nan
    out.loc[ok, "kruskal_fdr"] = multipletests(
        out.loc[ok, "kruskal_p"], method="fdr_bh"
    )[1]
    out["tumor_type_specific"] = (
        (out["n_patients_in_type"] >= 2)
        & (out["hit_frac_in_type"] >= 0.5)
        & (out["hit_frac_elsewhere"] == 0)
    )
    return out


tumor_specific = {p: tumor_type_specificity(trt[p]) for p in main_profiles}
for p, t in tumor_specific.items():
    t.to_parquet(results_path / f"tumor_type_specific_{p}.parquet", index=False)
    print(
        f"{p}: {int(t['tumor_type_specific'].sum())} tumor-type-specific pairs; {(t['kruskal_fdr'] < 0.05).sum()} with Kruskal FDR<0.05"
    )


# In[ ]:


tumor_specificity_scatter_frames = []
tumor_specificity_heat_frames = []
for p in main_profiles:
    t = tumor_specific[p]
    tumor_specificity_scatter_frames.append(
        t.assign(neglog_p=-np.log10(t["kruskal_p"].clip(lower=1e-12)), profile=p)[
            ["delta", "neglog_p", "tumor_type_specific", "profile"]
        ]
    )
    sp = t.loc[t["tumor_type_specific"]]
    top = sp.reindex(sp["delta"].abs().sort_values(ascending=False).index).head(30)
    if len(top):
        d = trt[p].set_index(["treatment", "feature"])
        rows = []
        for _, r in top.iterrows():
            g = d.loc[(r["treatment"], r["feature"])]
            m = g.groupby("tumor_type")["coefficient"].mean()
            m.name = (
                f"{r['drug']} | {r['treatment'].rsplit('_', 1)[-1]} | {r['feature']}"
            )
            rows.append(m)
        heat = pd.DataFrame(rows).reindex(columns=["cNF", "pNF", "MPNST", "Other"])
        tumor_specificity_heat_frames.append(
            heat.rename_axis("pair")
            .reset_index()
            .melt(id_vars="pair", var_name="tumor_type", value_name="mean_coef")
            .assign(profile=p)
        )
save_plot_data(
    "20_tumor_type_specificity",
    scatter=pd.concat(tumor_specificity_scatter_frames, ignore_index=True),
    heatmap=pd.concat(tumor_specificity_heat_frames, ignore_index=True)
    if tumor_specificity_heat_frames
    else pd.DataFrame(columns=["pair", "tumor_type", "mean_coef", "profile"]),
)


# In[ ]:


# which tumor type owns the specific hits, and for which MOA?
ts_all = pd.concat(
    [
        t.loc[t["tumor_type_specific"]].assign(profile=p)
        for p, t in tumor_specific.items()
    ]
)
save_plot_data("20_tumor_type_specific_counts", ts_all=ts_all)
ts_all.groupby(["profile", "best_tumor_type", "therapeutic_category"]).size().rename(
    "n"
).reset_index().sort_values("n", ascending=False).head(15)


# ## 21. Four kinds of "interesting" and the readout shortlist
#
# Each (treatment, feature) pair is flagged separately for each definition:
#
# * **replicated** -- hit in >=3 patients, >=80% same sign
# * **tumor-type-specific** -- hit confined to one tumor type (section 20)
# * **dose-dependent** -- among patients with a hit at either dose, >=2 patients and >=75% show the same sign with a larger |effect| at the higher dose
# * **MOA-consistent** -- feature is hit by >=50% of the (>=2) drugs of that MOA
#
# The shortlist keeps pairs passing at least one and records which. Count
# behaviour and technical robustness are carried as columns rather than filters.

# In[ ]:


def dose_value(label):
    num = float("".join(c for c in label if c.isdigit() or c == "."))
    return num * (1000 if "uM" in label else 1)


dose_dep = {}
for p in main_profiles:
    d = trt[p]
    multi = d.groupby("drug")["dose"].nunique()
    d = d.loc[d["drug"].isin(multi[multi > 1].index)].assign(
        hit_i=lambda x: x["hit"].astype(int)
    )
    idx = ["patient", "drug", "feature"]
    coefw = d.pivot_table(index=idx, columns="dose", values="coefficient")
    hitw = d.pivot_table(index=idx, columns="dose", values="hit_i", aggfunc="max")
    cols = sorted(coefw.columns, key=dose_value)
    lo, hi = cols[0], cols[-1]
    t = pd.DataFrame(
        {"lo": coefw[lo], "hi": coefw[hi], "any_hit": hitw.max(axis=1) > 0}
    ).dropna(subset=["lo", "hi"])
    t["grows"] = (np.sign(t["lo"]) == np.sign(t["hi"])) & (
        t["hi"].abs() > t["lo"].abs()
    )
    g = (
        t.loc[t["any_hit"]]
        .groupby(level=["drug", "feature"])["grows"]
        .agg(n_patients_hit_any="size", n_grow="sum")
        .reset_index()
    )
    g["frac_grow"] = g["n_grow"] / g["n_patients_hit_any"]
    g["dose_dependent"] = (g["n_patients_hit_any"] >= 2) & (g["frac_grow"] >= 0.75)
    dose_dep[p] = g
    g.to_parquet(results_path / f"dose_dependent_{p}.parquet", index=False)
    print(f"{p}: {int(g['dose_dependent'].sum())} dose-dependent (drug, feature) pairs")


# In[ ]:


shortlists = []
for p in main_profiles:
    base = all_needles[p].loc[all_needles[p]["n_patients_hit"] >= 1].copy()
    base["replicated"] = (base["n_patients_hit"] >= 3) & (
        base["sign_concordance"] >= 0.8
    )
    ts = tumor_specific[p][
        ["treatment", "feature", "best_tumor_type", "tumor_type_specific"]
    ]
    base = base.merge(ts, on=["treatment", "feature"], how="left")
    dd = dose_dep[p][["drug", "feature", "dose_dependent"]]
    base = base.merge(dd, on=["drug", "feature"], how="left")
    mo = moa_needles.loc[
        (moa_needles["profile"] == p) & (moa_needles["moa_drugs"] > 1)
    ].copy()
    mo["moa_consistent"] = (mo["n_drugs"] >= 2) & (mo["frac_moa_drugs_hit"] >= 0.5)
    base = base.merge(
        mo[["therapeutic_category", "feature", "moa_consistent"]],
        on=["therapeutic_category", "feature"],
        how="left",
    )
    crit = ["replicated", "tumor_type_specific", "dose_dependent", "moa_consistent"]
    for c in crit:
        base[c] = base[c].fillna(False).astype(bool)
    base["n_criteria"] = base[crit].sum(axis=1)
    h = needle_rows[p]
    cs = (
        h.groupby(["treatment", "feature"])
        .agg(frac_covariate_safe=("covariate_safe", "mean"))
        .reset_index()
    )
    base = base.merge(cs, on=["treatment", "feature"], how="left")
    base["profile"] = p
    shortlists.append(
        base.loc[base["n_criteria"] >= 1].sort_values(
            ["n_criteria", "needle_score"], ascending=False
        )
    )
shortlist = pd.concat(shortlists)
shortlist.to_parquet(results_path / "readout_shortlist.parquet", index=False)
crit = ["replicated", "tumor_type_specific", "dose_dependent", "moa_consistent"]
print(
    shortlist.groupby("profile")[crit + ["n_criteria"]].agg(
        {**{c: "sum" for c in crit}, "n_criteria": "size"}
    )
)


# In[ ]:


counts = shortlist.groupby("profile")[crit].sum().T
combo = (
    shortlist.assign(
        combo=shortlist[crit].apply(
            lambda r: "+".join(c[:4] for c in crit if r[c]), axis=1
        )
    )
    .groupby(["profile", "combo"])
    .size()
    .rename("n")
    .reset_index()
)
save_plot_data(
    "21_definitions_of_interesting",
    counts=counts.rename_axis("criterion").reset_index(),
    combo=combo,
)


# In[ ]:


# the strongest candidates, ranked by how many definitions they pass
shortlist.sort_values(["n_criteria", "needle_score"], ascending=False).head(20)[
    ["profile", "treatment", "feature", "n_criteria"]
    + crit
    + ["n_patients_hit", "best_tumor_type", "frac_covariate_safe"]
]


# ## 22. Is MOA consistency more than chance?
#
# *Mean treatment signatures are correlated between all pairs of drugs. Is the
# average correlation between different drugs of the same MOA higher than
# between drugs of different MOAs? Tested by permuting MOA labels over drugs.*

# In[ ]:


moa_test_rows, moa_within_rows, perm_store = [], [], {}
for p in main_profiles:
    mean_sig = trt[p].pivot_table(
        index="feature", columns="treatment", values="coefficient", aggfunc="mean"
    )
    corr = mean_sig.corr()
    meta = (
        trt[p]
        .drop_duplicates("treatment")
        .set_index("treatment")[["drug", "therapeutic_category"]]
        .reindex(corr.index)
    )
    iu = np.triu_indices(len(corr), 1)
    c = corr.values[iu]
    drug_arr = meta["drug"].to_numpy(dtype=object)
    diff_drug = drug_arr[iu[0]] != drug_arr[iu[1]]

    def contrast(moa_arr):
        same = moa_arr[iu[0]] == moa_arr[iu[1]]
        return np.nanmean(c[diff_drug & same]) - np.nanmean(c[diff_drug & ~same])

    moa_arr = meta["therapeutic_category"].to_numpy(dtype=object)
    obs = contrast(moa_arr)
    moa_by_drug = meta.drop_duplicates("drug").set_index("drug")["therapeutic_category"]
    perms = []
    for _ in range(2000):
        shuffled = pd.Series(
            RNG.permutation(moa_by_drug.to_numpy(dtype=object)), index=moa_by_drug.index
        )
        perms.append(contrast(meta["drug"].map(shuffled).to_numpy(dtype=object)))
    perms = np.array(perms)
    perm_store[p] = (obs, perms)
    moa_test_rows.append(
        {
            "profile": p,
            "observed_within_minus_between": obs,
            "perm_mean": perms.mean(),
            "p_value": (1 + (perms >= obs).sum()) / (1 + len(perms)),
        }
    )
    same = moa_arr[iu[0]] == moa_arr[iu[1]]
    for moa in np.unique(moa_arr[iu[0]][same]):
        sel = diff_drug & same & (moa_arr[iu[0]] == moa)
        if sel.sum():
            moa_within_rows.append(
                {
                    "profile": p,
                    "therapeutic_category": moa,
                    "n_pairs": int(sel.sum()),
                    "mean_within_corr": np.nanmean(c[sel]),
                }
            )
moa_test = pd.DataFrame(moa_test_rows)
moa_within = pd.DataFrame(moa_within_rows)
moa_test.to_parquet(results_path / "moa_consistency_test.parquet", index=False)
moa_within.to_parquet(results_path / "moa_within_correlation.parquet", index=False)

perm_frames = [
    pd.DataFrame({"profile": p, "perm_value": perms})
    for p, (obs, perms) in perm_store.items()
]
save_plot_data(
    "22_moa_consistency_test",
    permutations=pd.concat(perm_frames, ignore_index=True),
    observed=moa_test,
    within_corr=moa_within,
)
moa_test


# ## 23. Thirty questions the models can answer
#
# Each question is answered numerically from the saved fit statistics and the
# tables built above. The answers are collected in `qa_answers.parquet`.

# In[ ]:


qa = []


def answer(n, question, text):
    qa.append({"n": n, "question": question, "answer": text})
    print(f"Q{n}. {question}\n    -> {text}\n")


# ## 24. Every technical variate, explored like treatment
#
# *Sections 4-22 mostly look at the treatment term. The technical model also fits
# `cell_count`, `organoid_count`, `cell_per_organoid_count`, well distance, x/y/z position and z depth for
# every (patient, treatment, feature). Here each term gets the same treatment: how often it is
# significant, how much variance it takes, where it acts (patient, treatment, feature family) and
# whether it replicates across patients.*
#
# Covariate coefficients are per-unit slopes and are not comparable across terms, so the effect-size
# axis is `term_pct_of_total_var`. A term-level "hit" is `pvalue_fdr < FDR_MAX` in a good model
# (`R2 > R2_MIN`, `adj R2 > R2_ADJ_MIN`); the coefficient cut-off is dropped because it is on the treatment scale.

# In[ ]:


# term-level call for every term (treatment included), all profiles
for p in profiles:
    lm[p]["term_hit"] = lm[p]["sig"] & lm[p]["good_model"]

term_summary = pd.concat(
    [
        lm[p]
        .groupby("term")
        .agg(
            n_models=("term_hit", "size"),
            frac_sig=("sig", "mean"),
            frac_term_hit=("term_hit", "mean"),
            median_pct_var=("term_pct_of_total_var", "median"),
            mean_pct_var=("term_pct_of_total_var", "mean"),
            frac_positive=("coefficient", lambda s: (s > 0).mean()),
        )
        .reindex(term_order)
        .assign(profile=p)
        .reset_index()
        for p in profiles
    ],
    ignore_index=True,
)
term_summary.to_parquet(results_path / "term_summary_all_terms.parquet", index=False)
save_plot_data("24_term_summary", summary=term_summary)
term_summary


# In[ ]:


# significance vs variance share, one panel per term
volcano_all_terms_frames = []
for p in main_profiles:
    for term in term_order:
        d = lm[p].loc[lm[p]["term"] == term]
        d = d.sample(min(len(d), 30000), random_state=0)
        volcano_all_terms_frames.append(
            pd.DataFrame(
                {
                    "profile": p,
                    "term": term,
                    "term_pct_of_total_var": d["term_pct_of_total_var"].to_numpy(),
                    "neglog_fdr": -np.log10(
                        d["pvalue_fdr"].clip(lower=1e-300)
                    ).to_numpy(),
                    "term_hit": d["term_hit"].to_numpy(),
                }
            )
        )
save_plot_data(
    "24_volcano_all_terms",
    points=pd.concat(volcano_all_terms_frames, ignore_index=True),
)


# In[ ]:


# where does each term act? term hit rate by patient and by treatment
for p in main_profiles:
    by_patient = (
        lm[p]
        .pivot_table(index="term", columns="patient", values="term_hit", aggfunc="mean")
        .reindex(term_order)
    )
    by_trt = (
        lm[p]
        .pivot_table(
            index="term",
            columns="treatment",
            values="term_pct_of_total_var",
            aggfunc="median",
        )
        .reindex(term_order)
    )
    by_patient.reset_index().to_parquet(
        results_path / f"term_hit_rate_by_patient_{p}.parquet", index=False
    )
    save_plot_data(
        f"24_term_by_patient_and_treatment_{p}",
        by_patient=by_patient.reset_index(),
        by_treatment=by_trt.reset_index(),
    )


# In[ ]:


# which parts of the feature space does each term touch? term x feature family, mean % variance
for p in main_profiles:
    family_tables = {}
    for fam in ["channel_label", "compartment_label", "Feature_type"]:
        m = (
            lm[p]
            .pivot_table(
                index="term",
                columns=fam,
                values="term_pct_of_total_var",
                aggfunc="mean",
            )
            .reindex(term_order)
        )
        m.reset_index().to_parquet(
            results_path / f"term_by_{fam}_{p}.parquet", index=False
        )
        family_tables[fam] = m.reset_index()
    save_plot_data(f"24_term_by_feature_family_{p}", **family_tables)


# In[ ]:


# do term effects replicate across patients? patients with a significant term per (treatment, feature)
rep_rows = []
for p in main_profiles:
    n_pat = (
        lm[p]
        .loc[lm[p]["term_hit"]]
        .groupby(["term", "treatment", "feature"])["patient"]
        .nunique()
        .rename("n_patients")
        .reset_index()
    )
    n_pat["profile"] = p
    rep_rows.append(n_pat)
rep = pd.concat(rep_rows)
rep.to_parquet(results_path / "term_patient_replication.parquet", index=False)

frac_frames = []
for p in main_profiles:
    x = rep.loc[rep["profile"] == p]
    frac = (
        x.assign(rep3=x["n_patients"] >= 3)
        .groupby("term")["rep3"]
        .mean()
        .reindex(term_order)
    )
    frac_frames.append(
        frac.rename("frac_replicated_3plus").reset_index().assign(profile=p)
    )
save_plot_data(
    "24_term_patient_replication", fraction=pd.concat(frac_frames, ignore_index=True)
)
