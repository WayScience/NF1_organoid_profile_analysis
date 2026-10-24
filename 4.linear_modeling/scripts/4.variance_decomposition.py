#!/usr/bin/env python
# coding: utf-8

# # Variance decomposition
#
# Where does the variance in the linear-model fits go, and why is so much of it residual? The linear models (`2.linear_modeling`, `3.linear_modeling_technical_vars`) attribute ~70-90% of each feature's variance to `residual`. This notebook works only from the saved fit statistics (`residual_pct`, `rsquared`, `term_pct_of_total_var`), so nothing is refit.
#
# Sections:
#
# 1. **Variance per term**: the distribution of % of total variance per model term (plus residual), for every fitted model set.
# 2. **Residual diagnostics**: how `residual_pct` is distributed, whether it concentrates in feature families, patients or treatments, what the technical covariates buy, and which of them matter.
# 3. **Variates vs residual**: the models (feature x patient x treatment) where the variates together explain more variance than the residual.
# 4. **Top models**: the highest-explained models, which variate dominates them and which feature families they come from.
#
# **Outputs.** Every figure is a page of one pdf, `figures/variance_decomposition/variance_decomposition.pdf` (dense boxplot outliers are rasterized so the file stays small; each page has a title and a subtitle saying what one box or bar is). The tables are parquet files in `results/decomposed_variance/`, `results/residual_diagnostics/`, `results/variate_vs_residual/` and `results/top_models/`.
#
# `residual_pct = SSR / SST * 100 = (1 - R^2) * 100`, per (patient, treatment, feature) model.

# In[1]:


import pathlib
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from notebook_init_utils import FigurePDFs, init_notebook

warnings.filterwarnings("ignore")
root_dir, in_notebook = init_notebook()

try:
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

sns.set_theme(style="whitegrid", context="talk")
pd.set_option("display.max_columns", 50)

lm_results_path = pathlib.Path(root_dir, "4.linear_modeling/results/linear_modeling")
decomposed_path = pathlib.Path(
    root_dir, "4.linear_modeling/results/decomposed_variance"
)

# decomposition, variate-vs-residual and top-model outputs
figures_path = pathlib.Path(
    root_dir, "4.linear_modeling/figures/variance_decomposition"
)
results_path = pathlib.Path(
    root_dir, "4.linear_modeling/results/variance_decomposition"
)
win_dir = pathlib.Path(root_dir, "4.linear_modeling/results/variate_vs_residual")
top_dir = pathlib.Path(root_dir, "4.linear_modeling/results/top_models")
# residual diagnostics outputs (its figures go in the same pdf as the decomposition)
residual_results_path = pathlib.Path(
    root_dir, "4.linear_modeling/results/residual_diagnostics"
)
for path in (
    figures_path,
    results_path,
    win_dir,
    top_dir,
    decomposed_path,
    residual_results_path,
):
    path.mkdir(parents=True, exist_ok=True)
# one multi-page pdf holding every figure; dense artists (boxplot outliers) are
# rasterized so the vector text/axes stay sharp but the file stays small
pdf_path = figures_path / "variance_decomposition.pdf"
pdfs = FigurePDFs(dpi=600)

# model name -> (parquet, profile, model type); the aggregated fits may not
# have been run yet, so every section skips files that do not exist
lm_files = {
    "organoid_norm": ("organoid_norm.parquet", "organoid", "base"),
    "organoid_norm_technical_model": (
        "organoid_norm_technical_model.parquet",
        "organoid",
        "technical",
    ),
    "single_cell_norm": ("sc_norm.parquet", "single_cell", "base"),
    "single_cell_norm_technical_model": (
        "sc_norm_technical_model.parquet",
        "single_cell",
        "technical",
    ),
    "organoid_agg": ("organoid_agg.parquet", "organoid_agg", "base"),
    "organoid_agg_technical_model": (
        "organoid_agg_technical_model.parquet",
        "organoid_agg",
        "technical",
    ),
    "single_cell_agg": ("single_cell_agg.parquet", "single_cell_agg", "base"),
    "single_cell_agg_technical_model": (
        "single_cell_agg_technical_model.parquet",
        "single_cell_agg",
        "technical",
    ),
}
lm_results_dict = {
    name: {
        "input_profile_path": lm_results_path / file_name,
        "output_profile_path": decomposed_path / file_name,
    }
    for name, (file_name, _, _) in lm_files.items()
}

# the technical-model outputs use the raw Metadata_* column names
rename_map = {
    "Metadata_Biology_PatientTumor": "patient",
    "Metadata_Experiment_Treatment": "treatment",
}
model_keys = ["patient", "treatment", "feature"]


# ## 1. Variance per term: where does the variance actually go?

# In[2]:


def plot_decomposed_variance(df, model_name, level_order=None, save_path=None):
    """
    Plots the distribution of % of total variance per model term (plus residual).

    Parameters:
    - df: long DataFrame with one row per (fitted model, term) and a
      `pct_variance` column.
    - model_name: Name of the model set, used in the subtitle.
    - level_order: List specifying the order of terms. Defaults to row order.
    - save_path: Path to save the plot (png, dpi=600). If None, not saved.
    """
    # horizontal boxplots so long term names (e.g. manhattan_distance_from_center)
    # sit on the y axis and cannot overlap; each box summarises every fitted
    # model (feature x patient x treatment) rather than a single mean
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        x="pct_variance",
        y="term",
        data=df,
        order=level_order,
        fliersize=1,
        linewidth=1,
        flierprops={"rasterized": True},
        ax=ax,
    )
    fig.suptitle(
        "Share of each model's variance explained, by model term", y=1.02, va="bottom"
    )
    ax.set_title(
        f"{model_name}: each box summarises all fitted models\n"
        "(patient x treatment x feature); 'residual' is the unexplained share",
        fontsize=10,
    )
    ax.set_xlabel("% of total variance per model")
    ax.set_ylabel("Model term")
    fig.tight_layout()

    if save_path:
        pdfs.savefig(fig, save_path, bbox_inches="tight")

    plt.show()


# In[3]:


for lm_name, lm_dict in lm_results_dict.items():
    if not lm_dict["input_profile_path"].exists():
        continue
    df = pd.read_parquet(lm_dict["input_profile_path"])
    # one row per (fitted model, term): the share of that model's total variance
    var_decomp = df[["term", "term_pct_of_total_var"]].rename(
        columns={"term_pct_of_total_var": "pct_variance"}
    )
    # the residual is a property of the whole model (repeated on every term
    # row), so take it from one row per model rather than from a term row
    model_key = [
        c
        for c in [
            "patient",
            "Metadata_Biology_PatientTumor",
            "treatment",
            "Metadata_Experiment_Treatment",
            "feature",
        ]
        if c in df.columns
    ]
    residual = (
        df.drop_duplicates(model_key)[["residual_pct"]]
        .rename(columns={"residual_pct": "pct_variance"})
        .assign(term="residual")
    )
    var_decomp = pd.concat([var_decomp, residual], ignore_index=True)
    lm_dict["output_profile_path"].parent.mkdir(parents=True, exist_ok=True)
    var_decomp.to_parquet(lm_dict["output_profile_path"], index=False)
    plot_decomposed_variance(var_decomp, lm_name, save_path=pdf_path)


# # 2. Residual diagnostics: why is the residual so high?
#
# Questions, in order:
#
# 1. How is `residual_pct` distributed across models? Is a high mean driven by a few bad models or is it uniformly high?
# 2. Is it concentrated in particular feature families (compartment, channel, feature type)?
# 3. Is it concentrated in particular patients or treatments?
# 4. How much does adding the technical covariates (position, distance from center) actually reduce it? If barely, the missing variance is not explained by any of the modeled spatial/count terms.
# 5. Which technical covariates matter?

# ## 2.0 Load one row per fitted model
#
# The fit statistics (`rsquared`, `residual_pct`) are repeated on every term
# row of a model, so keep one row per (patient, treatment, feature).

# In[4]:


model_dfs = []
for name, (file_name, profile, model_type) in lm_files.items():
    if not (lm_results_path / file_name).exists():
        continue
    df = pd.read_parquet(lm_results_path / file_name).rename(columns=rename_map)
    # the technical model repeats the treatment term name in the `term` column;
    # the fit statistics are identical across a model's rows either way
    per_model = df.drop_duplicates(model_keys)[
        model_keys
        + [
            "rsquared",
            "residual_pct",
            "Compartment",
            "Channel",
            "Feature_type",
            "Measurement",
        ]
    ].copy()
    per_model["model_name"] = name
    per_model["profile"] = profile
    per_model["model_type"] = model_type
    model_dfs.append(per_model)
    print(f"{name}: {per_model.shape[0]} models")

models = pd.concat(model_dfs, ignore_index=True)
models["Channel"] = models["Channel"].fillna("NoChannel")
models["Compartment"] = models["Compartment"].fillna("adjacent")
models.head()


# ## 2.1 Distribution of `residual_pct`
#
# If the mean is high because of a long tail of near-100% models, the fix is
# different than if the whole distribution sits at 80-90%.

# In[5]:


summary = models.groupby("model_name")["residual_pct"].describe(
    percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)
for cutoff in (80, 90, 95, 99):
    summary[f"frac_>{cutoff}%"] = models.groupby("model_name")["residual_pct"].apply(
        lambda s, c=cutoff: (s > c).mean()
    )
summary.reset_index().to_parquet(residual_results_path / "residual_pct_summary.parquet")

fig, ax = plt.subplots(figsize=(12, 6))
sns.histplot(
    data=models,
    x="residual_pct",
    hue="model_name",
    bins=60,
    element="step",
    stat="density",
    common_norm=False,
    ax=ax,
)
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), title="Model set")
fig.suptitle(
    "Unexplained (residual) variance across fitted models", y=1.02, va="bottom"
)
ax.set_title(
    "Each observation is one fitted model (patient x treatment x feature)\n"
    "residual_pct = (1 - R^2) x 100",
    fontsize=10,
)
ax.set_xlabel("residual_pct (% of total variance unexplained)")
fig.tight_layout()
pdfs.savefig(fig, pdf_path, bbox_inches="tight")
plt.show()


# # 3. Models where the variates outperform the residual
#
# A fitted model (feature x patient x treatment) "wins" when the variance explained by all its variates together exceeds the residual variance, i.e. `residual_pct < 50`. A term "wins" individually when its own share of total variance exceeds the residual share.

# In[6]:


win_summary = []
for lm_name, lm_dict in lm_results_dict.items():
    if not lm_dict["input_profile_path"].exists():
        continue
    df = pd.read_parquet(lm_dict["input_profile_path"])
    # explained vs residual variance is a property of the whole model
    # (repeated on every term row), so compare on the term rows directly:
    # the model-level explained share is 100 - residual_pct
    df["explained_all_pct"] = 100 - df["residual_pct"]
    df["variates_beat_residual"] = df["explained_all_pct"] > df["residual_pct"]
    df["term_beats_residual"] = df["term_pct_of_total_var"] > df["residual_pct"]

    model_key = [
        c
        for c in [
            "patient",
            "Metadata_Biology_PatientTumor",
            "treatment",
            "Metadata_Experiment_Treatment",
            "feature",
        ]
        if c in df.columns
    ]
    models_df = df.drop_duplicates(model_key)
    winners = models_df.loc[models_df["variates_beat_residual"]]
    print(
        f"{lm_name}: {len(winners)} / {len(models_df)} models "
        f"({len(winners) / len(models_df):.1%}) have variates > residual"
    )
    # terms that individually explain more variance than the residual
    term_wins = (
        df.groupby("term")["term_beats_residual"]
        .agg(n_wins="sum", frac_wins="mean")
        .sort_values("n_wins", ascending=False)
    )
    print(term_wins.round(3))
    winners.to_parquet(
        win_dir / f"{lm_name}_variates_beat_residual.parquet", index=False
    )
    win_summary.append(
        pd.DataFrame(
            {
                "model_name": [lm_name],
                "n_models": [len(models_df)],
                "n_variates_beat_residual": [len(winners)],
                "frac_variates_beat_residual": [len(winners) / len(models_df)],
            }
        )
    )

win_summary = pd.concat(win_summary, ignore_index=True)
win_summary.to_parquet(win_dir / "variates_beat_residual_summary.parquet", index=False)
win_summary


# # 4. The highest-explained models: which features and variates?
#
# For each LM the models with the largest explained variance (lowest `residual_pct`) are ranked. The top few are drawn as stacked variance decompositions; the larger top set is used to count which variate dominates each model and which feature families (compartment, channel, feature type) they come from.

# In[7]:


TOP_N_PLOT = 20  # models drawn as stacked bars
TOP_N_SUMMARY = 500  # models used to count dominant variates / feature families
family_cols = ["Compartment", "Channel", "Feature_type"]

for lm_name, lm_dict in lm_results_dict.items():
    if not lm_dict["input_profile_path"].exists():
        continue
    df = pd.read_parquet(lm_dict["input_profile_path"]).rename(columns=rename_map)
    df["model_id"] = df["patient"] + " | " + df["treatment"] + " | " + df["feature"]
    df[family_cols] = df[family_cols].fillna("None")

    # one row per model, ranked by explained variance (100 - residual_pct)
    models_df = df.drop_duplicates("model_id").sort_values("residual_pct")
    top_ids = models_df["model_id"].head(TOP_N_SUMMARY)
    top_df = df.loc[df["model_id"].isin(top_ids)]
    top_df.to_parquet(top_dir / f"{lm_name}_top_models.parquet", index=False)

    # 1. stacked variance decomposition of the very best models
    plot_ids = list(top_ids.head(TOP_N_PLOT))
    stacked = (
        top_df.loc[top_df["model_id"].isin(plot_ids)]
        .pivot(index="model_id", columns="term", values="term_pct_of_total_var")
        .loc[plot_ids]
    )
    stacked["residual"] = models_df.set_index("model_id").loc[plot_ids, "residual_pct"]
    fig, ax = plt.subplots(figsize=(14, 0.45 * TOP_N_PLOT + 2))
    stacked.plot(kind="barh", stacked=True, ax=ax, width=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("% of total variance")
    ax.set_ylabel("")
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), title="Term")
    fig.suptitle(
        f"Variance explained by each term in the {TOP_N_PLOT} best-explained models",
        y=1.02,
        va="bottom",
    )
    ax.set_title(
        f"{lm_name}: one bar per fitted model (patient | treatment | feature)\n"
        "bars sum to 100% of that model's variance",
        fontsize=10,
    )
    fig.tight_layout()
    pdfs.savefig(fig, pdf_path, bbox_inches="tight")
    plt.show()

    # 2 + 3. which variate dominates each top model, and which feature
    # families the top models come from (share of the top set vs. all models)
    dominant = (
        top_df.sort_values("term_pct_of_total_var", ascending=False)
        .drop_duplicates("model_id")
        .rename(columns={"term": "dominant_term"})
    )
    fig, axes = plt.subplots(1, 4, figsize=(28, 6))
    dominant["dominant_term"].value_counts().plot(kind="barh", ax=axes[0])
    axes[0].set_xlabel("n top models")
    for ax, col in zip(axes[1:], family_cols):
        share = pd.DataFrame(
            {
                f"top {TOP_N_SUMMARY}": dominant[col].value_counts(normalize=True),
                "all models": models_df[col].value_counts(normalize=True),
            }
        ).fillna(0)
        share.sort_values(f"top {TOP_N_SUMMARY}").plot(kind="barh", ax=ax)
        ax.set_xlabel("fraction of models")
        ax.legend(loc="upper left", bbox_to_anchor=(0, -0.15), fontsize=10)
        ax.set_title(col)
    axes[0].set_title("Dominant term")
    fig.suptitle(
        f"What drives the {TOP_N_SUMMARY} best-explained models?\n({lm_name})",
        y=1.03,
        va="bottom",
    )
    fig.tight_layout()
    pdfs.savefig(fig, pdf_path, bbox_inches="tight")
    plt.show()

    print(lm_name)
    print(
        dominant[["patient", "treatment", "feature", "dominant_term", "residual_pct"]]
        .head(10)
        .to_string(index=False)
    )


# In[8]:


pdfs.close()
