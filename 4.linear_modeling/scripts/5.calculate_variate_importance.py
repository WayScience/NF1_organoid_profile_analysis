#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from notebook_init_utils import init_notebook

warnings.filterwarnings("ignore")
root_dir, in_notebook = init_notebook()

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

results_path = pathlib.Path(root_dir, "4.linear_modeling/results/variate_importance")
results_path.mkdir(parents=True, exist_ok=True)

pd.set_option("display.max_columns", 50)


# In[ ]:


df = pd.read_parquet(
    pathlib.Path(
        root_dir, "4.linear_modeling/results/linear_modeling/sc_norm.parquet"
    ).resolve(strict=True)
)
df["model"] = df["patient"] + "_" + df["treatment"] + "_" + df["feature"]
df["hit"] = (df["pvalue_fdr"] < 0.05) & (df["coefficient"] > 0.1)


# In[3]:


# calculate the the number of hits for each term
term_hit_counts = df.groupby("term")["hit"].sum().reset_index()
term_hit_counts = term_hit_counts.rename(columns={"hit": "num_hits"})
term_hit_counts = term_hit_counts.sort_values(by="num_hits", ascending=False)

# calculate the total number of hits by feature (model)
feature_hit_counts = df.groupby("feature")["hit"].sum().reset_index()
feature_hit_counts = feature_hit_counts.rename(columns={"hit": "num_hits"})
feature_hit_counts = feature_hit_counts.sort_values(by="num_hits", ascending=False)
feature_hit_counts

# calculate the total number of hits for each feature and term combination
feature_term_hit_counts = df.groupby(["feature", "term"])["hit"].sum().reset_index()
feature_term_hit_counts = feature_term_hit_counts.rename(columns={"hit": "num_hits"})
feature_term_hit_counts = feature_term_hit_counts.sort_values(
    by="num_hits", ascending=False
)
feature_term_hit_counts


# # Variate importance
#
# Counts the hits of every model term in `sc_norm.parquet` (`pvalue_fdr < 0.05` and `coefficient > 0.1`) and builds the Venn / UpSet tables plotted in `6.plot_variate_importance`.
#
# ### Venn and UpSet tables: which features belong to which variate group?
#
# A feature belongs to a variate group (a model term) when it is a hit for that term (`pvalue_fdr < 0.05` and `coefficient > 0.1`, so increases only). The residual (epsilon) is not shown because every feature has one. Membership is not exclusive; overlaps are features that respond to more than one variate.
#
# The tables are computed for two model sets:
# * **original** (`sc_norm.parquet`): `treatment`, `cell_count`, `organoid_count`, `cell_per_organoid_count`.
# * **technical** (`sc_norm_technical_model.parquet`): the four terms above plus the position / depth covariates.
#
# The Venn tables use the four core terms for both model sets (a Venn is only legible up to four sets); the UpSet tables use every term of the model set. Note that the covariate coefficients are per raw unit, so the fixed `coefficient > 0.1` cut-off is not scale-free for them.
#
# Each model set is tabulated at six scopes, and membership is recomputed within each:
# * `all_models`: every model pooled.
# * `per_patient_treatment`: one Venn + UpSet per patient x treatment.
# * `per_patient`: one per patient, all treatments pooled.
# * `per_treatment`: one per treatment, all patients pooled.
# * `per_treatment_tumor_type`: one per treatment within each tumor type (`cNF` / `pNF` / `MPNST` / `Other`, looked up by patient id -- the same classification as `utils/r_plot_themes.r`'s `tumor_type_lookup`).
# * `per_tumor_type`: one per tumor type (`cNF` / `pNF` / `MPNST` / `Other`), all patients and treatments pooled.
#
# The tables are saved to `variate_hit_venn_regions_all_scopes.parquet`, `variate_hit_upset_counts_all_scopes.parquet` and `variate_hit_set_sizes_all_scopes.parquet`, and plotted in `6.plot_variate_importance` (R / ggplot2).

# In[ ]:


technical_path = pathlib.Path(
    root_dir,
    "4.linear_modeling/results/linear_modeling/sc_norm_technical_model.parquet",
)

# a feature belongs to a variate group (model term) when it is a hit for that term
CORE_TERMS = ["treatment", "cell_count", "organoid_count", "cell_per_organoid_count"]
TECHNICAL_TERMS = CORE_TERMS + [
    "manhattan_distance_from_center",
    "cell_x_position",
    "cell_y_position",
    "cell_z_position",
    "cell_z_depth",
]


def load_technical_hits(path):
    """Technical-model results with the same column names / term labels as the original model."""
    out = pd.read_parquet(path.resolve(strict=True)).rename(
        columns={
            "Metadata_Biology_PatientTumor": "patient",
            "Metadata_Experiment_Treatment": "treatment",
        }
    )
    out["term"] = out["term"].replace({"Metadata_Experiment_Treatment": "treatment"})
    out["hit"] = (out["pvalue_fdr"] < 0.05) & (out["coefficient"] > 0.1)
    return out


def slim_hits(hits):
    """Only the columns the plots need, plus tumor_type (cNF / pNF / MPNST / Other) looked up by patient id."""
    out = hits[["term", "patient", "treatment", "feature", "hit"]].copy()
    return out


def build_membership(hits, terms):
    """Boolean feature x term table: True when the feature is a hit for that term in any model."""
    return (
        hits.loc[hits["term"].isin(terms)]
        .assign(hit=lambda d: d["hit"].astype(bool))
        .pivot_table(index="feature", columns="term", values="hit", aggfunc="max")
        .reindex(columns=terms)
        .fillna(False)
        .astype(bool)
    )


def venn_regions(membership, terms):
    """Feature count per Venn region; `region_key` has one bit per term (in term order)."""
    key = membership[terms].apply(
        lambda r: "".join("1" if v else "0" for v in r), axis=1
    )
    out = key.value_counts().rename_axis("region_key").reset_index(name="n_features")
    out = out.loc[out["region_key"] != "0" * len(terms)].reset_index(drop=True)
    out["variates"] = out["region_key"].map(
        lambda k: " + ".join(t for t, b in zip(terms, k) if b == "1")
    )
    return out


def upset_combinations(membership, terms):
    """Features in exactly each term combination (largest first), with one boolean `in_<term>` flag per term."""
    combos = (
        membership.groupby(terms)
        .size()
        .rename("n_features")
        .reset_index()
        .loc[lambda d: d[terms].any(axis=1)]
        .sort_values("n_features", ascending=False)
        .reset_index(drop=True)
    )
    if combos.empty:
        return combos.assign(combination=[], treatment_specific=[])
    combos["combination"] = combos[terms].apply(
        lambda r: " + ".join(t for t in terms if r[t]), axis=1
    )
    combos["treatment_specific"] = combos["treatment"] & ~combos[terms[1:]].any(axis=1)
    # term flags are boolean; prefix them so the `treatment` term does not collide
    # with the `treatment` scope identifier column
    return combos.rename(columns={t: f"in_{t}" for t in terms})


def set_sizes(membership, terms):
    """Features that are a hit for each term, plus the features that are a hit for none of the terms."""
    return pd.DataFrame(
        {
            "term": terms,
            "set_size": membership[terms].sum().to_numpy(),
            "n_total": len(membership),
            "n_none": int((~membership[terms].any(axis=1)).sum()),
        }
    )


# In[ ]:


# every scope: the columns that define one Venn + UpSet pair. tumor_type = TUMOR_TYPE_DICT lookup by patient id
SCOPES = {
    "all_models": [],
    "per_patient_treatment": ["patient", "treatment"],
    "per_patient": ["patient"],
    "per_treatment": ["treatment"],
    "per_treatment_tumor_type": ["treatment", "tumor_type"],
    "per_tumor_type": ["tumor_type"],
}
venn_parquet = results_path / "variate_hit_venn_regions_all_scopes.parquet"
upset_parquet = results_path / "variate_hit_upset_counts_all_scopes.parquet"
set_size_parquet = results_path / "variate_hit_set_sizes_all_scopes.parquet"


def build_model_sets():
    """venn terms, upset terms, hits per model set (loaded only when the tables must be computed)."""
    return {
        "original": (CORE_TERMS, CORE_TERMS, slim_hits(df)),
        "technical": (
            CORE_TERMS,
            TECHNICAL_TERMS,
            slim_hits(load_technical_hits(technical_path)),
        ),
    }


def scope_frames(hits, keys):
    """Yield (key values, sub-frame) for one scope; a single all-data frame when there are no keys."""
    if not keys:
        yield (), hits
    else:
        yield from hits.groupby(keys, sort=True, observed=True)


# the tables are computed once; delete a parquet to recompute them
if all(p.exists() for p in (venn_parquet, upset_parquet, set_size_parquet)):
    print("variate hit tables already present; nothing to recompute")
else:
    venn_tables, upset_tables, size_tables = [], [], []
    for model, (venn_terms, upset_terms, hits) in build_model_sets().items():
        for scope, keys in SCOPES.items():
            groups = list(scope_frames(hits, keys))
            for values, sub in tqdm(groups, desc=f"{model} {scope}", leave=False):
                values = (values,) if isinstance(values, str) else tuple(values)
                ids = {"model_set": model, "scope": scope, **dict(zip(keys, values))}
                venn_membership = build_membership(sub, venn_terms)
                upset_membership = build_membership(sub, upset_terms)
                venn_tables.append(
                    venn_regions(venn_membership, venn_terms).assign(**ids)
                )
                size_tables.append(
                    set_sizes(venn_membership, venn_terms).assign(plot="venn", **ids)
                )
                size_tables.append(
                    set_sizes(upset_membership, upset_terms).assign(plot="upset", **ids)
                )
                combos = upset_combinations(upset_membership, upset_terms)
                if not combos.empty:
                    upset_tables.append(combos.assign(**ids))

    pd.concat(venn_tables, ignore_index=True).to_parquet(venn_parquet, index=False)
    pd.concat(upset_tables, ignore_index=True).to_parquet(upset_parquet, index=False)
    pd.concat(size_tables, ignore_index=True).to_parquet(set_size_parquet, index=False)

venn_all = pd.read_parquet(venn_parquet)
upset_all = pd.read_parquet(upset_parquet)
print(f"{len(venn_all):,} Venn region rows, {len(upset_all):,} UpSet combination rows")
upset_all.loc[
    (upset_all["model_set"] == "technical") & (upset_all["scope"] == "all_models"),
    ["combination", "n_features"],
].head(25)
