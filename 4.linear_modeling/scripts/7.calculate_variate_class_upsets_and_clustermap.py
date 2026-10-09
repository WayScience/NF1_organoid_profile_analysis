#!/usr/bin/env python
# coding: utf-8

# # Variate-class UpSets (sets = patients) and patient x treatment clustermaps
#
# **What it does.** Every (patient, treatment, feature) model is assigned to the variate groups it is a
# member of: `treatment` (T) and, by default, every technical covariate of the model: `cell_count` (N),
# `organoid_count` (O), `cell_per_organoid_count` (C), `manhattan_distance_from_center` (M),
# `cell_x_position` (X), `cell_y_position` (Y), `cell_z_position` (Z) and `cell_z_depth` (D). A feature is a member of a group when it is a hit for that model term
# (`pvalue_fdr < FDR` and effect `> EFFECT_MIN`). Each (patient, treatment, feature) therefore falls into
# exactly one intersection class (e.g. T, T+O, O+C); features in no group are dropped.
#
# * **Task A**: for every treatment x class, an UpSet plot whose sets are the patients (bars = number of
#   features with exactly that patient combination; dot matrix below), drawn in `8.plot_variate_class_upsets_and_clustermap` (R / ggplot2) from the saved combination table.
# * **Task B**: a clustermap of all patient x treatment rows by their per-class feature counts
#   (log1p, column min-max scaled, Ward), and a second version on per-row fractions.
# * **Task C**: which features are treatment-only (class `T`) and how often they recur across patients and treatments.
# * **Task D**: the treatment coefficients of the top shared treatment-only features across every (patient, treatment) model.
#
# **Input schema.** `results/linear_modeling/{sc,organoid}_norm_technical_model.parquet`: long format, one row per
# (patient, treatment, feature, term) with `coefficient` and `pvalue_fdr`. The covariates' SDs are
# recomputed from the raw profile (plus the well Manhattan distance table) (`data/profiles_3D/all_patients/0.normalized_profiles`).
#
# **Standardized count effects.** A count term's coefficient is a per-unit slope (cell_count: about 0.005),
# so it cannot be compared with the treatment coefficient (a DMSO-vs-treated contrast). Count coefficients
# are multiplied by the SD of that count within the patient's treatment + DMSO rows before the effect
# cut-off is applied, which puts every term on the same "feature SD per SD of the covariate" scale.
#
# **Outputs** (tables are parquet; every figure is drawn in `8.plot_variate_class_upsets_and_clustermap`):
# `results/variate_class_plots/<tag>/`
# (`upset_top_combinations.parquet`, `class_count_matrix.parquet`, the clustermap row / column orders and the Task C / D tables).
#
# Run as a script with `python scripts/7.calculate_variate_class_upsets_and_clustermap.py --help`.

# In[ ]:


import argparse
import gc
import pathlib
import warnings

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook
from scipy.cluster.hierarchy import leaves_list, linkage

warnings.filterwarnings("ignore")
root_dir, in_notebook = init_notebook()

if not in_notebook:
    import tqdm
else:
    import tqdm.notebook as tqdm

GROUP_SHORT = {
    "treatment": "T",
    "organoid_count": "O",
    "cell_per_organoid_count": "C",
    "cell_count": "N",
    "manhattan_distance_from_center": "M",
    "cell_x_position": "X",
    "cell_y_position": "Y",
    "cell_z_position": "Z",
    "cell_z_depth": "D",
}
COVARIATE_TERMS = [g for g in GROUP_SHORT if g != "treatment"]
# every term of the technical model; restrict with --groups for a smaller class space
DEFAULT_GROUPS = list(GROUP_SHORT)
# class order requested for the three-group (T, O, C) case
CLASS_ORDER_TOC = ["T", "T+O", "T+O+C", "O", "O+C", "C", "T+C"]

parser = argparse.ArgumentParser(
    description="Variate-class tables for UpSets and clustermaps."
)
parser.add_argument("--profile", choices=["sc", "organoid"], default="sc")
parser.add_argument(
    "--groups",
    nargs="+",
    default=DEFAULT_GROUPS,
    choices=list(GROUP_SHORT),
    help="variate groups (bit order = order given)",
)
parser.add_argument("--top-n", type=int, default=15, help="max combinations per UpSet")
parser.add_argument(
    "--min-subset-size", type=int, default=3, help="min features per shown combination"
)
parser.add_argument("--fdr-max", type=float, default=0.05)
parser.add_argument(
    "--effect-min",
    type=float,
    default=0.1,
    help="min standardized effect for membership",
)
parser.add_argument(
    "--top-features",
    type=int,
    default=40,
    help="features shown in the treatment-only plots",
)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args([] if in_notebook else None)
np.random.seed(args.seed)

tag = f"{args.profile}_" + "-".join(GROUP_SHORT[g] for g in args.groups)
lm_path = pathlib.Path(
    root_dir,
    f"4.linear_modeling/results/linear_modeling/{'sc' if args.profile == 'sc' else 'organoid'}_norm_technical_model.parquet",
).resolve(strict=True)
profile_dir = pathlib.Path(
    root_dir, "data/profiles_3D/all_patients/0.normalized_profiles"
)
manhattan_df = pd.read_parquet(
    pathlib.Path(
        root_dir,
        "4.linear_modeling/results/well_manhattan_distance/well_manhattan_distance.parquet",
    )
)
results_path = pathlib.Path(
    root_dir, f"4.linear_modeling/results/variate_class_plots/{tag}"
)
results_path.mkdir(parents=True, exist_ok=True)
print(f"run tag: {tag}")


# ## Membership table
# Standardize the count coefficients, call a term a member with `pvalue_fdr < FDR` and standardized effect
# `> EFFECT_MIN`, and pack the memberships into an integer code `sum(bit_i * in_group_i)`.

# In[2]:


def covariate_sds(profile):
    """SD of each covariate within every patient's (treatment + DMSO) rows, as used in the fits."""
    obj = "Cell" if profile == "sc" else "Organoid"
    keys = [
        "Metadata_Biology_PatientTumor",
        "Metadata_Experiment_Treatment",
        "Metadata_Experiment_Dose",
        "Metadata_Experiment_Unit",
        "Metadata_Experiment_Well",
    ]
    loc_cols = [
        f"Metadata_Location_{obj}_{c}"
        for c in ("CenterX", "CenterY", "CenterZ", "MinZ", "MaxZ")
    ]
    if profile == "sc":
        raw = pd.read_parquet(
            profile_dir / "sc_norm_norm_profile.parquet",
            columns=keys + loc_cols + ["Metadata_Object_WellSingleCellCount"],
        )
        org = pd.read_parquet(
            profile_dir / "organoid_norm_norm_profile.parquet",
            columns=[
                "Metadata_Biology_PatientTumor",
                "Metadata_Experiment_Well",
                "Metadata_WellOrganoidCount",
            ],
        ).drop_duplicates()
        raw = raw.merge(
            org,
            on=["Metadata_Biology_PatientTumor", "Metadata_Experiment_Well"],
            how="left",
        )
        raw = raw.dropna(subset=["Metadata_WellOrganoidCount"])
        raw["cell_count"] = raw["Metadata_Object_WellSingleCellCount"]
    else:
        raw = pd.read_parquet(
            profile_dir / "organoid_norm_norm_profile.parquet",
            columns=keys
            + loc_cols
            + ["Metadata_WellOrganoidCount", "Metadata_Object_OrganoidSingleCellCount"],
        )
        raw["cell_count"] = raw["Metadata_Object_OrganoidSingleCellCount"]
    raw = raw.merge(manhattan_df, on="Metadata_Experiment_Well", how="left")
    raw = raw.loc[raw["Metadata_Biology_PatientTumor"] != "NF0037_T1_CQ1"].copy()
    # a few rows (e.g. NF0018_T6) are missing the experimental metadata
    # (treatment, dose, tumor type) from an incomplete metadata join;
    # drop them so they do not form a spurious "nan" treatment group
    raw = raw.loc[raw["Metadata_Experiment_Treatment"].notna()].copy()
    raw["organoid_count"] = raw["Metadata_WellOrganoidCount"]
    raw["cell_per_organoid_count"] = raw["cell_count"] / raw["organoid_count"]
    raw["cell_x_position"] = raw[f"Metadata_Location_{obj}_CenterX"]
    raw["cell_y_position"] = raw[f"Metadata_Location_{obj}_CenterY"]
    raw["cell_z_position"] = raw[f"Metadata_Location_{obj}_CenterZ"]
    raw["cell_z_depth"] = (
        raw[f"Metadata_Location_{obj}_MaxZ"] - raw[f"Metadata_Location_{obj}_MinZ"]
    )
    raw[COVARIATE_TERMS] = raw[COVARIATE_TERMS].astype("float64")
    raw["patient"] = raw["Metadata_Biology_PatientTumor"]
    raw["treatment_full"] = (
        raw["Metadata_Experiment_Treatment"].astype(str)
        + "_"
        + raw["Metadata_Experiment_Dose"].astype(str)
        + raw["Metadata_Experiment_Unit"].astype(str)
    )
    dmso = raw.loc[
        raw["Metadata_Experiment_Treatment"] == "DMSO", "treatment_full"
    ].iloc[0]
    rows = []
    for patient, g in raw.groupby("patient"):
        g_dmso = g.loc[g["treatment_full"] == dmso]
        for combo in sorted(set(g["treatment_full"]) - {dmso}):
            sub = pd.concat([g_dmso, g.loc[g["treatment_full"] == combo]])
            rows.append(
                {
                    "patient": patient,
                    "treatment": combo,
                    **{f"sd_{c}": sub[c].std(ddof=1) for c in COVARIATE_TERMS},
                }
            )
    return pd.DataFrame(rows)


def class_order(groups):
    """Fixed class order: the requested one for T/O/C, otherwise grouped by the first member."""
    shorts = [GROUP_SHORT[g] for g in groups]
    if shorts == ["T", "O", "C"]:
        return CLASS_ORDER_TOC
    n = len(shorts)
    subsets = [[shorts[i] for i in range(n) if m >> i & 1] for m in range(1, 2**n)]
    first = lambda s: shorts.index(s[0])
    return [
        "+".join(s)
        for s in sorted(
            subsets, key=lambda s: (first(s), len(s), [shorts.index(x) for x in s])
        )
    ]


def build_membership(lm_path, groups, sds, fdr_max, effect_min):
    """Long table (patient, treatment, feature, code, class) of features that are in at least one group."""
    lm = pd.read_parquet(
        lm_path,
        columns=[
            "term",
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "feature",
            "coefficient",
            "pvalue_fdr",
        ],
    )
    lm = lm.rename(
        columns={
            "Metadata_Biology_PatientTumor": "patient",
            "Metadata_Experiment_Treatment": "treatment",
        }
    )
    lm["term"] = lm["term"].replace({"Metadata_Experiment_Treatment": "treatment"})
    lm = lm.loc[lm["term"].isin(groups)].merge(
        sds, on=["patient", "treatment"], how="left"
    )
    lm["effect"] = lm["coefficient"]
    for term in COVARIATE_TERMS:
        m = lm["term"] == term
        lm.loc[m, "effect"] = lm.loc[m, "coefficient"] * lm.loc[m, f"sd_{term}"]
    bits = {g: 1 << i for i, g in enumerate(groups)}
    lm["bit"] = lm["term"].map(bits) * (
        (lm["pvalue_fdr"] < fdr_max) & (lm["effect"] > effect_min)
    )
    code = (
        lm.groupby(["patient", "treatment", "feature"])["bit"]
        .sum()
        .rename("code")
        .reset_index()
    )
    code = code.loc[code["code"] > 0].copy()
    labels = {
        c: "+".join(GROUP_SHORT[g] for g in groups if c & bits[g])
        for c in code["code"].unique()
    }
    code["class"] = code["code"].map(labels)
    return code, lm


sds = covariate_sds(args.profile)
sds.to_parquet(results_path / "covariate_sds.parquet", index=False)
membership, lm_long = build_membership(
    lm_path, args.groups, sds, args.fdr_max, args.effect_min
)
membership.to_parquet(results_path / "membership.parquet", index=False)

classes = class_order(args.groups)
patients = sorted(membership["patient"].unique())
treatments = sorted(membership["treatment"].unique())
present = set(membership["class"].unique())
assert present <= set(classes), f"unexpected classes {present - set(classes)}"
print(
    f"{len(membership):,} memberships | {len(patients)} patients x {len(treatments)} treatments | classes present: {len(present)} of {len(classes)}"
)
print(
    "members per group:",
    {
        g: int(((membership["code"] & (1 << i)) > 0).sum())
        for i, g in enumerate(args.groups)
    },
)
membership["class"].value_counts().reindex(classes).fillna(0).astype(int).to_frame(
    "n_models"
)


# ## Task A: UpSet plots, sets = patients
# For one treatment and one class, every feature is assigned to the exact set of patients in which it has that
# class. Patient membership is packed into a bitmask per feature and combinations are counted with
# `np.unique`.

# In[ ]:


patient_bit = {p: 1 << i for i, p in enumerate(patients)}


def patient_combinations(membership):
    """Counts of features by exact patient combination, for every (treatment, class)."""
    m = membership.assign(bit=membership["patient"].map(patient_bit))
    masks = (
        m.groupby(["treatment", "class", "feature"])["bit"]
        .sum()
        .rename("mask")
        .reset_index()
    )
    out = {}
    for (trt, cls), g in masks.groupby(["treatment", "class"]):
        combo, n = np.unique(g["mask"].to_numpy(), return_counts=True)
        out[(trt, cls)] = pd.DataFrame({"mask": combo, "n_features": n}).sort_values(
            ["n_features", "mask"], ascending=[False, True], ignore_index=True
        )
    return out


# In[ ]:


SUMMARY_COLS = ["class", "treatment", "rank", "n_features", "n_patients", "patients"]
STATUS_COLS = ["class", "treatment", "status", "reason"]

summary_dir = results_path / "upset_intermediate" / "summary"
status_dir = results_path / "upset_intermediate" / "status"

combos_all = None

for cls in tqdm.tqdm(classes):
    (summary_dir / cls).mkdir(parents=True, exist_ok=True)
    (status_dir / cls).mkdir(parents=True, exist_ok=True)

    for trt in tqdm.tqdm(treatments, leave=False):
        summary_file = summary_dir / cls / f"{trt}.parquet"
        status_file = status_dir / cls / f"{trt}.parquet"

        if summary_file.exists() and status_file.exists():
            continue

        if combos_all is None:
            combos_all = patient_combinations(membership)

        combos = combos_all.pop(
            (trt, cls), None
        )  # pop: drop the reference from the dict once used
        shown = (
            None
            if combos is None
            else combos.loc[combos["n_features"] >= args.min_subset_size]
            .head(args.top_n)
            .copy()
        )
        del combos  # .copy() above means `shown` no longer references the full table

        summary_rows = []
        if shown is None or shown.empty:
            reason = (
                "no features in this class"
                if shown is None
                else f"no combination with >= {args.min_subset_size} features"
            )
            status_row = {
                "class": cls,
                "treatment": trt,
                "status": "skipped",
                "reason": reason,
            }
        else:
            status_row = {
                "class": cls,
                "treatment": trt,
                "status": "generated",
                "reason": "",
            }
            for rank, (_, r) in enumerate(shown.iterrows(), start=1):
                mask = int(r["mask"])
                summary_rows.append(
                    {
                        "class": cls,
                        "treatment": trt,
                        "rank": rank,
                        "n_features": int(r["n_features"]),
                        "n_patients": bin(mask).count("1"),
                        "patients": ";".join(
                            p for p in patients if mask & patient_bit[p]
                        ),
                    }
                )

        pd.DataFrame(summary_rows, columns=SUMMARY_COLS).to_parquet(
            summary_file, index=False
        )
        pd.DataFrame([status_row], columns=STATUS_COLS).to_parquet(
            status_file, index=False
        )

        del shown, summary_rows
        gc.collect()


# In[ ]:


def concat_dir(directory, columns):
    files = sorted(directory.rglob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    dfs = [d for d in dfs if not d.empty]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=columns)


concated_summary_path = results_path / "upset_top_combinations.parquet"
concated_status_path = results_path / "upset_panel_status.parquet"

if concated_summary_path.exists() and concated_status_path.exists():
    upset_summary = pd.read_parquet(concated_summary_path)
    upset_status = pd.read_parquet(concated_status_path)
else:
    upset_summary = concat_dir(summary_dir, SUMMARY_COLS)
    upset_status = concat_dir(status_dir, STATUS_COLS)

    upset_summary.to_parquet(concated_summary_path, index=False)
    upset_status.to_parquet(concated_status_path, index=False)

print(upset_status["status"].value_counts().to_string())
print(
    upset_status.loc[upset_status["status"] == "skipped"]
    .groupby("reason")
    .size()
    .to_string()
)
upset_status.groupby(["class", "status"]).size().unstack(fill_value=0).reindex(classes)


# ## Task B: clustermap of all patient x treatment class-count profiles
# One row per patient x treatment, one column per intersection class. `log1p`, then column-wise min-max scaling
# (`standard_scale=1`), Ward linkage on Euclidean distance. The second version uses per-row fractions.

# In[ ]:


def class_count_matrix(membership):
    counts = (
        membership.groupby(["patient", "treatment", "class"])
        .size()
        .unstack("class", fill_value=0)
    )
    all_rows = pd.MultiIndex.from_product(
        [patients, treatments], names=["patient", "treatment"]
    )
    observed = membership[["patient", "treatment"]].drop_duplicates()
    keep = all_rows.isin(pd.MultiIndex.from_frame(observed))
    return counts.reindex(all_rows[keep], fill_value=0).reindex(
        columns=[c for c in classes if c in counts.columns], fill_value=0
    )


def scaled_log_matrix(matrix, name):
    """log1p of the class counts / fractions, without the constant columns (they cannot be min-max scaled)."""
    mat = np.log1p(matrix)
    constant = mat.columns[mat.nunique() < 2]
    if len(constant):
        print(f"{name}: dropping constant class columns {list(constant)}")
        mat = mat.drop(columns=constant)
    return mat


def cluster_orders(matrix, name):
    """Ward / Euclidean dendrogram order of the rows (patient|treatment) and columns (classes) of the
    column min-max scaled log1p matrix, as used for the clustermap drawn in the plot notebook."""
    mat = scaled_log_matrix(matrix, name)
    scaled = ((mat - mat.min()) / (mat.max() - mat.min())).to_numpy()
    keys = [f"{p}|{t}" for p, t in mat.index]
    row_order = pd.DataFrame(
        {"row_order": [keys[i] for i in leaves_list(linkage(scaled, method="ward"))]}
    )
    col_order = pd.DataFrame(
        {
            "column_order": [
                mat.columns[i] for i in leaves_list(linkage(scaled.T, method="ward"))
            ]
        }
    )
    return row_order, col_order


counts_matrix = class_count_matrix(membership)
counts_matrix.reset_index().to_parquet(
    results_path / "class_count_matrix.parquet", index=False
)
print(f"count matrix: {counts_matrix.shape}")
row_order, col_order = cluster_orders(counts_matrix, "clustermap_counts")
row_order.to_parquet(results_path / "clustermap_counts_row_order.parquet", index=False)
col_order.to_parquet(
    results_path / "clustermap_counts_column_order.parquet", index=False
)

nonempty = counts_matrix.loc[counts_matrix.sum(axis=1) > 0]
fractions = nonempty.div(nonempty.sum(axis=1), axis=0)
fractions.reset_index().to_parquet(
    results_path / "class_fraction_matrix.parquet", index=False
)
print(
    f"fraction matrix: {fractions.shape} ({len(counts_matrix) - len(nonempty)} empty rows dropped)"
)
row_order_f, col_order_f = cluster_orders(fractions, "clustermap_fractions")
row_order_f.to_parquet(
    results_path / "clustermap_fractions_row_order.parquet", index=False
)
col_order_f.to_parquet(
    results_path / "clustermap_fractions_column_order.parquet", index=False
)


# ## Task C: which features are treatment-only (class `T`)?
# Every (patient, treatment, feature) in class `T` is a feature that is a hit for the treatment term and for none of the covariates in the run's variate groups. Feature names follow `compartment_channel_featuretype_measurement` (`AreaSizeShape` features have no channel). Outputs:
# * `treatment_only_hits.parquet`: one row per (patient, treatment, feature) with the parsed name parts.
# * `treatment_only_feature_recurrence.parquet`: one row per feature with the number of patients, treatments and patient x treatment pairs in which it is treatment-only.
# * `treatment_only_feature_by_treatment.parquet`: for the `--top-features` most recurrent features, the number of patients in which each is treatment-only, per treatment.
#
# `8.plot_variate_class_upsets_and_clustermap` plots the most recurrent features (the rows of `treatment_only_feature_by_treatment.parquet`), a channel x feature-type heatmap of distinct treatment-only features and the per-treatment heatmap.

# In[ ]:


def parse_feature(name):
    compartment, channel, feature_type, measurement = name.split("_", 3)
    if feature_type == "AreaSizeShape":
        channel, measurement = None, channel
    return compartment, channel, feature_type, measurement


PARTS = ["compartment", "channel", "feature_type", "measurement"]
treatment_only = membership.loc[membership["class"] == "T"].copy()
if treatment_only.empty:
    print("no treatment-only (class T) memberships; is 'treatment' in --groups?")
else:
    treatment_only[PARTS] = pd.DataFrame(
        [parse_feature(f) for f in treatment_only["feature"]],
        index=treatment_only.index,
    )
    treatment_only.to_parquet(results_path / "treatment_only_hits.parquet", index=False)
    treatment_only["channel"] = treatment_only["channel"].fillna("none")

    recurrence = (
        treatment_only.groupby(["feature"] + PARTS)
        .agg(
            n_patients=("patient", "nunique"),
            n_treatments=("treatment", "nunique"),
            n_patient_treatments=("patient", "size"),
        )
        .reset_index()
        .sort_values(
            ["n_patient_treatments", "n_patients", "feature"],
            ascending=[False, False, True],
            ignore_index=True,
        )
    )
    recurrence.to_parquet(
        results_path / "treatment_only_feature_recurrence.parquet", index=False
    )
    print(
        f"{len(recurrence):,} distinct treatment-only features from {len(treatment_only):,} (patient, treatment, feature) hits"
    )

    top = recurrence.head(args.top_features)
    per_treatment = (
        treatment_only.loc[treatment_only["feature"].isin(top["feature"])]
        .groupby(["feature", "treatment"])["patient"]
        .nunique()
        .unstack("treatment", fill_value=0)
        .reindex(index=top["feature"])
    )
    per_treatment.reset_index().to_parquet(
        results_path / "treatment_only_feature_by_treatment.parquet", index=False
    )
    recurrence.head(args.top_features)


# ## Task D: distribution of the top 10 shared treatment-only features
# Take the 10 most recurrent treatment-only features (Task C ranking) and look at their treatment coefficients over
# every (patient, treatment) model, hit or not. Outputs (tables only; plotted in `8.plot_variate_class_upsets_and_clustermap`):
# * `top_shared_features_coefficients.parquet`: one row per (patient, treatment, feature) with coefficient, FDR and whether it is a treatment-only hit.
# * `top_shared_features_space_distribution.parquet`: compartment / channel / feature-type composition of the top features vs all modelled features.
# * `top_shared_features_median_by_patient.parquet` / `top_shared_features_median_by_treatment.parquet`: median treatment coefficient per feature x patient / feature x treatment.
# * `top_shared_features_hit_count_by_patient.parquet` / `top_shared_features_hit_count_by_treatment.parquet`: number of treatment-only hits per feature x patient / feature x treatment.

# In[ ]:


N_SHARED = 10
if treatment_only.empty:
    print("no treatment-only hits; skipping Task D")
else:
    top_shared = recurrence.head(N_SHARED)["feature"].tolist()
    shared = lm_long.loc[
        (lm_long["term"] == "treatment") & lm_long["feature"].isin(top_shared),
        ["patient", "treatment", "feature", "coefficient", "pvalue_fdr"],
    ].copy()
    only_keys = treatment_only[["patient", "treatment", "feature"]].assign(
        treatment_only=True
    )
    shared = shared.merge(only_keys, on=["patient", "treatment", "feature"], how="left")
    shared["treatment_only"] = shared["treatment_only"].fillna(False).astype(bool)
    shared.to_parquet(
        results_path / "top_shared_features_coefficients.parquet", index=False
    )

    # where the top shared features sit in feature space vs all modelled features
    parts_all = pd.DataFrame(
        [parse_feature(f) for f in lm_long["feature"].unique()],
        columns=PARTS,
        index=lm_long["feature"].unique(),
    ).fillna({"channel": "none"})
    parts_top = parts_all.loc[top_shared]
    space = []
    for part in ["compartment", "channel", "feature_type"]:
        frac_all = parts_all[part].value_counts(normalize=True)
        frac_top = (
            parts_top[part]
            .value_counts(normalize=True)
            .reindex(frac_all.index)
            .fillna(0)
        )
        both = pd.DataFrame(
            {"all modelled features": frac_all, f"top {N_SHARED} shared": frac_top}
        )
        space.append(both.assign(part=part).rename_axis("value").reset_index())
    pd.concat(space, ignore_index=True).to_parquet(
        results_path / "top_shared_features_space_distribution.parquet", index=False
    )

    # median coefficient / treatment-only hit count tables over patients and over treatments
    for by in ("patient", "treatment"):
        med = (
            shared.groupby(["feature", by])["coefficient"]
            .median()
            .unstack(by)
            .reindex(top_shared)
        )
        n_hits = (
            shared.loc[shared["treatment_only"]]
            .groupby(["feature", by])
            .size()
            .unstack(by)
            .reindex(index=top_shared, columns=med.columns)
            .fillna(0)
            .astype(int)
        )
        med.reset_index().to_parquet(
            results_path / f"top_shared_features_median_by_{by}.parquet", index=False
        )
        n_hits.reset_index().to_parquet(
            results_path / f"top_shared_features_hit_count_by_{by}.parquet",
            index=False,
        )
    shared.groupby("feature")["coefficient"].describe().reindex(top_shared)
