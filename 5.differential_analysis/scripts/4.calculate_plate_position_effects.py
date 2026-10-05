#!/usr/bin/env python
# coding: utf-8

# # Calculate plate position effects
# Do well-level morphology profiles differ only because of where the well sits on the plate?
#
# **Plate layout.** Only rows B-G and columns 2-11 of the 96-well plate are used; the true plate edge (rows A/H, columns 1/12) is empty.
# "Outer" wells are the outermost ring of used wells (rows B and G, columns 2 and 11); all other wells are "inner".
#
# **1. Inner vs outer.** Every plate uses the same layout, so treatment and position are confounded (e.g. Staurosporine is only in the corners).
# Position is therefore only compared **within a plate x treatment group that has both inner and outer wells**.
# For each group, the difference is mean(outer) - mean(inner) per feature; the observed effect is the mean of those differences across groups.
# The null distribution comes from shuffling the inner/outer labels within each group.
# - Global statistic: root mean square (RMS) of the per-feature effects.
# - Per-feature: two-sided permutation p-value, Benjamini-Hochberg q-value.
#
# **2. Sanity check: DMSO column 4 vs column 9.** On every plate, DMSO occupies eight wells: C4-F4 (column 4) and D9-G9 (column 9). Other wells in those columns are not DMSO and are excluded.
# If position does not matter, DMSO wells in the same column should be no more similar than DMSO wells in different columns.
# Per plate: mean Pearson correlation of within-column pairs minus between-column pairs; the null shuffles the column labels (exact for each plate, sampled for the pooled statistic).
#
# Profiles are the normalized, feature-selected, well-aggregated profiles (normalized to each plate's DMSO).

# In[1]:


import itertools
import pathlib

import numpy as np
import pandas as pd
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


# In[2]:


profiles_dir = pathlib.Path(
    root_dir, "data/profiles_3D/all_patients/2.aggregated_profiles"
)
patient_ids_path = pathlib.Path(root_dir, "data/patient_IDs.txt")
results_dir = pathlib.Path(root_dir, "5.differential_analysis/results")
results_dir.mkdir(parents=True, exist_ok=True)

profile_dict = {
    "organoid": pathlib.Path(profiles_dir, "organoid_norm_sc_agg_profiles.parquet"),
    "single_cell": pathlib.Path(profiles_dir, "sc_norm_sc_agg_profiles.parquet"),
}
well_position_output_path = pathlib.Path(results_dir, "plate_position_wells.parquet")
position_global_output_path = pathlib.Path(
    results_dir, "plate_position_global_test.parquet"
)
position_null_output_path = pathlib.Path(results_dir, "plate_position_null.parquet")
position_feature_output_path = pathlib.Path(
    results_dir, "plate_position_feature_tests.parquet"
)
dmso_pair_output_path = pathlib.Path(
    results_dir, "dmso_column_pair_correlations.parquet"
)
dmso_test_output_path = pathlib.Path(results_dir, "dmso_column_tests.parquet")

CONTROL = "DMSO"
OUTER_ROWS = ["B", "G"]
OUTER_COLUMNS = [2, 11]
DMSO_WELLS = ["C4", "D4", "E4", "F4", "D9", "E9", "F9", "G9"]
DMSO_COLUMNS = sorted({int(well[1:]) for well in DMSO_WELLS})
N_PERMUTATIONS = 10_000
SEED = 0
group_cols = [
    "Metadata_Biology_PatientTumor",
    "Metadata_Experiment_Treatment",
    "Metadata_Experiment_Dose",
]

rng = np.random.default_rng(SEED)
patient_ids = patient_ids_path.read_text().split()


# In[3]:


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty_like(adjusted)
    q_values[order] = np.minimum(adjusted, 1)
    return q_values


def load_wells(profile_path: pathlib.Path) -> tuple[pd.DataFrame, list]:
    """Load well profiles for the analysis patients and annotate plate position."""
    df = pd.read_parquet(profile_path)
    df = df.loc[df["Metadata_Biology_PatientTumor"].isin(patient_ids)].dropna(
        subset=["Metadata_Experiment_Well", "Metadata_Experiment_Treatment"]
    )
    df = df.reset_index(drop=True)
    df["Metadata_Plate_Row"] = df["Metadata_Experiment_Well"].str[0]
    df["Metadata_Plate_Column"] = df["Metadata_Experiment_Well"].str[1:].astype(int)
    df["Metadata_Plate_Position"] = np.where(
        df["Metadata_Plate_Row"].isin(OUTER_ROWS)
        | df["Metadata_Plate_Column"].isin(OUTER_COLUMNS),
        "outer",
        "inner",
    )
    feature_cols = [col for col in df.columns if not col.startswith("Metadata_")]
    return df, feature_cols


def mixed_position_groups(df: pd.DataFrame, feature_cols: list) -> list:
    """Profiles and outer-minus-inner contrast weights for groups with both positions."""
    groups = []
    for key, group_df in df.groupby(group_cols):
        is_outer = (group_df["Metadata_Plate_Position"] == "outer").to_numpy()
        if is_outer.all() or not is_outer.any():
            continue
        weights = np.where(is_outer, 1 / is_outer.sum(), -1 / (~is_outer).sum())
        groups.append(
            {
                "key": key,
                "profiles": group_df[feature_cols].to_numpy(),
                "weights": weights,
            }
        )
    return groups


def mean_contrast(groups: list, permute: bool = False) -> np.ndarray:
    """Mean over groups of mean(outer) - mean(inner), per feature."""
    total = np.zeros(groups[0]["profiles"].shape[1])
    for group in groups:
        weights = rng.permutation(group["weights"]) if permute else group["weights"]
        total += weights @ group["profiles"]
    return total / len(groups)


def within_minus_between(correlations: np.ndarray, labels: np.ndarray) -> float:
    """Mean correlation of same-label pairs minus mean of different-label pairs."""
    upper = np.triu_indices(len(labels), k=1)
    same = (labels[:, None] == labels[None, :])[upper]
    values = correlations[upper]
    return values[same].mean() - values[~same].mean()


# ## Load wells

# In[4]:


well_dict = {}
for compartment, profile_path in profile_dict.items():
    well_dict[compartment] = load_wells(profile_path)
    df, feature_cols = well_dict[compartment]
    print(
        f"{compartment}: {len(df)} wells, {len(feature_cols)} features, "
        f"{(df['Metadata_Plate_Position'] == 'outer').sum()} outer wells"
    )

well_position_df = (
    well_dict["single_cell"][0][
        ["Metadata_Plate_Row", "Metadata_Plate_Column", "Metadata_Plate_Position"]
    ]
    .drop_duplicates()
    .sort_values(["Metadata_Plate_Row", "Metadata_Plate_Column"])
    .reset_index(drop=True)
)
well_position_df.to_parquet(well_position_output_path, index=False)


# ## 1. Inner vs outer wells (within plate x treatment)

# In[5]:


global_rows, null_list, feature_list = [], [], []
for compartment, (df, feature_cols) in well_dict.items():
    groups = mixed_position_groups(df, feature_cols)
    observed = mean_contrast(groups)
    null = np.vstack(
        [
            mean_contrast(groups, permute=True)
            for _ in tqdm(range(N_PERMUTATIONS), desc=compartment)
        ]
    )
    observed_rms = np.sqrt(np.mean(observed**2))
    null_rms = np.sqrt(np.mean(null**2, axis=1))
    global_rows.append(
        {
            "compartment": compartment,
            "n_groups": len(groups),
            "n_plates": len({group["key"][0] for group in groups}),
            "n_features": len(feature_cols),
            "observed_rms_effect": observed_rms,
            "null_median_rms_effect": np.median(null_rms),
            "p": (1 + (null_rms >= observed_rms).sum()) / (1 + N_PERMUTATIONS),
        }
    )
    null_list.append(
        pd.DataFrame({"compartment": compartment, "null_rms_effect": null_rms})
    )
    feature_p = (1 + (np.abs(null) >= np.abs(observed)).sum(axis=0)) / (
        1 + N_PERMUTATIONS
    )
    feature_list.append(
        pd.DataFrame(
            {
                "compartment": compartment,
                "feature": feature_cols,
                "outer_minus_inner": observed,
                "p": feature_p,
                "q": bh_fdr(feature_p),
            }
        )
    )

position_global_df = pd.DataFrame(global_rows)
position_null_df = pd.concat(null_list, ignore_index=True)
position_feature_df = pd.concat(feature_list, ignore_index=True)
position_global_df.to_parquet(position_global_output_path, index=False)
position_null_df.to_parquet(position_null_output_path, index=False)
position_feature_df.to_parquet(position_feature_output_path, index=False)
print(position_global_df.round(4).to_string(index=False))
print(
    position_feature_df.assign(significant=lambda x: x["q"] < 0.05)
    .groupby("compartment")["significant"]
    .agg(["sum", "size"])
    .rename(columns={"sum": "features_q_below_0.05", "size": "n_features"})
    .to_string()
)


# ## 2. Sanity check: DMSO column 4 vs column 9

# In[6]:


pair_list, dmso_test_rows = [], []
for compartment, (df, feature_cols) in well_dict.items():
    plate_nulls, plate_observed = [], []
    for patient, plate_df in df.loc[
        df["Metadata_Experiment_Treatment"] == CONTROL
    ].groupby("Metadata_Biology_PatientTumor"):
        plate_df = plate_df.loc[plate_df["Metadata_Experiment_Well"].isin(DMSO_WELLS)]
        plate_df = plate_df.reset_index(drop=True)
        labels = plate_df["Metadata_Plate_Column"].to_numpy()
        correlations = np.corrcoef(plate_df[feature_cols].to_numpy())

        wells = plate_df["Metadata_Experiment_Well"].to_numpy()
        for i, j in itertools.combinations(range(len(wells)), 2):
            pair_list.append(
                {
                    "compartment": compartment,
                    "Metadata_Biology_PatientTumor": patient,
                    "well_a": wells[i],
                    "well_b": wells[j],
                    "pair_type": (
                        f"within column {labels[i]}"
                        if labels[i] == labels[j]
                        else "between columns"
                    ),
                    "correlation": correlations[i, j],
                }
            )

        # Exact null: every split of the wells into groups of the observed sizes.
        n_first = (labels == DMSO_COLUMNS[0]).sum()
        null = []
        for first_idx in itertools.combinations(range(len(labels)), n_first):
            permuted = np.full(len(labels), DMSO_COLUMNS[1])
            permuted[list(first_idx)] = DMSO_COLUMNS[0]
            null.append(within_minus_between(correlations, permuted))
        null = np.array(null)
        observed = within_minus_between(correlations, labels)
        plate_nulls.append(null)
        plate_observed.append(observed)
        dmso_test_rows.append(
            {
                "compartment": compartment,
                "Metadata_Biology_PatientTumor": patient,
                "n_wells": len(labels),
                "within_minus_between": observed,
                "p": (null >= observed - 1e-12).mean(),
                "n_permutations": len(null),
            }
        )

    # Pooled over plates: mean of the per-plate statistics, null sampled per plate.
    pooled_observed = np.mean(plate_observed)
    pooled_null = np.mean(
        [rng.choice(null, N_PERMUTATIONS) for null in plate_nulls], axis=0
    )
    dmso_test_rows.append(
        {
            "compartment": compartment,
            "Metadata_Biology_PatientTumor": "all plates",
            "n_wells": int(
                sum(
                    row["n_wells"]
                    for row in dmso_test_rows
                    if row["compartment"] == compartment
                )
            ),
            "within_minus_between": pooled_observed,
            "p": (1 + (pooled_null >= pooled_observed).sum()) / (1 + N_PERMUTATIONS),
            "n_permutations": N_PERMUTATIONS,
        }
    )

dmso_pair_df = pd.DataFrame(pair_list)
dmso_test_df = pd.DataFrame(dmso_test_rows)
dmso_pair_df.to_parquet(dmso_pair_output_path, index=False)
dmso_test_df.to_parquet(dmso_test_output_path, index=False)
print(dmso_test_df.round(3).to_string(index=False))
