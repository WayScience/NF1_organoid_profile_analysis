#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import skimage.io
from microfilm.microplot import Micropanel, microshow
from montage_utils.montage_utils import (
    create_and_save_two_image_panel,
    filter_out_diagonal_correlations,
    filter_out_self_correlations,
    generate_image_paths_from_combination_string,
    make_multi_channel_image_array,
    retrieve_quadrant_info,
)
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


viability_df = pd.read_parquet(
    pathlib.Path(f"{root_dir}/data/viabilities/combined_platemaps.parquet")
)
pairs_df = pd.read_parquet(
    pathlib.Path(
        f"{root_dir}/1.EDA/results/correlation/3D_sc_correlation_pairs.parquet"
    )
)
samples_df = pd.read_parquet(
    pathlib.Path(
        f"{root_dir}/1.EDA/results/correlation/3D_sc_correlation_samples.parquet"
    )
)

# sc_norm/agg matches this notebook's previous default input
# (3D_2.aggregated_profiles_sc_norm_sc_agg_profiles.parquet)
normalization_variant = "sc_norm"
profile_type = "agg"
pairs_df = pairs_df.loc[
    (pairs_df["normalization_variant"] == normalization_variant)
    & (pairs_df["profile_type"] == profile_type)
].reset_index(drop=True)
samples_df = samples_df.loc[
    (samples_df["normalization_variant"] == normalization_variant)
    & (samples_df["profile_type"] == profile_type)
].reset_index(drop=True)

figures_base_dir = pathlib.Path(f"{root_dir}/1.EDA/figures/").resolve()
montage_figure_base_dir = pathlib.Path(f"{figures_base_dir}/montages/").resolve()
image_base_dir = bandicoot_check(
    pathlib.Path(
        f"{os.path.expanduser('~')}/mnt/bandicoot/NF1_organoid_data/data/"
    ).resolve(),
    root_dir,
)
image_base_dir = image_base_dir / "data/"
montage_figure_base_dir.mkdir(parents=True, exist_ok=True)


# In[3]:


# pairs_df is already long-format (one row per sample pair), unlike the old
# wide correlation-matrix format this notebook used to read -- no melt
# needed, just rename to the group1/group2 convention montage_utils expects
df_long = pairs_df.rename(columns={"sample_i": "group1", "sample_j": "group2"})


# In[4]:


metadata_cols = [c for c in samples_df.columns if c.startswith("Metadata_")]

# per-sample metadata table, one row per sample, keyed by sample_index
sample_meta = samples_df[["sample_index"] + metadata_cols].copy()

# merge for group1
df_long_with_meta = df_long.merge(
    sample_meta.add_prefix("group1_").rename(
        columns={"group1_sample_index": "sample_index"}
    ),
    left_on="group1",
    right_on="sample_index",
    how="left",
).drop(columns=["sample_index"])

# merge for group2
df_long_with_meta = df_long_with_meta.merge(
    sample_meta.add_prefix("group2_").rename(
        columns={"group2_sample_index": "sample_index"}
    ),
    left_on="group2",
    right_on="sample_index",
    how="left",
).drop(columns=["sample_index"])


# In[5]:


# aggregate viability to one row per patient/treatment/dose
viability_summary = viability_df.groupby(
    ["Metadata_Biology_PatientTumor", "Treatment", "Dose"], as_index=False
).agg(
    {
        "Metadata_Viability_percentage": "mean",
        "min_max_viability": "mean",
    }
)

# sanity check: this should now be duplicate-free
dupe_check = viability_summary.groupby(
    ["Metadata_Biology_PatientTumor", "Treatment", "Dose"]
).size()
assert dupe_check.max() == 1, "still duplicates after aggregation"

# merge in group1 viability scores
df_long_with_meta_and_viability = df_long_with_meta.merge(
    viability_summary.rename(
        columns={
            "Metadata_Biology_PatientTumor": "group1_patient_id",
            "Treatment": "group1_Treatment",
            "Dose": "group1_Dose",
            "Metadata_Viability_percentage": "group1_viability",
            "min_max_viability": "group1_min_max_viability",
        }
    ),
    left_on=[
        "group1_Metadata_Biology_PatientTumor",
        "group1_Metadata_Experiment_Treatment",
        "group1_Metadata_Experiment_Dose",
    ],
    right_on=["group1_patient_id", "group1_Treatment", "group1_Dose"],
    how="left",
)

# merge in group2 viability scores
df_long_with_meta_and_viability = df_long_with_meta_and_viability.merge(
    viability_summary.rename(
        columns={
            "Metadata_Biology_PatientTumor": "group2_patient_id",
            "Treatment": "group2_Treatment",
            "Dose": "group2_Dose",
            "Metadata_Viability_percentage": "group2_viability",
            "min_max_viability": "group2_min_max_viability",
        }
    ),
    left_on=[
        "group2_Metadata_Biology_PatientTumor",
        "group2_Metadata_Experiment_Treatment",
        "group2_Metadata_Experiment_Dose",
    ],
    right_on=["group2_patient_id", "group2_Treatment", "group2_Dose"],
    how="left",
)


# In[6]:


df_long_with_meta_and_viability.dropna(inplace=True)
df_long_with_meta_and_viability["group1_group2_viability_diff"] = abs(
    df_long_with_meta_and_viability["group1_min_max_viability"]
    - df_long_with_meta_and_viability["group2_min_max_viability"]
)


# In[7]:


df_long_with_meta_and_viability = filter_out_diagonal_correlations(
    df_long_with_meta_and_viability
)


# In[8]:


high_correlation_cutoff = 0.8
low_correlation_cutoff = -0.3
similar_viability_difference_cutoff = 0.1
dissimilar_viability_difference_cutoff = 0.9


# In[9]:


# --- assign each pair to a quadrant based on correlation and viability difference ---
# using x=0 and y=0 as the dividing lines (adjust if you want to divide on the
# midpoint of your cutoffs instead, e.g. x_divider = (high_correlation_cutoff + low_correlation_cutoff) / 2)

# quick function to use in the apply method to assign quadrants


def assign_quadrant(row):
    corr = row["correlation"]
    diff = row["group1_group2_viability_diff"]

    if corr > high_correlation_cutoff:
        x_side = "right"
    elif corr < low_correlation_cutoff:
        x_side = "left"
    else:
        x_side = "middle"

    if diff > dissimilar_viability_difference_cutoff:
        y_side = "top"
    elif diff < similar_viability_difference_cutoff:
        y_side = "bottom"
    else:
        y_side = "middle"

    # return f"{y_side}_{x_side}"
    return (
        "middle" if (x_side == "middle" or y_side == "middle") else f"{y_side}_{x_side}"
    )


df_long_with_meta_and_viability["quadrant"] = df_long_with_meta_and_viability.apply(
    assign_quadrant, axis=1
)
df_long_with_meta_and_viability = filter_out_self_correlations(
    df_long_with_meta_and_viability
)


# In[10]:


# save the long df
df_long_with_meta_and_viability.to_parquet(
    pathlib.Path(
        f"{root_dir}/1.EDA/results/correlation/3D_sc_correlation_pairs_{normalization_variant}_{profile_type}_with_meta_and_viability.parquet"
    )
)


# In[11]:


threshold_definitions_dict = {  # bottom right
    "high_correlation_similar_viability": {
        "correlation_threshold": high_correlation_cutoff,
        "viability_diff_threshold": similar_viability_difference_cutoff,
    },
    "high_correlation_dissimilar_viability": {  # upper right
        "correlation_threshold": high_correlation_cutoff,
        "viability_diff_threshold": dissimilar_viability_difference_cutoff,
    },
    "low_correlation_similar_viability": {  # bottom left
        "correlation_threshold": low_correlation_cutoff,
        "viability_diff_threshold": similar_viability_difference_cutoff,
    },
    "low_correlation_dissimilar_viability": {  # top left
        "correlation_threshold": low_correlation_cutoff,
        "viability_diff_threshold": dissimilar_viability_difference_cutoff,
    },
}


# In[12]:


high_correlation_similar_viability_df = filter_out_self_correlations(  # bottom right
    df_long_with_meta_and_viability.loc[
        (
            df_long_with_meta_and_viability["correlation"]
            >= threshold_definitions_dict["high_correlation_similar_viability"][
                "correlation_threshold"
            ]
        )
        & (
            df_long_with_meta_and_viability["group1_group2_viability_diff"]
            <= threshold_definitions_dict["high_correlation_similar_viability"][
                "viability_diff_threshold"
            ]
        )
    ]
)
high_correlation_dissimilar_viability_df = filter_out_self_correlations(  # upper right
    df_long_with_meta_and_viability.loc[
        (
            df_long_with_meta_and_viability["correlation"]
            >= threshold_definitions_dict["high_correlation_dissimilar_viability"][
                "correlation_threshold"
            ]
        )
        & (
            df_long_with_meta_and_viability["group1_group2_viability_diff"]
            >= threshold_definitions_dict["high_correlation_dissimilar_viability"][
                "viability_diff_threshold"
            ]
        )
    ]
)
low_correlation_similar_viability_df = filter_out_self_correlations(  # bottom left
    df_long_with_meta_and_viability.loc[
        (
            df_long_with_meta_and_viability["correlation"]
            <= threshold_definitions_dict["low_correlation_similar_viability"][
                "correlation_threshold"
            ]
        )
        & (
            df_long_with_meta_and_viability["group1_group2_viability_diff"]
            <= threshold_definitions_dict["low_correlation_similar_viability"][
                "viability_diff_threshold"
            ]
        )
    ]
)
low_correlation_dissimilar_viability_df = filter_out_self_correlations(  # upper left
    df_long_with_meta_and_viability.loc[
        (
            df_long_with_meta_and_viability["correlation"]
            <= threshold_definitions_dict["low_correlation_dissimilar_viability"][
                "correlation_threshold"
            ]
        )
        & (
            df_long_with_meta_and_viability["group1_group2_viability_diff"]
            >= threshold_definitions_dict["low_correlation_dissimilar_viability"][
                "viability_diff_threshold"
            ]
        )
    ]
)
print(
    f"High Correlation & Similar Viability: {len(high_correlation_similar_viability_df)} pairs"
)
print(
    f"High Correlation & Dissimilar Viability: {len(high_correlation_dissimilar_viability_df)} pairs"
)
print(
    f"Low Correlation & Similar Viability: {len(low_correlation_similar_viability_df)} pairs"
)
print(
    f"Low Correlation & Dissimilar Viability: {len(low_correlation_dissimilar_viability_df)} pairs"
)


# In[13]:


high_correlation_dissimilar_viability_list = retrieve_quadrant_info(
    high_correlation_dissimilar_viability_df
)
high_correlation_similar_viability_list = retrieve_quadrant_info(
    high_correlation_similar_viability_df
)
low_correlation_dissimilar_viability_list = retrieve_quadrant_info(
    low_correlation_dissimilar_viability_df
)
low_correlation_similar_viability_list = retrieve_quadrant_info(
    low_correlation_similar_viability_df
)

# retrieve_quadrant_info adds a "combinations" column to each df (matching
# the combination_string built from the same row) -- use it to look up each
# pair's correlation/viability_diff value for the individual montage
# filenames below.
combination_to_values = {}
for _df in (
    high_correlation_dissimilar_viability_df,
    high_correlation_similar_viability_df,
    low_correlation_dissimilar_viability_df,
    low_correlation_similar_viability_df,
):
    combination_to_values.update(
        dict(
            zip(
                _df["combinations"],
                zip(_df["correlation"], _df["group1_group2_viability_diff"]),
            )
        )
    )


# In[14]:


number_of_choices = 5  # number of random choices to make from each quadrant list
random.seed(42)  # for reproducibility
dict_of_randomly_selected_images = {
    "high_correlation_dissimilar_viability": random.sample(
        high_correlation_dissimilar_viability_list,
        min(number_of_choices, len(high_correlation_dissimilar_viability_list)),
    ),
    "high_correlation_similar_viability": random.sample(
        high_correlation_similar_viability_list,
        min(number_of_choices, len(high_correlation_similar_viability_list)),
    ),
    "low_correlation_dissimilar_viability": random.sample(
        low_correlation_dissimilar_viability_list,
        min(number_of_choices, len(low_correlation_dissimilar_viability_list)),
    ),
    "low_correlation_similar_viability": random.sample(
        low_correlation_similar_viability_list,
        min(number_of_choices, len(low_correlation_similar_viability_list)),
    ),
}


# In[15]:


for quadrent_of_correlation_and_viability in tqdm.tqdm(
    dict_of_randomly_selected_images.keys(),
    desc="Processing correlation pairs",
    leave=True,
    total=len(dict_of_randomly_selected_images.keys()),
):
    for combination_string in tqdm.tqdm(
        dict_of_randomly_selected_images[quadrent_of_correlation_and_viability],
        desc=f"Processing {quadrent_of_correlation_and_viability} pairs",
        leave=False,
        total=len(
            dict_of_randomly_selected_images[quadrent_of_correlation_and_viability]
        ),
    ):
        correlation_value, viability_diff_value = combination_to_values[
            combination_string
        ]
        # "-" instead of "." for the decimal point: a "." anywhere but
        # right before the real extension breaks tools that read the
        # extension as everything after the first "." in the filename.
        correlation_str = f"{correlation_value:.3f}".replace(".", "-")
        viability_diff_str = f"{viability_diff_value:.3f}".replace(".", "-")
        output_image_path = (
            pathlib.Path(f"{montage_figure_base_dir}")
            / f"{quadrent_of_correlation_and_viability}"
            / f"{combination_string}_corr{correlation_str}_viabilitydiff{viability_diff_str}.png"
        )
        # if output_image_path.exists():
        #     continue
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        image_1_file_path, image_2_file_path = (
            generate_image_paths_from_combination_string(
                image_base_dir=image_base_dir, combination_string=combination_string
            )
        )
        image_label1 = " ".join(combination_string.split("__")[:4])
        image_label1 = f"{'_'.join(image_label1.split('_')[:2])} {' '.join(image_label1.split('_')[3:5])}"
        image_label2 = " ".join(combination_string.split("__")[-4:])
        image_label2 = f"{'_'.join(image_label2.split('_')[:2])} {' '.join(image_label2.split('_')[3:5])}"
        # disable stdout for the plots here, since they are being saved to disk and not displayed in the notebook
        create_and_save_two_image_panel(
            image_1_array=make_multi_channel_image_array(image_1_file_path),
            image_2_array=make_multi_channel_image_array(image_2_file_path),
            image_1_label=image_label1,
            image_2_label=image_label2,
            output_path=output_image_path,
        )
        plt.close("all")
