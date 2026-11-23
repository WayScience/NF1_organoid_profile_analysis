#!/usr/bin/env python
# coding: utf-8

# # Plot Mutual Information Between 2D and 3D Profiles
#
# Plot the 2D ↔ 3D mutual information computed in notebook 6b.
# 2D and 3D profiles are paired at the aggregated level (patient/tumor × treatment), so each value describes how much information a 2D profile shares with a 3D profile.
# All values are shown as the information coefficient of correlation $r_I = \sqrt{1 - 2^{-2I}}$, which runs from 0 (2D and 3D measure different things) to 1 (they measure the same thing).
#
# Each resolution (organoid, single-cell) gets one PDF at 600 dpi, with one plot per page:
#
# - **Per-feature MI, two pages per stage.**
#   - Violin + boxplot of each 3D feature's best-match $r_I$ with any 2D feature, for every 3D type × 2D projection,
#     with each feature overlaid as a jittered point.
#     The title reports a Kruskal–Wallis test across 3D types, using `REFERENCE_PROJECTION` as the 2D side.
#   - Fraction of features that are redundant (best-match MI above the permutation null's 95th percentile), in both directions:
#     3D features explained by 2D (●) and 2D features explained by 3D (△).
#
# - **Controls, one page.** The same per-feature violins for two control comparisons from 6b:
#   MIP vs. Middle Slice (two 2D projections of the same stack) and ZedProfiler vs. SAM-Med3D (two 3D feature extractors on the same masked objects).
#   They show how much information two profiles of the *same* objects share, as a reference for the 2D ↔ 3D values.
#   The violins show each feature of the second profile (Middle Slice, SAM-Med3D) and its best match in the first.
#
# Whole-profile MI (permutation-corrected $r_I$ at `PRIMARY_N_COMPONENTS` PCs, its range across PC counts, and BH-adjusted permutation p-values) is saved to parquet but not plotted.
# Pairwise two-sided Mann–Whitney U tests between 3D types (with Cliff's delta and BH-adjusted p-values) are saved to parquet but not plotted.
# Features within a set are not independent, so per-feature p-values are optimistic. Use the effect sizes to judge how large a difference is.

# In[1]:


import itertools
import textwrap
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from notebook_init_utils.figure_pdf import FigurePDFs
from notebook_init_utils.notebook_init_utils import init_notebook
from notebook_init_utils.result_cache import is_fresh, mark_fresh
from scipy.stats import false_discovery_control, kruskal, mannwhitneyu

root_dir, in_notebook = init_notebook()


# In[2]:


_analysis_dir = root_dir / "2.2d_vs_3d_analysis"
_mi_dir = _analysis_dir / "results" / "mutual_information"
_stats_dir = _analysis_dir / "results" / "mutual_information_stats"
_figures_dir = _analysis_dir / "figures" / "mutual_information"
results_root = (_analysis_dir / "results").resolve()
# Caching: a figure is redrawn only when its input tables or this script change
CACHE_STEP = "7b.plot_mutual_information"
CODE_PATH = (_analysis_dir / "scripts" / "7b.plot_mutual_information.py").resolve(
    strict=True
)
PROFILE_MI_PATH = _mi_dir / "all_profile_mi.parquet"
FEATURE_MI_PATH = _mi_dir / "all_feature_mi_summary.parquet"

RESOLUTIONS = {"organoid": "Organoid", "sc": "Single-Cell"}
STAGES = {"normal": "Normalized", "fs": "Feature-selected", "agg": "Aggregated"}
# 2D group labels written by 6b, in plotting order
PROJECTIONS = ["MIP", "Middle Slice", "Middle N Slices"]
PROJECTION_COLORS = {
    "MIP": "#4C72B0",
    "Middle Slice": "#55A868",
    "Middle N Slices": "#C44E52",
}
PRIMARY_N_COMPONENTS = 5  # PC count reported in the whole-profile stats
REFERENCE_PROJECTION = "MIP"  # 2D side for the pairwise tests between 3D types
# Adjusted p-value thresholds for significance stars, most significant first
STAR_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
DPI = 600
JITTER_WIDTH = 0.25  # half-width of the horizontal jitter on per-feature points
JITTER_SEED = 0  # fixed so the jittered points are the same on every run
COMPARISON_2D_VS_3D = "2D vs 3D"  # comparison_type of the main analysis in 6b
CONTROL_COLORS = {
    "MIP vs Middle Slice": "#8172B3",
    "ZedProfiler vs SAM-Med3D": "#937860",
}
AGG_MODE_LABELS = {"from_fs": "from FS", "from_norm": "from Norm"}
# Font sizes (pt) shared by every figure
FONT_SIZES = {"title": 18, "label": 16, "tick": 14, "legend": 14, "annotation": 13}


# In[3]:


def label_3d(group_label: str, agg_mode: str) -> str:
    # Short 3D profile-type label; agg stage carries its aggregation mode
    variant = {"from_fs": "\nfrom FS", "from_norm": "\nfrom Norm"}.get(agg_mode, "")
    return f"{group_label}{variant}"


def significance_stars(p_adj: float) -> str:
    # Return significance stars for an adjusted p-value, or ``ns``
    for threshold, stars in STAR_THRESHOLDS:
        if p_adj < threshold:
            return stars
    return "ns"


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_3d"] = [
        label_3d(g, m) for g, m in zip(df["group_label_3d"], df["agg_mode_3d"])
    ]
    return df


def order_3d_labels(df: pd.DataFrame) -> List[str]:
    ordered = (
        df[["label_3d", "group_label_3d", "agg_mode_3d"]]
        .drop_duplicates()
        .sort_values(["group_label_3d", "agg_mode_3d"])
    )
    return ordered["label_3d"].tolist()


# ## Whole-profile MI
#
# `p_value` in the 6b output is a permutation p-value per comparison. Here it's BH-adjusted across all comparisons within a resolution, at `PRIMARY_N_COMPONENTS`.
# With 200 permutations, the smallest possible p-value is $1/201 \approx 0.005$.

# In[4]:


def profile_stats(profile_df: pd.DataFrame) -> pd.DataFrame:
    # Primary-c r_I per comparison, its range across c, and BH-adjusted p
    primary = profile_df.loc[profile_df["n_components"] == PRIMARY_N_COMPONENTS].copy()
    spread = (
        profile_df.groupby("comparison")["info_correlation"]
        .agg(r_i_min="min", r_i_max="max")
        .reset_index()
    )
    primary = primary.merge(spread, on="comparison")
    primary["p_adj_bh"] = false_discovery_control(primary["p_value"], method="bh")
    primary["significance"] = primary["p_adj_bh"].map(significance_stars)
    return primary


# ## Per-feature MI
#
# For each 3D feature, `max_info_correlation` is the $r_I$ of its best-matching 2D feature.
# High values mean that single 2D feature already captures what the 3D feature measures.

# In[5]:


def pairwise_tests(
    groups: Dict[str, np.ndarray], labels: List[str]
) -> Tuple[pd.DataFrame, float]:
    # Kruskal-Wallis across groups and pairwise Mann-Whitney U tests.
    # Cliff's delta for (a, b) is P(a > b) - P(a < b), from -1 to 1;
    # positive means 3D type ``a`` shares more information with 2D.
    kruskal_p = kruskal(*groups.values()).pvalue if len(groups) > 1 else np.nan
    records = []
    for label_a, label_b in itertools.combinations(labels, 2):
        a, b = groups[label_a], groups[label_b]
        result = mannwhitneyu(a, b, alternative="two-sided")
        records.append(
            {
                "group_a": label_a,
                "group_b": label_b,
                "n_a": len(a),
                "n_b": len(b),
                "u_statistic": result.statistic,
                "p_value": result.pvalue,
                "cliffs_delta": 2 * result.statistic / (len(a) * len(b)) - 1,
            }
        )
    stats_df = pd.DataFrame(records)
    if not stats_df.empty:
        stats_df["p_adj_bh"] = false_discovery_control(stats_df["p_value"], method="bh")
        stats_df["significance"] = stats_df["p_adj_bh"].map(significance_stars)
    return stats_df, kruskal_p


def grouped_positions(labels: List[str], projections: List[str]) -> Dict:
    # x position for each (3D type, 2D projection), grouped by 3D type
    width = len(projections) + 1
    return {
        (label, proj): i * width + j
        for i, label in enumerate(labels)
        for j, proj in enumerate(projections)
    }


def feature_figure(n_positions: int):
    # One wide page for a grouped per-feature plot
    width = max(12, 0.6 * n_positions + 7)
    return plt.subplots(figsize=(width, 8), layout="constrained")


def set_grouped_xticks(
    ax, labels, projections, positions, ticklabels, xlabel="3D profile type"
):
    centers = [np.mean([positions[(l, p)] for p in projections]) for l in labels]
    # Wrap long 3D-type names so neighboring tick labels don't overlap
    wrapped = [
        "\n".join(textwrap.fill(line, 14) for line in t.split("\n")) for t in ticklabels
    ]
    ax.set_xticks(centers)
    ax.set_xticklabels(wrapped, fontsize=FONT_SIZES["tick"])
    ax.set_xlabel(xlabel, fontsize=FONT_SIZES["label"])
    ax.tick_params(axis="y", labelsize=FONT_SIZES["tick"])


def color_handles(names, colors=PROJECTION_COLORS):
    return [plt.Rectangle((0, 0), 1, 1, color=colors[n], alpha=0.55) for n in names]


def plot_feature_violins(
    df_3d,
    labels,
    projections,
    positions,
    title,
    group_field="label_3d",
    hue_field="group_label_2d",
    colors=PROJECTION_COLORS,
    xlabel="3D profile type",
    legend_title="2D projection",
    ylabel="Best-match $r_I$ with any 2D feature\n(per 3D feature)",
):
    # Violin + boxplot + jittered points of best-match r_I per (group, hue),
    # grouped on the x-axis
    fig, ax = feature_figure(len(positions))
    rng = np.random.default_rng(JITTER_SEED)
    for (label, proj), pos in positions.items():
        values = df_3d.loc[
            (df_3d[group_field] == label) & (df_3d[hue_field] == proj),
            "max_info_correlation",
        ].to_numpy()
        if len(values) == 0:
            continue
        parts = ax.violinplot([values], positions=[pos], showextrema=False, widths=0.85)
        parts["bodies"][0].set_facecolor(colors[proj])
        parts["bodies"][0].set_alpha(0.55)
        # Rasterized so PDFs with thousands of points stay small
        ax.scatter(
            pos + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, len(values)),
            values,
            s=4,
            color=colors[proj],
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
        ax.boxplot(
            [values],
            positions=[pos],
            widths=0.15,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white"},
            medianprops={"color": "black"},
            zorder=3,
        )
    ticklabels = []
    for label in labels:
        n = df_3d.loc[df_3d[group_field] == label].groupby(hue_field).size()
        ticklabels.append(f"{label}\n(n = {int(n.max()):,})" if len(n) else label)
    set_grouped_xticks(ax, labels, projections, positions, ticklabels, xlabel)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel, fontsize=FONT_SIZES["label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.legend(
        color_handles(projections, colors),
        projections,
        title=legend_title,
        title_fontsize=FONT_SIZES["legend"],
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    return fig


def plot_redundancy(df_stage, labels, projections, positions, title):
    fig, ax = feature_figure(len(positions))
    frac = df_stage.groupby(["label_3d", "group_label_2d", "modality"])[
        "redundant"
    ].mean()
    for (label, proj), pos in positions.items():
        color = PROJECTION_COLORS[proj]
        for modality, marker, face in (("3D", "o", color), ("2D", "^", "none")):
            key = (label, proj, modality)
            if key in frac.index:
                ax.scatter(
                    pos,
                    frac[key],
                    marker=marker,
                    s=90,
                    facecolors=face,
                    edgecolors=color,
                    linewidths=1.5,
                )
    set_grouped_xticks(ax, labels, projections, positions, labels)
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Fraction of features redundant", fontsize=FONT_SIZES["label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"])
    ax.grid(axis="y", alpha=0.3)
    marker_handles = [
        Line2D([], [], marker="o", color="gray", linestyle="", markersize=10),
        Line2D(
            [],
            [],
            marker="^",
            color="gray",
            markerfacecolor="none",
            linestyle="",
            markersize=10,
        ),
    ]
    ax.legend(
        color_handles(projections) + marker_handles,
        projections + ["3D features\nexplained by 2D", "2D features\nexplained by 3D"],
        frameon=False,
        fontsize=FONT_SIZES["legend"],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
    )
    return fig


def plot_feature_pages(df_stage, stage, resolution_label):
    # Two pages per stage: per-feature r_I violins, then redundancy fractions
    df_3d = df_stage.loc[df_stage["modality"] == "3D"]
    labels = order_3d_labels(df_stage)
    projections = [p for p in PROJECTIONS if p in set(df_stage["group_label_2d"])]
    positions = grouped_positions(labels, projections)

    ref = df_3d.loc[df_3d["group_label_2d"] == REFERENCE_PROJECTION]
    groups = {
        label: ref.loc[ref["label_3d"] == label, "max_info_correlation"].to_numpy()
        for label in labels
    }
    groups = {k: v for k, v in groups.items() if len(v)}
    stats_df, kruskal_p = pairwise_tests(groups, list(groups))

    if np.isnan(kruskal_p):
        kw_text = "Kruskal-Wallis: n/a"
    elif kruskal_p > 0:
        kw_text = (
            f"Kruskal-Wallis (3D types, 2D {REFERENCE_PROJECTION}) p = {kruskal_p:.2e}"
        )
    else:
        # p underflows to 0 for very large differences; report the float floor
        kw_text = f"Kruskal-Wallis (3D types, 2D {REFERENCE_PROJECTION}) p < 1e-300"
    figs = [
        plot_feature_violins(
            df_3d,
            labels,
            projections,
            positions,
            f"2D-3D Mutual Information per Feature: {resolution_label}, "
            f"{STAGES[stage]}\n{kw_text}",
        ),
        plot_redundancy(
            df_stage,
            labels,
            projections,
            positions,
            f"Redundant Features: {resolution_label}, {STAGES[stage]}",
        ),
    ]
    return figs, stats_df, kruskal_p


# ## Controls
#
# Control per-feature MI is plotted against stage. The agg stage is split by 3D aggregation mode.
# MIP vs. Middle Slice has a single 2D agg profile, so its agg value is shown in both agg columns.
# Whole-profile permutation p-values are BH-adjusted across the controls of a resolution, separately from the 2D ↔ 3D comparisons, and saved to parquet.

# In[6]:


def stage_column(stage: str, agg_mode: str) -> str:
    if stage != "agg":
        return STAGES[stage]
    return f"{STAGES[stage]}\n({AGG_MODE_LABELS[agg_mode]})"


STAGE_COLUMNS = [
    stage_column(stage, mode)
    for stage in STAGES
    for mode in (AGG_MODE_LABELS if stage == "agg" else [None])
]


def control_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Add the control name and stage column; repeat single agg rows for both modes
    agg_none = (df["stage"] == "agg") & (df["agg_mode_3d"] == "none")
    expanded = pd.concat(
        [df.loc[~agg_none]]
        + [df.loc[agg_none].assign(agg_mode_3d=mode) for mode in AGG_MODE_LABELS],
        ignore_index=True,
    )
    expanded["stage_column"] = [
        stage_column(s, m) for s, m in zip(expanded["stage"], expanded["agg_mode_3d"])
    ]
    expanded["control"] = expanded["comparison_type"].str.removeprefix("Control: ")
    return expanded


def plot_control_page(control_features, resolution_label):
    # Per-feature violins for the control comparisons
    features = control_columns(
        control_features.loc[control_features["modality"] == "3D"]
    )
    cols = [c for c in STAGE_COLUMNS if c in set(features["stage_column"])]
    controls = [c for c in CONTROL_COLORS if c in set(features["control"])]
    return plot_feature_violins(
        features,
        cols,
        controls,
        grouped_positions(cols, controls),
        f"Control Mutual Information per Feature: {resolution_label}",
        group_field="stage_column",
        hue_field="control",
        colors=CONTROL_COLORS,
        xlabel="Stage",
        legend_title="Control",
        ylabel="Best-match $r_I$ with the first profile\n"
        "(per feature of the second profile)",
    )


# ## Make the figures

# In[7]:


inputs = [PROFILE_MI_PATH, FEATURE_MI_PATH]
missing = [p.name for p in inputs if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing {missing}. Run notebook 6b first.")

_figures_dir.mkdir(parents=True, exist_ok=True)
_stats_dir.mkdir(parents=True, exist_ok=True)
pdf = FigurePDFs(dpi=DPI)
profile_all = feature_all = None

for resolution, resolution_label in RESOLUTIONS.items():
    pdf_path = _figures_dir / f"{resolution}_mutual_information.pdf"
    profile_stats_path = _stats_dir / f"{resolution}_profile_mi_stats.parquet"
    feature_stats_path = _stats_dir / f"{resolution}_pairwise_mann_whitney.parquet"
    control_stats_path = _stats_dir / f"{resolution}_control_profile_mi_stats.parquet"
    outputs = [pdf_path, profile_stats_path, feature_stats_path, control_stats_path]
    if is_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs):
        print(f"Using cached result: {pdf_path.name}")
        continue

    if profile_all is None:
        profile_all = add_labels(pd.read_parquet(PROFILE_MI_PATH))
        feature_all = add_labels(pd.read_parquet(FEATURE_MI_PATH))
    profile_res = profile_all.loc[profile_all["resolution"] == resolution]
    feature_res = feature_all.loc[feature_all["resolution"] == resolution]
    # Main 2D vs 3D comparisons; controls are plotted on their own pages
    is_main = profile_res["comparison_type"] == COMPARISON_2D_VS_3D
    profile_df, control_profile_df = profile_res.loc[is_main], profile_res.loc[~is_main]
    is_main = feature_res["comparison_type"] == COMPARISON_2D_VS_3D
    feature_df, control_feature_df = feature_res.loc[is_main], feature_res.loc[~is_main]
    if profile_df.empty:
        print(f"SKIPPED (no data): {resolution}")
        continue

    # Whole-profile MI: saved, not plotted
    stats = profile_stats(profile_df)
    stats.assign(label_3d=stats["label_3d"].str.replace("\n", " ")).to_parquet(
        profile_stats_path, index=False
    )

    # Two pages per stage: per-feature MI
    stage_stats = []
    for stage in STAGES:
        df_stage = feature_df.loc[feature_df["stage"] == stage]
        if df_stage.empty:
            continue
        figs, stats_df, kruskal_p = plot_feature_pages(
            df_stage, stage, resolution_label
        )
        for fig in figs:
            pdf.savefig(fig, pdf_path)
            plt.close(fig)
        if not stats_df.empty:
            stage_stats.append(
                stats_df.assign(
                    resolution=resolution,
                    stage=stage,
                    reference_projection=REFERENCE_PROJECTION,
                    kruskal_p_value=kruskal_p,
                    # Line breaks in the labels are only for the axes; store plain names
                    group_a=stats_df["group_a"].str.replace("\n", " "),
                    group_b=stats_df["group_b"].str.replace("\n", " "),
                )
            )
    pd.concat(stage_stats, ignore_index=True).to_parquet(
        feature_stats_path, index=False
    )

    # One page: control comparisons
    control_stats = (
        pd.DataFrame()
        if control_profile_df.empty
        else profile_stats(control_profile_df)
    )
    control_stats.to_parquet(control_stats_path, index=False)
    if control_stats.empty or control_feature_df.empty:
        print(f"No control comparisons for {resolution}; rerun notebook 6b")
    else:
        fig = plot_control_page(control_feature_df, resolution_label)
        pdf.savefig(fig, pdf_path)
        plt.close(fig)

    pdf.close(pdf_path)
    mark_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs)
    print(
        f"Wrote {pdf_path.name}, {profile_stats_path.name}, "
        f"{feature_stats_path.name}, {control_stats_path.name}"
    )
