#!/usr/bin/env python
# coding: utf-8

# # Plot Mutual Information Between 2D and 3D Profiles
#
# Plot the 2D ↔ 3D mutual information computed in notebook 6b.
# 2D and 3D profiles are paired at the aggregated level (patient/tumor × treatment), so each value describes how much information a 2D profile shares with a 3D profile.
# All values are shown as the information coefficient of correlation $r_I = \sqrt{1 - 2^{-2I}}$, which runs from 0 (2D and 3D measure different things) to 1 (they measure the same thing).
#
# Each resolution (organoid, single-cell) gets one PDF at 600 dpi:
#
# - **Page 1, whole-profile MI.** One heatmap per stage: 3D profile types (rows) × 2D projections (columns).
#   Each cell shows permutation-corrected $r_I$ at `PRIMARY_N_COMPONENTS` PCs, with the range across all PC counts underneath as a sensitivity check.
#   Stars are Benjamini–Hochberg adjusted permutation p-values within the resolution.
# - **One page per stage, per-feature MI.**
#   - Top: violin + boxplot of each 3D feature's best-match $r_I$ with any 2D feature, for every 3D type × 2D projection.
#   - Middle: fraction of features that are redundant (best-match MI above the permutation null's 95th percentile), in both directions:
#     3D features explained by 2D (●) and 2D features explained by 3D (△).
#   - Bottom: pairwise two-sided Mann–Whitney U tests between 3D types, using `REFERENCE_PROJECTION` as the 2D side.
#     Color is Cliff's delta (column vs. row), and stars are BH-adjusted p-values.
#
# Features within a set are not independent, so per-feature p-values are optimistic. Use the effect sizes to judge how large a difference is.

# In[1]:


import itertools
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
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
PRIMARY_N_COMPONENTS = 5  # PC count shown in the profile-level heatmap cells
REFERENCE_PROJECTION = "MIP"  # 2D side for the pairwise tests between 3D types
# Adjusted p-value thresholds for significance stars, most significant first
STAR_THRESHOLDS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]
DPI = 600


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


def plot_profile_heatmaps(stats: pd.DataFrame, resolution_label: str):
    stages = [s for s in STAGES if s in set(stats["stage"])]
    heights = [stats.loc[stats["stage"] == s, "label_3d"].nunique() for s in stages]
    fig, axes = plt.subplots(
        1,
        len(stages),
        figsize=(5.5 * len(stages) + 1, 1.0 * max(heights) + 2.5),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for ax, stage in zip(axes[0], stages):
        df = stats.loc[stats["stage"] == stage]
        rows = order_3d_labels(df)
        cols = [p for p in PROJECTIONS if p in set(df["group_label_2d"])]
        matrix = np.full((len(rows), len(cols)), np.nan)
        for rec in df.itertuples():
            i, j = rows.index(rec.label_3d), cols.index(rec.group_label_2d)
            matrix[i, j] = rec.info_correlation
            star = "" if rec.significance == "ns" else rec.significance
            text = f"{rec.info_correlation:.2f}{star}\n({rec.r_i_min:.2f}-{rec.r_i_max:.2f})"
            color = "white" if rec.info_correlation > 0.6 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color=color)
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([r.replace("\n", " ") for r in rows], fontsize=7)
        ax.set_title(STAGES[stage], fontsize=10)
    fig.suptitle(
        f"2D-3D Mutual Information, Whole Profile: {resolution_label}\n"
        f"$r_I$ at {PRIMARY_N_COMPONENTS} PCs (range across PC counts); "
        "BH-adjusted permutation p: * < 0.05, ** < 0.01, *** < 0.001",
        fontsize=10,
    )
    colorbar = fig.colorbar(image, ax=axes[0].tolist(), fraction=0.03, pad=0.02)
    colorbar.set_label("$r_I$ (permutation-corrected)", fontsize=8)
    return fig


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


def plot_feature_violins(ax, df_3d, labels, projections, positions, title):
    for (label, proj), pos in positions.items():
        values = df_3d.loc[
            (df_3d["label_3d"] == label) & (df_3d["group_label_2d"] == proj),
            "max_info_correlation",
        ].to_numpy()
        if len(values) == 0:
            continue
        parts = ax.violinplot([values], positions=[pos], showextrema=False, widths=0.85)
        parts["bodies"][0].set_facecolor(PROJECTION_COLORS[proj])
        parts["bodies"][0].set_alpha(0.55)
        ax.boxplot(
            [values],
            positions=[pos],
            widths=0.15,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "white"},
            medianprops={"color": "black"},
        )
    centers, ticklabels = [], []
    for label in labels:
        pos = [positions[(label, p)] for p in projections]
        n = df_3d.loc[df_3d["label_3d"] == label].groupby("group_label_2d").size()
        centers.append(np.mean(pos))
        ticklabels.append(f"{label}\n(n = {int(n.max()):,})" if len(n) else label)
    ax.set_xticks(centers)
    ax.set_xticklabels(ticklabels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Best-match $r_I$ with any 2D feature\n(per 3D feature)", fontsize=8)
    ax.set_title(title, fontsize=10)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PROJECTION_COLORS[p], alpha=0.55)
        for p in projections
    ]
    ax.legend(
        handles,
        [f"2D {p}" for p in projections],
        frameon=False,
        fontsize=7,
        ncol=len(projections),
        loc="upper right",
    )


def plot_redundancy(ax, df_stage, labels, projections, positions):
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
                    s=28,
                    facecolors=face,
                    edgecolors=color,
                    linewidths=1,
                )
    ax.set_xticks([np.mean([positions[(l, p)] for p in projections]) for l in labels])
    ax.set_xticklabels([])
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Fraction of features\nredundant", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(
        [
            Line2D([], [], marker="o", color="gray", linestyle=""),
            Line2D(
                [], [], marker="^", color="gray", markerfacecolor="none", linestyle=""
            ),
        ],
        ["3D features explained by 2D", "2D features explained by 3D"],
        frameon=False,
        fontsize=7,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.02),
    )


def plot_pairwise_matrix(ax, stats_df: pd.DataFrame, labels: List[str]):
    # Lower-triangle heatmap of Cliff's delta with BH-adjusted significance stars
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.full((len(labels), len(labels)), np.nan)
    stars = {}
    for row in stats_df.itertuples():
        i, j = index[row.group_a], index[row.group_b]
        # Row = group_b, column = group_a: delta is column vs. row
        matrix[j, i] = row.cliffs_delta
        stars[(j, i)] = row.significance

    image = ax.imshow(
        matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1)
    )
    for (r, c), text in stars.items():
        # White text on strongly colored cells so the stars stay readable
        color = "white" if abs(matrix[r, c]) > 0.5 else "black"
        ax.text(c, r, text, ha="center", va="center", fontsize=6, color=color)
    short = [label.replace("\n", " ") for label in labels]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(short, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(
        f"3D types vs. 2D {REFERENCE_PROJECTION}: pairwise Mann-Whitney U\n"
        "(BH-adjusted): * < 0.05, ** < 0.01, *** < 0.001",
        fontsize=9,
    )
    return image


def plot_feature_page(df_stage, stage, resolution_label):
    # One page: violins, redundancy fractions, pairwise matrix
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

    width = max(8, 0.55 * len(positions) + 3)
    fig, (ax_violin, ax_red, ax_matrix) = plt.subplots(
        3,
        1,
        figsize=(width, width * 1.35),
        gridspec_kw={"height_ratios": [1, 0.45, 0.9]},
    )
    if np.isnan(kruskal_p):
        kw_text = "Kruskal-Wallis: n/a"
    elif kruskal_p > 0:
        kw_text = (
            f"Kruskal-Wallis (3D types, 2D {REFERENCE_PROJECTION}) p = {kruskal_p:.2e}"
        )
    else:
        # p underflows to 0 for very large differences; report the float floor
        kw_text = f"Kruskal-Wallis (3D types, 2D {REFERENCE_PROJECTION}) p < 1e-300"
    plot_feature_violins(
        ax_violin,
        df_3d,
        labels,
        projections,
        positions,
        f"2D-3D Mutual Information per Feature: {resolution_label}, "
        f"{STAGES[stage]}\n{kw_text}",
    )
    plot_redundancy(ax_red, df_stage, labels, projections, positions)
    ax_red.set_xlim(ax_violin.get_xlim())
    if stats_df.empty:
        ax_matrix.axis("off")
    else:
        image = plot_pairwise_matrix(ax_matrix, stats_df, list(groups))
        colorbar = fig.colorbar(image, ax=ax_matrix, fraction=0.04, pad=0.02)
        colorbar.set_label("Cliff's delta (column vs. row)", fontsize=8)
    fig.tight_layout()
    return fig, stats_df, kruskal_p


# ## Make the figures

# In[6]:


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
    outputs = [pdf_path, profile_stats_path, feature_stats_path]
    if is_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs):
        print(f"Using cached result: {pdf_path.name}")
        continue

    if profile_all is None:
        profile_all = add_labels(pd.read_parquet(PROFILE_MI_PATH))
        feature_all = add_labels(pd.read_parquet(FEATURE_MI_PATH))
    profile_df = profile_all.loc[profile_all["resolution"] == resolution]
    feature_df = feature_all.loc[feature_all["resolution"] == resolution]
    if profile_df.empty:
        print(f"SKIPPED (no data): {resolution}")
        continue

    # Page 1: whole-profile MI
    stats = profile_stats(profile_df)
    stats.assign(label_3d=stats["label_3d"].str.replace("\n", " ")).to_parquet(
        profile_stats_path, index=False
    )
    fig = plot_profile_heatmaps(stats, resolution_label)
    pdf.savefig(fig, pdf_path)
    plt.close(fig)

    # One page per stage: per-feature MI
    stage_stats = []
    for stage in STAGES:
        df_stage = feature_df.loc[feature_df["stage"] == stage]
        if df_stage.empty:
            continue
        fig, stats_df, kruskal_p = plot_feature_page(df_stage, stage, resolution_label)
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

    pdf.close(pdf_path)
    mark_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs)
    print(
        f"Wrote {pdf_path.name}, {profile_stats_path.name}, {feature_stats_path.name}"
    )
