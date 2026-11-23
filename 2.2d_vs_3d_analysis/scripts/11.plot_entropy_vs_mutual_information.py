#!/usr/bin/env python
# coding: utf-8

# # Plot Shannon Entropy vs. Mutual Information
#
# Compare each 3D feature's Shannon entropy (notebook 6) with how much information it shares with 2D (notebook 6b).
# For every 3D feature, the y-axis is the information coefficient of correlation $r_I$ with its best-matching 2D feature from `REFERENCE_PROJECTION`.
# The x-axis is that 3D feature's Shannon entropy (bits, 50 histogram bins).
#
# Each resolution (organoid, single-cell) gets one PDF at 600 dpi, with one 2D density heatmap per page: one page per stage × 3D profile type.
# Color is the number of features in each bin (log scale), and the title reports Spearman's $\rho$ between entropy and $r_I$.
# The Spearman statistics are also saved to parquet.

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from notebook_init_utils.figure_pdf import FigurePDFs
from notebook_init_utils.notebook_init_utils import init_notebook
from notebook_init_utils.result_cache import is_fresh, mark_fresh
from scipy.stats import spearmanr

root_dir, in_notebook = init_notebook()


# In[ ]:


_analysis_dir = root_dir / "2.2d_vs_3d_analysis"
_entropy_dir = _analysis_dir / "results" / "entropy"
_stats_dir = _analysis_dir / "results" / "mutual_information_stats"
_figures_dir = _analysis_dir / "figures" / "entropy_vs_mutual_information"
results_root = (_analysis_dir / "results").resolve()
# Caching: a figure is redrawn only when its input tables or this script change
CACHE_STEP = "11.plot_entropy_vs_mutual_information"
CODE_PATH = (
    _analysis_dir / "scripts" / "11.plot_entropy_vs_mutual_information.py"
).resolve(strict=True)
FEATURE_MI_PATH = (
    _analysis_dir / "results" / "mutual_information" / "all_feature_mi_summary.parquet"
)
# Per-profile entropy tables from notebook 6, named "<profile>_entropy.parquet"
ENTROPY_SUFFIX = "_entropy.parquet"
ENTROPY_PATHS = sorted(
    p for p in _entropy_dir.glob(f"3D_*{ENTROPY_SUFFIX}") if "per_patient" not in p.name
)

RESOLUTIONS = {"organoid": "Organoid", "sc": "Single-Cell"}
STAGES = {"normal": "Normalized", "fs": "Feature-selected", "agg": "Aggregated"}
AGG_MODES = {"none": "", "from_fs": " (from FS)", "from_norm": " (from Norm)"}
REFERENCE_PROJECTION = "MIP"  # 2D side of the best-match r_I, as in notebook 7b
N_BINS = 40  # bins per axis of the 2D histogram
ENTROPY_MAX_BITS = np.log2(50)  # 50 histogram bins in notebook 6
FONT_SIZES = {"title": 18, "label": 16, "tick": 14, "legend": 14}
DPI = 600


# ## Join entropy to per-feature MI
#
# Each MI row names its 3D profile (`profile_3d`), which matches the entropy file name, so the two tables join on profile and feature name.

# In[ ]:


def load_entropy(paths) -> pd.DataFrame:
    frames = [
        pd.read_parquet(p, columns=["feature_name", "entropy"]).assign(
            profile_3d=p.name.removesuffix(ENTROPY_SUFFIX)
        )
        for p in paths
    ]
    return pd.concat(frames, ignore_index=True)


def join_entropy_mi(mi_df: pd.DataFrame, entropy_df: pd.DataFrame) -> pd.DataFrame:
    # 3D features vs. the reference 2D projection, with each feature's entropy
    # 2D vs 3D comparisons only (6b also writes control comparisons)
    mi_3d = mi_df.loc[
        (mi_df["comparison_type"] == "2D vs 3D")
        & (mi_df["modality"] == "3D")
        & (mi_df["group_label_2d"] == REFERENCE_PROJECTION)
    ]
    joined = mi_3d.merge(
        entropy_df, on=["profile_3d", "feature_name"], how="left", validate="1:1"
    )
    n_missing = joined["entropy"].isna().sum()
    if n_missing:
        print(f"{n_missing} 3D features have no entropy value and are dropped")
    return joined.dropna(subset=["entropy", "max_info_correlation"])


# ## Plot
#
# One page per stage × 3D profile type. All pages share axis limits so they can be compared.

# In[ ]:


def plot_entropy_mi_heatmap(df: pd.DataFrame, title: str):
    x = df["entropy"].to_numpy()
    y = df["max_info_correlation"].to_numpy()
    rho, p_value = spearmanr(x, y)
    fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")
    *_, image = ax.hist2d(
        x,
        y,
        bins=N_BINS,
        range=[[0, ENTROPY_MAX_BITS], [0, 1]],
        cmap="viridis",
        norm=LogNorm(),
        cmin=1,
    )
    ax.set_xlabel("Shannon entropy of 3D feature (bits)", fontsize=FONT_SIZES["label"])
    ax.set_ylabel(
        f"Best-match $r_I$ with any 2D {REFERENCE_PROJECTION} feature",
        fontsize=FONT_SIZES["label"],
    )
    ax.tick_params(labelsize=FONT_SIZES["tick"])
    ax.set_title(
        f"{title}\nSpearman $\\rho$ = {rho:.2f} (p = {p_value:.1e}), n = {len(df):,}",
        fontsize=FONT_SIZES["title"],
    )
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Number of features", fontsize=FONT_SIZES["label"])
    colorbar.ax.tick_params(labelsize=FONT_SIZES["tick"])
    return fig, rho, p_value


# In[ ]:


inputs = [FEATURE_MI_PATH, *ENTROPY_PATHS]
if not FEATURE_MI_PATH.exists() or not ENTROPY_PATHS:
    raise FileNotFoundError("Missing MI or entropy results. Run notebooks 6 and 6b.")

_figures_dir.mkdir(parents=True, exist_ok=True)
_stats_dir.mkdir(parents=True, exist_ok=True)
pdf = FigurePDFs(dpi=DPI)
joined_all = None

for resolution, resolution_label in RESOLUTIONS.items():
    pdf_path = _figures_dir / f"{resolution}_entropy_vs_mutual_information.pdf"
    stats_path = _stats_dir / f"{resolution}_entropy_vs_mi_spearman.parquet"
    outputs = [pdf_path, stats_path]
    if is_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs):
        print(f"Using cached result: {pdf_path.name}")
        continue

    if joined_all is None:
        joined_all = join_entropy_mi(
            pd.read_parquet(FEATURE_MI_PATH), load_entropy(ENTROPY_PATHS)
        )
    joined = joined_all.loc[joined_all["resolution"] == resolution]
    if joined.empty:
        print(f"SKIPPED (no data): {resolution}")
        continue

    records = []
    for stage, stage_label in STAGES.items():
        df_stage = joined.loc[joined["stage"] == stage]
        groups = df_stage.groupby(["group_label_3d", "agg_mode_3d"], sort=True)
        for (group_label, agg_mode), df in groups:
            title = (
                f"{resolution_label}, {stage_label}: "
                f"{group_label}{AGG_MODES.get(agg_mode, '')}"
            )
            fig, rho, p_value = plot_entropy_mi_heatmap(df, title)
            pdf.savefig(fig, pdf_path)
            plt.close(fig)
            records.append(
                {
                    "resolution": resolution,
                    "stage": stage,
                    "group_label_3d": group_label,
                    "agg_mode_3d": agg_mode,
                    "reference_projection": REFERENCE_PROJECTION,
                    "n_features": len(df),
                    "spearman_rho": rho,
                    "p_value": p_value,
                }
            )
    pd.DataFrame(records).to_parquet(stats_path, index=False)
    pdf.close(pdf_path)
    mark_fresh(CACHE_STEP, results_root, CODE_PATH, outputs, inputs)
    print(f"Wrote {pdf_path.name}, {stats_path.name}")
