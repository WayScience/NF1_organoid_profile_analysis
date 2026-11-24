#!/usr/bin/env python
# coding: utf-8

# # UMAP with example images framed around the plot
#
# This notebook:
# 1. Computes a 2-D UMAP embedding of your features.
# 2. Picks a set of points from the embedding (k-means representatives, farthest-point sampling, random, or your own indices).
# 3. Draws the UMAP scatter in the middle and lines the selected points' images up around it in a rectangle, with a leader line from each image to its point.
#
# Points are matched to border slots with a minimum-total-length assignment, which guarantees that leader lines never cross.
#
# It runs as-is on the scikit-learn **digits** dataset. To use your own data, edit only the **Load data** cell.

# In[1]:


# !pip install umap-learn scikit-learn matplotlib pillow

import os
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import umap
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from microfilm.colorify import multichannel_to_rgb
from montage_utils.montage_utils import (
    create_and_save_single_image_panel,
    make_multi_channel_image_array,
)
from notebook_init_utils.notebook_init_utils import bandicoot_check, init_notebook
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

root_dir, init_notebook = init_notebook()

images_base_dir = bandicoot_check(
    bandicoot_mount_path=pathlib.Path(os.path.expanduser("~/mnt/bandicoot")),
    root_dir=root_dir,
)
RNG_SEED = 0

# Output figures and a local, in-repo cache of rendered multi-channel composites.
# The cache lets small plot tweaks re-run fast: each composite is built from the
# raw channel z-stacks only once, then reloaded from this directory thereafter.
figures_dir = root_dir / "figures" / "assembled_figure_3" / "figures"
image_cache_dir = root_dir / "figures" / "assembled_figure_3" / "data" / "image_cache"
figures_dir.mkdir(parents=True, exist_ok=True)
image_cache_dir.mkdir(parents=True, exist_ok=True)

# All image-frame figures are written as pages of this one multi-page PDF.
umap_image_frame_pdf_path = figures_dir / "umap_image_frame.pdf"
# The Figure 3 panel (top-down single-cell crops) is written separately, and the same cells
# drawn as 3D isometric volume renders to their own PDF, for comparison only: the z-step
# (10x the xy pixel size) makes tilted views smeared, so they are not used in the figure.
umap_image_frame_sc_pdf_path = figures_dir / "umap_image_frame_sc_mip.pdf"
umap_image_frame_3d_pdf_path = figures_dir / "umap_image_frame_3d.pdf"


# ## Load data

# In[2]:


umap_data_file_path = pathlib.Path(
    f"{root_dir}/1.EDA/results/umap/3D_scfs_umap.parquet"
).resolve(strict=True)
umap_data = pd.read_parquet(umap_data_file_path)


# In[3]:


# Stage subdirectory holding the display-ready (illum-corrected) channel z-stacks.
image_stage = "zstack_images"


def _raw_image_filepath_channelx(row):
    well_fov = row["Metadata_Experiment_WellFOV"].replace("_", "-")
    patient_tumor = row["Metadata_Biology_PatientTumor"]
    return f"{images_base_dir}/data/{patient_tumor}/{image_stage}/{well_fov}"


umap_data[f"Metadata_raw_image_filepath"] = umap_data.apply(
    lambda row: _raw_image_filepath_channelx(row), axis=1
)
umap_data.head()


# In[4]:


# Wire the real UMAP data into the names the plotting cells expect.
# embedding: (N, 2) UMAP coordinates; images: aligned well-FOV directory paths
# (each rendered on demand into a multi-channel composite by `_load_image`).
embedding = umap_data[["UMAP1", "UMAP2"]].to_numpy()
images = umap_data["Metadata_raw_image_filepath"].to_numpy()

# Point-coloring schemes requested: tumor type, patient, treatment (drug + dose).
labels_tumor = umap_data["Metadata_Biology_TumorType"].to_numpy()
labels_patient = umap_data["Metadata_Biology_PatientTumor"].to_numpy()
labels_treatment = (
    umap_data["Metadata_Experiment_Treatment"].astype(str)
    + " "
    + umap_data["Metadata_Experiment_Dose"].map(
        lambda d: f"{d:g}uM" if pd.notna(d) else "NA"
    )
).to_numpy()

# Default labels used by any single-labeling helper call.
labels = labels_tumor

# ---------------------------------------------------------------------------
# Canonical palettes (copied from utils/r_plot_themes.r so the Python figures
# match the manuscript R figures). A gray fallback covers any label missing
# from a palette (e.g. a NaN tumor type).
# ---------------------------------------------------------------------------
MISSING_COLOR = "#CCCCCC"

tumor_type_palette = {
    "cNF": "#1B9E77",
    "pNF": "#D95F02",
    "MPNST": "#7570B3",
    "Other": "#999999",
}

# tab20_palette_for_patients applied in the canonical patient order from
# tumor_type_lookup (r_plot_themes.r).
_patient_order = [
    "NF0014_T1",
    "NF0014_T2",
    "NF0016_T1",
    "NF0018_T6",
    "NF0021_T1",
    "NF0030_T1",
    "NF0035_T1",
    "NF0037_T1",
    "NF0040_T1",
    "NF0055_T1",
    "SARCO219_T2",
    "SARCO361_T1",
]
_tab20_patients = [
    "#1f77b4",
    "#aec7e8",
    "#ff7f0e",
    "#ffbb78",
    "#2ca02c",
    "#98df8a",
    "#d62728",
    "#ff9896",
    "#9467bd",
    "#c5b0d5",
    "#8c564b",
    "#c49c94",
    "#e377c2",
    "#f7b6d2",
    "#7f7f7f",
    "#c7c7c7",
    "#bcbd22",
    "#dbdb8d",
    "#17becf",
    "#9edae5",
]
patient_palette = {p: _tab20_patients[i] for i, p in enumerate(_patient_order)}

# custom_treatment_palette (r_plot_themes.r), keyed by drug name.
drug_palette = {
    "DMSO": "#A6A6A6",
    "Staurosporine": "#3F468C",
    "Fimepinostat": "#3DCCA8",
    "Copanlisib": "#3DCACC",
    "Everolimus": "#3DA4CC",
    "Rapamycin": "#3D7DCC",
    "Sapanisertib": "#3D57CC",
    "Vistusertib": "#493DCC",
    "Panobinostat": "#CC8029",
    "ARV-825": "#CCAB29",
    "Imatinib": "#6047CC",
    "Nilotinib": "#8347CC",
    "Cabozantinib": "#A647CC",
    "Linsitinib": "#CA47CC",
    "Binimetinib": "#D92B7F",
    "Mirdametinib": "#D92B51",
    "Trametinib": "#D9342B",
    "Selumetinib": "#D9622B",
    "Onalespib": "#6CA642",
    "Digoxin": "#BF3078",
    "Ketotifen": "#238C83",
    "Trabectedin": "#388C5B",
}
# Treatment labels are "<drug> <dose>uM"; color them by the drug (the canonical
# palette is per-drug), so both doses of a drug share its color.
treatment_palette = {
    lab: drug_palette.get(str(lab).rsplit(" ", 1)[0], MISSING_COLOR)
    for lab in pd.unique(labels_treatment)
}


# In[5]:


# plot the loaded data
plt.figure(figsize=(10, 6))
plt.scatter(umap_data["UMAP1"], umap_data["UMAP2"])
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of 3D SCFS Data")
plt.show()


# ## Choose which points to show
#
# - `"kmeans"`: cluster the embedding into `n` groups and take the real point closest to each centre — good coverage of the dense regions.
# - `"fps"`: farthest-point sampling — good coverage of the whole shape, including outliers.
# - `"per_label"`: `n_per_label` random points from each class.
# - `"random"`
# - or just pass a list of indices you care about.

# In[6]:


def select_points(
    embedding, n=24, method="kmeans", labels=None, n_per_label=2, seed=RNG_SEED
):
    rng = np.random.default_rng(seed)
    N = len(embedding)
    if method == "kmeans":
        km = KMeans(n_clusters=n, n_init=10, random_state=seed).fit(embedding)
        idx = []
        for c in km.cluster_centers_:
            d = np.linalg.norm(embedding - c, axis=1)
            d[idx] = np.inf  # avoid duplicates
            idx.append(int(np.argmin(d)))
        return np.array(idx)
    if method == "fps":
        idx = [int(rng.integers(N))]
        d = np.linalg.norm(embedding - embedding[idx[0]], axis=1)
        for _ in range(n - 1):
            i = int(np.argmax(d))
            idx.append(i)
            d = np.minimum(d, np.linalg.norm(embedding - embedding[i], axis=1))
        return np.array(idx)
    if method == "per_label":
        assert labels is not None, "per_label needs labels"
        idx = []
        for u in np.unique(labels):
            members = np.flatnonzero(labels == u)
            idx.extend(
                rng.choice(members, size=min(n_per_label, len(members)), replace=False)
            )
        return np.array(idx)
    if method == "random":
        return rng.choice(N, size=n, replace=False)
    raise ValueError(method)


# ## Layout helpers
#
# The scatter is scaled into a box (keeping its aspect ratio). Images are spread over the four sides of a larger rectangle around it so neighbours (including at corners) never overlap; they shrink automatically if there are too many. Points are matched to slots with the minimum-total-length assignment (Hungarian algorithm). A minimum-length matching never contains two crossing segments, so leader lines don't cross.

# In[7]:


def label_colors(labels, palette=None):
    """Map categorical labels to colors.

    Returns (colors, lut): `colors` is a per-point RGBA/colorlist aligned to
    `labels` (or None if labels is None), and `lut` is an ordered
    {label: color} dict for building a legend.

    If `palette` (a {label: color} dict, e.g. a canonical palette from
    utils/r_plot_themes.r) is given, it is used directly and any label missing
    from it falls back to MISSING_COLOR. Otherwise colors are generated: up to
    10 categories use tab10, up to 20 use tab20, and more are drawn from the
    combined tab20/tab20b/tab20c (60 distinct colors).
    """
    if labels is None:
        return None, {}
    labels = np.asarray(labels)
    uniq = list(pd.unique(labels))
    if palette is not None:
        fallback = globals().get("MISSING_COLOR", "#CCCCCC")
        lut = {lab: palette.get(lab, fallback) for lab in uniq}
    else:
        n = len(uniq)
        if n <= 10:
            colors_list = [plt.get_cmap("tab10")(i) for i in range(n)]
        elif n <= 20:
            colors_list = [plt.get_cmap("tab20")(i) for i in range(n)]
        else:
            base = (
                list(plt.get_cmap("tab20").colors)
                + list(plt.get_cmap("tab20b").colors)
                + list(plt.get_cmap("tab20c").colors)
            )
            colors_list = [base[i % len(base)] for i in range(n)]
        lut = {lab: colors_list[i] for i, lab in enumerate(uniq)}
    colors = [lut[lab] for lab in labels]
    return colors, lut


# Channel -> color for the multi-channel composite, matching
# montage_utils.create_and_save_single_image_panel (sorted channels 405, 488, 555, 640).
COMPOSITE_CMAPS = ["pure_cyan", "pure_green", "pure_magenta", "pure_red"]
# Per-channel percentile contrast stretch (low, high) applied before colorizing.
# Clipping the low 3% to black and the top 0.5% to full brightens dim signal
# while keeping the background clean (plain min-max leaves composites too dim).
COMPOSITE_CONTRAST = (3.0, 99.5)


def _composite_rgb(channel_array):
    """Contrast-stretch each channel (per COMPOSITE_CONTRAST), colorize with
    COMPOSITE_CMAPS, and blend into one uint8 RGB image."""
    arr = channel_array.astype(np.float64)
    lo_p, hi_p = COMPOSITE_CONTRAST
    stretched = np.empty_like(arr)
    for k in range(arr.shape[0]):
        a = arr[k]
        p_lo, p_hi = np.percentile(a, [lo_p, hi_p])
        stretched[k] = np.clip((a - p_lo) / (p_hi - p_lo + 1e-9), 0.0, 1.0)
    rgb = multichannel_to_rgb(
        stretched, cmaps=COMPOSITE_CMAPS[: arr.shape[0]], rescale_type="min_max"
    )[0]
    return (np.clip(np.asarray(rgb)[..., :3], 0.0, 1.0) * 255).astype(np.uint8)


def _cache_path(img_dir):
    """Local cache filename for a well-FOV directory's composite: keyed by
    patient / stage / well-FOV and the contrast setting, so changing the
    contrast invalidates stale renders automatically."""
    p = pathlib.Path(img_dir)
    lo_p, hi_p = COMPOSITE_CONTRAST
    key = f"{p.parent.parent.name}__{p.parent.name}__{p.name}__c{lo_p:g}-{hi_p:g}"
    return image_cache_dir / f"{key}.png"


def _load_image(img):
    """Load an image for display in the UMAP frame.

    A path to a well-FOV *directory* of channel tiffs is rendered as the
    multi-channel composite (same pipeline/colors as
    montage_utils.create_and_save_single_image_panel: middle z-slice per
    channel via `make_multi_channel_image_array`, colorized via
    `multichannel_to_rgb`). Renders are cached as PNGs under
    `image_cache_dir` and reloaded on later runs. An array or single-image
    path falls back to a contrast-stretched grayscale view.
    """
    is_path = isinstance(img, (str, bytes, pathlib.Path)) or hasattr(img, "__fspath__")
    if is_path and os.path.isdir(img):
        cache_fp = _cache_path(img)
        if cache_fp.exists():
            try:
                return np.asarray(Image.open(cache_fp).convert("RGB"))
            except Exception:
                pass  # corrupt/partial cache entry -> rebuild below
        composite = _composite_rgb(make_multi_channel_image_array(pathlib.Path(img)))
        try:
            Image.fromarray(composite).save(cache_fp)
        except Exception:
            pass  # caching is best-effort; never fail the plot over it
        return composite
    # fallback: single image path or raw array -> grayscale
    if is_path:
        from PIL import ImageSequence

        im = Image.open(img)
        frames = [np.asarray(f) for f in ImageSequence.Iterator(im)]
        arr = np.max(np.stack(frames, axis=0), axis=0) if len(frames) > 1 else frames[0]
    else:
        arr = np.asarray(img)
    if arr.ndim == 2 and arr.dtype != np.uint8:
        arr = arr.astype(np.float64)
        lo, hi = np.percentile(arr, [1, 99])
        if hi > lo:
            arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        else:
            arr = arr - arr.min()
            peak = arr.max()
            arr = arr / peak if peak else arr
        arr = (arr * 255).astype(np.uint8)
    return arr


def _side_counts(n, hw, hh):
    """Split n images over top/bottom/left/right to maximise the smallest spacing.
    Top/bottom rows include the corners; left/right columns sit between them."""
    best, best_sp = None, -1
    for t in range(1, n + 1):
        for b in range(max(1, t - 1), min(t + 1, n - t) + 1):
            rest = n - t - b
            if rest < 0:
                continue
            l, r = rest // 2 + rest % 2, rest // 2
            sp = min(
                2 * hw / (t - 1) if t > 1 else 4 * hw,
                2 * hw / (b - 1) if b > 1 else 4 * hw,
                2 * hh / (l + 1),
                2 * hh / (r + 1),
            )
            if sp > best_sp + 1e-12:
                best, best_sp = (t, b, l, r), sp
    return best, best_sp


def _rect_slots(center, hw, hh, counts):
    t, b, l, r = counts
    row = lambda k: np.linspace(-hw, hw, k) if k > 1 else np.array([0.0])
    col = lambda k: np.linspace(-hh, hh, k + 2)[1:-1]
    pts = (
        [(x, hh) for x in row(t)]
        + [(x, -hh) for x in row(b)]
        + [(-hw, y) for y in col(l)]
        + [(hw, y) for y in col(r)]
    )
    return np.array(pts) + center


def _per_side_counts(per_side):
    """`per_side` as (top / bottom, left / right) slot counts: an int gives the same
    count on every side, a pair (n_tb, n_lr) sets the rows and columns separately."""
    if np.isscalar(per_side):
        return int(per_side), int(per_side)
    n_tb, n_lr = per_side
    return int(n_tb), int(n_lr)


def _n_corner_slots(per_side):
    """Number of slots in the fixed corner layout: each side's count plus 4 corners."""
    n_tb, n_lr = _per_side_counts(per_side)
    return 2 * n_tb + 2 * n_lr + 4


def _rect_slots_corners(center, hw, hh, per_side):
    """`per_side` slots evenly along each side of the rectangle (see _per_side_counts),
    plus one at each corner."""
    n_tb, n_lr = _per_side_counts(per_side)
    xs = np.linspace(-hw, hw, n_tb + 2)
    ys = np.linspace(-hh, hh, n_lr + 2)[1:-1]
    pts = (
        [(x, hh) for x in xs]
        + [(x, -hh) for x in xs]
        + [(-hw, y) for y in ys]
        + [(hw, y) for y in ys]
    )
    return np.array(pts) + center


def _assign_slots(points, slots):
    """Minimum total Euclidean length matching. A min-length matching between two
    point sets never has crossing segments, so leader lines don't cross."""
    from scipy.optimize import linear_sum_assignment

    D = np.linalg.norm(points[:, None, :] - slots[None, :, :], axis=2)
    _, cols = linear_sum_assignment(D)
    return cols  # assignment[i] = slot index for point i


def _frame_geometry(
    embedding,
    n,
    img_size,
    gap,
    frame_aspect,
    scatter_fill,
    auto_grow,
    max_img_size,
    per_side=None,
    min_spacing=1.08,
):
    """Scaled scatter coordinates and the ring of image slots around them.
    Returns (E, center, slots, s, hw, hh): E is the embedding rescaled to fill the
    ring interior, s the crop side length, and hw / hh the ring half-extents."""
    # normalise embedding into a box, preserving aspect ratio
    lo, hi = embedding.min(0), embedding.max(0)
    scale = (hi - lo).max()
    E = (embedding - lo) / scale  # in [0, w] x [0, h], max(w, h) = 1
    w, h = E.max(0)
    center = np.array([w / 2, h / 2])

    # Ring half-extents. `frame_aspect` stretches the horizontal extent only.
    def ring_half(s):
        return (w / 2 + gap + s / 2) * frame_aspect, h / 2 + gap + s / 2

    # Fit the crop side length `s` to the number of crops: shrink it when the
    # ring is crowded, and (when `auto_grow`) enlarge it toward `max_img_size`
    # when few crops leave spare room -- so fewer crops are drawn larger.
    # `spacing` is the smallest gap between neighbouring slots; keep it
    # >= s * min_spacing so crops never touch (1 = edge to edge). A one-step look-ahead before growing
    # avoids oscillation at the crossover (spacing/s falls as s grows).
    s = img_size
    for _ in range(400):
        hw, hh = ring_half(s)
        counts, spacing = _ring_spacing(n, hw, hh, per_side)
        if spacing < s * min_spacing:
            s *= 0.95
            continue
        if auto_grow and s < max_img_size:
            s_next = min(s * 1.03, max_img_size)
            hw_n, hh_n = ring_half(s_next)
            _, spacing_n = _ring_spacing(n, hw_n, hh_n, per_side)
            if spacing_n >= s_next * min_spacing:
                s = s_next
                continue
        break
    s = min(s, max_img_size)
    hw, hh = ring_half(s)
    counts, _ = _ring_spacing(n, hw, hh, per_side)

    if per_side:
        slots = _rect_slots_corners(center, hw, hh, per_side)
    else:
        slots = _rect_slots(center, hw, hh, counts)

    # Scale the scatter outward so its edges nearly meet the image ring, removing
    # the white gap between the UMAP and the images (per-axis, so it also widens
    # the embedding to fill the wide frame).
    inner_hw = hw - s / 2 - gap
    inner_hh = hh - s / 2 - gap
    fx = scatter_fill * inner_hw / (w / 2) if w else 1.0
    fy = scatter_fill * inner_hh / (h / 2) if h else 1.0
    E = center + (E - center) * np.array([fx, fy])

    return E, center, slots, s, hw, hh


def _ring_spacing(n, hw, hh, per_side=None):
    """Smallest spacing between neighbouring slots: the fixed corner layout when
    `per_side` is given, otherwise the best split of n images over the four sides."""
    if per_side:
        n_tb, n_lr = _per_side_counts(per_side)
        return None, min(2 * hw / (n_tb + 1), 2 * hh / (n_lr + 1))
    return _side_counts(n, hw, hh)


def select_near_slots(
    embedding,
    images,
    per_side,
    density_quantile=0.75,
    k=20,
    lateral_weight=3.0,
    **geometry,
):
    """One real point per frame slot, close to its slot (so leader lines are short),
    with every pick from a different well-FOV. Each side slot "looks" straight
    inward: sideways offset along its side costs `lateral_weight` times more than
    depth, so picks spread along each edge instead of all crowding the nearest tip
    of the embedding; corner slots use plain distance. Isolated points (k-th
    neighbour farther than `density_quantile` of all points) and points without an
    image directory are not eligible."""
    n = _n_corner_slots(per_side)
    E, center, slots, s, hw, hh = _frame_geometry(
        embedding, n, per_side=per_side, **geometry
    )
    rel = slots - center
    on_tb = np.isclose(np.abs(rel[:, 1]), hh)  # top / bottom rows
    on_lr = np.isclose(np.abs(rel[:, 0]), hw)  # left / right columns
    kth = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E)[0][:, -1]
    has_image = {p: os.path.isdir(p) for p in pd.unique(images)}
    eligible = (kth <= np.quantile(kth, density_quantile)) & np.array(
        [has_image[p] for p in images]
    )
    while True:
        cand = np.flatnonzero(eligible)
        dx = np.abs(slots[:, None, 0] - E[cand][None, :, 0])
        dy = np.abs(slots[:, None, 1] - E[cand][None, :, 1])
        wx = np.where(on_tb & ~on_lr, lateral_weight, 1.0)[:, None]
        wy = np.where(on_lr & ~on_tb, lateral_weight, 1.0)[:, None]
        D = np.hypot(wx * dx, wy * dy)
        _, cols = linear_sum_assignment(D)
        picks = cand[cols]  # picks[j] = point for slot j
        seen, dup = {}, False
        for j in np.argsort(D[np.arange(n), cols]):  # nearest keeps its FOV
            if images[picks[j]] in seen:
                eligible[picks[j]] = False
                dup = True
            seen[images[picks[j]]] = j
        if not dup:
            return picks


def umap_with_image_frame(
    embedding,
    images,
    idx,
    labels=None,
    palette=None,
    img_size=0.11,  # image side length, relative to the larger side of the scatter box
    gap=0.015,  # space between the (scaled) scatter and the image ring
    frame_aspect=2.6,  # >1 widens the image ring horizontally -> wide/rectangular figure
    scatter_fill=0.95,  # scale the scatter out to fill this fraction of the ring interior
    auto_grow=True,  # enlarge crops toward `max_img_size` when few are shown
    max_img_size=0.30,  # cap on crop side length (relative) when auto-growing
    label_bar_loc="top",  # solid label-color bar inside each crop: "top"/"bottom"/None
    label_bar_frac=0.14,  # bar height as a fraction of the crop height
    point_size=6,
    figsize=(32, 13),
    cmap_images="gray",  # used for 2-D (grayscale) images
    legend_title="label",
    title_fontsize=30,
    legend_fontsize=17,
    show_index=False,
    title=None,
    ax=None,
    per_side=None,  # fixed layout: crops per side (int, or (top/bottom, left/right)) + corners
    loader=None,  # image loader (defaults to `_load_image`, the 2D composite)
    crop_labels=None,  # {point index: text} written beside each crop, on its outer side
    legend_x=1.005,  # legend's left edge, in axes fraction (move right to clear labels)
    legend_loc="right",  # "right": a column beside the frame; "bottom": rows under it
    legend_ncol=None,  # legend columns (default: 1, or 2 for > 14 labels; 6 at the bottom)
    crop_label_fontsize=15,
    min_spacing=1.08,  # slot spacing / crop size: lower packs the crops closer together
):
    loader = _load_image if loader is None else loader
    if idx is None:  # pick the real points that lie closest to the frame slots
        idx = select_near_slots(
            embedding,
            images,
            per_side=per_side,
            img_size=img_size,
            gap=gap,
            frame_aspect=frame_aspect,
            scatter_fill=scatter_fill,
            auto_grow=auto_grow,
            max_img_size=max_img_size,
        )
    idx = np.asarray(idx)
    n = len(idx)
    E, center, slots, s, hw, hh = _frame_geometry(
        embedding,
        n,
        img_size,
        gap,
        frame_aspect,
        scatter_fill,
        auto_grow,
        max_img_size,
        per_side,
        min_spacing,
    )

    pts = E[idx]
    assignment = _assign_slots(pts, slots)

    colors, lut = label_colors(labels, palette=palette)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # background scatter
    ax.scatter(
        E[:, 0],
        E[:, 1],
        s=point_size,
        linewidths=0,
        c=colors if colors is not None else "0.6",
        alpha=0.6,
        zorder=1,
    )
    # highlighted points
    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=point_size * 12,
        facecolors="none",
        edgecolors="k",
        linewidths=1.2,
        zorder=4,
    )

    for j, i in enumerate(idx):
        sx, sy = slots[assignment[j]]
        px, py = pts[j]

        # leader line ends at the image edge rather than its centre
        d = np.array([px - sx, py - sy])
        t = min(
            (s / 2) / abs(d[0]) if d[0] else np.inf,
            (s / 2) / abs(d[1]) if d[1] else np.inf,
        )
        ex, ey = sx + d[0] * t, sy + d[1] * t
        ax.plot([px, ex], [py, ey], color="0.35", lw=0.7, zorder=2)

        img = loader(images[i])
        # keep image aspect inside the s x s slot
        ih, iw = img.shape[:2]
        if iw >= ih:
            dw, dh = s / 2, s / 2 * ih / iw
        else:
            dw, dh = s / 2 * iw / ih, s / 2
        ax.imshow(
            img,
            extent=(sx - dw, sx + dw, sy - dh, sy + dh),
            cmap=cmap_images if img.ndim == 2 else None,
            interpolation="nearest" if max(ih, iw) < 64 else "antialiased",
            zorder=3,
        )

        edge = colors[i] if colors is not None else "0.3"
        # solid label-color bar inside the crop (top or bottom), so each crop's
        # category is readable even where the thin border is hard to see.
        if label_bar_loc and colors is not None:
            bar_h = 2 * dh * label_bar_frac
            bar_y = sy + dh - bar_h if label_bar_loc == "top" else sy - dh
            ax.add_patch(
                Rectangle(
                    (sx - dw, bar_y),
                    2 * dw,
                    bar_h,
                    fill=True,
                    facecolor=colors[i],
                    edgecolor="none",
                    alpha=0.95,
                    zorder=4.5,
                )
            )
        ax.add_patch(
            Rectangle(
                (sx - dw, sy - dh),
                2 * dw,
                2 * dh,
                fill=False,
                edgecolor=edge,
                lw=2.0,
                zorder=5,
            )
        )
        if crop_labels and int(i) in crop_labels:
            # outer side of the crop, so labels never run into neighbouring crops:
            # above the top row, below the bottom row, outside the side columns
            rel_x, rel_y = sx - center[0], sy - center[1]
            gap_ = s * 0.04
            if np.isclose(abs(rel_y), hh):  # top / bottom rows (corners included)
                top_row = rel_y > 0
                pos = (sx, sy + dh + gap_) if top_row else (sx, sy - dh - gap_)
                ha, va = "center", ("bottom" if top_row else "top")
            else:  # side columns
                right_col = rel_x > 0
                pos = (sx + dw + gap_, sy) if right_col else (sx - dw - gap_, sy)
                ha, va = ("left" if right_col else "right"), "center"
            ax.text(
                *pos,
                crop_labels[int(i)],
                ha=ha,
                va=va,
                fontsize=crop_label_fontsize,
                linespacing=1.15,
                zorder=6,
            )
        if show_index:
            ax.text(
                sx - dw,
                sy + dh,
                f" {i}",
                ha="left",
                va="top",
                fontsize=11,
                zorder=6,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none", alpha=0.8),
            )

    if lut:
        handles = [
            plt.Line2D([], [], marker="o", ls="", color=c, label=str(l))
            for l, c in lut.items()
        ]
        if legend_loc == "bottom":
            # rows centred under the frame, below the bottom row's crop labels (their
            # most lines of crop_label_fontsize), so the frame can use the full width
            from matplotlib.transforms import offset_copy

            label_lines = [str(t).count("\n") + 1 for t in (crop_labels or {}).values()]
            n_lines = max(label_lines, default=0)
            below = n_lines * 1.15 * crop_label_fontsize + 12
            ncol = legend_ncol or 6
            where = dict(
                loc="upper center",
                bbox_to_anchor=(0.5, 0),
                bbox_transform=offset_copy(
                    ax.transAxes, fig=fig, y=-below, units="points"
                ),
            )
        else:
            ncol = legend_ncol or (1 if len(lut) <= 14 else 2)
            where = dict(loc="upper left", bbox_to_anchor=(legend_x, 1))
        ax.legend(
            handles=handles,
            **where,
            frameon=False,
            title=legend_title,
            fontsize=legend_fontsize,
            title_fontsize=legend_fontsize + 2,
            ncol=ncol,
            handletextpad=0.4,
            labelspacing=0.3,
            borderaxespad=0.0,
        )

    # Tight limits around the ring (small `pad`) to minimise surrounding white space.
    pad = s * 0.12
    ax.set_xlim(center[0] - hw - s / 2 - pad, center[0] + hw + s / 2 + pad)
    ax.set_ylim(center[1] - hh - s / 2 - pad, center[1] + hh + s / 2 + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=title_fontsize)
    return fig, ax


# ## Plots: wide image frame, colored by canonical palettes
#
# Each metadata scheme (tumor type, patient, treatment) is drawn twice, from two
# point-selection strategies shown in **separate plots**:
#
# - **closest** — k-means representatives: the real point closest to each cluster
#   centre (covers the dense, typical regions).
# - **farthest** — farthest-point sampling: spreads across the whole embedding,
#   including the extremes/outliers.
#
# Colors come from `utils/r_plot_themes.r` (`tumor_type_palette`,
# `tab20_palette_for_patients`, `custom_treatment_palette`). The frame is wide,
# the composites use the brighter `COMPOSITE_CONTRAST`, and renders are cached.
#
# ### Colored by tumor type

# In[8]:


# Two point selections, each shown in its own plot for every labeling.
# `N_IMAGES` is the knob for crop size: lower it to show fewer crops, and
# `umap_with_image_frame(auto_grow=True)` enlarges each crop toward
# `max_img_size` to use the freed space.
N_IMAGES = 44
idx_closest = select_points(
    embedding, n=N_IMAGES, method="kmeans"
)  # closest to cluster centres
idx_farthest = select_points(
    embedding, n=N_IMAGES, method="fps"
)  # farthest-point sampling
SELECTIONS = {"closest": idx_closest, "farthest": idx_farthest}


# Single multi-page PDF: every figure below is appended as one page.
# Opened here and closed in the final cell.
pdf = PdfPages(umap_image_frame_pdf_path)


def frame_and_save(labels_arr, palette, legend_title, scheme):
    """Draw the wide image frame for `labels_arr` from both the closest and
    farthest point selections, appending each as a page to the shared `pdf`."""
    for sel_name, sel_idx in SELECTIONS.items():
        fig, ax = umap_with_image_frame(
            embedding,
            images,
            sel_idx,
            labels=labels_arr,
            palette=palette,
            legend_title=legend_title,
            title=f"UMAP — {legend_title} ({sel_name} points)",
        )
        pdf.savefig(fig, dpi=600, bbox_inches="tight")
        plt.show()


frame_and_save(labels_tumor, tumor_type_palette, "Tumor type", "tumor_type")


# ### Colored by patient

# In[9]:


# frame_and_save(labels_patient, patient_palette, "Patient", "patient")


# ### Colored by treatment (drug + dose)

# In[10]:


frame_and_save(
    labels_treatment, treatment_palette, "Treatment (drug + dose)", "treatment"
)


# ### All three together
#
# One figure stacking the same wide image frame colored by tumor type, patient,
# and treatment. It reuses the shared points and the cached composites, so it adds
# little runtime on top of the individual plots.

# ## Figure 3 panel: single-cell crops beside their cells
#
# The figure panel shows 22 crops: 5 along the top and bottom rows and 4 down each side, plus one at each corner (`PER_SIDE`), so the ring is filled in, with points coloured by patient and each crop labelled with its patient and treatment.
#
# **Which cells.** A small integer program (`select_constrained`) picks cells that are typical of their area of the embedding. The embedding is split into islands (`crop_areas`): the large left cluster gives exactly `LEFT_CLUSTER_PICKS` crops, all from `LEFT_CLUSTER_PATIENT` (SARCO219_T2); every other island of at least `MIN_ISLAND_SIZE` cells gives at least one; the rest spread over `MAIN_SUBREGIONS` sub-regions of the main island, its middle included, at most one each. Every patient appears, and at least `MIN_PER_GROUP` MEK-inhibitor and cytotoxic crops come from at least `MIN_GROUP_AREAS` different areas. Only QC-passing, non-isolated cells of typical size and depth that touch at most `MAX_TOUCHING_NEIGHBORS` other cells (where cells touch, the segmentation splits them along straight lines that slice through a crop) are eligible, no two crops share a well-FOV, and crops too dim to show their cell (`MIN_CROP_SIGNAL`) are dropped and the selection solved again. `MANUAL_PICKS` adds hand-picked cells. Crops are then matched to frame slots so leader lines are short and never cross.
#
# **Each crop.** The cell seen top-down at full xy resolution: each channel's maximum-intensity projection over the cell's own z-range (`load_cell_mip`), in a square window around its xy bounding box plus `CROP_PAD`. Hot pixels are removed and the field-of-view background subtracted. Channels are pseudocoloured DNA cyan, ER green, AGP magenta and Mito red, each stretched over the cell's own signal (`BLACK_PCT` to `WHITE_PCT` percentile inside the cell's footprint, but never over less than `MIN_RANGE_FRAC` of the field of view's signal range, so a channel the cell lacks is not stretched into noise), with a soft shoulder above `TONE_KNEE` so the brightest pixels roll off instead of saturating. Neighbouring cells are dimmed to `NEIGHBOR_DIM` rather than cut away, the footprint of the cell's segmentation mask is outlined, and a `SCALE_BAR_UM` scale bar added. Each render is zoomed onto the footprint (`CROP_MARGIN` around it), so the cell fills a fixed-size square black tile and all crops are the same size. Renders are cached like the 2D composites.
#
# **3D comparison.** The same cells are also drawn as 3D isometric volume renders (`render_volume`) to `umap_image_frame_3d.pdf`, to show why the figure uses top-down views: the z-step is 10x the xy pixel size, so any tilted view mostly shows interpolated, axially blurred signal and the cells look smeared.
#

# In[11]:


# Voxel size (µm) from the microscopy metadata; every crop is resampled to cubic voxels of
# ISO_UM (twice the xy pixel size: 2 x 2 binning in xy, shape-preserving cubic (PCHIP)
# interpolation along z), so the renders are isotropic.
XY_UM = float(umap_data["Metadata_Microscopy_XResolutionUm"].iloc[0])
Z_UM = float(umap_data["Metadata_Microscopy_ZResolutionUm"].iloc[0])
assert umap_data["Metadata_Microscopy_XResolutionUm"].nunique() == 1
assert umap_data["Metadata_Microscopy_ZResolutionUm"].nunique() == 1
ISO_UM = 2 * XY_UM
CROP_PAD = 0.03  # margin around the cell, as a fraction of its extent along each axis
CROP_MARGIN = (
    0.06  # black spacing around the cell in its tile, as a fraction of its size
)
CHANNEL_GAIN = (
    0.9  # peak brightness per channel (screen blending keeps overlaps off white)
)
SIGNAL_GAMMA = 0.85  # < 1 lifts dim, thin structures
# Contrast: each channel is stretched over the cell's own projected signal, from its
# BLACK_PCT percentile inside the cell (out-of-focus haze, which a maximum projection piles
# up) to its WHITE_PCT percentile; never over less than MIN_RANGE_FRAC of the field of
# view's signal range, so a channel the cell lacks stays dark instead of becoming noise
BLACK_PCT = 25
WHITE_PCT = 99.5
MIN_RANGE_FRAC = 0.25
TONE_KNEE = 0.8  # tone curve: linear up to this output level, then a soft shoulder
RENDER_PX = 720  # render size (px) of the volume, about its printed size at 300 dpi
# Mask edges: the cell mask is smoothed with this Gaussian sigma (µm) before use, so the
# 1 µm z-steps of the segmentation do not show as stair-steps in tilted views
MASK_SMOOTH_UM = 0.5
# A crop is too dim to show when even its brightest channel's 99.9th percentile reaches
# less than this fraction of that channel's field-of-view signal range.
MIN_CROP_SIGNAL = 0.5
SCALE_BAR_UM = 5  # one scale-bar length on every crop, so the crops compare directly
RENDER_VERSION = 20  # bump to re-render cached 3D crops after changing render_volume
MIP_VERSION = 1  # bump to re-render cached top-down crops after changing render_mip
NEIGHBOR_DIM = (
    0.15  # top-down crops: brightness kept outside the cell (neighbours dimmed)
)
# Crops show cells touching at most this many others: where cells touch, the segmentation
# splits them along straight lines, which slice through the crop
MAX_TOUCHING_NEIGHBORS = 1
PER_SIDE = (5, 4)  # crops along the top / bottom rows and left / right columns (plus
# one per corner: 22 in all), so the rows fill their width like the columns do
# Pseudocolour per stain, keyed by the channel's wavelength in the file names
# (config/channel_mapping.toml): DNA 405 cyan, ER 488 green, AGP 555 magenta, Mito 640 red
CHANNEL_ORDER = ["405", "488", "555", "640"]
CHANNEL_RGB = np.array(
    [
        [0, 1, 1],  # 405 DNA: cyan
        [0, 1, 0],  # 488 ER: green
        [1, 0, 1],  # 555 AGP: magenta
        [1, 0, 0],  # 640 Mito: red
    ],
    dtype=np.float32,
)
# View: isometric corner view. The cube is turned VIEW_AZIM degrees about the optical
# axis, then tilted VIEW_TILT degrees off top-down (arctan(sqrt(2)) = 54.74), so one cube
# diagonal points at the viewer and three faces show equally.
VIEW_AZIM = 45
VIEW_TILT = -float(np.degrees(np.arctan(np.sqrt(2))))
# Outline colours: the crop's bounding box and the cell mask
BOX_RGB = (110, 110, 110)
MASK_OUTLINE_RGB = (225, 225, 225)
MASK_OUTLINE_ALPHA = 0.7  # outline blended over the cell, so it does not hide its edge

# Where the crops come from. The embedding splits into islands (DBSCAN, ISLAND_EPS): the
# large left cluster (almost all SARCO219_T2) gives exactly LEFT_CLUSTER_PICKS crops, all
# from LEFT_CLUSTER_PATIENT; every other island of at least MIN_ISLAND_SIZE cells gives at
# least one; the remaining crops spread over MAIN_SUBREGIONS k-means sub-regions of the
# main island (its middle included), at most one per sub-region.
ISLAND_EPS = 0.5
MIN_ISLAND_SIZE = 100
LEFT_CLUSTER_PATIENT = "SARCO219_T2"
LEFT_CLUSTER_PICKS = 2
MAIN_SUBREGIONS = 12
# Treatment groups the panel must sample, each from several areas
MEK_INHIBITORS = {"Binimetinib", "Mirdametinib", "Selumetinib", "Trametinib"}
CYTOTOXIC = {"Staurosporine", "Digoxin", "Trabectedin"}
MIN_PER_GROUP = 3  # at least this many crops per group ...
MIN_GROUP_AREAS = (
    3  # ... from at least this many different areas (islands / sub-regions)
)
# Cherry-picking: {slot index: umap_data row index} pins a cell to a slot (slots run
# along the top row left to right, then the bottom row, then the left and right columns).
MANUAL_PICKS = {}


def _channel_paths(img_dir):
    """The four stain tiffs of a well-FOV in CHANNEL_ORDER (DNA, ER, AGP, Mito)."""
    found = {p.stem.split("_")[-1]: p for p in pathlib.Path(img_dir).glob("*.tif*")}
    missing = [ch for ch in CHANNEL_ORDER if ch not in found]
    if missing:
        raise FileNotFoundError(f"{img_dir} is missing channel(s) {missing}")
    return [found[ch] for ch in CHANNEL_ORDER]


def _mask_path(row):
    p = pathlib.Path(row["Metadata_raw_image_filepath"])
    return p.parent.parent / "segmentation_masks" / p.name / "cell_mask.tiff"


def load_cell_crop(row):
    """One cell, masked, as (channels, z, y, x): read from a cube centred on its nucleus
    that holds its whole bounding box, then trimmed to a cube around the extent of its mask
    plus CROP_PAD, so the cell fills the crop and every crop has the same shape. Only the
    cell's own voxels (its label in cell_mask.tiff) keep their signal. Background is
    subtracted, hot pixels removed, and the cube resampled to isotropic ISO_UM voxels
    (2 x 2 binning in xy, then PCHIP interpolation along z: cubic but without the
    overshoot of a cubic spline, so no bright halos or dark rings); parts of the cube outside
    the image are zero-padded so every crop stays a full cube. Also returns each
    channel's field-of-view signal range (99.95th percentile above background), which
    keeps a channel the cell lacks from being stretched into noise, and the soft cell
    mask (z, y, x) on the same grid."""
    from scipy.interpolate import PchipInterpolator
    from scipy.ndimage import gaussian_filter, median_filter

    cx, cy = (
        row["Metadata_Location_Nuclei_CenterX"],
        row["Metadata_Location_Nuclei_CenterY"],
    )
    cz = row["Metadata_Location_Nuclei_CenterZ"]
    # half-size (µm) reaching the farthest edge of the cell's bounding box, in any axis
    reach_um = max(
        XY_UM
        * max(
            cx - row["Metadata_Location_Cell_MinX"],
            row["Metadata_Location_Cell_MaxX"] - cx,
        ),
        XY_UM
        * max(
            cy - row["Metadata_Location_Cell_MinY"],
            row["Metadata_Location_Cell_MaxY"] - cy,
        ),
        Z_UM
        * max(
            cz - row["Metadata_Location_Cell_MinZ"],
            row["Metadata_Location_Cell_MaxZ"] - cz,
        ),
    ) * (1 + CROP_PAD)
    half, half_z = int(np.ceil(reach_um / XY_UM)), int(np.ceil(reach_um / Z_UM))
    cxi, cyi, czi = int(round(cx)), int(round(cy)), int(round(cz))
    win = (2 * half_z + 1, 2 * half + 1, 2 * half + 1)

    def read_window(path):
        """The cube window of a (z, y, x) tiff, zero-padded where it leaves the image,
        plus the full pages it was read from (for field-of-view statistics)."""
        with tifffile.TiffFile(path) as tif:
            n_z = len(tif.pages)
            z0, z1 = max(0, czi - half_z), min(n_z, czi + half_z + 1)
            pages = tif.asarray(key=range(z0, z1))
        if pages.ndim == 2:
            pages = pages[None]
        ny, nx = pages.shape[1:]
        y0, y1 = max(0, cyi - half), min(ny, cyi + half + 1)
        x0, x1 = max(0, cxi - half), min(nx, cxi + half + 1)
        out = np.zeros(win, dtype=np.float32)
        out[
            z0 - (czi - half_z) : z1 - (czi - half_z),
            y0 - (cyi - half) : y1 - (cyi - half),
            x0 - (cxi - half) : x1 - (cxi - half),
        ] = pages[:, y0:y1, x0:x1]
        return out, pages

    b = int(round(ISO_UM / XY_UM))  # xy binning factor

    def resample(a):
        nz_, ny_, nx_ = a.shape[0], a.shape[1] // b * b, a.shape[2] // b * b
        a = a[:, :ny_, :nx_].reshape(nz_, ny_ // b, b, nx_ // b, b).mean(axis=(2, 4))
        z_in = np.arange(nz_) * Z_UM
        z_out = np.arange(int(round(z_in[-1] / ISO_UM)) + 1) * ISO_UM
        return PchipInterpolator(z_in, a, axis=0)(z_out).astype(np.float32)

    mask_win, _ = read_window(_mask_path(row))
    # soft mask: fraction of each resampled voxel inside the cell, smoothed so its surface
    # is rounded rather than stepped, then sharpened back to a narrow edge
    mask = resample((mask_win == row["Metadata_Object_ObjectID"]).astype(np.float32))
    mask = gaussian_filter(mask, MASK_SMOOTH_UM / ISO_UM)
    mask = np.clip((mask - 0.3) / 0.4, 0, 1)
    chans, fov_range = [], []
    for p in _channel_paths(row["Metadata_raw_image_filepath"]):
        crop, pages = read_window(p)
        sample = pages[:, ::4, ::4].astype(np.float32)  # every 16th pixel is plenty
        bg = np.percentile(sample, 50)  # field-of-view background
        fov_range.append(np.percentile(sample, 99.95) - bg)
        crop = np.clip(median_filter(crop, size=(1, 3, 3)) - bg, 0, None)  # hot pixels
        chans.append(np.clip(resample(crop), 0, None) * mask)
    vol = np.stack(chans)
    # zoom in: trim to a cube centred on the cell's own extent (where its mask is), its
    # side the cell's longest axis plus CROP_PAD on each end, so every crop is a cube and
    # draws the same isometric box; zero-padded where the cube leaves the window
    inside = np.nonzero(mask > 0.05)
    if inside[0].size:
        lo_ = np.array([ix.min() for ix in inside])
        hi_ = np.array([ix.max() + 1 for ix in inside])
        side = int(np.ceil((hi_ - lo_).max() * (1 + 2 * CROP_PAD)))
        start = (lo_ + hi_) // 2 - side // 2
        src = tuple(
            slice(max(0, a), min(n, a + side)) for a, n in zip(start, mask.shape)
        )
        dst = tuple(slice(s_.start - a, s_.stop - a) for s_, a in zip(src, start))
        cube = np.zeros((vol.shape[0], side, side, side), dtype=np.float32)
        cube_mask = np.zeros((side, side, side), dtype=np.float32)
        cube[(slice(None), *dst)] = vol[(slice(None), *src)]
        cube_mask[dst] = mask[src]
        vol, mask = cube, cube_mask
    return vol, np.array(fov_range), mask


def tone_map(proj, fov_range, inside, gamma=SIGNAL_GAMMA):
    """Pseudocolour a (channels, y, x) projection to RGB in [0, 1]. Each channel is
    stretched over the cell's own signal (BLACK_PCT to WHITE_PCT inside `inside`, over at
    least MIN_RANGE_FRAC of the field's range); a soft shoulder above TONE_KNEE compresses
    the brightest pixels instead of clipping; channels are screen-blended in CHANNEL_RGB."""
    c, H_, W_ = proj.shape
    chans = np.zeros_like(proj, dtype=np.float32)
    for k in range(c):
        vals = proj[k][inside] if inside.any() else proj[k].ravel()
        lo_k = np.percentile(vals, BLACK_PCT)
        hi_k = max(np.percentile(vals, WHITE_PCT), lo_k + MIN_RANGE_FRAC * fov_range[k])
        x = np.clip((proj[k] - lo_k) / (hi_k - lo_k + 1e-9), 0, None) ** gamma
        x = np.where(
            x < TONE_KNEE,
            x,
            TONE_KNEE
            + (1 - TONE_KNEE) * (1 - np.exp(-(x - TONE_KNEE) / (1 - TONE_KNEE))),
        )
        chans[k] = CHANNEL_GAIN * x
    rgb = 1 - np.prod(
        1 - chans.reshape(c, -1)[:, :, None] * CHANNEL_RGB[:c, None, :], axis=0
    )
    return rgb.reshape(H_, W_, 3)


def _rotation(elev, azim):
    """Rotation for a view turned `azim` degrees about the optical (z) axis, then
    tilted `elev` degrees about the horizontal (x) axis. Acts on (x, y, z) vectors."""
    a, e = np.deg2rad(azim), np.deg2rad(elev)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(e), -np.sin(e)], [0, np.sin(e), np.cos(e)]])
    return rx @ rz


def render_volume(
    vol,
    fov_range,
    mask=None,
    label=None,
    elev=VIEW_TILT,
    azim=VIEW_AZIM,
    out=RENDER_PX,
    hi_pct=99.5,
    cue_min=0.7,
    gamma=SIGNAL_GAMMA,
):
    """Maximum-intensity render of an isotropic crop, seen from an isometric corner view
    (VIEW_AZIM, VIEW_TILT): nearer voxels brighter (depth cue), the crop's bounding box as
    a wireframe (front edges only: in an isometric view the hidden ones meet them at the
    centre; drawn behind the cell), the outline of the cell `mask` (its projected
    silhouette), channels screen-blended so overlaps do not clip to white, and a strip
    underneath holding any `label` lines (centred) and a µm scale bar (right). The view
    is zoomed onto the cell's silhouette, so the cell fills a fixed `out` x `out` tile. Returns RGBA uint8."""
    from scipy.ndimage import affine_transform, binary_erosion

    c, nz, ny, nx = vol.shape
    R = _rotation(elev, azim)
    ctr = np.array([nx / 2, ny / 2, nz / 2])
    corners = (
        np.array([[x, y, z] for x in (0, nx) for y in (0, ny) for z in (0, nz)]) - ctr
    )
    qc = (R @ corners.T).T
    lo3, hi3 = qc.min(0), qc.max(0)
    lo, hi = lo3[:2], hi3[:2]
    # the rotated volume is resampled (trilinear) on a grid aligned with the view, one
    # sample per voxel, so every ray is evenly sampled: no gaps or striping at any angle.
    # Array axes are (z, y, x) and view axes (depth, screen y, screen x), hence the flips.
    flip = np.eye(3)[::-1]
    shape = tuple((np.ceil(hi3 - lo3).astype(int) + 1)[::-1])
    matrix = flip @ R.T @ flip
    offset = flip @ (R.T @ lo3 + ctr)
    depth = np.linspace(0, 1, shape[0])  # 0 = nearest
    cue = (cue_min + (1 - cue_min) * (1 - depth)).astype(np.float32)[:, None, None]
    proj = np.stack(
        [
            (
                affine_transform(vol[k], matrix, offset, output_shape=shape, order=1)
                * cue
            ).max(0)
            for k in range(c)
        ]
    )
    H_, W_ = proj.shape[1:]
    # the mask's silhouette in this view (projection grid), for contrast and outline
    sil_proj = (
        affine_transform(mask, matrix, offset, output_shape=shape, order=1).max(0)
        if mask is not None
        else proj.max(0)
    )
    inside = sil_proj > 0.5
    rgb = Image.fromarray(
        (tone_map(proj, fov_range, inside, gamma) * 255).astype(np.uint8)
    )
    # zoom: the view is framed on the cell's silhouette (plus CROP_MARGIN on each side), not
    # on the whole bounding box, so the cell fills its tile; the tile is a fixed `out` x `out`
    # square, its lower `strip` px holding the label lines and scale bar
    labels = list(label or [])
    line = int(out * 0.13)
    strip = (len(labels) + 1) * line
    view_w, view_h = out, out - strip
    ys, xs = np.nonzero(inside) if inside.any() else np.mgrid[0:H_, 0:W_].reshape(2, -1)
    span_x, span_y = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    sc_out = min(view_w / span_x, view_h / span_y) / (
        1 + 2 * CROP_MARGIN
    )  # px per voxel
    # top-left of the view, in projection-grid (voxel) units
    ox = (xs.min() + xs.max() + 1) / 2 - view_w / sc_out / 2
    oy = (ys.min() + ys.max() + 1) / 2 - view_h / sc_out / 2
    full = (int(round(W_ * sc_out)), int(round(H_ * sc_out)))
    at = (int(round(-ox * sc_out)), int(round(-oy * sc_out)))
    render = Image.new("RGB", (view_w, view_h), (0, 0, 0))
    render.paste(rgb.resize(full, Image.LANCZOS), at)
    # bounding box wireframe, behind the cell; only front edges (the farthest corner's
    # edges are hidden; depth 0 = nearest). Edges leaving the view are clipped.
    box = Image.new("RGB", (view_w, view_h), (0, 0, 0))
    draw = ImageDraw.Draw(box)
    pc = (qc[:, :2] - lo - np.array([ox, oy])) * sc_out
    far = np.argmax(qc[:, 2])
    for i in range(8):
        for j in range(i + 1, 8):
            box_edge = np.sum(np.abs(corners[i] - corners[j]) > 1e-6) == 1
            if box_edge and far not in (i, j):
                draw.line(
                    [tuple(pc[i]), tuple(pc[j])], fill=BOX_RGB, width=max(1, out // 360)
                )
    px = np.asarray(box).copy()
    rp = np.asarray(render)
    np.maximum(px, rp, out=px)
    if mask is not None:
        # the mask's silhouette in this view, on the same zoomed grid: the cell sits in
        # front of the box lines, and its outline is drawn
        sil_img = Image.new("L", (view_w, view_h), 0)
        sil_img.paste(
            Image.fromarray((sil_proj * 255).astype(np.uint8)).resize(
                full, Image.BILINEAR
            ),
            at,
        )
        sil = np.asarray(sil_img) > 127
        px[sil] = rp[sil]
        edge = sil & ~binary_erosion(sil, iterations=max(1, out // 360))
        px[edge] = (
            MASK_OUTLINE_ALPHA * np.array(MASK_OUTLINE_RGB)
            + (1 - MASK_OUTLINE_ALPHA) * px[edge]
        ).astype(np.uint8)
    # black strip underneath: label lines (centred) and scale bar (bottom line, right;
    # orthographic view, so screen-parallel lengths are true at any depth)
    tile = Image.new("RGB", (out, out), (0, 0, 0))
    tile.paste(Image.fromarray(px), (0, 0))
    font = ImageFont.truetype(
        str(
            pathlib.Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        ),
        int(line * 0.62),
    )
    draw = ImageDraw.Draw(tile)
    for k, text_ in enumerate(labels):
        draw.text(
            (out // 2, view_h + line * k + line // 2),
            text_,
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
    bar_um = SCALE_BAR_UM
    bar_px = bar_um / ISO_UM * sc_out
    x1, yb = out - int(out * 0.05), out - line // 2
    bar_h = max(2, out // 120)
    draw.rectangle([x1 - bar_px, yb - bar_h, x1, yb + bar_h], fill=(255, 255, 255))
    draw.text(
        (x1 - bar_px - out // 45, yb),
        f"{bar_um:g} µm",
        fill=(255, 255, 255),
        font=font,
        anchor="rm",
    )
    return np.asarray(tile.convert("RGBA"))


crop_signal_path = image_cache_dir / "cell_crop_signal.parquet"


def _crop_key(i, kind="mip"):
    """Cache key of the `kind` ("mip": top-down, "volume": 3D isometric) render of row i."""
    row = umap_data.iloc[int(i)]
    p = pathlib.Path(row["Metadata_raw_image_filepath"])
    tag = (
        f"cellmip__v{MIP_VERSION}"
        if kind == "mip"
        else f"cellvolume__v{RENDER_VERSION}"
    )
    return f"{p.parent.parent.name}__{p.name}__obj{int(row['Metadata_Object_ObjectID'])}__{tag}"


def load_cell_mip(row):
    """One cell seen top-down at full xy resolution (no binning): each channel's
    maximum-intensity projection over the cell's own z-range, in a square window around
    its xy bounding box (plus CROP_PAD on each side), zero-padded where it leaves the
    image. Hot pixels are removed and the field-of-view background subtracted. Returns the
    projection (channels, y, x), each channel's field-of-view signal range (99.95th
    percentile above background) and the cell's footprint (its mask projected over z)."""
    from scipy.ndimage import median_filter

    x0b, x1b = (
        int(row["Metadata_Location_Cell_MinX"]),
        int(row["Metadata_Location_Cell_MaxX"]),
    )
    y0b, y1b = (
        int(row["Metadata_Location_Cell_MinY"]),
        int(row["Metadata_Location_Cell_MaxY"]),
    )
    z0b, z1b = (
        int(row["Metadata_Location_Cell_MinZ"]),
        int(row["Metadata_Location_Cell_MaxZ"]),
    )
    half = int(np.ceil(max(x1b - x0b, y1b - y0b) * (1 + 2 * CROP_PAD) / 2))
    cx, cy = (x0b + x1b) // 2, (y0b + y1b) // 2

    def read_window(path, reduce):
        """The cell's z-range of a (z, y, x) tiff reduced over z by `reduce`, in the
        square window (zero-padded outside the image), plus the pages it was read from."""
        with tifffile.TiffFile(path) as tif:
            n_z = len(tif.pages)
            pages = tif.asarray(key=range(max(0, z0b), min(n_z, z1b + 1)))
        if pages.ndim == 2:
            pages = pages[None]
        ny, nx = pages.shape[1:]
        y0, y1 = max(0, cy - half), min(ny, cy + half + 1)
        x0, x1 = max(0, cx - half), min(nx, cx + half + 1)
        out = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float32)
        out[
            y0 - (cy - half) : y1 - (cy - half), x0 - (cx - half) : x1 - (cx - half)
        ] = reduce(pages[:, y0:y1, x0:x1])
        return out, pages

    footprint, _ = read_window(
        _mask_path(row), lambda a: (a == row["Metadata_Object_ObjectID"]).any(0)
    )
    chans, fov_range = [], []
    for p in _channel_paths(row["Metadata_raw_image_filepath"]):
        mip, pages = read_window(p, lambda a: a.max(0))
        sample = pages[:, ::4, ::4].astype(np.float32)  # every 16th pixel is plenty
        bg = np.percentile(sample, 50)  # field-of-view background
        fov_range.append(np.percentile(sample, 99.95) - bg)
        chans.append(np.clip(median_filter(mip, size=3) - bg, 0, None))  # hot pixels
    return np.stack(chans), np.array(fov_range), footprint > 0


def render_mip(
    mip, fov_range, footprint, label=None, out=RENDER_PX, gamma=SIGNAL_GAMMA
):
    """Top-down render of a cell's maximum-intensity projection: channels pseudocoloured
    by tone_map (contrast set inside the cell's footprint), pixels outside the cell dimmed
    to NEIGHBOR_DIM (not cut away, so the cell's edge is not sliced), the footprint's
    outline, and a strip underneath holding any `label` lines (centred) and a µm scale bar
    (right). The view is zoomed onto the footprint (CROP_MARGIN around it), so the cell
    fills a fixed `out` x `out` tile. Returns RGBA uint8."""
    from scipy.ndimage import binary_erosion, gaussian_filter

    H_, W_ = footprint.shape
    rgb = tone_map(mip, fov_range, footprint, gamma)
    weight = NEIGHBOR_DIM + (1 - NEIGHBOR_DIM) * gaussian_filter(
        footprint.astype(np.float32), 2
    )
    rgb = Image.fromarray((rgb * weight[..., None] * 255).astype(np.uint8))
    labels = list(label or [])
    line = int(out * 0.13)
    strip = (len(labels) + 1) * line
    view_w, view_h = out, out - strip
    ys, xs = (
        np.nonzero(footprint)
        if footprint.any()
        else np.mgrid[0:H_, 0:W_].reshape(2, -1)
    )
    span_x, span_y = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    sc_out = min(view_w / span_x, view_h / span_y) / (
        1 + 2 * CROP_MARGIN
    )  # px per pixel
    ox = (xs.min() + xs.max() + 1) / 2 - view_w / sc_out / 2
    oy = (ys.min() + ys.max() + 1) / 2 - view_h / sc_out / 2
    full = (int(round(W_ * sc_out)), int(round(H_ * sc_out)))
    at = (int(round(-ox * sc_out)), int(round(-oy * sc_out)))
    view = Image.new("RGB", (view_w, view_h), (0, 0, 0))
    view.paste(rgb.resize(full, Image.LANCZOS), at)
    sil_img = Image.new("L", (view_w, view_h), 0)
    sil_img.paste(
        Image.fromarray((footprint * 255).astype(np.uint8)).resize(
            full, Image.BILINEAR
        ),
        at,
    )
    sil = np.asarray(sil_img) > 127
    px = np.asarray(view).copy()
    edge = sil & ~binary_erosion(sil, iterations=max(1, out // 360))
    px[edge] = (
        MASK_OUTLINE_ALPHA * np.array(MASK_OUTLINE_RGB)
        + (1 - MASK_OUTLINE_ALPHA) * px[edge]
    ).astype(np.uint8)
    tile = Image.new("RGB", (out, out), (0, 0, 0))
    tile.paste(Image.fromarray(px), (0, 0))
    font = ImageFont.truetype(
        str(
            pathlib.Path(mpl.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        ),
        int(line * 0.62),
    )
    draw = ImageDraw.Draw(tile)
    for k, text_ in enumerate(labels):
        draw.text(
            (out // 2, view_h + line * k + line // 2),
            text_,
            fill=(255, 255, 255),
            font=font,
            anchor="mm",
        )
    bar_px = SCALE_BAR_UM / XY_UM * sc_out
    x1, yb = out - int(out * 0.05), out - line // 2
    bar_h = max(2, out // 120)
    draw.rectangle([x1 - bar_px, yb - bar_h, x1, yb + bar_h], fill=(255, 255, 255))
    draw.text(
        (x1 - bar_px - out // 45, yb),
        f"{SCALE_BAR_UM:g} µm",
        fill=(255, 255, 255),
        font=font,
        anchor="rm",
    )
    return np.asarray(tile.convert("RGBA"))


def _crop_signals():
    if crop_signal_path.exists():
        df = pd.read_parquet(crop_signal_path)
        return dict(zip(df["key"], df["signal"]))
    return {}


def render_cell(i, kind="mip"):
    """Render (and cache) the cell in umap_data row i, top-down ("mip") or as a 3D
    isometric volume ("volume"); returns its signal: the largest, over channels, of the
    crop's 99.9th percentile as a fraction of that channel's field-of-view signal range.
    Signals are kept in cell_crop_signal.parquet."""
    key = _crop_key(i, kind)
    signals = _crop_signals()
    cache_fp = image_cache_dir / f"{key}.png"
    if key in signals and cache_fp.exists():
        return signals[key]
    row = umap_data.iloc[int(i)]
    if kind == "mip":
        img, fov_range, footprint = load_cell_mip(row)
        vals = [
            img[k][footprint] if footprint.any() else img[k].ravel()
            for k in range(len(img))
        ]
        Image.fromarray(render_mip(img, fov_range, footprint)).save(cache_fp)
    else:
        img, fov_range, mask = load_cell_crop(row)
        vals = list(img)
        Image.fromarray(render_volume(img, fov_range, mask=mask)).save(cache_fp)
    signal = float(
        max(np.percentile(v, 99.9) / (fov_range[k] + 1e-9) for k, v in enumerate(vals))
    )
    signals[key] = signal
    pd.DataFrame({"key": list(signals), "signal": list(signals.values())}).to_parquet(
        crop_signal_path
    )
    return signal


def cell_mip_loader(i):
    """Image loader for umap_with_image_frame (images = row indices): the cached
    top-down render of the cell in umap_data row i."""
    render_cell(i, "mip")
    return np.asarray(
        Image.open(image_cache_dir / f"{_crop_key(i, 'mip')}.png").convert("RGBA")
    )


def cell_volume_loader(i):
    """Image loader for umap_with_image_frame (images = row indices): the cached 3D
    isometric volume render of the cell in umap_data row i (comparison figure only)."""
    render_cell(i, "volume")
    return np.asarray(
        Image.open(image_cache_dir / f"{_crop_key(i, 'volume')}.png").convert("RGBA")
    )


def crop_areas(embedding):
    """Area of every cell: its island (DBSCAN on the embedding), with the main island
    split into MAIN_SUBREGIONS k-means sub-regions. Returns (area, left_area, islands):
    area ids per cell (-1 = noise / islands under MIN_ISLAND_SIZE), the left cluster's
    area id, and the area ids of the other islands that must each give a crop."""
    from sklearn.cluster import DBSCAN

    island = DBSCAN(eps=ISLAND_EPS, min_samples=15).fit_predict(embedding)
    ids, sizes = np.unique(island[island >= 0], return_counts=True)
    big = ids[sizes >= MIN_ISLAND_SIZE]
    main = ids[np.argmax(sizes)]
    patient = umap_data["Metadata_Biology_PatientTumor"].to_numpy()
    left = max(
        (i for i in big if i != main),
        key=lambda i: np.sum((island == i) & (patient == LEFT_CLUSTER_PATIENT)),
    )
    area = np.where(np.isin(island, big), island, -1)
    in_main = island == main
    sub = KMeans(
        n_clusters=MAIN_SUBREGIONS, n_init=10, random_state=RNG_SEED
    ).fit_predict(embedding[in_main])
    area[in_main] = island.max() + 1 + sub
    return area, left, [i for i in big if i not in (main, left)]


def select_constrained(embedding, per_side, k=20, density_quantile=0.75, per_area=60):
    """One cell per frame slot, chosen jointly (integer program): each pick is a typical
    cell of its area (close to the area's centre), subject to: exactly
    LEFT_CLUSTER_PICKS from the left cluster, all from LEFT_CLUSTER_PATIENT; at least
    one from every other island; at most one per main-island sub-region; every patient
    at least once; at least MIN_PER_GROUP MEK-inhibitor and cytotoxic picks, each group
    from at least MIN_GROUP_AREAS areas; one cell per well-FOV; MANUAL_PICKS included.
    Eligible cells pass QC (have a parent organoid), are not isolated in the embedding,
    have a typical size and depth, touch at most MAX_TOUCHING_NEIGHBORS other cells, and
    have an image directory and cell mask. Slots are
    assigned afterwards by umap_with_image_frame (shortest, non-crossing leader lines)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    n = _n_corner_slots(per_side)
    df = umap_data
    area, left, islands = crop_areas(embedding)
    patient = df["Metadata_Biology_PatientTumor"].to_numpy()
    treatment = df["Metadata_Experiment_Treatment"].to_numpy()
    kth = (
        NearestNeighbors(n_neighbors=k + 1)
        .fit(embedding)
        .kneighbors(embedding)[0][:, -1]
    )
    size = df["Metadata_Location_Cell_MaxX"] - df["Metadata_Location_Cell_MinX"]
    depth = df["Metadata_Location_Cell_MaxZ"] - df["Metadata_Location_Cell_MinZ"]
    touching = df[
        "Metadata_Neighbors_CellNoChannelNeighborsNeighborsCountAdjacent"
    ].to_numpy(float)
    has_files = {
        p: os.path.isdir(p) and _mask_path({"Metadata_raw_image_filepath": p}).exists()
        for p in pd.unique(images)
    }
    eligible = (
        (area >= 0)
        & (kth <= np.quantile(kth, density_quantile))
        & ~df["Metadata_cqc_missing_parent_organoid"].to_numpy(bool)
        & size.between(*size.quantile([0.2, 0.9])).to_numpy()
        & depth.between(8, depth.quantile(0.9)).to_numpy()
        & (touching <= MAX_TOUCHING_NEIGHBORS)
        & np.array([has_files[p] for p in images])
        & ((area != left) | (patient == LEFT_CLUSTER_PATIENT))
    )
    eligible[list(EXCLUDE)] = False
    # typicality: distance to the area's centre, relative to the area's spread
    cost_all = np.full(len(df), np.inf)
    cand = set(MANUAL_PICKS.values())
    for a in np.unique(area[area >= 0]):
        m = area == a
        centre = embedding[m].mean(0)
        d = np.linalg.norm(embedding - centre, axis=1)
        d /= np.median(d[m]) + 1e-9
        cost_all[m & eligible] = d[m & eligible]
        idx = np.flatnonzero(m & eligible)
        cand.update(idx[np.argsort(d[idx])][:per_area])
        for mask_ in (
            np.isin(treatment, list(MEK_INHIBITORS)),
            np.isin(treatment, list(CYTOTOXIC)),
        ):
            g = idx[mask_[idx]]
            cand.update(g[np.argsort(d[g])][:10])
    for p_ in np.unique(patient):  # every patient needs a few candidates
        idx = np.flatnonzero((patient == p_) & eligible)
        cand.update(idx[np.argsort(cost_all[idx])][:20])
    cand = np.array(sorted(cand))
    cost_all[list(MANUAL_PICKS.values())] = 0.0

    nv = len(cand)
    groups = {
        "MEKi": np.isin(treatment, list(MEK_INHIBITORS)),
        "cytotoxic": np.isin(treatment, list(CYTOTOXIC)),
    }
    areas_ = np.unique(area[cand])
    n_y = len(groups) * len(areas_)
    rows, lb, ub = [], [], []

    def add(coeffs, lo_, hi_):
        rows.append(coeffs)
        lb.append(lo_)
        ub.append(hi_)

    add({v: 1 for v in range(nv)}, n, n)  # n picks in all
    for v, c in enumerate(cand):
        if c in MANUAL_PICKS.values():
            add({v: 1}, 1, 1)
    by = lambda key: pd.Series(range(nv)).groupby(key).apply(list).to_dict()
    for vs in by(images[cand]).values():  # one cell per well-FOV
        add({v: 1 for v in vs}, 0, 1)
    for vs in by(patient[cand]).values():  # every patient
        add({v: 1 for v in vs}, 1, np.inf)
    for a, vs in by(area[cand]).items():
        if a == left:
            add({v: 1 for v in vs}, LEFT_CLUSTER_PICKS, LEFT_CLUSTER_PICKS)
        elif a in islands:
            add({v: 1 for v in vs}, 1, 2)
        else:  # main-island sub-region
            add({v: 1 for v in vs}, 0, 1)
    for g, mask_ in enumerate(groups.values()):
        add({v: 1 for v in range(nv) if mask_[cand[v]]}, MIN_PER_GROUP, np.inf)
        for r, a in enumerate(areas_):
            y = nv + g * len(areas_) + r
            coeffs = {v: -1 for v in range(nv) if mask_[cand[v]] and area[cand[v]] == a}
            coeffs[y] = 1
            add(coeffs, -np.inf, 0)  # an area counts only if a group pick lies in it
        add(
            {nv + g * len(areas_) + r: 1 for r in range(len(areas_))},
            MIN_GROUP_AREAS,
            np.inf,
        )
    A = lil_matrix((len(rows), nv + n_y))
    for i, coeffs in enumerate(rows):
        for v, a in coeffs.items():
            A[i, v] = a
    res = milp(
        np.concatenate([cost_all[cand], np.zeros(n_y)]),
        constraints=LinearConstraint(A.tocsr(), lb, ub),
        integrality=np.ones(nv + n_y),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"crop selection failed: {res.message}")
    return cand[res.x[:nv] > 0.5]


# Patients named with their tumour type, e.g. "SARCO219_T2 (MPNST)", in the panel's
# legend and crop labels
_tumor_type = dict(
    zip(
        umap_data["Metadata_Biology_PatientTumor"],
        umap_data["Metadata_Biology_TumorType"],
    )
)


def patient_with_tumor(patient, sep=" "):
    """`patient` followed by its tumour type in brackets, e.g. "SARCO219_T2 (MPNST)";
    `sep` goes between the two (a newline stacks them, for narrow crop labels)."""
    tumor = _tumor_type.get(patient)
    return patient if tumor is None or pd.isna(tumor) else f"{patient}{sep}({tumor})"


labels_patient_tumor = np.array([patient_with_tumor(p) for p in labels_patient])
patient_tumor_palette = {patient_with_tumor(p): c for p, c in patient_palette.items()}


def crop_label(i):
    """Patient, its tumour type and treatment (with dose) of the cell in umap_data row i,
    one per line, e.g. "NF0014_T1\n(pNF)\nTrametinib 10 µM"; written beside its crop.
    Three short lines, so neighbouring labels along the top and bottom rows never meet."""
    row = umap_data.iloc[int(i)]
    dose, unit = row["Metadata_Experiment_Dose"], row["Metadata_Experiment_Unit"]
    treatment = row["Metadata_Experiment_Treatment"]
    if not (pd.isna(dose) or pd.isna(unit)):
        treatment = f"{treatment} {dose:g} {str(unit).replace('uM', 'µM')}"
    patient = patient_with_tumor(row["Metadata_Biology_PatientTumor"], sep="\n")
    return f"{patient}\n{treatment}"


# crops packed nearly edge to edge (min_spacing) and as large as that allows
FRAME_3D = dict(
    per_side=PER_SIDE, frame_aspect=1.0, max_img_size=0.8, gap=0.0, min_spacing=1.02
)
# Select, render each pick, and drop picks too dim to show (MIN_CROP_SIGNAL); repeat
# until every crop shows its cell.
EXCLUDE = set()
while True:
    idx_panel = select_constrained(embedding, per_side=PER_SIDE)
    dim = {
        int(i)
        for i in idx_panel
        if i not in MANUAL_PICKS.values() and render_cell(i, "mip") < MIN_CROP_SIGNAL
    }
    if not dim:
        break
    EXCLUDE |= dim

print(
    umap_data.iloc[idx_panel][
        [
            "Metadata_Biology_PatientTumor",
            "Metadata_Experiment_Treatment",
            "Metadata_Experiment_Dose",
        ]
    ]
    .assign(row=idx_panel)
    .to_string()
)

# The figure panel (top-down crops), then the same cells as 3D isometric renders
for pdf_path, loader in [
    (umap_image_frame_sc_pdf_path, cell_mip_loader),
    (umap_image_frame_3d_pdf_path, cell_volume_loader),
]:
    with PdfPages(pdf_path) as pdf:
        fig, ax = umap_with_image_frame(
            embedding,
            np.arange(
                len(umap_data)
            ),  # images = row indices; the loader crops each cell
            idx_panel,
            labels=labels_patient_tumor,
            palette=patient_tumor_palette,
            legend_title="Patient",
            loader=loader,
            figsize=(26, 19),
            point_size=8,
            legend_fontsize=24,
            label_bar_loc=None,
            crop_labels={int(i): crop_label(i) for i in idx_panel},
            crop_label_fontsize=17,
            legend_loc="bottom",
            legend_ncol=4,
            **FRAME_3D,
        )
        pdf.savefig(fig, dpi=300, bbox_inches="tight")
        plt.show()
