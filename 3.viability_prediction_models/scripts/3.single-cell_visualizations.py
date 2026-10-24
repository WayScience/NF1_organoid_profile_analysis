#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
import tifffile
from microfilm.microplot import microshow
from notebook_init_utils import bandicoot_check, init_notebook

root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


start_time = time.time()


# In[3]:


# Raw 3D image stacks/segmentation masks live on the external Bandicoot
# mount (~/mnt/bandicoot/NF1_organoid_data/data/<patient>/...), not in this
# repo - see notebook_init_utils.bandicoot_check, which falls back to
# root_dir if the mount isn't present on this machine (in which case the
# image-loading cells below will just report missing files).
image_base_dir = bandicoot_check(
    pathlib.Path("~/mnt/bandicoot").expanduser().resolve(), root_dir
)

# Standard Cell Painting wavelength -> marker convention (Hoechst/DNA=405,
# ER=488, AGP=555, Mito=640). The raw images on disk are only labeled by
# wavelength, not marker name - this mapping is an assumption based on the
# four marker prefixes (DNA/ER/AGP/Mito) used throughout every feature name
# in this project. Verify against the actual imaging protocol if a crop's
# colors look inconsistent with its feature's marker.
CHANNEL_MARKER_MAP = {
    "405": ("DNA", "pure_blue"),
    "488": ("ER", "pure_green"),
    "555": ("AGP", "pure_red"),
    "640": ("Mito", "pure_magenta"),
}

# basicpy_zstack_images/ (illumination-corrected) was tried first, but its
# pixel values were compressed to a near-flat ~0-30 range at every location
# checked - unusable for a human-readable crop. zstack_images/ (raw,
# uncorrected) keeps the camera's native intensity range and is what's used
# for every crop below.
CROP_PAD_PX = 5

crop_output_dir = pathlib.Path(
    f"{root_dir}/3.viability_prediction_models/figures/single_cell_crops"
)
crop_output_dir.mkdir(parents=True, exist_ok=True)


# In[4]:


# Load the hits saved from visualize_model_results.ipynb:
#   - top_feature_extreme_objects.parquet: the single highest/lowest
#     specimen per (model, feature) - the "hits" to visualize here.
#   - top_feature_hits_with_cells.parquet: each hit already resolved down
#     to its identified cells (child cells for an organoid-level hit, or
#     the cell itself for an sc-level hit).
model_results_dir = pathlib.Path(
    root_dir, "3.viability_prediction_models/model_results"
)

hits_df = pq.read_table(
    model_results_dir / "top_feature_extreme_objects.parquet"
).to_pandas()
hit_cells_df = pq.read_table(
    model_results_dir / "top_feature_hits_with_cells.parquet"
).to_pandas()

print(f"Loaded {len(hits_df)} hits and {len(hit_cells_df)} hit-to-cell rows")
hits_df.head()


# In[5]:


def get_feature_compartment(feature: str) -> str:
    """A feature's compartment (Organoid/Cell/Cytoplasm/Nuclei) is its
    first underscore-delimited token."""
    return feature.split("_")[0]


def get_hit_bbox(hit: pd.Series):
    """hits_df now carries each hit's own min/max x/y/z bounding box
    (added upstream in visualize_model_results.ipynb, which reads it once
    from the QC parquet) - pull it straight from the row instead of
    re-reading that parquet a second time here."""
    return (
        (
            int(hit["Metadata_MinZ"]),
            int(hit["Metadata_MinY"]),
            int(hit["Metadata_MinX"]),
        ),
        (
            int(hit["Metadata_MaxZ"]),
            int(hit["Metadata_MaxY"]),
            int(hit["Metadata_MaxX"]),
        ),
    )


def crop_zstack(stack: np.ndarray, bbox, pad: int = CROP_PAD_PX) -> np.ndarray:
    """Crop a (Z, Y, X) stack to a bounding box, padded and clamped to the
    stack's own extent."""
    (z0, y0, x0), (z1, y1, x1) = bbox
    z0, y0, x0 = max(z0 - pad, 0), max(y0 - pad, 0), max(x0 - pad, 0)
    z1 = min(z1 + pad + 1, stack.shape[0])
    y1 = min(y1 + pad + 1, stack.shape[1])
    x1 = min(x1 + pad + 1, stack.shape[2])
    return stack[z0:z1, y0:y1, x0:x1]


def max_project(stack: np.ndarray) -> np.ndarray:
    """Max-intensity-project a (Z, Y, X) stack down to its (Y, X) plane -
    the standard way to collapse a z-stack into one image per channel/mask
    for display, keeping the brightest signal at each pixel across depth."""
    return stack.max(axis=0)


def build_channel_path(patient: str, well_fov: str, wavelength: str) -> pathlib.Path:
    return (
        image_base_dir
        / "data"
        / patient
        / "zstack_images"
        / well_fov
        / f"{well_fov}_{wavelength}.tif"
    )


def build_mask_path(patient: str, well_fov: str, compartment: str) -> pathlib.Path:
    return (
        image_base_dir
        / "data"
        / patient
        / "segmentation_masks"
        / well_fov
        / f"{compartment.lower()}_mask.tiff"
    )


# In[6]:


# For each of the first HITS_PER_COMPARTMENT hits per feature compartment:
# crop every channel's raw z-stack (plus the segmentation mask) to that
# specimen's bounding box, max-intensity-project each channel across z,
# composite them with microfilm, and save alongside a binary panel
# highlighting just this object's mask pixels (there can be other
# organoids/cells inside the same padded crop).
#
# Restricted to random_split hits only - lopo/loto hold out a whole
# patient/treatment group per fold, so their top features (and hence
# their hits) reflect cross-group generalization; random_split's 70/30
# holdout is the one comparable to a conventional single train/test model,
# so its hits are the ones worth visualizing in isolation here.
#
# hits_df sorts each profile type's features together, and within the
# single-cell profile type all "Cell_*" features sort before
# "Cytoplasm_*"/"Nuclei_*" - so grouping only by Metadata_profile_type (or
# a plain .head(N)) systematically missed the Cytoplasm/Nuclei
# compartments. Group by the feature's own compartment instead so
# Organoid, Cell, Cytoplasm, and Nuclei are all represented.
HITS_PER_COMPARTMENT = 3

random_split_hits = hits_df[hits_df["Metadata_split_method"] == "random_split"].copy()
random_split_hits["compartment"] = random_split_hits.apply(
    lambda hit: (
        "Organoid"
        if hit["Metadata_profile_type"] == "organoid_norm_sc_consensus_profiles"
        else get_feature_compartment(hit["feature"])
    ),
    axis=1,
)
hits_to_plot = random_split_hits.groupby("compartment", group_keys=False).head(
    HITS_PER_COMPARTMENT
)

for _, hit in hits_to_plot.iterrows():
    patient = hit["Metadata_Biology_PatientTumor"]
    well_fov = hit["Metadata_Experiment_WellFOV"]
    object_id = int(hit["Metadata_Object_ObjectID"])
    compartment = hit["compartment"]
    print(well_fov)
    bbox = get_hit_bbox(hit)

    mip_images, cmaps, marker_names = [], [], []
    for wavelength, (marker, cmap) in CHANNEL_MARKER_MAP.items():
        channel_path = build_channel_path(patient, well_fov, wavelength)
        if not channel_path.exists():
            break
        stack = tifffile.imread(channel_path)
        crop = crop_zstack(stack, bbox)
        mip_images.append(max_project(crop))
        cmaps.append(cmap)
        marker_names.append(marker)
    else:
        mask_path = build_mask_path(patient, well_fov, compartment)
        mask_highlight = None
        mask_note = None
        if mask_path.exists():
            mask_stack = tifffile.imread(mask_path)
            mask_crop = crop_zstack(mask_stack, bbox)
            mask_highlight = max_project(mask_crop == object_id)
            if not mask_highlight.any():
                # For sc-level compartments (Cell/Cytoplasm/Nuclei),
                # Metadata_Object_ObjectID in sc_flagged_outliers.parquet
                # doesn't reliably match the raw per-FOV *_mask.tiff label
                # values for every well - some wells carry what looks
                # like a well-level cumulative object count instead of
                # the FOV-local segmentation label (e.g. multiples of a
                # fixed well-level offset), so `mask == object_id` finds
                # no pixels at all. That mismatch originates upstream of
                # this repo (wherever sc_flagged_outliers.parquet is
                # built) and can't be corrected here - surface it loudly
                # instead of silently saving a blank mask panel.
                mask_note = (
                    f"WARNING: object {object_id} not found in {mask_path.name} for "
                    f"{patient} {well_fov} - Metadata_Object_ObjectID likely doesn't "
                    "match this well's raw mask labels (see note in this cell)."
                )
                print(mask_note)

        fig, axes = plt.subplots(
            1, 2 if mask_highlight is not None else 1, figsize=(8, 4)
        )
        axes = np.atleast_1d(axes)

        microshow(images=mip_images, cmaps=cmaps, ax=axes[0])
        axes[0].set_title(f"{'+'.join(marker_names)} (MIP)", fontsize=9)

        if mask_highlight is not None:
            if mask_note is not None:
                axes[1].imshow(
                    np.zeros_like(mask_highlight), cmap="gray", vmin=0, vmax=1
                )
                axes[1].set_title(
                    f"{compartment} mask: object {object_id} NOT FOUND\n(ObjectID/mask-label mismatch)",
                    fontsize=8,
                    color="red",
                )
            else:
                axes[1].imshow(mask_highlight, cmap="gray")
                axes[1].set_title(
                    f"{compartment} mask (object {object_id})", fontsize=9
                )
            axes[1].axis("off")

        fig.suptitle(
            f"{hit['feature']} = {hit['value']:.3g} ({hit['edge_direction']}) — "
            f"{patient} {well_fov} obj {object_id} — {hit['Metadata_split_method']}",
            fontsize=9,
        )
        fig.tight_layout()

        out_name = (
            f"{hit['Metadata_split_method']}_{compartment}_{hit['feature']}_"
            f"{hit['edge_direction']}_{patient}_{well_fov}_obj{object_id}.png"
        )
        fig.savefig(crop_output_dir / out_name, dpi=150)
        plt.show()
        plt.close(fig)
