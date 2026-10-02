"""Color palettes and level orders ported from ``utils/r_plot_themes.r``.

Keep these in sync with the R file. The lookups are keyed by the (lower-cased)
column name so any plot colored/faceted by that column picks up the same colors
as the static R figures.
"""

import re

MOA_PALETTE = {
    "BRD4 inhibitor": "#93152A",
    "receptor tyrosine kinase inhibitor": "#BA3924",
    "tyrosine kinase inhibitor": "#D08543",
    "MEK1/2 inhibitor": "#A1961A",
    "IGF-1R inhibitor": "#9FC62A",
    "mTOR inhibitor": "#1FAD23",
    "PI3K inhibitor": "#32D06A",
    "PI3K and HDAC inhibitor": "#15937C",
    "HDAC inhibitor": "#24A5BA",
    "Apoptosis induction": "#438CD0",
    "DNA binding": "#1A24A1",
    "HSP90 inhibitor": "#532AC6",
    "histamine H1\nreceptor antagonist": "#AD1FA6",
    "Na+/K+ pump inhibitor": "#D03294",
    "Control": "#444444",
}

TREATMENT_MOA_MAP = {
    "DMSO": "Control",
    "Staurosporine": "Apoptosis induction",
    "Fimepinostat": "PI3K and HDAC inhibitor",
    "Copanlisib": "PI3K inhibitor",
    "Everolimus": "mTOR inhibitor",
    "Rapamycin": "mTOR inhibitor",
    "Sapanisertib": "mTOR inhibitor",
    "Vistusertib": "mTOR inhibitor",
    "Panobinostat": "HDAC inhibitor",
    "ARV-825": "BRD4 inhibitor",
    "Imatinib": "tyrosine kinase inhibitor",
    "Nilotinib": "tyrosine kinase inhibitor",
    "Cabozantinib": "receptor tyrosine kinase inhibitor",
    "Linsitinib": "IGF-1R inhibitor",
    "Binimetinib": "MEK1/2 inhibitor",
    "Mirdametinib": "MEK1/2 inhibitor",
    "Trametinib": "MEK1/2 inhibitor",
    "Selumetinib": "MEK1/2 inhibitor",
    "Onalespib": "HSP90 inhibitor",
    "Digoxin": "Na+/K+ pump inhibitor",
    "Ketotifen": "histamine H1\nreceptor antagonist",
    "Trabectedin": "DNA binding",
}

# ported from the EDA tables' Metadata_Experiment_Class / Metadata_class column;
# every drug is a small molecule except the DMSO control and the ARV-825 PROTAC
TREATMENT_CLASS_MAP = {
    "DMSO": "Control",
    "ARV-825": "PROTAC",
}
TREATMENT_CLASS_DEFAULT = "Small Molecule"

MOA_ORDER = [
    "Control",
    "BRD4 inhibitor",
    "receptor tyrosine kinase inhibitor",
    "tyrosine kinase inhibitor",
    "MEK1/2 inhibitor",
    "IGF-1R inhibitor",
    "mTOR inhibitor",
    "PI3K inhibitor",
    "PI3K and HDAC inhibitor",
    "HDAC inhibitor",
    "Apoptosis induction",
    "DNA binding",
    "HSP90 inhibitor",
    "histamine H1\nreceptor antagonist",
    "Na+/K+ pump inhibitor",
]

TREATMENT_PALETTE = {
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

TREATMENT_ORDER = [
    "DMSO",
    "Staurosporine",
    "Fimepinostat",
    "Copanlisib",
    "Everolimus",
    "Rapamycin",
    "Sapanisertib",
    "Vistusertib",
    "Imatinib",
    "Nilotinib",
    "Cabozantinib",
    "Linsitinib",
    "Panobinostat",
    "ARV-825",
    "Onalespib",
    "Digoxin",
    "Ketotifen",
    "Trabectedin",
    "Binimetinib",
    "Mirdametinib",
    "Trametinib",
    "Selumetinib",
]

DOSE_PALETTE = {"1": "#CFCFCF", "10": "#4D4D4D"}

CHANNEL_PALETTE = {
    "DNA": "#0000AB",
    "AGP": "#b1001a",
    "Mito": "#B000B0",
    "ER": "#00D55B",
    "BF": "#FFFF00",
    "NoChannel": "#B09FB0",
}

COMPARTMENT_PALETTE = {
    "Cell": "#B000B0",
    "Cells": "#B000B0",  # alias: some tables use the plural
    "Cytoplasm": "#00D55B",
    "Nuclei": "#0000AB",
    "Organoid": "#B09FB0",
}

# RColorBrewer "Paired" (8) entries 1, 2, 3, 5 and 8 as used in the R theme
FEATURE_TYPE_PALETTE = {
    "AreaSizeShape": "#A6CEE3",
    "Colocalization": "#1F78B4",
    "Granularity": "#B2DF8A",
    "Intensity": "#FB9A99",
    "Texture": "#FF7F00",
}

TUMOR_TYPE_LOOKUP = {
    "NF0014_T1": "cNF",
    "NF0014_T2": "pNF",
    "NF0016_T1": "pNF",
    "NF0018_T6": "cNF",
    "NF0021_T1": "cNF",
    "NF0030_T1": "Other",
    "NF0035_T1": "cNF",
    "NF0037_T1": "cNF",
    "NF0040_T1": "Other",
    "NF0055_T1": "pNF",
    "SARCO219_T2": "MPNST",
    "SARCO361_T1": "MPNST",
}

TUMOR_TYPE_PALETTE = {
    "cNF": "#1B9E77",
    "pNF": "#D95F02",
    "MPNST": "#7570B3",
    "Other": "#999999",
}

TUMOR_TYPE_ORDER = list(TUMOR_TYPE_PALETTE)

# tab20, assigned to patients in sorted order
TAB20_PALETTE = [
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a",
    "#d62728", "#ff9896", "#9467bd", "#c5b0d5", "#8c564b", "#c49c94",
    "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d",
    "#17becf", "#9edae5",
]  # fmt: skip

# ported from 6.plot_variate_importance.r's `term_colors` / `term_order`
TERM_ORDER = [
    "treatment",
    "cell_count",
    "organoid_count",
    "cell_per_organoid_count",
    "manhattan_distance_from_center",
    "cell_x_position",
    "cell_y_position",
    "cell_z_position",
    "cell_z_depth",
]

# user-defined split of TERM_ORDER for the UpSet "term group" toggle
BIOLOGICAL_TERMS = [
    "treatment",
    "cell_count",
    "organoid_count",
    "cell_per_organoid_count",
]
TECHNICAL_TERMS = [
    "manhattan_distance_from_center",
    "cell_x_position",
    "cell_y_position",
    "cell_z_position",
    "cell_z_depth",
]

TERM_COLORS = {
    "treatment": "#d95f02",
    "cell_count": "#1b9e77",
    "organoid_count": "#7570b3",
    "cell_per_organoid_count": "#e7298a",
    "manhattan_distance_from_center": "#66a61e",
    "cell_x_position": "#e6ab02",
    "cell_y_position": "#a6761d",
    "cell_z_position": "#666666",
    "cell_z_depth": "#1f78b4",
    # synthetic group-identity sets for the UpSet "combined groups" view
    "Biological": "#d95f02",
    "Technical": "#377eb8",
}

NORMALIZATION_VARIANT_LABELS = {
    "organoid_norm": "Organoid (ZEDProfiler)",
    "sammed_organoid_norm": "Organoid (SAM-med)",
    "sc_norm": "Single-cell (ZEDProfiler)",
    "sammed_sc_norm": "Single-cell (SAM-med)",
    "nucleocentric_morphem_norm": "Nucleocentric (MorphEm)",
    "sammed_nucleocentric_norm": "Nucleocentric (SAM-med)",
}

_PALETTES = {
    "treatment": TREATMENT_PALETTE,
    "moa": MOA_PALETTE,
    "dose": DOSE_PALETTE,
    "channel": CHANNEL_PALETTE,
    "compartment": COMPARTMENT_PALETTE,
    "feature_type": FEATURE_TYPE_PALETTE,
    "tumor_type": TUMOR_TYPE_PALETTE,
}
# columns whose legend/facet/axis levels are sorted alphabetically (case-insensitive)
_ALPHABETICAL = {"treatment", "patient_tumor", "held_out_group", "patient"}
_ORDERS = {
    "moa": MOA_ORDER,
    "tumor_type": TUMOR_TYPE_ORDER,
}
# levels that have no defined color (e.g. doses other than 1/10) get these
_FALLBACK = TAB20_PALETTE


def palette_for(column: str, levels: list[str]) -> dict[str, str] | None:
    """Color map for the given levels of ``column`` (None if no palette applies).

    ``levels`` may contain duplicates (e.g. one entry per row rather than per
    distinct value); assigning by position over a duplicate-laden list would
    make a level's color depend on how many times it repeats, so every branch
    below de-duplicates first.
    """
    name = column.lower()
    if name in ("patient_tumor", "held_out_group", "patient"):
        unique_levels = sorted(set(levels))
        return {
            lvl: TAB20_PALETTE[i % len(TAB20_PALETTE)]
            for i, lvl in enumerate(unique_levels)
        }
    base = _PALETTES.get(name)
    if base is None:
        return None
    colors = {}
    unknown = sorted({lvl for lvl in levels if lvl not in base})
    for i, lvl in enumerate(unknown):
        colors[lvl] = _FALLBACK[i % len(_FALLBACK)]
    colors.update({lvl: base[lvl] for lvl in levels if lvl in base})
    return colors


def order_for(column: str, levels: list[str]) -> list[str] | None:
    """Level order for ``column``: alphabetical for treatment/patient, R order for MOA/tumor type."""
    if column.lower() in _ALPHABETICAL:
        return sorted(levels, key=str.lower)
    order = _ORDERS.get(column.lower())
    if order is None:
        return None
    return [lvl for lvl in order if lvl in levels] + [
        lvl for lvl in levels if lvl not in order
    ]


# ---------------------------------------------------------------------------
# Dropdown labels: turn code-shaped names (file stems, profile_type values)
# into plain English. Multi-token keys are matched greedily, longest first,
# against underscore-split tokens; an empty mapped phrase drops the token
# (e.g. "profiles", "embeddings", which carry no information for a human).
# ---------------------------------------------------------------------------
_PHRASES: dict[tuple[str, ...], str] = {
    ("patient", "specific"): "Per-patient",
    ("max", "projection"): "Max projection",
    ("maxproj",): "Max projection",
    ("middle", "n", "slice"): "Middle N-slice",
    ("middle", "slice"): "Middle slice",
    ("midslice",): "Mid-slice",
    ("single", "cell", "agg"): "Single-cell (aggregated)",
    ("single", "cell"): "Single-cell",
    ("organoid", "norm", "technical", "model"): "Organoid (technical model)",
    ("sc", "norm", "technical", "model"): "Single-cell (technical model)",
    ("sammed", "organoid", "norm"): "Organoid (SAM-med)",
    ("sammed", "sc", "norm"): "Single-cell (SAM-med)",
    ("sammed", "nucleocentric", "norm"): "Nucleocentric (SAM-med)",
    ("nucleocentric", "morphem", "norm"): "Nucleocentric (MorphEm)",
    ("nucleocentric", "norm"): "Nucleocentric",
    ("organoid", "norm"): "Organoid (ZEDProfiler)",
    ("sc", "norm"): "Single-cell (ZEDProfiler)",
    ("organoid", "agg"): "Organoid (aggregated)",
    ("organoid", "consensus"): "Organoid (consensus)",
    ("organoid", "fs"): "Organoid (feature-selected)",
    ("sc", "agg"): "Single-cell (aggregated)",
    ("sc", "consensus"): "Single-cell (consensus)",
    ("sc", "fs"): "Single-cell (feature-selected)",
    ("scagg",): "Single-cell (aggregated)",
    ("scconsensus",): "Single-cell (consensus)",
    ("scfs",): "Single-cell (feature-selected)",
    ("sammed",): "SAM-med",
    ("fold", "metrics"): "fold metrics",
    ("summary", "metrics"): "summary metrics",
    ("predicted", "viabilities"): "predicted viabilities",
    ("feature", "importances"): "feature importances",
    ("correlation", "pairs"): "correlation pairs",
    ("correlation", "samples"): "correlation samples",
    ("cell", "counts"): "cell counts",
    ("held", "out", "group"): "held-out group",
    ("random", "split"): "Random split",
    ("lopo",): "LOPO (leave one patient out)",
    ("loto",): "LOTO (leave one treatment out)",
    ("combined",): "",
    ("profiles",): "",
    ("embeddings",): "",
    ("fs",): "feature-selected",
    ("agg",): "aggregated",
    ("norm",): "normalized",
    ("sc",): "single-cell",
    ("raw",): "(raw)",
    ("umap",): "UMAP",
    ("pca",): "PCA",
}

_DIMENSION_RE = re.compile(r"\d+[dD]$")


def humanize_label(name: str) -> str:
    """Turn a code-shaped dataset/profile name (a file stem or raw data value
    like ``organoid_norm`` or ``patient_specific_2D_maxproj_scfs_umap``) into a
    plain-English label for display in a dropdown. Falls back to title-casing
    unrecognized tokens, so it degrades gracefully on names not covered above.
    """
    if not isinstance(name, str) or not name:
        return str(name)
    if " " in name:
        return name  # already humanized (raw keys never contain spaces); idempotent
    # pipeline-step prefixes carry no meaning for a reader (e.g. the
    # "1.feature_selected_profiles_" in PCA/UMAP file stems)
    s = re.sub(r"\d+\.(feature_selected|aggregated|consensus)_profiles_", "", name)
    tokens = [t for t in s.split("_") if t]
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        for width in (4, 3, 2, 1):
            if i + width > n:
                continue
            key = tuple(t.lower() for t in tokens[i : i + width])
            if key in _PHRASES:
                phrase = _PHRASES[key]
                if phrase:
                    out.append(phrase)
                i += width
                break
        else:
            tok = tokens[i]
            out.append(
                tok.upper() if _DIMENSION_RE.fullmatch(tok) else tok.capitalize()
            )
            i += 1
    label = re.sub(r"\s+", " ", " ".join(out)).strip()
    if not label:
        return name
    return label[0].upper() + label[1:]
