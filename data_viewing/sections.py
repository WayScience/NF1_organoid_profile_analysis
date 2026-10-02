"""One render function per analysis type, grouped into one function per module."""

import re

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from background import subpanel_background
from data_io import (
    EDA_RESULTS,
    EXCLUDED_PATIENTS,
    Dataset,
    load_barcode_platemap,
    load_dataset,
    load_platemaps,
    parquet_columns,
    registry,
)
from palettes import (
    BIOLOGICAL_TERMS,
    TECHNICAL_TERMS,
    TERM_ORDER,
    TREATMENT_CLASS_DEFAULT,
    TREATMENT_CLASS_MAP,
    TREATMENT_MOA_MAP,
    TUMOR_TYPE_LOOKUP,
    humanize_label,
    palette_for,
)
from plots import (
    COL_TRACK_NAMES,
    NONE,
    ROW_TRACK_NAMES,
    annotated_significance_heatmap,
    apply_global_filters,
    correlation_heatmap,
    explorer,
    local_subset,
    missing_notice,
    platemap_heatmap,
    png_download,
    regroup_combos_by_identity,
    upset_plot,
)

Filters = dict[str, list[str]]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def dataset_section(
    datasets: dict[str, Dataset],
    key: str,
    filters: Filters,
    produced_by: str,
    directory,
    defaults_for=None,
    kinds=None,
    columns: tuple[str, ...] | None = None,
    prepare=None,
) -> None:
    """Pick one of several result files and explore it."""
    if not datasets:
        missing_notice(key.replace("_", " "), produced_by, directory)
        return
    label = st.selectbox(
        "Dataset", list(datasets), key=f"{key}_dataset", format_func=humanize_label
    )
    ds = datasets[label]
    df = load_dataset(str(ds.path), columns)
    if prepare:
        df = prepare(df, label)
    defaults = defaults_for(label, df) if defaults_for else {}
    explorer(df, f"{key}_{label}", filters, kinds=kinds, defaults=defaults, title=label)


def _first(df: pd.DataFrame, *names: str) -> str | None:
    return next((n for n in names if n in df.columns), None)


# ---------------------------------------------------------------------------
# 0.Overview -- static experiment design (not computed, just config/platemaps/*)
# ---------------------------------------------------------------------------
def platemap_section(filters: Filters) -> None:
    plates = load_platemaps()
    if not plates:
        missing_notice("platemaps", "(none)", "config/platemaps")
        return
    barcodes = load_barcode_platemap()
    name = st.selectbox(
        "Platemap", list(plates), key="overview_platemap", format_func=humanize_label
    )
    plate = plates[name]
    patients = sorted(
        barcodes.loc[barcodes["platemap_number"] == name, "patient_tumor"]
    )
    if patients:
        st.caption(f"Run on {len(patients)} patient(s): {', '.join(patients)}")
    palette = palette_for("treatment", sorted(plate["Treatment"].unique())) or {}
    fig = platemap_heatmap(plate, palette, title=humanize_label(name))
    st.plotly_chart(fig, width="stretch", key="overview_platemap_chart")
    png_download(fig, "overview_platemap", f"{name}_layout")


def drugs_section(filters: Filters) -> None:
    plates = load_platemaps()
    if not plates:
        missing_notice("platemaps", "(none)", "config/platemaps")
        return
    combined = pd.concat(plates.values(), ignore_index=True)
    drugs = (
        combined[combined["Treatment"] != "DMSO"]
        .drop_duplicates(["Treatment", "Dose", "Unit"])
        .assign(
            moa=lambda d: d["Treatment"].map(TREATMENT_MOA_MAP),
            drug_class=lambda d: (
                d["Treatment"].map(TREATMENT_CLASS_MAP).fillna(TREATMENT_CLASS_DEFAULT)
            ),
        )
        .sort_values(["moa", "Treatment", "Dose"])
        .rename(
            columns={
                "Treatment": "treatment",
                "Dose": "dose",
                "Unit": "unit",
                "drug_class": "class",
            }
        )
    )
    drugs = local_subset(
        drugs[["treatment", "class", "moa", "dose", "unit"]], "overview_drugs"
    )
    st.caption(f"{drugs['treatment'].nunique()} drugs across {len(plates)} platemap(s)")
    st.dataframe(drugs, width="stretch", hide_index=True)


def patients_section(filters: Filters) -> None:
    barcodes = load_barcode_platemap()
    if barcodes.empty:
        missing_notice("patients", "(none)", "config/platemaps/barcode_platemap.csv")
        return
    st.caption(f"{len(barcodes)} patient tumor samples")
    st.dataframe(
        barcodes.rename(
            columns={"platemap_number": "platemap", "tumor_type": "tumor manifestation"}
        ).sort_values("patient_tumor"),
        width="stretch",
        hide_index=True,
    )


OVERVIEW_SECTIONS = {
    "Platemap": platemap_section,
    "Drugs": drugs_section,
    "Patients & tumor manifestations": patients_section,
}


# ---------------------------------------------------------------------------
# 1.EDA
# ---------------------------------------------------------------------------
_DIMENSION_RE = re.compile(r"(?:^|_)(2D|3D)(?:_|$)")


def _dimension_of(key: str) -> str | None:
    """ "2D"/"3D" token in a dataset key (e.g. ``patient_specific_2D_maxproj_scfs_umap``
    -> "2D"), or None if the key doesn't carry one."""
    m = _DIMENSION_RE.search(key)
    return m.group(1) if m else None


def umap_section(filters: Filters) -> None:
    datasets = registry()["umap"]
    dims = sorted({d for d in (_dimension_of(k) for k in datasets) if d})
    if dims:
        dim = st.radio("Dimensionality", dims, horizontal=True, key="umap_dim")
        datasets = {k: v for k, v in datasets.items() if _dimension_of(k) == dim}
    if not datasets:
        missing_notice("umap", "1.EDA/scripts/0.generate_umap.py", EDA_RESULTS / "umap")
        return

    label = st.selectbox(
        "Dataset", list(datasets), key="umap_dataset", format_func=humanize_label
    )
    df = load_dataset(str(datasets[label].path))

    is_per_patient = label.startswith("patient_specific")
    individual = True
    if is_per_patient:
        individual = st.checkbox(
            "Individual patients (one plot per patient, each with its own axes "
            "-- combining them is misleading since each patient has its own "
            "independent UMAP fit)",
            value=True,
            key="umap_pp_individual",
        )

    facet_by_patient = is_per_patient and individual
    defaults = {
        "kind": "scatter",
        "x": "UMAP1",
        "y": "UMAP2",
        "color": (
            _first(df, "treatment")
            if facet_by_patient
            else _first(df, "treatment", "patient_tumor")
        ),
        "facet": "patient_tumor" if facet_by_patient else None,
    }
    explorer(
        df,
        f"umap_{label}_{'ind' if facet_by_patient else 'comb'}",
        filters,
        kinds=["scatter", "histogram", "heatmap"],
        defaults=defaults,
        title=label,
        free_facet_axes=facet_by_patient,
    )


def pca_section(filters: Filters) -> None:
    datasets = registry()["pca"]
    dims = sorted({d for d in (_dimension_of(k) for k in datasets) if d})
    if dims:
        dim = st.radio("Dimensionality", dims, horizontal=True, key="pca_dim")
        datasets = {k: v for k, v in datasets.items() if _dimension_of(k) == dim}
    if not datasets:
        missing_notice("PCA", "1.EDA/scripts/2.generate_pca.py", EDA_RESULTS / "pca")
        return
    label = st.selectbox(
        "Dataset", list(datasets), key="pca_dataset", format_func=humanize_label
    )
    path = datasets[label].path
    df = load_dataset(str(path))
    variance_path = path.with_name(
        path.name.replace("_embeddings", "_explained_variance")
    )
    variance = None
    if variance_path.exists():
        variance = pd.read_parquet(variance_path).iloc[0]
    x_col, y_col = st.columns(2)
    pcs = [c for c in df.columns if c.startswith("PC") and c[2:].isdigit()]
    x_pc = x_col.selectbox("X component", pcs, index=0, key="pca_x_pc")
    y_pc = y_col.selectbox(
        "Y component", pcs, index=min(1, len(pcs) - 1), key="pca_y_pc"
    )
    fig = explorer(
        df,
        f"pca_{label}",
        filters,
        kinds=["scatter", "histogram", "heatmap"],
        defaults={
            "kind": "scatter",
            "x": x_pc,
            "y": y_pc,
            "color": _first(df, "treatment", "patient_tumor"),
        },
        title=label,
    )
    if variance is not None:
        explained = variance.rename(lambda n: n.replace("_explained_variance", ""))
        st.caption(
            f"Explained variance: {x_pc} {explained.get(x_pc, float('nan')):.1%}, "
            f"{y_pc} {explained.get(y_pc, float('nan')):.1%}"
        )
        scree = px.bar(
            x=explained.index,
            y=explained.values,
            labels={"x": "Component", "y": "Explained variance ratio"},
            title="Scree plot",
        )
        subpanel_background("pca_scree")
        st.plotly_chart(scree, width="stretch", key=f"pca_scree_{label}")


def correlation_section(filters: Filters) -> None:
    corr_dir = EDA_RESULTS / "correlation"
    pair_files = (
        sorted(corr_dir.glob("*_correlation_pairs.parquet"))
        if corr_dir.exists()
        else []
    )
    matrix_files = (
        sorted(corr_dir.glob("*per_patient_correlation_matrices.parquet"))
        if corr_dir.exists()
        else []
    )
    if not pair_files and not matrix_files:
        missing_notice(
            "correlation matrices",
            "1.EDA/scripts/4.calculate_correlation_matrix.py",
            corr_dir,
        )
        return

    mode = st.radio(
        "Matrix type",
        ["Replicate/treatment-level (pairs)", "Per-patient single-cell"],
        horizontal=True,
        key="corr_mode",
    )
    subpanel_background(
        "corr_pairs" if mode.startswith("Replicate") else "corr_per_patient"
    )
    if mode.startswith("Replicate"):
        _pairs_heatmap(pair_files, filters)
    else:
        _per_patient_heatmap(matrix_files)


def _pairs_heatmap(pair_files, filters: Filters) -> None:
    if not pair_files:
        st.info("No `*_correlation_pairs.parquet` files found.")
        return
    pairs_path = st.selectbox(
        "Pairs file",
        pair_files,
        format_func=lambda p: humanize_label(p.stem),
        key="corr_pairs_file",
    )
    samples_path = pairs_path.with_name(pairs_path.name.replace("_pairs", "_samples"))
    if not samples_path.exists():
        st.info(f"Missing sample metadata file `{samples_path.name}`.")
        return
    pairs = pd.read_parquet(pairs_path)
    samples = load_dataset(str(samples_path))
    group_cols = [
        c for c in pairs.columns if c not in ("sample_i", "sample_j", "correlation")
    ]
    selection = {}
    cols = st.columns(len(group_cols))
    for col, name in zip(cols, group_cols):
        selection[name] = col.selectbox(
            name, sorted(pairs[name].unique()), key=f"corr_{name}"
        )
    pair_mask = np.ones(len(pairs), dtype=bool)
    sample_mask = np.ones(len(samples), dtype=bool)
    for name, value in selection.items():
        pair_mask &= (pairs[name] == value).to_numpy()
        sample_mask &= (samples[name] == value).to_numpy()
    pairs = pairs[pair_mask]
    samples = samples[sample_mask].reset_index(drop=True)

    n = int(samples["sample_index"].max()) + 1
    matrix = np.full((n, n), np.nan)
    matrix[pairs["sample_i"].to_numpy(), pairs["sample_j"].to_numpy()] = pairs[
        "correlation"
    ].to_numpy()
    matrix[pairs["sample_j"].to_numpy(), pairs["sample_i"].to_numpy()] = pairs[
        "correlation"
    ].to_numpy()

    filtered = local_subset(apply_global_filters(samples, filters), "corr")
    order_options = [
        c
        for c in filtered.columns
        if 1 < filtered[c].nunique() <= 100 and c not in group_cols
    ]
    order_by = st.selectbox(
        "Order/label samples by",
        order_options,
        index=0 if order_options else None,
        key="corr_order",
    )
    if order_by:
        filtered = filtered.sort_values(order_by, kind="stable")
    idx = filtered["sample_index"].to_numpy()
    if len(idx) == 0:
        st.warning("No samples left after subsetting.")
        return
    sub = matrix[np.ix_(idx, idx)]
    labels = (
        filtered[order_by].astype(str).to_numpy()
        if order_by
        else np.arange(len(idx)).astype(str)
    )
    labels = [f"{i}: {v}" for i, v in enumerate(labels)]

    c1, c2 = st.columns(2)
    cluster = c1.checkbox(
        "Hierarchical clustering (rows & cols)", key="corr_pairs_cluster"
    )
    track_options = [c for c in order_options if filtered[c].nunique() <= 60]
    tracks = c2.multiselect(
        "Color bars (rows & cols)", track_options, key="corr_pairs_tracks"
    )
    meta = (
        filtered[tracks].reset_index(drop=True) if tracks else pd.DataFrame(index=idx)
    )
    fig = correlation_heatmap(
        sub,
        labels,
        meta,
        [(t, t, t) for t in tracks],
        cluster,
        title=" | ".join(f"{k}={v}" for k, v in selection.items()),
    )
    st.plotly_chart(fig, width="content", key="corr_pairs_chart")
    png_download(fig, "corr_pairs", "correlation_heatmap")


def _per_patient_heatmap(matrix_files) -> None:
    if not matrix_files:
        st.info("No per-patient correlation matrix file found.")
        return
    df = pd.read_parquet(matrix_files[0])
    df = df[~df["patient"].astype(str).isin(EXCLUDED_PATIENTS)]
    c1, c2 = st.columns(2)
    variant = c1.selectbox(
        "Normalization variant",
        sorted(df["variant"].unique()),
        key="corr_variant",
        format_func=humanize_label,
    )
    patient = c2.selectbox(
        "Patient",
        sorted(df.loc[df["variant"] == variant, "patient"].unique()),
        key="corr_patient",
    )
    row = df[(df["variant"] == variant) & (df["patient"] == patient)].iloc[0]
    n = int(row["n_samples"])
    matrix = np.asarray(row["correlation"]).reshape(n, n)
    treatments = np.asarray(row["treatment"])
    order_treatments = st.multiselect(
        "Keep treatments", sorted(set(treatments)), key="corr_pp_treat"
    )
    keep = (
        np.isin(treatments, order_treatments)
        if order_treatments
        else np.ones(n, dtype=bool)
    )
    idx = np.where(keep)[0]
    idx = idx[np.argsort(treatments[idx], kind="stable")]

    c3, c4 = st.columns(2)
    cluster = c3.checkbox(
        "Hierarchical clustering (rows & cols)", key="corr_pp_cluster"
    )
    show_track = c4.checkbox("Color bar: treatment", value=True, key="corr_pp_track")
    meta = pd.DataFrame({"treatment": treatments[idx]})
    tracks = [("Treatment", "treatment", "treatment")] if show_track else []
    order_note = "clustered" if cluster else "sorted by treatment"
    fig = correlation_heatmap(
        matrix[np.ix_(idx, idx)],
        [str(i) for i in idx],
        meta,
        tracks,
        cluster,
        title=f"{patient} - {humanize_label(variant)} ({len(idx)} cells, {order_note})",
    )
    st.plotly_chart(fig, width="content", key="corr_pp_chart")
    png_download(fig, "corr_pp", "correlation_heatmap_per_patient")


def cell_counts_section(filters: Filters) -> None:
    def defaults(label, df):
        y = _first(df, "n_cells", "total_cells", "mean_cells_per_organoid")
        return {
            "kind": "box",
            "x": _first(df, "treatment"),
            "y": y,
            "color": _first(df, "patient_tumor"),
        }

    dataset_section(
        registry()["cell_counts"],
        "cell counts",
        filters,
        "1.EDA/scripts/7.generate_cell_counts.py",
        EDA_RESULTS / "cell_counts",
        defaults,
    )


def area_volume_section(filters: Filters) -> None:
    def defaults(label, df):
        return {
            "kind": "violin",
            "x": "treatment",
            "y": _first(df, "volume", "area"),
            "color": _first(df, "treatment"),
            "facet": _first(df, "patient_tumor"),
        }

    dataset_section(
        registry()["area_vs_volume"],
        "area and volume",
        filters,
        "1.EDA/scripts/17.calculate_area_volume_by_patient_treatment.py",
        EDA_RESULTS / "area_vs_volume",
        defaults,
    )


def neighbors_section(filters: Filters) -> None:
    def defaults(label, df):
        measure = next(
            (
                c
                for c in df.columns
                if c.startswith(("Organoid_", "Nuclei_"))
                and pd.api.types.is_numeric_dtype(df[c])
            ),
            None,
        )
        return {
            "kind": "box",
            "x": "treatment",
            "y": measure,
            "color": _first(df, "patient_tumor"),
        }

    dataset_section(
        registry()["neighbors"],
        "neighbors",
        filters,
        "1.EDA/scripts/13.calculate_neighbor_features.py",
        EDA_RESULTS / "neighbors",
        defaults,
    )


def intensity_section(filters: Filters) -> None:
    def defaults(label, df):
        return {
            "kind": "box",
            "x": "treatment",
            "y": "value",
            "color": _first(df, "patient", "patient_tumor"),
            "facet": _first(df, "channel"),
        }

    dataset_section(
        registry()["intensity"],
        "intensity",
        filters,
        "1.EDA/scripts/15.calculate_intensity_values.py",
        EDA_RESULTS / "intensity",
        defaults,
    )


def count_viability_section(filters: Filters) -> None:
    def defaults(label, df):
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        x = _first(df, "mean_cell_count", "mean_cells_per_organoid") or (
            numeric[0] if numeric else None
        )
        y = next((c for c in df.columns if "iab" in c.lower() and c in numeric), None)
        return {
            "kind": "scatter",
            "x": x,
            "y": y,
            "color": _first(df, "patient_tumor", "treatment"),
        }

    dataset_section(
        registry()["count_viability"],
        "count vs viability",
        filters,
        "1.EDA/scripts/10.calculate_count_viability_join.py",
        EDA_RESULTS / "count_viability",
        defaults,
    )


EDA_SECTIONS = {
    "UMAP": umap_section,
    "PCA": pca_section,
    "Correlation heatmaps": correlation_section,
    "Cell counts": cell_counts_section,
    "Area & volume": area_volume_section,
    "Neighbors": neighbors_section,
    "Intensity": intensity_section,
    "Count vs viability": count_viability_section,
}


# ---------------------------------------------------------------------------
# 3.viability_prediction_models
# ---------------------------------------------------------------------------
def _viability_dataset(name: str) -> Dataset | None:
    return registry()["viability_models"].get(name)


def model_performance_section(filters: Filters) -> None:
    datasets = {
        k: v
        for k, v in registry()["viability_models"].items()
        if k in ("combined_fold_metrics", "combined_summary_metrics")
    }

    def defaults(label, df):
        return {
            "kind": "box" if "fold" in label else "bar",
            "x": "profile_type",
            "y": "R2",
            "color": _first(df, "shuffle_status"),
            "facet": _first(df, "split_method"),
        }

    dataset_section(
        datasets,
        "model performance",
        filters,
        "3.viability_prediction_models/scripts/1.viability_prediction.py",
        "3.viability_prediction_models/model_results",
        defaults,
    )


def predicted_vs_actual_section(filters: Filters) -> None:
    ds = _viability_dataset("combined_predicted_viabilities")
    if ds is None:
        missing_notice(
            "predicted viabilities",
            "3.viability_prediction_models/scripts/1.viability_prediction.py",
            "3.viability_prediction_models/model_results",
        )
        return
    # the file has ~23k feature columns; read only the metadata and target columns
    keep = tuple(
        c
        for c in parquet_columns(str(ds.path))
        if c.startswith("Metadata_")
        or c in ("Actual_Viability", "Predicted_Viability", "min_max_viability")
    )
    df = load_dataset(str(ds.path), keep)
    explorer(
        df,
        "vpred",
        filters,
        kinds=["scatter", "box", "violin", "histogram", "heatmap"],
        defaults={
            "kind": "scatter",
            "x": "Actual_Viability",
            "y": "Predicted_Viability",
            "color": "patient_tumor",
            "facet": _first(df, "split_method"),
        },
        title="Predicted vs actual viability",
    )


def feature_importance_section(filters: Filters) -> None:
    ds = _viability_dataset("combined_feature_importances")
    if ds is None:
        missing_notice(
            "feature importances",
            "3.viability_prediction_models/scripts/1.viability_prediction.py",
            "3.viability_prediction_models/model_results",
        )
        return
    df = load_dataset(str(ds.path))
    df = apply_global_filters(df, filters)
    c1, c2, c3, c4 = st.columns(4)
    selectors = {}
    for col, name in zip(
        (c1, c2, c3, c4),
        ("profile_type", "split_method", "shuffle_status", "image_mode"),
    ):
        if name in df.columns:
            options = sorted(df[name].astype(str).unique())
            selectors[name] = col.multiselect(
                name,
                options,
                default=options[:1] if name == "profile_type" else options,
                key=f"fi_{name}",
                format_func=humanize_label,
            )
    for name, values in selectors.items():
        if values:
            df = df[df[name].astype(str).isin(values)]
    df = local_subset(df, "fi")
    if df.empty:
        st.warning("No rows left after subsetting.")
        return
    c5, c6 = st.columns(2)
    top_n = c5.slider("Top N features", 5, 100, 25, key="fi_topn")
    color = c6.selectbox(
        "Color by",
        [NONE, "split_method", "shuffle_status", "profile_type", "held_out_group"],
        key="fi_color",
        format_func=lambda v: v if v == NONE else humanize_label(v),
    )
    color = None if color == NONE or color not in df.columns else color
    top = df.groupby("feature")["importance"].mean().nlargest(top_n).index
    grouped = df[df["feature"].isin(top)]
    group = ["feature"] + ([color] if color else [])
    grouped = grouped.groupby(group, observed=True)["importance"].mean().reset_index()
    fig = px.bar(
        grouped,
        x="importance",
        y="feature",
        color=color,
        barmode="group",
        orientation="h",
        category_orders={"feature": list(top)},
        title=f"Top {top_n} features by mean importance",
    )
    fig.update_layout(height=max(450, 22 * top_n), template="plotly_white")
    st.plotly_chart(fig, width="stretch", key="fi_chart")
    png_download(fig, "fi", "feature_importances")


VIABILITY_SECTIONS = {
    "Model performance": model_performance_section,
    "Predicted vs actual": predicted_vs_actual_section,
    "Feature importances": feature_importance_section,
}


# ---------------------------------------------------------------------------
# 4.linear_modeling
# ---------------------------------------------------------------------------
def _normalize_lm_df(df: pd.DataFrame) -> pd.DataFrame:
    """Column/value fixups shared by every linear-modeling results table.

    The "_technical_model" files keep their raw Metadata_-prefixed schema, so
    ``canonicalize_columns`` maps their patient column to "patient_tumor" (not
    "patient") and their treatment-term rows still read the raw column name.
    """
    if "patient" not in df.columns and "patient_tumor" in df.columns:
        df = df.rename(columns={"patient_tumor": "patient"})
    if "term" in df.columns:
        df["term"] = df["term"].replace({"Metadata_Experiment_Treatment": "treatment"})
    if "tumor_type" not in df.columns and "patient" in df.columns:
        df["tumor_type"] = df["patient"].map(TUMOR_TYPE_LOOKUP)
    return df


def _linear_model_data(key: str, extra_defaults=None):
    datasets = registry()["linear_modeling"]
    if not datasets:
        missing_notice(
            "linear-model results",
            "4.linear_modeling/scripts/0.linear_modeling.py",
            "4.linear_modeling/results/linear_modeling",
        )
        return None, None
    label = st.selectbox(
        "Model results",
        list(datasets),
        key=f"{key}_dataset",
        format_func=humanize_label,
    )
    df = _normalize_lm_df(load_dataset(str(datasets[label].path)))
    if "pvalue_fdr" in df.columns:
        df["neg_log10_pvalue_fdr"] = -np.log10(df["pvalue_fdr"].clip(lower=1e-300))
        df["significant_fdr_0.05"] = np.where(
            df["pvalue_fdr"] < 0.05, "significant", "n.s."
        )
    if "coefficient" in df.columns:
        df["abs_coefficient"] = df["coefficient"].abs()
    return label, df


def lm_volcano_section(filters: Filters) -> None:
    label, df = _linear_model_data("lmvol")
    if df is None:
        return
    explorer(
        df,
        f"lmvol_{label}",
        filters,
        kinds=["scatter", "heatmap", "histogram"],
        defaults={
            "kind": "scatter",
            "x": "coefficient",
            "y": "neg_log10_pvalue_fdr",
            "color": "significant_fdr_0.05",
            "facet": _first(df, "term"),
        },
        title=f"{label}: volcano",
    )


def lm_effects_section(filters: Filters) -> None:
    label, df = _linear_model_data("lmeff")
    if df is None:
        return
    explorer(
        df,
        f"lmeff_{label}",
        filters,
        kinds=["box", "violin", "bar", "histogram", "scatter"],
        defaults={
            "kind": "box",
            "x": _first(df, "therapeutic_category", "treatment"),
            "y": "coefficient",
            "color": _first(df, "term"),
            "facet": _first(df, "patient"),
        },
        title=f"{label}: effect sizes",
    )


def lm_fit_section(filters: Filters) -> None:
    label, df = _linear_model_data("lmfit")
    if df is None:
        return
    explorer(
        df,
        f"lmfit_{label}",
        filters,
        kinds=["histogram", "box", "violin", "bar", "scatter"],
        defaults={
            "kind": "histogram",
            "x": "rsquared_adj",
            "color": _first(df, "Feature_type"),
            "facet": _first(df, "Compartment"),
        },
        title=f"{label}: model fit",
    )


# ported from 6.plot_variate_importance.r's `scopes` list: scope name -> its group columns
UPSET_SCOPES = {
    "all_models": [],
    "per_patient_treatment": ["patient", "treatment"],
    "per_patient": ["patient"],
    "per_treatment": ["treatment"],
    "per_treatment_tumor_type": ["treatment", "tumor_type"],
    "per_tumor_type": ["tumor_type"],
}

# profile x model_set -> the linear_modeling dataset that holds it
UPSET_PROFILE_DATASETS = {
    ("organoid", "original"): "organoid_norm",
    ("organoid", "technical"): "organoid_norm_technical_model",
    ("sc", "original"): "sc_norm",
    ("sc", "technical"): "sc_norm_technical_model",
}


def _build_membership(
    hits: pd.DataFrame, terms: list[str], index_cols: list[str]
) -> pd.DataFrame:
    """One row per (index_cols..., feature), one boolean column per term: True
    when any model at that index is a hit for that term. Ported from
    ``5.calculate_variate_importance.py``'s ``build_membership()``."""
    return (
        hits.loc[hits["term"].isin(terms), index_cols + ["term", "hit"]]
        .pivot_table(index=index_cols, columns="term", values="hit", aggfunc="max")
        .reindex(columns=terms)
        .fillna(False)
        .astype(bool)
    )


def _upset_combinations(membership: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    """Features per exact term combination, largest first. Ported from
    ``5.calculate_variate_importance.py``'s ``upset_combinations()``."""
    combos = (
        membership.groupby(terms, observed=True)
        .size()
        .rename("n_features")
        .reset_index()
    )
    combos = (
        combos.loc[combos[terms].any(axis=1)]
        .sort_values("n_features", ascending=False)
        .reset_index(drop=True)
    )
    if combos.empty:
        return combos.assign(combination=[], treatment_specific=[])
    combos["combination"] = combos[terms].apply(
        lambda r: " + ".join(t for t in terms if r[t]), axis=1
    )
    other = [t for t in terms if t != "treatment"]
    combos["treatment_specific"] = (
        combos["treatment"] & ~combos[other].any(axis=1)
        if "treatment" in terms and other
        else pd.Series(False, index=combos.index)
    )
    return combos.rename(columns={t: f"in_{t}" for t in terms})


def _set_sizes(membership: pd.DataFrame, terms: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"term": terms, "set_size": membership[terms].sum().to_numpy()})


def lm_upset_section(filters: Filters) -> None:
    """UpSet plots of which variates (model terms) a feature is a 'hit' for,
    computed on the fly (ported from ``5.calculate_variate_importance.py`` /
    ``6.plot_variate_importance.r``'s ``plot_upset()``): a feature is a hit for
    a term when a model has ``pvalue_fdr < threshold`` and ``coefficient >
    min`` for it. Below the plot, an overlap matrix shows how many features
    a chosen combination shares between every pair of groups in the scope
    (e.g. MPNST vs pNF for the tumor-type scope).
    """
    c1, c2 = st.columns(2)
    profile = c1.radio(
        "Profile", ["organoid", "sc"], horizontal=True, key="upset_profile"
    )
    model_set = c2.radio(
        "Model set", ["original", "technical"], horizontal=True, key="upset_model_set"
    )
    subpanel_background("lm_upset_profile")
    dataset_key = UPSET_PROFILE_DATASETS[(profile, model_set)]
    ds = registry()["linear_modeling"].get(dataset_key)
    if ds is None:
        missing_notice(
            f"{profile} ({model_set}) linear-model results",
            "4.linear_modeling/scripts/0.linear_modeling.py",
            "4.linear_modeling/results/linear_modeling",
        )
        return
    df = _normalize_lm_df(load_dataset(str(ds.path)))
    df = apply_global_filters(df, filters)
    if df.empty:
        st.warning("No rows left after subsetting.")
        return
    feature_meta = df.drop_duplicates("feature")[
        ["feature", "Feature_type", "Channel", "Compartment"]
    ]

    c3, c4 = st.columns(2)
    fdr_max = c3.slider("FDR threshold", 0.001, 0.25, 0.05, key="upset_fdr")
    coef_min = c4.slider(
        "Min coefficient (increases only, matches 5.calculate_variate_importance.py)",
        0.0,
        1.0,
        0.1,
        key="upset_coef",
    )
    hits = df.assign(hit=(df["pvalue_fdr"] < fdr_max) & (df["coefficient"] > coef_min))
    terms = [t for t in TERM_ORDER if t in hits["term"].unique()]

    scope = st.selectbox(
        "Scope", list(UPSET_SCOPES), key="upset_scope", format_func=humanize_label
    )
    group_cols = UPSET_SCOPES[scope]
    index_cols = group_cols + ["feature"]
    membership_all = _build_membership(hits, terms, index_cols)
    if membership_all.empty:
        st.warning("No hits at the current thresholds.")
        return

    group_label = ""
    if group_cols:
        cols = st.columns(len(group_cols))
        selection = {}
        for col_widget, name in zip(cols, group_cols):
            values = sorted(membership_all.index.get_level_values(name).unique())
            selection[name] = col_widget.selectbox(name, values, key=f"upset_{name}")
        mask = pd.Series(True, index=membership_all.index)
        for name, value in selection.items():
            mask &= membership_all.index.get_level_values(name) == value
        membership = membership_all[mask]
        membership.index = membership.index.get_level_values("feature")
        group_label = " | ".join(f"{k}={v}" for k, v in selection.items())
    else:
        membership = membership_all

    combos = _upset_combinations(membership, terms)
    sizes = _set_sizes(membership, terms)
    if combos.empty:
        st.warning("No term combinations for this selection.")
        return

    c5, c6 = st.columns([2, 1])
    top_n = c5.slider("Top N combinations", 5, 40, 25, key="upset_topn")
    term_group = c6.radio(
        "Variate group",
        ["Individual variates", "Grouped (Biological vs Technical)"],
        horizontal=True,
        key="upset_term_group",
    )
    if term_group == "Grouped (Biological vs Technical)":
        groups = {
            "Biological": [t for t in BIOLOGICAL_TERMS if t in terms],
            "Technical": [t for t in TECHNICAL_TERMS if t in terms],
        }
        groups = {name: g_terms for name, g_terms in groups.items() if g_terms}
        if len(groups) < 2:
            st.warning(
                "This model set doesn't have both biological and technical terms."
            )
            return
        combos, sizes = regroup_combos_by_identity(combos, groups)
        term_order = list(groups)
    else:
        term_order = terms

    n_total = len(membership)
    n_none = int((~membership[terms].any(axis=1)).sum())
    st.caption(f"{n_total:,} features ({n_none:,} not a hit for any term)")

    title = f"{profile} {model_set} model, {scope}" + (
        f" ({group_label})" if group_label else ""
    )
    fig = upset_plot(sizes, combos, term_order, top_n=top_n, title=title)
    if fig is None:
        st.warning("No term combinations to show.")
        return
    st.plotly_chart(fig, width="stretch", key="upset_chart")
    png_download(fig, "upset", f"upset_{profile}_{model_set}_{scope}")

    if not group_cols:
        return
    st.divider()
    st.caption(f"Feature overlap across {' x '.join(group_cols)} groups in this scope")
    group_keys = membership_all.index.to_frame(index=False)[
        group_cols
    ].drop_duplicates()
    group_keys["_label"] = (
        group_keys[group_cols[0]].astype(str)
        if len(group_cols) == 1
        else group_keys[group_cols].astype(str).agg(" | ".join, axis=1)
    )
    n_groups = len(group_keys)
    if n_groups < 2:
        st.info("Only one group in this scope; nothing to compare.")
        return
    if n_groups > 60:
        st.caption(
            f"{n_groups} groups -- rendering the full pairwise matrix, cell counts "
            "hidden for legibility (hover to read a value)."
        )

    combo_options = combos["combination"].tolist()
    chosen_combo = st.selectbox(
        "Combination to compare across groups", combo_options, key="upset_overlap_combo"
    )
    combo_terms = term_order if term_group == "Individual variates" else list(groups)
    pattern_row = combos.loc[combos["combination"] == chosen_combo].iloc[0]
    match_mask = pd.Series(True, index=membership_all.index)
    for t in combo_terms:
        want = bool(pattern_row[f"in_{t}"])
        col = (
            membership_all[t]
            if term_group == "Individual variates"
            else membership_all[
                [
                    c
                    for c in (
                        BIOLOGICAL_TERMS if t == "Biological" else TECHNICAL_TERMS
                    )
                    if c in terms
                ]
            ].any(axis=1)
        )
        match_mask &= col == want
    matched = membership_all[match_mask].index.to_frame(index=False)

    sets_by_group = {}
    for _, row in group_keys.iterrows():
        sub_mask = pd.Series(True, index=matched.index)
        for c in group_cols:
            sub_mask &= matched[c] == row[c]
        sets_by_group[row["_label"]] = set(matched.loc[sub_mask, "feature"])
    labels = group_keys["_label"].tolist()
    overlap = pd.DataFrame(
        [[len(sets_by_group[a] & sets_by_group[b]) for b in labels] for a in labels],
        index=labels,
        columns=labels,
    )
    overlap_fig = px.imshow(
        overlap,
        text_auto=n_groups <= 60,
        color_continuous_scale="Viridis",
        aspect="auto",
        labels=dict(color="shared features"),
        title=f"Shared features for '{chosen_combo}' across {' x '.join(group_cols)}",
    )
    overlap_fig.update_layout(
        template="plotly_white", height=max(400, 40 * n_groups + 150)
    )
    event = st.plotly_chart(
        overlap_fig,
        width="stretch",
        key="upset_overlap_chart",
        on_select="rerun",
        selection_mode="points",
    )
    png_download(overlap_fig, "upset_overlap", f"overlap_{scope}_{chosen_combo}")

    # a click on a cell updates the two pickers below; they also work by hand.
    # Widgets ignore `index=` once their key already holds a value, so a new
    # click has to be pushed into session_state *before* the widgets are
    # created below -- but only once per click, or it would fight the user's
    # own dropdown choice on every later, unrelated rerun.
    for key in ("upset_overlap_row", "upset_overlap_col"):
        if key in st.session_state and st.session_state[key] not in labels:
            del st.session_state[key]  # stale value from a previous scope

    points = event.selection.points if event else []
    clicked = (points[0]["y"], points[0]["x"]) if points else None
    if clicked and (clicked[0] not in labels or clicked[1] not in labels):
        clicked = None  # stale selection from a previous scope/combination
    if clicked and st.session_state.get("upset_overlap_last_click") != clicked:
        st.session_state["upset_overlap_row"] = clicked[0]
        st.session_state["upset_overlap_col"] = clicked[1]
        st.session_state["upset_overlap_last_click"] = clicked

    c7, c8 = st.columns(2)
    row_label = c7.selectbox("Row group", labels, key="upset_overlap_row")
    col_label = c8.selectbox("Column group", labels, key="upset_overlap_col")
    clicked_features = (
        sets_by_group[row_label]
        if row_label == col_label
        else sets_by_group[row_label] & sets_by_group[col_label]
    )
    header = (
        f"**{len(clicked_features):,} features** in **{row_label}**"
        if row_label == col_label
        else f"**{len(clicked_features):,} features** shared between "
        f"**{row_label}** and **{col_label}**"
    )
    st.markdown(header)
    if not clicked_features:
        return
    feat_df = feature_meta[feature_meta["feature"].isin(clicked_features)]
    with st.expander(f"Feature names ({len(feat_df):,})"):
        st.dataframe(feat_df, width="stretch", hide_index=True)

    melted = feat_df.melt(
        id_vars="feature",
        value_vars=["Compartment", "Channel", "Feature_type"],
        var_name="attribute",
        value_name="value",
    )
    melted["value"] = melted["value"].fillna("Other")
    counts = (
        melted.groupby(["attribute", "value"], observed=True)
        .size()
        .reset_index(name="count")
    )
    detail_fig = px.bar(
        counts,
        x="value",
        y="count",
        color="attribute",
        facet_col="attribute",
        title="Selected features by Compartment / Channel / Feature type",
    )
    detail_fig.update_xaxes(matches=None, showticklabels=True)
    detail_fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    detail_fig.update_layout(template="plotly_white", showlegend=False, height=400)
    st.plotly_chart(detail_fig, width="stretch", key="upset_overlap_detail_chart")
    png_download(
        detail_fig, "upset_overlap_detail", f"overlap_detail_{row_label}_{col_label}"
    )


def _annotation_legends(tracks: list[tuple[str, list[str], str]]) -> None:
    """Small colored-swatch legends for the heatmap's annotation strips (Plotly
    heatmaps have no native per-category legend for a discrete color track)."""
    cols = st.columns(len(tracks))
    for col, (title, categories, palette_key) in zip(cols, tracks):
        colors = palette_for(palette_key, categories) or {}
        swatches = "".join(
            f'<span style="display:inline-block;width:10px;height:10px;'
            f"background:{colors.get(c, '#cccccc')};margin-right:4px;"
            f'border:1px solid #999;"></span>{c}<br>'
            for c in categories
        )
        with col:
            st.markdown(f"**{title}**<br>{swatches}", unsafe_allow_html=True)


def lm_model_variates_section(filters: Filters) -> None:
    """Where is treatment significant on its own vs alongside covariates?

    Feature x (patient, treatment) yes/no heatmap -- ported from
    ``6.plot_variate_importance.r``'s ``cooccurrence_heatmap()`` -- with
    Patient, Tumor type and Treatment column annotations and a Feature type
    row annotation. A toggle switches the criterion between ``treatment``
    being a hit on its own vs together with at least one term from the
    chosen covariate group (biological or technical, in any combination);
    ``hit = pvalue_fdr < threshold & coefficient > min``, matching
    ``5.calculate_variate_importance.py``. A single-model drill-down table
    (its 'unique variate signature') is below the heatmap.
    """
    label, df = _linear_model_data("lmvar")
    if df is None:
        return
    df = apply_global_filters(df, filters)
    if df.empty:
        st.warning("No rows left after subsetting.")
        return

    c1, c2 = st.columns(2)
    fdr_max = c1.slider("FDR threshold", 0.001, 0.25, 0.05, key="lmvar_fdr")
    coef_min = c2.slider(
        "Min coefficient (increases only, matches 5.calculate_variate_importance.py)",
        0.0,
        1.0,
        0.1,
        key="lmvar_coef",
    )
    covariate_group = st.radio(
        "Covariate group (defines '+ group' below)",
        ["Biological", "Technical"],
        horizontal=True,
        key="lmvar_group",
    )
    group_terms = {"Biological": BIOLOGICAL_TERMS, "Technical": TECHNICAL_TERMS}[
        covariate_group
    ]
    group_terms = [
        t for t in group_terms if t != "treatment" and t in df["term"].unique()
    ]
    if not group_terms:
        st.warning(f"No {covariate_group.lower()} covariates in this dataset.")
        return
    criterion = st.radio(
        "Criterion",
        ["Treatment only", f"Treatment + {covariate_group.lower()}"],
        horizontal=True,
        key="lmvar_criterion",
    )

    hits = df.assign(hit=(df["pvalue_fdr"] < fdr_max) & (df["coefficient"] > coef_min))
    model_keys = ["patient", "treatment", "feature"]
    treatment_hit = (
        hits.loc[hits["term"] == "treatment"]
        .set_index(model_keys)["hit"]
        .rename("treatment_hit")
    )
    group_hit = (
        hits.loc[hits["term"].isin(group_terms)]
        .groupby(model_keys)["hit"]
        .any()
        .rename("group_hit")
    )
    models = pd.concat([treatment_hit, group_hit], axis=1).fillna(False).reset_index()
    # "Treatment only" excludes models where the covariate group is also a hit;
    # "Treatment + group" is every treatment hit, whether or not the group is
    # also significant -- a superset of "Treatment only", not an intersection
    models["treatment_only"] = models["treatment_hit"] & ~models["group_hit"]
    selected_col = (
        "treatment_only" if criterion == "Treatment only" else "treatment_hit"
    )

    model_meta = df.drop_duplicates(model_keys)[
        model_keys + ["drug", "Feature_type", "Channel", "Compartment", "tumor_type"]
    ]
    models = models.merge(model_meta, on=model_keys, how="left")
    sig = models.loc[models[selected_col]]
    n_hits = len(sig)
    st.caption(
        f"{n_hits:,} of {len(models):,} (patient, treatment, feature) models are "
        f"significant under **{criterion}**"
    )
    if sig.empty:
        st.warning("No models meet this criterion at the current thresholds.")
        return
    if n_hits > 20_000:
        st.info(
            f"{n_hits:,} significant models is a lot to render as a heatmap -- "
            "raise the FDR/coefficient thresholds to narrow it down."
        )
        return

    with st.expander("Heatmap: color bars & clustering", expanded=False):
        c6, c7 = st.columns(2)
        cluster_rows = c6.checkbox(
            "Cluster rows (features)", value=True, key="lmvar_cluster_rows"
        )
        cluster_cols = c6.checkbox(
            "Cluster columns (patient x treatment)",
            value=True,
            key="lmvar_cluster_cols",
        )
        row_tracks = c7.multiselect(
            "Row color bars",
            ROW_TRACK_NAMES,
            default=ROW_TRACK_NAMES,
            key="lmvar_row_tracks",
        )
        col_tracks = c7.multiselect(
            "Column color bars",
            COL_TRACK_NAMES,
            default=COL_TRACK_NAMES,
            key="lmvar_col_tracks",
        )

    fig = annotated_significance_heatmap(
        sig,
        criterion,
        legend_title=criterion,
        cluster_rows=cluster_rows,
        cluster_cols=cluster_cols,
        row_tracks=row_tracks,
        col_tracks=col_tracks,
    )
    if fig is None:
        st.warning("Nothing to plot.")
    else:
        st.plotly_chart(fig, width="content", key="lmvar_heatmap")
        png_download(fig, "lmvar_heatmap", f"{label}_{criterion.replace(' ', '_')}")
        all_legends = {
            "Patient": ("Patient", sorted(sig["patient"].unique()), "patient"),
            "Tumor type": (
                "Tumor type",
                sorted(sig["tumor_type"].dropna().unique()),
                "tumor_type",
            ),
            "Treatment": ("Treatment", sorted(sig["drug"].unique()), "treatment"),
            "Feature type": (
                "Feature type",
                sorted(sig["Feature_type"].fillna("Other").unique()),
                "feature_type",
            ),
            "Channel": (
                "Channel",
                sorted(sig["Channel"].fillna("Other").unique()),
                "channel",
            ),
            "Compartment": (
                "Compartment",
                sorted(sig["Compartment"].fillna("Other").unique()),
                "compartment",
            ),
        }
        shown = [all_legends[t] for t in col_tracks + row_tracks if t in all_legends]
        if shown:
            _annotation_legends(shown)

    st.divider()
    st.caption("Drill into one model's full term signature")
    subpanel_background("lm_variates_drilldown")
    c3, c4, c5 = st.columns(3)
    patient = c3.selectbox(
        "Patient", sorted(df["patient"].unique()), key="lmvar_patient"
    )
    df_p = df[df["patient"] == patient]
    treatment = c4.selectbox(
        "Treatment", sorted(df_p["treatment"].unique()), key="lmvar_treatment"
    )
    df_pt = df_p[df_p["treatment"] == treatment]
    feature = c5.selectbox(
        "Feature", sorted(df_pt["feature"].unique()), key="lmvar_feature"
    )
    model = df_pt[df_pt["feature"] == feature].sort_values("term")
    if model.empty:
        st.warning("No rows for this patient / treatment / feature combination.")
        return
    model = model.assign(
        hit=(model["pvalue_fdr"] < fdr_max) & (model["coefficient"] > coef_min)
    )
    unique_variates = model.loc[model["hit"], "term"].tolist()
    model["hit"] = model["hit"].astype(str)
    st.markdown(
        "**Unique variates (significant terms) for this model:** "
        + (
            ", ".join(f"`{t}`" for t in unique_variates)
            if unique_variates
            else "_none_"
        )
    )
    st.dataframe(
        model[["term", "coefficient", "pvalue_fdr", "hit", "rsquared", "rsquared_adj"]],
        width="stretch",
        hide_index=True,
    )


LINEAR_MODELING_SECTIONS = {
    "Volcano (effects vs significance)": lm_volcano_section,
    "Effect sizes": lm_effects_section,
    "Model fit": lm_fit_section,
    "UpSet (variate combinations)": lm_upset_section,
    "Unique variates per model": lm_model_variates_section,
}
