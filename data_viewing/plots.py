"""Generic interactive plot explorer: subset, color and facet by any column."""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from data_io import GLOBAL_FILTER_COLUMNS, _natural_key
from palettes import TERM_COLORS, order_for, palette_for
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import pdist, squareform

NONE = "(none)"
MAX_FACETS = 30
CONTINUOUS_MIN_UNIQUE = 12
PNG_DPI = 600
CSS_DPI = 96  # plotly lays figures out at 96 px per inch
PLOT_KINDS = ["scatter", "box", "violin", "histogram", "bar", "heatmap"]


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------
def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]


def categorical_columns(df: pd.DataFrame, max_unique: int = 200) -> list[str]:
    """Columns usable for color/facet/subset: non-numeric or low-cardinality."""
    cols = []
    for c in df.columns:
        if pd.api.types.is_list_like(df[c].iloc[0]) if len(df) else False:
            continue
        n = df[c].nunique(dropna=True)
        is_numeric = pd.api.types.is_numeric_dtype(
            df[c]
        ) and not pd.api.types.is_bool_dtype(df[c])
        if (not is_numeric and n <= max_unique) or (
            is_numeric and n <= CONTINUOUS_MIN_UNIQUE
        ):
            cols.append(c)
    return cols


def _order_categories(df: pd.DataFrame, col: str) -> None:
    """Sort category levels naturally (so dose 0.1 < 1 < 10) for stable legends/facets."""
    if col not in df.columns or pd.api.types.is_numeric_dtype(df[col]):
        return
    levels = sorted(df[col].dropna().astype(str).unique(), key=_natural_key)
    df[col] = pd.Categorical(df[col].astype(str), categories=levels, ordered=True)


# ---------------------------------------------------------------------------
# Subsetting
# ---------------------------------------------------------------------------
def apply_global_filters(
    df: pd.DataFrame, filters: dict[str, list[str]]
) -> pd.DataFrame:
    """Apply sidebar filters wherever the column exists (silently skipped otherwise).

    ``patient_tumor`` resolves to the ``patient`` column when a table (e.g. a
    normalized linear-model result) only has the latter.
    """
    for col, selected in filters.items():
        if col not in df.columns and col == "patient_tumor" and "patient" in df.columns:
            col = "patient"
        if selected and col in df.columns:
            df = df[df[col].astype(str).isin(selected)]
    return df


def local_subset(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Section-level subset: keep/exclude values of any categorical column."""
    with st.expander("Subset rows by any metadata column", expanded=False):
        cols = [c for c in categorical_columns(df) if c not in GLOBAL_FILTER_COLUMNS]
        cols = _global_columns_in(df) + cols
        chosen = st.multiselect("Columns to subset", cols, key=f"{key}_sub_cols")
        for col in chosen:
            values = sorted(df[col].dropna().astype(str).unique(), key=_natural_key)
            mode = st.radio(
                f"{col}",
                ["keep", "exclude"],
                horizontal=True,
                key=f"{key}_sub_mode_{col}",
            )
            picked = st.multiselect(
                f"{col} values",
                values,
                key=f"{key}_sub_vals_{col}",
                label_visibility="collapsed",
            )
            if picked:
                mask = df[col].astype(str).isin(picked)
                df = df[mask] if mode == "keep" else df[~mask]
    return df


def _global_columns_in(df: pd.DataFrame) -> list[str]:
    return [c for c in GLOBAL_FILTER_COLUMNS if c in df.columns]


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
def png_download(fig: go.Figure, key: str, filename: str) -> None:
    """Render a 600 dpi PNG on demand (only PNG is offered, per project convention)."""
    if not st.button("Prepare PNG (600 dpi)", key=f"{key}_png_prep"):
        return
    width = int(fig.layout.width or 1000)
    height = int(fig.layout.height or 700)
    try:
        with st.spinner("Rendering PNG..."):
            png = fig.to_image(
                format="png", width=width, height=height, scale=PNG_DPI / CSS_DPI
            )
    except Exception as err:  # kaleido needs a Chrome install
        st.error(
            f"PNG export failed: {err}. Run `plotly_get_chrome` once to install Chrome."
        )
        return
    st.download_button(
        "Download PNG",
        png,
        file_name=f"{filename}.png",
        mime="image/png",
        key=f"{key}_png_dl",
    )


def missing_notice(dataset_label: str, produced_by: str, path) -> None:
    st.info(
        f"No results for **{dataset_label}** yet. Run `{produced_by}` to generate "
        f"`{path}`."
    )


# ---------------------------------------------------------------------------
# Generic explorer
# ---------------------------------------------------------------------------
def explorer(
    df: pd.DataFrame,
    key: str,
    filters: dict[str, list[str]],
    kinds: list[str] | None = None,
    defaults: dict | None = None,
    title: str = "",
    height: int = 650,
    free_facet_axes: bool = False,
) -> go.Figure | None:
    """Interactive plot with plot-type, axis, color, facet and subset controls.

    Parameters
    ----------
    df : data with canonical metadata columns.
    key : unique widget-key prefix for this section.
    filters : shared sidebar filters (column -> selected values).
    kinds : allowed plot types (defaults to all).
    defaults : initial values for ``kind``, ``x``, ``y``, ``color``, ``facet``,
        ``shape``.
    free_facet_axes : give each facet panel its own x/y range instead of a
        shared one (e.g. per-patient UMAPs, whose coordinates are independent
        fits and aren't comparable across panels).
    """
    kinds = kinds or PLOT_KINDS
    defaults = defaults or {}
    df = apply_global_filters(df, filters)
    df = local_subset(df, key)
    if df.empty:
        st.warning("No rows left after subsetting.")
        return None
    st.caption(f"{len(df):,} rows after subsetting")

    numeric = numeric_columns(df)
    categorical = categorical_columns(df)
    all_cols = [c for c in df.columns if not pd.api.types.is_list_like(df[c].iloc[0])]

    def pick(label, options, default, col, allow_none=True):
        opts = ([NONE] if allow_none else []) + list(options)
        idx = opts.index(default) if default in opts else 0
        return col.selectbox(label, opts, index=idx, key=f"{key}_{label}")

    c1, c2, c3 = st.columns(3)
    kind = pick(
        "Plot type", kinds, defaults.get("kind", kinds[0]), c1, allow_none=False
    )
    x = pick(
        "X",
        all_cols,
        defaults.get("x"),
        c2,
        allow_none=kind in ("histogram", "box", "violin", "bar"),
    )
    y_options = numeric if kind in ("box", "violin", "bar") else all_cols
    y = pick("Y", y_options, defaults.get("y"), c3, allow_none=kind == "histogram")

    c4, c5, c6, c7 = st.columns(4)
    color = pick("Color", all_cols, defaults.get("color"), c4)
    facet = pick("Facet", categorical, defaults.get("facet"), c5)
    shape_options = [c for c in ("dose", "tumor_type") if c in df.columns]
    shape = pick("Shape", shape_options, defaults.get("shape"), c6, allow_none=True)
    facet_wrap = c7.slider(
        "Facets per row", 1, 8, 3, key=f"{key}_wrap", disabled=facet == NONE
    )

    with st.expander("Appearance", expanded=False):
        a1, a2, a3, a4 = st.columns(4)
        opacity = a1.slider("Opacity", 0.05, 1.0, 0.6, key=f"{key}_opacity")
        size = a2.slider("Point size", 1, 15, 4, key=f"{key}_size")
        max_points = a3.number_input(
            "Max scatter points", 1000, 500_000, 50_000, step=5000, key=f"{key}_maxpts"
        )
        agg = a4.selectbox(
            "Aggregation (bar)", ["mean", "median", "sum", "count"], key=f"{key}_agg"
        )
        b1, b2, b3 = st.columns(3)
        as_cat = b1.checkbox(
            "Treat color as categorical", value=True, key=f"{key}_ascat"
        )
        log_x = b2.checkbox("Log X", key=f"{key}_logx")
        log_y = b3.checkbox("Log Y", key=f"{key}_logy")
        nbins = st.slider("Histogram bins", 5, 200, 50, key=f"{key}_bins")
        background = st.checkbox(
            "Scatter: show all points in every facet (grey), color only that facet's points",
            key=f"{key}_background",
            disabled=facet == NONE,
        )

    plot_df = df.copy()
    x = None if x == NONE else x
    y = None if y == NONE else y
    color = None if color == NONE else color
    facet = None if facet == NONE else facet
    shape = None if shape == NONE else shape

    if facet:
        n_facets = plot_df[facet].nunique()
        if n_facets > MAX_FACETS:
            top = plot_df[facet].value_counts().head(MAX_FACETS).index
            st.warning(
                f"{facet} has {n_facets} levels; showing the {MAX_FACETS} largest."
            )
            plot_df = plot_df[plot_df[facet].isin(top)]
    for col in (x, color, facet, shape):
        if col and (col != color or as_cat or col in ("dose",)):
            _order_categories(plot_df, col)
    if color and as_cat and pd.api.types.is_numeric_dtype(plot_df[color]):
        plot_df[color] = plot_df[color].astype(str)

    common = dict(
        color=color, facet_col=facet, facet_col_wrap=facet_wrap if facet else 0
    )
    common.update(_palette_kwargs(plot_df, x, color, facet, shape))
    if not facet:
        common.pop("facet_col_wrap")
    log = dict(log_x=log_x, log_y=log_y)

    try:
        fig = _build(
            kind,
            plot_df,
            x,
            y,
            common,
            log,
            opacity,
            size,
            max_points,
            agg,
            nbins,
            background and facet is not None,
            shape,
        )
    except Exception as err:
        st.error(f"Could not draw this combination: {err}")
        return None

    fig.update_layout(title=title, height=height, template="plotly_white")
    if facet:
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        if free_facet_axes:
            fig.update_xaxes(matches=None, showticklabels=True)
            fig.update_yaxes(matches=None, showticklabels=True)
    st.plotly_chart(fig, width="stretch", key=f"{key}_chart")
    png_download(fig, key, key)
    return fig


def _palette_kwargs(df: pd.DataFrame, x, color, facet, shape=None) -> dict:
    """R-theme colors for the color column and R-defined orders for x/color/facet/shape."""
    kwargs: dict = {}
    if color and not (
        pd.api.types.is_numeric_dtype(df[color])
        and df[color].nunique() > CONTINUOUS_MIN_UNIQUE
    ):
        levels = [str(v) for v in df[color].dropna().unique()]
        colors = palette_for(color, levels)
        if colors:
            kwargs["color_discrete_map"] = colors
    orders = {}
    for col in (x, color, facet, shape):
        if col and not pd.api.types.is_numeric_dtype(df[col]):
            levels = [str(v) for v in df[col].dropna().unique()]
            order = order_for(col, levels)
            if order:
                orders[col] = order
    if orders:
        kwargs["category_orders"] = orders
    return kwargs


def _build(
    kind,
    df,
    x,
    y,
    common,
    log,
    opacity,
    size,
    max_points,
    agg,
    nbins,
    background=False,
    shape=None,
):
    if kind == "scatter":
        if x is None or y is None:
            raise ValueError("choose both X and Y")
        if len(df) > max_points:
            st.caption(f"Showing a random {max_points:,} of {len(df):,} points.")
            df = df.sample(int(max_points), random_state=0)
        fig = px.scatter(
            df,
            x=x,
            y=y,
            symbol=shape,
            opacity=opacity,
            render_mode="webgl",
            **common,
            **log,
        )
        fig.update_traces(marker=dict(size=size))
        if background:
            _add_grey_background(fig, df, x, y, size)
        return fig
    if kind in ("box", "violin"):
        if y is None:
            raise ValueError("choose a numeric Y")
        fn = px.box if kind == "box" else px.violin
        kwargs = dict(box=True) if kind == "violin" else {}
        return fn(df, x=x, y=y, **kwargs, **common, log_y=log["log_y"])
    if kind == "histogram":
        col = x or y
        if col is None:
            raise ValueError("choose X")
        return px.histogram(
            df,
            x=col,
            nbins=nbins,
            barmode="overlay",
            opacity=0.65,
            **common,
            log_x=log["log_x"],
        )
    if kind == "bar":
        if y is None:
            raise ValueError("choose a numeric Y")
        group = [c for c in (x, common["color"], common["facet_col"]) if c]
        if not group:
            raise ValueError("choose X, color or facet to group by")
        grouped = (
            df.groupby(group, observed=True)[y].agg(agg).reset_index()
            if agg != "count"
            else df.groupby(group, observed=True)[y].count().reset_index()
        )
        return px.bar(grouped, x=x, y=y, barmode="group", **common, log_y=log["log_y"])
    if kind == "heatmap":
        if x is None or y is None:
            raise ValueError("choose X and Y")
        return px.density_heatmap(
            df,
            x=x,
            y=y,
            z=None,
            facet_col=common["facet_col"],
            facet_col_wrap=common.get("facet_col_wrap", 0) or None,
            nbinsx=nbins,
            nbinsy=nbins,
            color_continuous_scale="Viridis",
        )
    raise ValueError(f"unknown plot type {kind}")


# ---------------------------------------------------------------------------
# UpSet plot (ported from 6.plot_variate_importance.r's plot_upset())
# ---------------------------------------------------------------------------
def regroup_combos_by_identity(
    combos: pd.DataFrame, groups: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse individual-term combinations onto group identities (e.g. every
    biological term folded into one "Biological" set, every technical term
    into one "Technical" set): a combination is "in" a group's set if any of
    its member terms is in the original combination.

    Returns ``(combos, sizes)`` shaped like ``upset_df``/``sizes_df`` filtered
    to ``plot == "upset"``, so the result plugs straight into ``upset_plot``.
    """
    work = combos.copy()
    group_names = list(groups)
    for name, terms in groups.items():
        cols = [f"in_{t}" for t in terms if f"in_{t}" in work.columns]
        work[f"in_{name}"] = work[cols].fillna(False).any(axis=1) if cols else False
    keep_cols = [f"in_{g}" for g in group_names]
    regrouped = work.groupby(keep_cols, dropna=False)["n_features"].sum().reset_index()
    regrouped = regrouped[regrouped[keep_cols].fillna(False).any(axis=1)].reset_index(
        drop=True
    )
    regrouped["combination"] = regrouped[keep_cols].apply(
        lambda r: " + ".join(g for g, c in zip(group_names, keep_cols) if bool(r[c])),
        axis=1,
    )
    regrouped["treatment_specific"] = False
    regrouped = regrouped.sort_values("n_features", ascending=False).reset_index(
        drop=True
    )
    sizes = pd.DataFrame(
        {
            "term": group_names,
            "set_size": [
                int(work.loc[work[f"in_{g}"], "n_features"].sum()) for g in group_names
            ],
        }
    )
    return regrouped, sizes


def upset_plot(
    sizes: pd.DataFrame,
    combos: pd.DataFrame,
    term_order: list[str],
    top_n: int = 25,
    title: str = "",
) -> go.Figure | None:
    """Bar chart of the top term combinations (by feature count) plus a set-size
    bar chart and dot matrix showing which terms make up each combination.

    Parameters
    ----------
    sizes : one row per term, with a ``set_size`` column (``sizes_df`` filtered
        to ``plot == "upset"`` for the chosen model_set/scope/group).
    combos : one row per term combination, with ``in_<term>`` booleans,
        ``n_features`` and ``treatment_specific`` (``upset_df`` filtered the
        same way).
    """
    terms = [t for t in term_order if t in sizes["term"].to_numpy()]
    if not terms or combos.empty:
        return None

    shown = (
        combos.sort_values("n_features", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    n_combo = len(shown)
    n_terms = len(terms)
    x = list(range(1, n_combo + 1))

    fig = make_subplots(
        rows=2,
        cols=3,
        row_heights=[0.42, 0.58],
        column_widths=[0.16, 0.22, 0.62],
        horizontal_spacing=0.02,
        vertical_spacing=0.03,
        specs=[
            [{}, {}, {}],
            [{}, {}, {}],
        ],
    )

    # top-right: bar chart of features per combination (log scale), orange = treatment-only
    bar_colors = [
        "#d95f02" if v else "#555555" for v in shown["treatment_specific"].fillna(False)
    ]
    fig.add_trace(
        go.Bar(
            x=x,
            y=shown["n_features"],
            marker_color=bar_colors,
            text=shown["n_features"],
            textposition="outside",
            showlegend=False,
            hovertext=shown["combination"],
            hoverinfo="text+y",
        ),
        row=1,
        col=3,
    )
    # explicit headroom above the tallest bar: on a log axis, kaleido's static PNG
    # renderer flips "outside" text upside-down if it sits too close to the top of
    # the subplot, so round the range up to the next whole decade above the max
    max_n = float(shown["n_features"].max())
    y_top = 10 ** (math.floor(math.log10(max_n)) + 1) if max_n > 0 else 10
    fig.update_yaxes(
        type="log",
        range=[math.log10(0.8), math.log10(y_top)],
        title_text="features in exactly this combination (log)",
        row=1,
        col=3,
    )
    fig.update_xaxes(showticklabels=False, range=[0.4, n_combo + 0.6], row=1, col=3)

    # bottom-left: set-size bars (horizontal, largest terms first)
    term_sizes = sizes.set_index("term").reindex(terms)["set_size"]
    fig.add_trace(
        go.Bar(
            x=term_sizes.to_numpy(),
            y=list(range(n_terms)),
            orientation="h",
            marker_color="#555555",
            text=term_sizes.to_numpy(),
            textposition="outside",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_xaxes(autorange="reversed", title_text="features in group", row=2, col=1)
    fig.update_yaxes(range=[n_terms - 0.5, -0.5], showticklabels=False, row=2, col=1)

    # bottom-middle: term labels, colored to match the dot matrix
    fig.add_trace(
        go.Scatter(
            x=[0] * n_terms,
            y=list(range(n_terms)),
            mode="text",
            text=terms,
            textposition="middle right",
            textfont=dict(color=[TERM_COLORS.get(t, "black") for t in terms], size=13),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=2,
    )
    fig.update_xaxes(visible=False, range=[0, 1], row=2, col=2)
    fig.update_yaxes(visible=False, range=[n_terms - 0.5, -0.5], row=2, col=2)

    # bottom-right: dot matrix (grey = not in combination, colored + connected = in combination)
    bg_x = [j + 1 for j in range(n_combo) for _ in range(n_terms)]
    bg_y = [i for _ in range(n_combo) for i in range(n_terms)]
    fig.add_trace(
        go.Scatter(
            x=bg_x,
            y=bg_y,
            mode="markers",
            marker=dict(size=13, color="#dddddd"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=3,
    )
    for j, (_, row) in enumerate(shown.iterrows()):
        on_terms = [t for t in terms if bool(row.get(f"in_{t}"))]
        if not on_terms:
            continue
        on_idx = [terms.index(t) for t in on_terms]
        if len(on_idx) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[j + 1, j + 1],
                    y=[min(on_idx), max(on_idx)],
                    mode="lines",
                    line=dict(color="black", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=2,
                col=3,
            )
        fig.add_trace(
            go.Scatter(
                x=[j + 1] * len(on_idx),
                y=on_idx,
                mode="markers",
                marker=dict(
                    size=13,
                    color=[TERM_COLORS.get(t, "black") for t in on_terms],
                    line=dict(color="black", width=1),
                ),
                showlegend=False,
                hovertext=on_terms,
                hoverinfo="text",
            ),
            row=2,
            col=3,
        )
    fig.update_xaxes(showticklabels=False, range=[0.4, n_combo + 0.6], row=2, col=3)
    fig.update_yaxes(range=[n_terms - 0.5, -0.5], showticklabels=False, row=2, col=3)

    fig.update_layout(
        title=title,
        height=max(500, 90 * n_terms + 260),
        template="plotly_white",
        bargap=0.25,
        showlegend=False,
    )
    return fig


def _discrete_colorscale(
    categories: list[str], palette: dict[str, str], fallback: str = "#cccccc"
) -> tuple[list, dict]:
    """A piecewise-constant Plotly colorscale over ``[0, 1]``, one flat step per
    category, plus the ``category -> z`` map to look values up by. Lets a
    ``go.Heatmap`` (which only takes a continuous colorscale) render a
    categorical color strip, e.g. an annotation track."""
    n = len(categories)
    if n == 0:
        return [[0, fallback], [1, fallback]], {}
    colors = [palette.get(c, fallback) for c in categories]
    if n == 1:
        return [[0, colors[0]], [1, colors[0]]], {categories[0]: 0.5}
    scale = []
    for i, color in enumerate(colors):
        scale += [[i / n, color], [(i + 1) / n, color]]
    cat_to_z = {c: (i + 0.5) / n for i, c in enumerate(categories)}
    return scale, cat_to_z


def _annotation_strip(
    axis_categories: list[str],
    labels: list[str],
    palette: dict[str, str],
    orientation: str,
    name: str,
) -> go.Heatmap:
    """One color-coded annotation strip (a single-row or single-column
    ``go.Heatmap``) for the given category labels, e.g. a "Patient" or
    "Feature type" track alongside a main heatmap. ``axis_categories`` are the
    main heatmap's own column (or row) ids, in the same order as ``labels``
    (the attribute value to color each one by, e.g. each column's patient)."""
    categories = sorted(set(labels))
    scale, cat_to_z = _discrete_colorscale(categories, palette)
    z_values = [cat_to_z[label] for label in labels]
    if orientation == "row":  # a top strip, one column per main-heatmap column
        z, x, y = [z_values], axis_categories, [name]
        text = [labels]
    else:  # a left strip, one row per main-heatmap row
        z, x, y = [[v] for v in z_values], [name], axis_categories
        text = [[label] for label in labels]
    return go.Heatmap(
        z=z,
        x=x,
        y=y,
        colorscale=scale,
        zmin=0,
        zmax=1,
        showscale=False,
        showlegend=False,
        text=text,
        hovertemplate=f"{name}: %{{text}}<extra></extra>",
    )


def _hclust(arr: np.ndarray) -> tuple[list[int], np.ndarray | None]:
    """Leaf order and linkage matrix from average-linkage hierarchical
    clustering on binary (Jaccard) distance, matching R's ``hclust(dist(mat,
    method="binary"), method="average")``. Falls back to the original order
    (and no linkage, so no dendrogram) when there are too few rows to
    cluster, or the distance matrix has non-finite entries (e.g. two
    all-zero rows, for which binary/Jaccard distance is undefined)."""
    n = arr.shape[0]
    if n < 3:
        return list(range(n)), None
    try:
        d = pdist(arr, metric="jaccard")
        if not np.all(np.isfinite(d)):
            return list(range(n)), None
        z = linkage(d, method="average")
        return leaves_list(z).tolist(), z
    except Exception:
        return list(range(n)), None


def _cluster_from_similarity(sim: np.ndarray) -> tuple[list[int], np.ndarray | None]:
    """Average-linkage leaf order and linkage matrix for a symmetric similarity
    matrix (e.g. a correlation matrix), clustered on ``1 - similarity`` as the
    distance. Missing pairs (NaN) are treated as maximally dissimilar rather
    than dropped, so the matrix stays square. Falls back to the original order
    (no dendrogram) when there are too few rows to cluster."""
    n = sim.shape[0]
    if n < 3:
        return list(range(n)), None
    dist = 1 - np.nan_to_num(sim, nan=-1.0)
    np.fill_diagonal(dist, 0)
    try:
        d = squareform(dist, checks=False)
        if not np.all(np.isfinite(d)):
            return list(range(n)), None
        z = linkage(d, method="average")
        return leaves_list(z).tolist(), z
    except Exception:
        return list(range(n)), None


def _dendrogram_traces(
    z: np.ndarray | None, leaf_axis: str, color: str = "#999999"
) -> list[go.Scatter]:
    """Line traces for a dendrogram of linkage matrix ``z``.

    ``leaf_axis="x"``: leaves along x (0..n-1), merge height along y, root at
    the top -- for a strip above the heatmap.
    ``leaf_axis="y"``: leaves along y (0..n-1), merge height mirrored onto x
    so the root is at x=0 (far left) and the leaves sit at x=max distance,
    right next to the heatmap -- for a strip to its left.
    """
    if z is None:
        return []
    dd = dendrogram(z, no_plot=True)
    max_dist = max((max(ys) for ys in dd["dcoord"]), default=1) or 1
    traces = []
    for xs, ys in zip(dd["icoord"], dd["dcoord"]):
        leaf_pos = [(v - 5) / 10 for v in xs]
        if leaf_axis == "x":
            x, y = leaf_pos, ys
        else:
            x, y = [max_dist - v for v in ys], leaf_pos
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color, width=1),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    return traces


ROW_TRACK_NAMES = ["Compartment", "Channel", "Feature type"]
COL_TRACK_NAMES = ["Patient", "Tumor type", "Treatment"]


def annotated_significance_heatmap(
    sig: pd.DataFrame,
    title: str,
    legend_title: str = "significant",
    cluster_rows: bool = True,
    cluster_cols: bool = True,
    row_tracks: list[str] | None = None,
    col_tracks: list[str] | None = None,
) -> go.Figure | None:
    """Feature x (patient, treatment) yes/no significance matrix with Patient,
    Tumor type, Treatment (top) and Compartment, Channel, Feature type (left)
    color-coded annotation strips, ported from
    ``6.plot_variate_importance.r``'s ``cooccurrence_heatmap()``. ``sig``
    is one row per (feature, patient, treatment) pair that is significant
    under the chosen criterion, with columns ``feature``, ``Feature_type``,
    ``Channel``, ``Compartment``, ``patient``, ``treatment``, ``drug``,
    ``tumor_type``.

    ``cluster_rows``/``cluster_cols`` toggle hierarchical clustering on each
    axis independently (unclustered keeps the original row/column order).
    ``row_tracks``/``col_tracks`` pick which annotation strips to draw on each
    axis (from ``ROW_TRACK_NAMES``/``COL_TRACK_NAMES``; ``None`` draws all).
    """
    if sig.empty:
        return None
    row_tracks = ROW_TRACK_NAMES if row_tracks is None else row_tracks
    col_tracks = COL_TRACK_NAMES if col_tracks is None else col_tracks
    sig = sig.assign(
        patient_treatment=sig["patient"] + " | " + sig["treatment"],
        Feature_type=sig["Feature_type"].fillna("Other"),
        Channel=sig["Channel"].fillna("Other"),
        Compartment=sig["Compartment"].fillna("Other"),
    )
    features = sig.drop_duplicates("feature")[
        ["feature", "Feature_type", "Channel", "Compartment"]
    ].reset_index(drop=True)
    columns = sig.drop_duplicates("patient_treatment")[
        ["patient_treatment", "patient", "drug", "tumor_type"]
    ].reset_index(drop=True)
    feat_list = features["feature"].tolist()
    col_list = columns["patient_treatment"].tolist()

    feat_idx = {f: i for i, f in enumerate(feat_list)}
    col_idx = {c: i for i, c in enumerate(col_list)}
    arr = np.zeros((len(feat_list), len(col_list)), dtype=int)
    arr[
        sig["feature"].map(feat_idx).to_numpy(),
        sig["patient_treatment"].map(col_idx).to_numpy(),
    ] = 1

    row_order, row_z = (
        _hclust(arr) if cluster_rows else (list(range(arr.shape[0])), None)
    )
    col_order, col_z = (
        _hclust(arr.T) if cluster_cols else (list(range(arr.shape[1])), None)
    )
    arr = arr[np.ix_(row_order, col_order)]
    features = features.iloc[row_order].reset_index(drop=True)
    columns = columns.iloc[col_order].reset_index(drop=True)
    feat_list = features["feature"].tolist()
    col_list = columns["patient_treatment"].tolist()

    row_annotations_all = [
        ("Compartment", features["Compartment"].tolist(), "compartment"),
        ("Channel", features["Channel"].tolist(), "channel"),
        ("Feature type", features["Feature_type"].tolist(), "feature_type"),
    ]
    col_annotations_all = [
        ("Patient", columns["patient"].tolist(), "patient"),
        ("Tumor type", columns["tumor_type"].tolist(), "tumor_type"),
        ("Treatment", columns["drug"].tolist(), "treatment"),
    ]
    row_annotations = [t for t in row_annotations_all if t[0] in row_tracks]
    col_annotations = [t for t in col_annotations_all if t[0] in col_tracks]
    n_row_strips = len(row_annotations)
    n_col_strips = len(col_annotations)
    top_strip_frac = 0.018
    left_strip_frac = 0.03
    dendro_top_frac = 0.08 if col_z is not None else 0.001
    dendro_left_frac = 0.08 if row_z is not None else 0.001
    row_heights = (
        [dendro_top_frac]
        + [top_strip_frac] * n_col_strips
        + [1 - dendro_top_frac - top_strip_frac * n_col_strips]
    )
    col_widths = (
        [dendro_left_frac]
        + [left_strip_frac] * n_row_strips
        + [1 - dendro_left_frac - left_strip_frac * n_row_strips]
    )

    fig = make_subplots(
        rows=n_col_strips + 2,
        cols=n_row_strips + 2,
        row_heights=row_heights,
        column_widths=col_widths,
        horizontal_spacing=0.003,
        vertical_spacing=0.003,
        specs=[[{}] * (n_row_strips + 2)] * (n_col_strips + 2),
    )
    main_row, main_col = n_col_strips + 2, n_row_strips + 2

    for trace in _dendrogram_traces(col_z, "x"):
        fig.add_trace(trace, row=1, col=main_col)
    for trace in _dendrogram_traces(row_z, "y"):
        fig.add_trace(trace, row=main_row, col=1)
    for i, (name, values, palette_key) in enumerate(col_annotations, start=2):
        fig.add_trace(
            _annotation_strip(
                col_list, values, palette_for(palette_key, values) or {}, "row", name
            ),
            row=i,
            col=main_col,
        )
    for i, (name, values, palette_key) in enumerate(row_annotations, start=2):
        fig.add_trace(
            _annotation_strip(
                feat_list, values, palette_for(palette_key, values) or {}, "col", name
            ),
            row=main_row,
            col=i,
        )
    fig.add_trace(
        go.Heatmap(
            z=arr,
            x=col_list,
            y=feat_list,
            colorscale=[[0, "white"], [0.5, "white"], [0.5, "#d95f02"], [1, "#d95f02"]],
            zmin=0,
            zmax=1,
            showscale=True,
            colorbar=dict(
                title=legend_title,
                tickvals=[0.25, 0.75],
                ticktext=["no", "yes"],
                len=0.5,
            ),
            hovertemplate="%{y}<br>%{x}<extra></extra>",
        ),
        row=main_row,
        col=main_col,
    )

    main_axis_num = main_row * main_col
    # the color strips are categorical heatmaps on the same string categories as
    # the main heatmap, so `matches` keeps their axis in lockstep; the
    # dendrograms plot numeric leaf positions instead, so instead they get an
    # explicit range spanning the same [-0.5, n-0.5] span Plotly gives a
    # category axis with n categories, which lines them up just as exactly
    for r in range(2, main_row):
        fig.update_xaxes(matches=f"x{main_axis_num}", row=r, col=main_col)
        fig.update_xaxes(showticklabels=False, row=r, col=main_col)
    for c in range(2, main_col):
        fig.update_yaxes(matches=f"y{main_axis_num}", row=main_row, col=c)
        fig.update_yaxes(showticklabels=False, row=main_row, col=c)
        # the row (left-strip) track name is its single x tick label; vertical
        # text fits the narrow strip width far better than horizontal
        fig.update_xaxes(tickangle=90, row=main_row, col=c)
    fig.update_xaxes(
        range=[-0.5, len(col_list) - 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1,
        col=main_col,
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, row=1, col=main_col
    )
    fig.update_yaxes(
        range=[-0.5, len(feat_list) - 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=1,
    )
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False, row=main_row, col=1
    )
    # feature names are dropped (the Compartment/Channel/Feature type strips
    # carry the per-row identity instead); hover still shows the feature name
    fig.update_yaxes(showticklabels=False, row=main_row, col=main_col)
    fig.update_xaxes(showticklabels=False, row=main_row, col=main_col)
    size = max(280, min(900, 4 * len(feat_list) + 140))
    fig.update_layout(
        template="plotly_white",
        title=f"{title} ({len(feat_list):,} features x {len(col_list):,} patient-treatments)",
        width=size,
        height=size,
        showlegend=False,
    )
    return fig


def platemap_heatmap(
    plate: pd.DataFrame,
    palette: dict[str, str],
    title: str = "",
) -> go.Figure:
    """Well-grid view of a plate layout: one colored cell per well, grouped by
    ``Treatment`` (``palette`` maps treatment name -> color), with dose/unit
    in the hover text. ``plate`` has ``WellRow``, ``WellCol``, ``Treatment``,
    ``Dose``, ``Unit`` columns (one row per well, as in
    ``config/platemaps/platemap*.csv``). Row A is at the top, matching the
    physical plate."""
    rows = sorted(plate["WellRow"].unique())
    cols = sorted(plate["WellCol"].unique(), key=int)
    treatments = sorted(plate["Treatment"].unique())
    scale, cat_to_z = _discrete_colorscale(treatments, palette)
    row_idx = {r: i for i, r in enumerate(rows)}
    col_idx = {c: i for i, c in enumerate(cols)}
    z = np.full((len(rows), len(cols)), np.nan)
    text = np.full((len(rows), len(cols)), "", dtype=object)
    for _, well in plate.iterrows():
        i, j = row_idx[well["WellRow"]], col_idx[well["WellCol"]]
        z[i, j] = cat_to_z[well["Treatment"]]
        text[i, j] = (
            f"{well['WellRow']}{well['WellCol']}<br>{well['Treatment']}"
            f"<br>{well['Dose']}{well['Unit']}"
        )
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[str(c) for c in cols],
            y=rows,
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=9),
            colorscale=scale,
            zmin=0,
            zmax=1,
            showscale=False,
            hoverinfo="text",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_yaxes(autorange="reversed", title_text="Row")
    fig.update_xaxes(title_text="Column", side="top")
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=max(320, 70 * len(rows) + 80),
    )
    return fig


def correlation_heatmap(
    matrix: np.ndarray,
    labels: list[str],
    meta: pd.DataFrame,
    tracks: list[tuple[str, str, str]],
    cluster: bool,
    zmin: float = -1,
    zmax: float = 1,
    title: str = "",
) -> go.Figure:
    """Symmetric correlation-style matrix with optional average-linkage
    hierarchical clustering and optional color-coded annotation strips along
    both edges. Since rows and columns represent the same samples, one
    clustering order and one set of tracks is applied to both axes.

    Parameters
    ----------
    matrix : square, symmetric similarity matrix (may contain NaN for missing
        pairs).
    labels : axis tick labels, aligned row-for-row with ``matrix``.
    meta : per-sample metadata, aligned row-for-row with ``matrix``; supplies
        the values for each selected track.
    tracks : ``(display name, column in meta, palette lookup key)`` for every
        annotation strip to draw (already filtered to what the caller wants
        shown -- pass ``[]`` for none).
    cluster : reorder rows/columns by hierarchical clustering instead of
        keeping the given order.
    """
    n = matrix.shape[0]
    order = list(range(n))
    z = None
    if cluster:
        order, z = _cluster_from_similarity(matrix)
    mat = matrix[np.ix_(order, order)]
    labels = [labels[i] for i in order]
    meta = meta.iloc[order].reset_index(drop=True)
    tracks = [t for t in tracks if t[1] in meta.columns]

    n_tracks = len(tracks)
    strip_frac = 0.02
    dendro_frac = 0.08 if z is not None else 0.001
    main_frac = max(0.2, 1 - dendro_frac - strip_frac * n_tracks)

    fig = make_subplots(
        rows=n_tracks + 2,
        cols=n_tracks + 2,
        row_heights=[dendro_frac] + [strip_frac] * n_tracks + [main_frac],
        column_widths=[dendro_frac] + [strip_frac] * n_tracks + [main_frac],
        horizontal_spacing=0.003,
        vertical_spacing=0.003,
        specs=[[{}] * (n_tracks + 2)] * (n_tracks + 2),
    )
    main_row = main_col = n_tracks + 2

    for trace in _dendrogram_traces(z, "x"):
        fig.add_trace(trace, row=1, col=main_col)
    for trace in _dendrogram_traces(z, "y"):
        fig.add_trace(trace, row=main_row, col=1)
    for i, (name, col, palette_key) in enumerate(tracks, start=2):
        values = meta[col].astype(str).tolist()
        palette = palette_for(palette_key, values) or {}
        fig.add_trace(
            _annotation_strip(labels, values, palette, "row", name),
            row=i,
            col=main_col,
        )
        fig.add_trace(
            _annotation_strip(labels, values, palette, "col", name),
            row=main_row,
            col=i,
        )

    fig.add_trace(
        go.Heatmap(
            z=mat,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            hovertemplate="%{y}<br>%{x}<br>%{z:.2f}<extra></extra>",
        ),
        row=main_row,
        col=main_col,
    )

    main_axis_num = main_row * main_col
    for r in range(2, main_row):
        fig.update_xaxes(
            matches=f"x{main_axis_num}", showticklabels=False, row=r, col=main_col
        )
    for c in range(2, main_col):
        fig.update_yaxes(
            matches=f"y{main_axis_num}", showticklabels=False, row=main_row, col=c
        )
        # the row (left-strip) track name is its single x tick label; vertical
        # text fits the narrow strip width far better than horizontal
        fig.update_xaxes(tickangle=90, row=main_row, col=c)
    fig.update_xaxes(
        range=[-0.5, n - 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=1,
        col=main_col,
    )
    fig.update_yaxes(
        showticklabels=False, showgrid=False, zeroline=False, row=1, col=main_col
    )
    fig.update_yaxes(
        range=[-0.5, n - 0.5],
        showticklabels=False,
        showgrid=False,
        zeroline=False,
        row=main_row,
        col=1,
    )
    fig.update_xaxes(
        showticklabels=False, showgrid=False, zeroline=False, row=main_row, col=1
    )
    show_labels = n <= 60
    fig.update_xaxes(showticklabels=show_labels, row=main_row, col=main_col)
    fig.update_yaxes(showticklabels=show_labels, row=main_row, col=main_col)
    size = max(500, min(1200, 20 * n + 200))
    fig.update_layout(
        template="plotly_white",
        title=title,
        width=size,
        height=size,
        showlegend=False,
    )
    return fig


def _add_grey_background(
    fig: go.Figure, df: pd.DataFrame, x: str, y: str, size: int
) -> None:
    """Draw every point in grey behind the colored points of each facet panel."""
    panels = {(t.xaxis, t.yaxis) for t in fig.data}
    grey = [
        go.Scattergl(
            x=df[x],
            y=df[y],
            mode="markers",
            xaxis=xaxis,
            yaxis=yaxis,
            marker=dict(size=size, color="lightgrey", opacity=0.25),
            showlegend=False,
            hoverinfo="skip",
        )
        for xaxis, yaxis in sorted(panels, key=str)
    ]
    n_colored = len(fig.data)
    fig.add_traces(grey)
    # background traces first so the colored points draw on top
    n_grey = len(grey)
    fig.data = fig.data[n_colored:] + fig.data[:n_colored]
    assert len(fig.data) == n_colored + n_grey
