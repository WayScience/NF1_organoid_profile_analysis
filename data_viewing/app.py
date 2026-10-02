"""NF1 organoid profile viewer.

Run from anywhere inside the repo with:
    streamlit run data_viewing/app.py
"""

import plotly.io as pio
import streamlit as st
from background import module_background, section_background
from data_io import (
    BUCKET_SYNC_ERROR,
    GLOBAL_FILTER_COLUMNS,
    all_filterable_paths,
    global_filter_options,
)
from sections import (  # noqa: F401 -- VIABILITY_SECTIONS kept for the commented-out tab below
    EDA_SECTIONS,
    LINEAR_MODELING_SECTIONS,
    OVERVIEW_SECTIONS,
    VIABILITY_SECTIONS,
)

pio.templates.default = "plotly_white"
st.set_page_config(page_title="NF1 organoid data viewer", layout="wide")
st.title("NF1 organoid profile data viewer")

if BUCKET_SYNC_ERROR:
    st.error(
        f"**Couldn't load data from the HF Bucket** -- every tab below will show "
        f'"no results" until this is fixed.\n\n{BUCKET_SYNC_ERROR}',
        icon="🚨",
    )

# ---------------------------------------------------------------------------
# Shared sidebar filters (applied wherever a table has the column)
# ---------------------------------------------------------------------------
st.sidebar.header("Subset (all tabs)")
st.sidebar.caption(
    "Filters apply to every plot whose table has that column. "
    "Leave empty to keep everything."
)
options = global_filter_options(all_filterable_paths())
filters: dict[str, list[str]] = {}
for column in GLOBAL_FILTER_COLUMNS:
    if column in options:
        filters[column] = st.sidebar.multiselect(
            column, options[column], key=f"global_{column}"
        )
if st.sidebar.button("Clear all filters"):
    for column in GLOBAL_FILTER_COLUMNS:
        st.session_state.pop(f"global_{column}", None)
    st.rerun()

# ---------------------------------------------------------------------------
# One tab per module, one selectable section per analysis type
# ---------------------------------------------------------------------------
MODULES = {
    "0.Overview": OVERVIEW_SECTIONS,
    "1.EDA": EDA_SECTIONS,
    # "3.viability_prediction_models" is hidden for now (its results need another
    # pass); the module import and its sections.py code are kept intact -- restore
    # this tab by uncommenting the line below.
    # "3.viability_prediction_models": VIABILITY_SECTIONS,
    "4.linear_modeling": LINEAR_MODELING_SECTIONS,
}

tabs = st.tabs(list(MODULES))
for tab, (module, sections) in zip(tabs, MODULES.items()):
    with tab:
        module_background(module)
        # only the selected section renders, so heavy tables load on demand
        section = st.radio(
            "Analysis", list(sections), horizontal=True, key=f"section_{module}"
        )
        st.subheader(section)
        section_background(section)
        sections[section](filters)
