"""Dataset registry, loading and metadata harmonization for the data_viewing app.

Every results table the app can show is registered here with its path. Metadata
columns are renamed to canonical names (e.g. ``Metadata_Experiment_Treatment``
and ``Metadata_treatment`` both become ``treatment``) so one set of filters
works across every module.

Data lives in an HF Bucket (see ``sync_data.py``). Two ways to get it into a
running app, picked automatically:

- An HF Space with the bucket mounted as a storage volume (Space Settings ->
  Storage Buckets) -- set ``DATA_DIR`` to that mount path.
- Anywhere else (e.g. Streamlit Community Cloud, which can't mount buckets):
  this module downloads it with ``sync_bucket()`` into ``DATA_DIR`` every
  time the process starts, via ``st.cache_resource`` so it only actually
  transfers once per process (``sync_bucket`` also diffs, so even that one
  call is cheap if ``DATA_DIR`` already matches). The bucket (see
  ``HF_BUCKET`` below) is **private**, so this requires an ``HF_TOKEN``
  secret/env var with read access to it -- without one, the download fails,
  every section shows "no results", and a sidebar warning names the error.
"""

import os
import pathlib

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from palettes import TREATMENT_MOA_MAP, TUMOR_TYPE_LOOKUP

DATA_DIR = pathlib.Path(
    os.environ.get("DATA_DIR", pathlib.Path(__file__).resolve().parent / "data")
)

# default matches sync_data.py's push target, so this works with zero extra
# config beyond HF_TOKEN (the bucket is private) once deployed.
HF_BUCKET = os.environ.get("HF_BUCKET", "lippincm/NF1_3D_organoid_data_viewing-storage")


@st.cache_resource(show_spinner="Downloading precomputed results from the HF Bucket...")
def _sync_bucket_once(bucket: str, dest: str) -> None:
    """Download ``hf://buckets/<bucket>/data`` into ``dest``, once per process.

    Always attempts the sync rather than skipping when ``dest`` already has
    *some* files: a previous attempt (e.g. before ``HF_TOKEN`` was set) can
    leave it partially populated -- missing just one subfolder was exactly
    that bug -- and ``sync_bucket`` only transfers what's actually missing
    or changed, so a fully-up-to-date ``dest`` costs one cheap diff, not a
    re-download. ``st.cache_resource`` only caches a *successful* call, so a
    failure (e.g. bad ``HF_TOKEN``) retries on the next process start rather
    than silently staying broken for its lifetime.
    """
    from huggingface_hub import sync_bucket

    pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    sync_bucket(f"hf://buckets/{bucket}/data", dest)


BUCKET_SYNC_ERROR: str | None = None
try:
    _sync_bucket_once(HF_BUCKET, str(DATA_DIR))
except Exception as err:  # no/bad HF_TOKEN, offline, unreachable, ...
    from huggingface_hub import get_token

    token_state = "an HF_TOKEN is set" if get_token() else "no HF_TOKEN is set"
    BUCKET_SYNC_ERROR = (
        f"Could not sync data from the HF Bucket `{HF_BUCKET}` ({token_state}): {err}"
    )


EDA_RESULTS = DATA_DIR / "eda"
VIABILITY_RESULTS = DATA_DIR / "viability_models"
LINEAR_MODELING_RESULTS = DATA_DIR / "linear_modeling" / "models"
VARIATE_IMPORTANCE_RESULTS = DATA_DIR / "linear_modeling" / "variate_importance"
PLATEMAP_CONFIG_DIR = DATA_DIR / "platemaps"
PLATEMAPS = PLATEMAP_CONFIG_DIR / "combined_platemaps.parquet"

# ---------------------------------------------------------------------------
# Canonical metadata names
# ---------------------------------------------------------------------------
# raw column name -> canonical name. Anything else with a ``Metadata_`` prefix
# just has the prefix stripped.
CANONICAL_COLUMNS = {
    "Metadata_Biology_PatientTumor": "patient_tumor",
    "Metadata_patient_tumor": "patient_tumor",
    "Metadata_held_out_group": "held_out_group",
    "Metadata_Experiment_Treatment": "treatment",
    "Metadata_treatment": "treatment",
    "Metadata_Experiment_Dose": "dose",
    "Metadata_dose": "dose",
    "Metadata_Experiment_Unit": "dose_unit",
    "Metadata_dose_unit": "dose_unit",
    "Metadata_Experiment_Class": "class",
    "Metadata_class": "class",
    "Metadata_Experiment_Target": "target",
    "Metadata_target": "target",
    "Metadata_Experiment_TherapeuticCategories": "therapeutic_category",
    "Metadata_therapeutic_categories": "therapeutic_category",
    "Metadata_Experiment_Well": "well",
    "Metadata_Well": "well",
    "Metadata_modality": "modality",
    "Metadata_image_mode": "image_mode",
    # linear-modeling / platemap tables use un-prefixed names
    "patient": "patient",
    "patient_id": "patient_tumor",
    "Treatment": "treatment",
    "Dose": "dose",
}

# NF0037_T1_CQ1 is a separate analysis (see data_viewing/background.py's
# PROFILE_GLOSSARY) and must never show up alongside the pooled patients, in
# any module -- dropped here so every table is covered, not just the ones an
# upstream script already excludes it from.
EXCLUDED_PATIENTS = {"NF0037_T1_CQ1"}

# Metadata fields offered in the shared sidebar filters (when present in a table).
GLOBAL_FILTER_COLUMNS = [
    "patient_tumor",
    "treatment",
    "dose",
    "class",
    "target",
    "therapeutic_category",
    "moa",
    "tumor_type",
    "well",
    "image_mode",
    "modality",
]


def canonicalize_columns(columns: list[str]) -> dict[str, str]:
    """Map raw column names to canonical names, avoiding duplicate targets."""
    mapping: dict[str, str] = {}
    taken: set[str] = set()
    # explicit canonical names first so they win over prefix-stripped ones
    for col in columns:
        if col in CANONICAL_COLUMNS and CANONICAL_COLUMNS[col] not in taken:
            mapping[col] = CANONICAL_COLUMNS[col]
            taken.add(mapping[col])
    for col in columns:
        if col in mapping:
            continue
        new = col.removeprefix("Metadata_")
        if new in taken or new in columns and new != col:
            new = col  # keep the raw name rather than collide
        mapping[col] = new
        taken.add(new)
    return mapping


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``moa`` and ``tumor_type`` using the lookups from the R theme."""
    if "treatment" in df.columns and "moa" not in df.columns:
        df["moa"] = df["treatment"].astype(str).map(TREATMENT_MOA_MAP)
    if "patient_tumor" in df.columns and "tumor_type" not in df.columns:
        df["tumor_type"] = df["patient_tumor"].astype(str).map(TUMOR_TYPE_LOOKUP)
    return df


def _exclude_patients(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows for ``EXCLUDED_PATIENTS`` wherever a patient column is present."""
    for col in ("patient_tumor", "patient"):
        if col in df.columns:
            df = df[~df[col].astype(str).isin(EXCLUDED_PATIENTS)]
    return df


def _harmonize(df: pd.DataFrame) -> pd.DataFrame:
    mapping = canonicalize_columns(list(df.columns))
    df = df.rename(columns=mapping)
    # mixed-type object columns (e.g. dose) break filtering/plotting
    for col in ("dose", "well"):
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].astype(str)
    return _exclude_patients(_add_derived(df))


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
class Dataset:
    """A parquet file plus the script that produces it (shown when missing)."""

    def __init__(self, label: str, path: pathlib.Path, produced_by: str):
        self.label = label
        self.path = path
        self.produced_by = produced_by

    @property
    def exists(self) -> bool:
        return self.path.exists()


def _glob(
    directory: pathlib.Path, pattern: str, produced_by: str
) -> dict[str, Dataset]:
    files = sorted(directory.glob(pattern)) if directory.exists() else []
    return {f.stem: Dataset(f.stem, f, produced_by) for f in files}


@st.cache_data(show_spinner="Loading data...")
def load_dataset(path: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Read a parquet file (optionally a column subset) and harmonize metadata."""
    df = pd.read_parquet(path, columns=list(columns) if columns else None)
    return _harmonize(df)


@st.cache_data(show_spinner=False)
def parquet_columns(path: str) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


@st.cache_data(show_spinner="Scanning metadata values...")
def global_filter_options(paths: tuple[str, ...]) -> dict[str, list[str]]:
    """Union of values for each global filter column across the given files."""
    values: dict[str, set[str]] = {c: set() for c in GLOBAL_FILTER_COLUMNS}
    for path in paths:
        raw_cols = pq.ParquetFile(path).schema_arrow.names
        mapping = canonicalize_columns(raw_cols)
        wanted = [r for r, c in mapping.items() if c in GLOBAL_FILTER_COLUMNS]
        if not wanted:
            continue
        df = _exclude_patients(
            _add_derived(pd.read_parquet(path, columns=wanted).rename(columns=mapping))
        )
        for col in df.columns:
            values[col].update(df[col].dropna().astype(str).unique())
    return {c: sorted(v, key=_natural_key) for c, v in values.items() if v}


def _natural_key(value: str):
    try:
        return (0, float(value), value)
    except ValueError:
        return (1, 0.0, value)


def registry() -> dict[str, dict[str, Dataset]]:
    """All datasets grouped by analysis type key."""
    return {
        "umap": {
            **_glob(
                EDA_RESULTS / "umap", "*.parquet", "1.EDA/scripts/0.generate_umap.py"
            ),
            **_glob(
                EDA_RESULTS / "umap" / "patient_specific",
                "*.parquet",
                "1.EDA/scripts/0.generate_umap.py",
            ),
        },
        "pca": _glob(
            EDA_RESULTS / "pca",
            "*_embeddings.parquet",
            "1.EDA/scripts/2.generate_pca.py",
        ),
        "cell_counts": _glob(
            EDA_RESULTS / "cell_counts",
            "*.parquet",
            "1.EDA/scripts/7.generate_cell_counts.py",
        ),
        "area_vs_volume": {
            k: v
            for k, v in _glob(
                EDA_RESULTS / "area_vs_volume",
                "*_raw.parquet",
                "1.EDA/scripts/17.calculate_area_volume_by_patient_treatment.py",
            ).items()
        },
        "neighbors": _glob(
            EDA_RESULTS / "neighbors",
            "*.parquet",
            "1.EDA/scripts/13.calculate_neighbor_features.py",
        ),
        "intensity": _glob(
            EDA_RESULTS / "intensity",
            "*.parquet",
            "1.EDA/scripts/15.calculate_intensity_values.py",
        ),
        "count_viability": _glob(
            EDA_RESULTS / "count_viability",
            "*_joined.parquet",
            "1.EDA/scripts/10.calculate_count_viability_join.py",
        ),
        "viability_models": _glob(
            VIABILITY_RESULTS,
            "combined_*.parquet",
            "3.viability_prediction_models/scripts/1.viability_prediction.py",
        ),
        "linear_modeling": {
            k: v
            for k, v in _glob(
                LINEAR_MODELING_RESULTS,
                "*.parquet",
                "4.linear_modeling/scripts/0.linear_modeling.py",
            ).items()
            if "feature_name_mapping" not in k
        },
        "variate_importance": _glob(
            VARIATE_IMPORTANCE_RESULTS,
            "variate_hit_*_all_scopes.parquet",
            "4.linear_modeling/scripts/5.calculate_variate_importance.py",
        ),
    }


@st.cache_data(show_spinner=False)
def load_platemaps() -> dict[str, pd.DataFrame]:
    """Named well -> treatment layouts (``config/platemaps/platemap*.csv``),
    keyed by file stem (``platemap1``, ``platemap2``, ...)."""
    if not PLATEMAP_CONFIG_DIR.exists():
        return {}
    files = sorted(PLATEMAP_CONFIG_DIR.glob("platemap[0-9]*.csv"))
    return {f.stem: pd.read_csv(f) for f in files}


@st.cache_data(show_spinner=False)
def load_barcode_platemap() -> pd.DataFrame:
    """Which named platemap each patient was run on, plus its tumor type."""
    path = PLATEMAP_CONFIG_DIR / "barcode_platemap.csv"
    if not path.exists():
        return pd.DataFrame(columns=["patient_tumor", "platemap_number", "tumor_type"])
    df = pd.read_csv(path).rename(columns={"patient_tumor_barcode": "patient_tumor"})
    df = df[~df["patient_tumor"].astype(str).isin(EXCLUDED_PATIENTS)]
    df["tumor_type"] = df["patient_tumor"].map(TUMOR_TYPE_LOOKUP)
    return df.reset_index(drop=True)


def all_filterable_paths() -> tuple[str, ...]:
    """Files scanned to populate the shared sidebar filters (small/medium ones)."""
    reg = registry()
    keep = []
    for group in (
        "umap",
        "cell_counts",
        "area_vs_volume",
        "neighbors",
        "count_viability",
    ):
        keep += [str(d.path) for d in reg[group].values()]
    keep += [
        str(d.path)
        for k, d in reg["viability_models"].items()
        if k in ("combined_fold_metrics", "combined_summary_metrics")
    ]
    return tuple(keep)
