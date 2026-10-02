"""Move this app's precomputed data between the main repo, the local
``data/`` folder, and the HF Bucket it's deployed from.

``data/`` is .gitignore'd on purpose -- it's never committed. Locally it's
also exactly what ``data_io.py``'s ``DATA_DIR`` reads by default, so
``download`` doubles as "get the app runnable locally."

    python sync_data.py upload   [--source /path/to/main/repo] [--bucket ns/name]
    python sync_data.py download [--bucket ns/name] [--dest data]

``upload``: stage the trimmed set of results data_io.py actually reads from
a sibling checkout of the main analysis repo into ``data/``, then push
``data/`` to the bucket.
``download``: pull the bucket down into ``data/`` (or ``--dest``) -- the
same thing the running app does on its own the first time it starts without
a local ``data/`` already there (see data_io.py's ``_sync_bucket_once``).
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import sync_bucket

DEFAULT_BUCKET = "lippincm/NF1_3D_organoid_data_viewing-storage"

# (source path relative to the main repo's root, destination path relative to
# data/, kind) -- "dir" copies the whole directory, "file" copies one file,
# "glob:<pattern>" copies only the top-level files in that directory matching
# the pattern. Kept in sync with data_io.py's EDA_RESULTS/VIABILITY_RESULTS/etc.
MANIFEST: list[tuple[str, str, str]] = [
    ("1.EDA/results/umap", "eda/umap", "dir"),
    ("1.EDA/results/pca", "eda/pca", "dir"),
    ("1.EDA/results/correlation", "eda/correlation", "dir"),
    ("1.EDA/results/cell_counts", "eda/cell_counts", "dir"),
    ("1.EDA/results/area_vs_volume", "eda/area_vs_volume", "dir"),
    ("1.EDA/results/neighbors", "eda/neighbors", "dir"),
    ("1.EDA/results/intensity", "eda/intensity", "dir"),
    ("1.EDA/results/count_viability", "eda/count_viability", "dir"),
    (
        "3.viability_prediction_models/model_results",
        "viability_models",
        "glob:combined_*.parquet",
    ),
    ("4.linear_modeling/results/linear_modeling", "linear_modeling/models", "dir"),
    (
        "4.linear_modeling/results/variate_importance",
        "linear_modeling/variate_importance",
        "dir",
    ),
    ("config/platemaps", "platemaps", "dir"),
    (
        "data/viabilities/combined_platemaps.parquet",
        "platemaps/combined_platemaps.parquet",
        "file",
    ),
    # not read by the app yet -- staged ahead of a future 2D-vs-3D section
    ("2.2d_vs_3d_analysis/results", "2d_vs_3d", "dir"),
]

_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc")


def _dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def stage_from_main_repo(source_root: Path, data_dir: Path) -> None:
    """Copy only what data_io.py reads from ``source_root`` into ``data_dir``."""
    for src_rel, dest_rel, kind in MANIFEST:
        src = source_root / src_rel
        dest = data_dir / dest_rel
        if kind == "dir":
            if not src.is_dir():
                print(f"  skip (missing): {src_rel}")
                continue
            if dest.exists():
                shutil.rmtree(dest)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, dest, ignore=_IGNORE)
            print(f"  staged dir:  {src_rel} -> data/{dest_rel}")
        elif kind == "file":
            if not src.is_file():
                print(f"  skip (missing): {src_rel}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"  staged file: {src_rel} -> data/{dest_rel}")
        elif kind.startswith("glob:"):
            if not src.is_dir():
                print(f"  skip (missing): {src_rel}")
                continue
            pattern = kind.removeprefix("glob:")
            dest.mkdir(parents=True, exist_ok=True)
            matches = sorted(src.glob(pattern))
            for match in matches:
                shutil.copy2(match, dest / match.name)
            print(
                f"  staged {len(matches)} file(s) matching '{pattern}': "
                f"{src_rel} -> data/{dest_rel}"
            )
        else:
            raise ValueError(f"Unknown manifest kind: {kind}")


def cmd_upload(args: argparse.Namespace) -> None:
    source_root = args.source.resolve()
    if not (source_root / "1.EDA").is_dir():
        raise SystemExit(
            f"'{source_root}' doesn't look like the main repo (no 1.EDA/ dir) -- "
            "pass --source /path/to/NF1_organoid_profile_analysis"
        )
    data_dir = args.dest.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Staging from {source_root.name}/ into {data_dir.name}/ ...")
    stage_from_main_repo(source_root, data_dir)
    print(f"Staged {_dir_size(data_dir) / 1e9:.2f} GB.")

    uri = f"hf://buckets/{args.bucket}/data"
    print(f"Pushing {data_dir.name}/ -> {uri} ...")
    sync_bucket(str(data_dir), uri, delete=args.delete)
    print("Upload done.")


def cmd_download(args: argparse.Namespace) -> None:
    data_dir = args.dest.resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    uri = f"hf://buckets/{args.bucket}/data"
    print(f"Pulling {uri} -> {data_dir.name}/ ...")
    sync_bucket(uri, str(data_dir), delete=args.delete)
    print(f"Download done. {data_dir.name}/ is {_dir_size(data_dir) / 1e9:.2f} GB.")


def main() -> None:
    this_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    up = sub.add_parser(
        "upload", help="stage from the main repo, then push to the bucket"
    )
    up.add_argument("--source", type=Path, default=this_dir.parent)
    up.add_argument("--dest", type=Path, default=this_dir / "data")
    up.add_argument("--bucket", default=DEFAULT_BUCKET)
    up.add_argument(
        "--delete",
        action="store_true",
        help="also remove bucket files not present locally",
    )
    up.set_defaults(func=cmd_upload)

    down = sub.add_parser("download", help="pull the bucket into data/ for local use")
    down.add_argument("--dest", type=Path, default=this_dir / "data")
    down.add_argument("--bucket", default=DEFAULT_BUCKET)
    down.add_argument(
        "--delete",
        action="store_true",
        help="also remove local files not present in the bucket",
    )
    down.set_defaults(func=cmd_download)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
