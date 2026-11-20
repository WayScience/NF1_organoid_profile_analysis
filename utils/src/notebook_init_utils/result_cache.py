"""
Skip-if-unchanged result caching for analysis steps.

A cached output is reused only when every one of the following holds:

- the output file exists,
- the step's code file (its mirrored script) has not changed, and
- every input file has the same path, size and modification time as when the
  output was last written.

Each step keeps a small manifest, ``<results_dir>/_cache_manifests/<step>.parquet``,
that maps each output to the fingerprint it was produced from. Checking the
manifest, rather than only the presence of an output file, is what stops a
rerun on new data from silently reusing results computed from old data.

Set the environment variable ``RESULTS_CACHE_FORCE=1`` to ignore every cache
and recompute.

The R counterpart lives in ``utils/r_result_cache.r`` and uses the same
manifest layout.
"""

import hashlib
import os
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

MANIFEST_DIRNAME = "_cache_manifests"
FORCE_ENV_VAR = "RESULTS_CACHE_FORCE"


def _input_signature(path: Path) -> str:
    """Return a string that changes whenever the file is rewritten or replaced."""
    stat = path.stat()
    return f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"


def fingerprint(code_path: Path, inputs: Iterable[Path]) -> str:
    """
    Fingerprint a step's code and inputs.

    Input files are identified by path, size and mtime rather than by content,
    because the aggregated profiles run to tens of gigabytes and hashing them
    on every run would cost more than the caching saves.

    Parameters
    ----------
    code_path : pathlib.Path
        The step's script. Its content is hashed, so any edit to the code,
        including its constants, invalidates the cache.
    inputs : Iterable[pathlib.Path]
        Every file the step reads.

    Returns
    -------
    str
        A hex digest.
    """
    digest = hashlib.md5()
    digest.update(Path(code_path).read_bytes())
    for input_path in sorted(Path(path) for path in inputs):
        digest.update(_input_signature(input_path).encode())
    return digest.hexdigest()


def _manifest_path(results_dir: Path, step: str) -> Path:
    return Path(results_dir) / MANIFEST_DIRNAME / f"{step}.parquet"


def _read_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    manifest = pd.read_parquet(manifest_path)
    return dict(zip(manifest["output"], manifest["fingerprint"]))


def is_fresh(
    step: str,
    results_dir: Path,
    code_path: Path,
    outputs: Sequence[Path],
    inputs: Sequence[Path],
) -> bool:
    """
    Return True if all outputs exist and were produced from the current code and inputs.

    Parameters
    ----------
    step : str
        Name of the analysis step, used to name its manifest (e.g. ``"3.calculate_x"``).
    results_dir : pathlib.Path
        The module's results directory, where the manifest is kept.
    code_path : pathlib.Path
        The step's script, hashed as part of the fingerprint.
    outputs : Sequence[pathlib.Path]
        The files this step writes.
    inputs : Sequence[pathlib.Path]
        The files this step reads.

    Returns
    -------
    bool
        True if the outputs can be reused without recomputing.
    """
    if os.environ.get(FORCE_ENV_VAR, "") == "1":
        return False
    recorded = _read_manifest(_manifest_path(results_dir, step))
    current = fingerprint(code_path, inputs)
    return all(
        Path(output).exists() and recorded.get(str(output)) == current
        for output in outputs
    )


def mark_fresh(
    step: str,
    results_dir: Path,
    code_path: Path,
    outputs: Sequence[Path],
    inputs: Sequence[Path],
) -> None:
    """
    Record that the outputs were just written from the current code and inputs.

    Call this immediately after each output is written. A run killed partway
    through then keeps the outputs it finished and recomputes only the rest.
    """
    manifest_path = _manifest_path(results_dir, step)
    recorded = _read_manifest(manifest_path)
    current = fingerprint(code_path, inputs)
    recorded.update({str(output): current for output in outputs})
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"output": list(recorded.keys()), "fingerprint": list(recorded.values())}
    ).to_parquet(manifest_path, index=False)
