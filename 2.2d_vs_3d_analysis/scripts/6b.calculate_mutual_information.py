#!/usr/bin/env python
# coding: utf-8

# # Calculate Mutual Information Between 2D and 3D Profiles
#
# Mutual information (MI) is a measure of the amount of information that one random variable contains about another.
# It quantifies the reduction in uncertainty about one variable given knowledge of the other.
#
# Here we want to know **whether 2D and 3D profiles measure the same or different things**, i.e. how much redundant information exists between them.
#
# - The overlap $I(\text{2D}; \text{3D})$ quantifies the amount of information shared between the 2D and 3D profiles.
# - The 3D unique information $H(\text{3D} \mid \text{2D})$ is the information present in the 3D profile but not shared with the 2D profile.
# - The 2D unique information $H(\text{2D} \mid \text{3D})$ is the information present in the 2D profile but not shared with the 3D profile.
#
# We run two separate analyses:
#
# 1. **Whole-profile MI**: one number per 2D × 3D comparison, computed on the top principal components of each profile.
# 2. **Per-feature MI**: a matrix of MI between every 2D feature and every 3D feature, which shows *which* features are redundant.
#
# Both use the Kraskov–Stögbauer–Grassberger (KSG) $k$-nearest-neighbour estimator rather than histogram binning.

# ## Mutual information
#
# For two random variables $X$ (2D profile) and $Y$ (3D profile) with joint density $f(x, y)$:
#
# $$
# I(X;Y) = \iint f(x,y)\,\log\frac{f(x,y)}{f(x)\,f(y)}\,dx\,dy
# = D_{\mathrm{KL}}\!\big(f_{XY}\,\|\,f_X f_Y\big)
# $$
#
# i.e. how far the joint distribution is from what it would be if 2D and 3D were independent. Equivalently:
#
# $$
# I(X;Y) = H(X) - H(X\mid Y) = H(X) + H(Y) - H(X,Y)
# $$
#
# Two properties matter for this comparison:
#
# - $I(X;Y) \ge 0$, with equality only when 2D and 3D are independent.
# - MI is unchanged by any invertible transform of $X$ or $Y$ (rescaling, standardizing, change of units).
#   Differential entropy $H$ is *not*: rescaling a feature by $a$ shifts its entropy by $\log|a|$.
#
# ### Why not $I/H(\text{2D})$ and $I/H(\text{3D})$?
#
# For continuous features, $H$ is a differential entropy. It depends on units and can be negative, so ratios like $I/H$ are not meaningful
# (the histogram entropies from notebook 6 depend on the bin count in the same way).
# Instead we report the **information coefficient of correlation**:
#
# $$
# r_I = \sqrt{1 - 2^{-2I}} \in [0, 1]
# $$
#
# with $I$ in bits. For jointly Gaussian variables this recovers $|\rho|$ exactly, so it can be read like a correlation: 0 means 2D and 3D share nothing, and values near 1 mean they measure the same thing.

# ## Pairing 2D and 3D profiles
#
# MI needs **paired** observations: the same unit measured in both 2D and 3D.
# Single cells and organoids are segmented separately in 2D and 3D, so they can't be matched one to one.
# We therefore pair at the **aggregated** level: every profile is collapsed to the median feature value per group
# (patient/tumor × treatment, plus dose when both files have it), and 2D and 3D groups are inner-joined on those keys.
#
# - The `agg` stage files are already aggregated. They are re-grouped on the shared keys, which is a no-op if they're already at that level.
# - The `normal` and `fs` stages are single-cell or organoid level, so they're aggregated here, on the fly, to the same keys.
# - 2D and 3D are only compared at the same resolution (organoid ↔ organoid, single cell ↔ single cell) and the same stage.
#   The 2D `agg` stage is compared with both 3D aggregation modes (`from_fs`, `from_norm`).
#
# > **Check the join keys.** `JOIN_KEY_CANDIDATES` below lists the column names tried for each key. The pairing report prints how many groups matched. A low count usually means the 2D and 3D files spell patient or treatment values differently.

# In[ ]:


import hashlib
import inspect
import os
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from joblib import Parallel, delayed
from notebook_init_utils.notebook_init_utils import init_notebook
from notebook_init_utils.result_cache import FORCE_ENV_VAR, is_fresh, mark_fresh
from scipy.spatial import cKDTree
from scipy.special import digamma
from sklearn.decomposition import PCA
from tqdm.auto import tqdm

root_dir, in_notebook = init_notebook()


# In[ ]:


_2d_base_dir = root_dir / "data" / "profiles_2D" / "all_patients"
_3d_dir = root_dir / "data" / "profiles_3D" / "all_patients"
_3d_stage_dirs = {
    "normal": _3d_dir / "0.normalized_profiles",
    "fs": _3d_dir / "1.feature_selected_profiles",
    "agg": _3d_dir / "2.aggregated_profiles",
}
_mi_dir = root_dir / "2.2d_vs_3d_analysis" / "results" / "mutual_information"
_agg_dir = _mi_dir / "_aggregated"  # per-file median profiles, reused across runs
# Caching: an output is skipped when it was built from unchanged inputs and code.
# Each layer is keyed on only the functions and parameters it depends on (see
# `code_text`), so editing one analysis does not force the others to rerun.
CACHE_STEP = "6b.calculate_mutual_information"
AGG_CACHE_STEP = f"{CACHE_STEP}.aggregate"
PROFILE_CACHE_STEP = f"{CACHE_STEP}.profile"
FEATURE_CACHE_STEP = f"{CACHE_STEP}.feature"
# True: a comparison whose output files already exist is skipped, even if the
# code or inputs changed since. False: reuse only outputs whose fingerprint
# matches. Set RESULTS_CACHE_FORCE=1 to recompute everything either way.
SKIP_EXISTING = True
CODE_PATH = (
    root_dir / "2.2d_vs_3d_analysis" / "scripts" / "6b.calculate_mutual_information.py"
).resolve(strict=True)
results_root = (root_dir / "2.2d_vs_3d_analysis" / "results").resolve()

# 2D has three separate projection methods, each in its own directory
_2d_projections = [
    ("MIP", "max_projection"),
    ("Middle Slice", "middle_slice"),
    ("Middle N Slices", "middle_n_slice"),
]

# (resolution, stage, filename) for each 2D type x stage
_2d_stage_filenames = [
    ("organoid", "normal", "organoid_profiles.parquet"),
    ("organoid", "fs", "organoid_fs_profiles.parquet"),
    ("organoid", "agg", "organoid_agg_profiles.parquet"),
    ("sc", "normal", "sc_profiles.parquet"),
    ("sc", "fs", "sc_fs_profiles.parquet"),
    ("sc", "agg", "sc_agg_profiles.parquet"),
]

# (slug, group_label, resolution, file_prefix) for each 3D type
_3d_types = [
    ("organoid_handcrafted", "ZedProfiler", "organoid", "organoid"),
    ("organoid_sammed", "SAM-Med3D", "organoid", "sammed_organoid"),
    ("sc_handcrafted", "ZedProfiler", "sc", "sc"),
    ("sc_sammed", "SAM-Med3D", "sc", "sammed_sc"),
    (
        "sc_sammed_nucleocentric",
        "Nucleocentric SAM-Med3D",
        "sc",
        "sammed_nucleocentric",
    ),
    (
        "sc_nucleocentric_morphem",
        "Nucleocentric MorphEM",
        "sc",
        "nucleocentric_morphem",
    ),
]

profiles_2d: Dict[str, dict] = {}
for proj_label, proj_dir_name in _2d_projections:
    proj_dir = _2d_base_dir / proj_dir_name
    for resolution, stage, filename in _2d_stage_filenames:
        suffix = "" if stage == "normal" else f"_{stage}"
        profiles_2d[f"2D_{proj_dir_name}_{resolution}{suffix}"] = {
            "input": next(proj_dir.glob(filename)),
            "group_label": proj_label,
            "resolution": resolution,
            "stage": stage,
            "agg_mode": None,
        }

profiles_3d: Dict[str, dict] = {}
for slug, group_label, resolution, file_prefix in _3d_types:
    for stage, stage_dir in _3d_stage_dirs.items():
        # agg stage is written two ways (from_fs / from_norm); keep both
        modes = ["from_fs", "from_norm"] if stage == "agg" else [None]
        for mode in modes:
            path = sorted(
                f
                for f in stage_dir.glob("*.parquet")
                if f.name.startswith(f"{file_prefix}_")
                and (mode is None or f.stem.endswith(mode))
            )[0]
            mode_suffix = f"_{mode}" if mode is not None else ""
            suffix = ("" if stage == "normal" else f"_{stage}") + mode_suffix
            profiles_3d[f"3D_{slug}{suffix}"] = {
                "input": path,
                "group_label": group_label,
                "resolution": resolution,
                "stage": stage,
                "agg_mode": mode,
            }

# Every 2D x 3D comparison at matching resolution and stage
comparisons: List[dict] = []
for key_2d, cfg_2d in profiles_2d.items():
    for key_3d, cfg_3d in profiles_3d.items():
        if (cfg_2d["resolution"], cfg_2d["stage"]) != (
            cfg_3d["resolution"],
            cfg_3d["stage"],
        ):
            continue
        name = f"{key_2d}__vs__{key_3d}"
        comparisons.append(
            {
                "name": name,
                "key_2d": key_2d,
                "key_3d": key_3d,
                "cfg_2d": cfg_2d,
                "cfg_3d": cfg_3d,
                "profile_output": _mi_dir / f"{name}_profile_mi.parquet",
                "feature_output": _mi_dir / f"{name}_feature_mi.parquet",
                "feature_summary_output": _mi_dir
                / f"{name}_feature_mi_summary.parquet",
            }
        )
print(f"{len(comparisons)} 2D x 3D comparisons")


# In[ ]:


# --- Pairing ---------------------------------------------------------------
# canonical key -> column names to try, in order. 2D and 3D use different
# patient/tumor column names; patient_tumor columns are preferred because some
# patients (e.g. NF0014) have multiple tumors.
JOIN_KEY_CANDIDATES: Dict[str, List[str]] = {
    "Metadata_patient": [
        "Metadata_patient_tumor",
        "Metadata_Biology_PatientTumor",
        "Metadata_patient",
    ],
    "Metadata_treatment": [
        "Metadata_treatment",
        "Metadata_Treatment",
        "Metadata_Biology_Treatment",
        "Metadata_Experiment_Treatment",
    ],
    "Metadata_dose": [
        "Metadata_dose",
        "Metadata_Dose",
        "Metadata_Biology_Dose",
        "Metadata_Experiment_Dose",
    ],
}
REQUIRED_KEYS = ["Metadata_patient", "Metadata_treatment"]
MAX_NAN_FRACTION = 0.1  # drop features missing in >10% of paired groups

# --- KSG estimator ---------------------------------------------------------
K_NEIGHBORS = 3  # k in KSG; 3-10 is typical, smaller = less bias, more variance
SEED = 0

# --- Whole-profile MI ------------------------------------------------------
PCA_COMPONENTS = [2, 5, 10]  # PCs kept per side; several values = sensitivity check
N_PERMUTATIONS = 200  # permutation null for each profile-level MI

# --- Per-feature MI --------------------------------------------------------
N_FEATURE_NULL_PERMUTATIONS = 1  # shuffled copies of the full matrix for the null
# Worker processes for per-feature MI. The work is CPU-bound, so physical cores
# (half of the logical CPUs on SMT machines) is the useful maximum.
N_JOBS = max(1, (os.cpu_count() or 2) // 2)


# In[ ]:


def code_text(*funcs, **params) -> str:
    # Source of `funcs` plus parameter values: the cache fingerprint of a result
    sources = [inspect.getsource(f) for f in funcs]
    values = [f"{name}={value!r}" for name, value in sorted(params.items())]
    return "\n".join(sources + values)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    # Same rule as notebook 6: numeric columns without the Metadata_ prefix
    return [
        col
        for col in df.columns
        if not col.startswith("Metadata_") and pd.api.types.is_numeric_dtype(df[col])
    ]


def resolve_join_keys(columns: List[str]) -> Dict[str, str]:
    # Map each canonical key to the first candidate column present in the file
    resolved = {}
    for canonical, candidates in JOIN_KEY_CANDIDATES.items():
        for col in candidates:
            if col in columns:
                resolved[canonical] = col
                break
    return resolved


_agg_cache: Dict[Tuple[str, Tuple[str, ...]], pd.DataFrame] = {}


def load_aggregated(path: pathlib.Path, keys: List[str]) -> pd.DataFrame:
    # Load a profile and take the median of every feature per group of `keys`
    cache_key = (str(path), tuple(keys))
    if cache_key in _agg_cache:
        return _agg_cache[cache_key]

    # Disk cache: one file per input x join keys; several 2D inputs share a name
    tag = hashlib.md5(
        f"{path.relative_to(root_dir)}|{','.join(keys)}".encode()
    ).hexdigest()[:10]
    agg_path = _agg_dir / f"{path.stem}__{tag}.parquet"
    agg_code = code_text(
        load_aggregated,
        resolve_join_keys,
        get_feature_columns,
        JOIN_KEY_CANDIDATES=JOIN_KEY_CANDIDATES,
    )
    if is_fresh(AGG_CACHE_STEP, results_root, CODE_PATH, [agg_path], [path], agg_code):
        _agg_cache[cache_key] = pd.read_parquet(agg_path)
        return _agg_cache[cache_key]

    df = pd.read_parquet(path)
    key_map = resolve_join_keys(list(df.columns))
    for canonical in keys:
        col = df[key_map[canonical]]
        if pd.api.types.is_numeric_dtype(col):
            # 2D stores dose as int (1), 3D as float (1.0); format both as "1"
            col = col.map(lambda v: f"{v:g}")
        df[canonical] = col.astype(str).str.strip()
    features = get_feature_columns(df)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan)
    agg = df.groupby(keys)[features].median()

    _agg_dir.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(agg_path)
    mark_fresh(AGG_CACHE_STEP, results_root, CODE_PATH, [agg_path], [path], agg_code)
    _agg_cache[cache_key] = agg
    return agg


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    # Drop mostly missing / constant / overflowing features; impute the rest
    df = df.loc[:, df.isna().mean() <= MAX_NAN_FRACTION]
    df = df.fillna(df.median())
    with np.errstate(over="ignore", invalid="ignore"):
        std = df.std()
    return df.loc[:, np.isfinite(std) & (std > 0)].astype(float)


def pair_profiles(
    path_2d: pathlib.Path, path_3d: pathlib.Path
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    # Return aggregated 2D and 3D feature tables with identical, aligned rows
    keys_2d = resolve_join_keys(pq.read_schema(path_2d).names)
    keys_3d = resolve_join_keys(pq.read_schema(path_3d).names)
    keys = [k for k in JOIN_KEY_CANDIDATES if k in keys_2d and k in keys_3d]
    missing = [k for k in REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(
            f"Required join keys {missing} not found in both files. "
            f"Update JOIN_KEY_CANDIDATES.\n2D: {path_2d.name}\n3D: {path_3d.name}"
        )

    agg_2d = load_aggregated(path_2d, keys)
    agg_3d = load_aggregated(path_3d, keys)
    common = agg_2d.index.intersection(agg_3d.index)
    x_2d = clean_features(agg_2d.loc[common])
    x_3d = clean_features(agg_3d.loc[common])
    return x_2d, x_3d, keys


# ## The KSG estimator
#
# Histogram estimates of MI need a bin size, are biased, and break down beyond a few dimensions.
# The KSG estimator (Kraskov et al., 2004, estimator 1) works directly on the $N$ paired samples $(x_i, y_i)$:
#
# 1. For each point, find the distance $\varepsilon_i$ to its $k$-th nearest neighbour in the **joint** $(x, y)$ space, using the max-norm
#    $\lVert (x,y) \rVert = \max(\lVert x \rVert_\infty, \lVert y \rVert_\infty)$.
# 2. Count the neighbours *within that same distance* in each marginal space separately:
#    $n_x(i) = \#\{j \ne i : \lVert x_j - x_i \rVert < \varepsilon_i\}$, and $n_y(i)$ likewise.
# 3. Estimate
#
# $$
# \hat I(X;Y) = \psi(k) + \psi(N) - \big\langle \psi(n_x + 1) + \psi(n_y + 1) \big\rangle \quad \text{(nats)}
# $$
#
# where $\psi$ is the digamma function and $\langle\cdot\rangle$ averages over all points. Divide by $\ln 2$ for bits.
#
# **Intuition:** if 2D and 3D are strongly dependent, a point's joint neighbours are also its neighbours in each marginal space.
# The joint ball is then small relative to the marginal crowds, so $n_x$ and $n_y$ stay small and $\hat I$ is large.
# If they're independent, the joint ball has to reach out a long way, it sweeps up many marginal neighbours, and $\hat I \approx 0$.
#
# Features are z-scored first, which doesn't change MI but keeps the max-norm from being dominated by one large-scale feature.
# A $10^{-10}$ jitter breaks ties between identical values.

# In[ ]:


LN2 = np.log(2.0)


def _standardize(a: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    # z-score each column and add a tiny jitter to break ties
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    a = (a - a.mean(axis=0)) / a.std(axis=0)
    return a + 1e-10 * rng.standard_normal(a.shape)


def ksg_mi(
    x: np.ndarray, y: np.ndarray, k: int = 3, rng: np.random.Generator = None
) -> float:
    # KSG estimator 1 of I(X;Y) in bits; x is (N, p), y is (N, q)
    rng = rng if rng is not None else np.random.default_rng(SEED)
    x, y = _standardize(x, rng), _standardize(y, rng)
    n = x.shape[0]

    # 1) distance to the k-th neighbour in the joint space (k+1: includes self)
    joint = np.hstack([x, y])
    eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, -1]

    # 2) marginal neighbours strictly closer than eps (minus self)
    r = np.nextafter(eps, 0)
    n_x = cKDTree(x).query_ball_point(x, r=r, p=np.inf, return_length=True) - 1
    n_y = cKDTree(y).query_ball_point(y, r=r, p=np.inf, return_length=True) - 1

    # 3) KSG formula
    mi_nats = digamma(k) + digamma(n) - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    return mi_nats / LN2


def info_correlation(mi_bits: float) -> float:
    # Information coefficient of correlation r_I = sqrt(1 - 2^(-2I))
    return float(np.sqrt(1.0 - 2.0 ** (-2.0 * max(mi_bits, 0.0))))


# ### Sanity check against the Gaussian closed form
#
# For jointly Gaussian $X \in \mathbb{R}^p$, $Y \in \mathbb{R}^q$ with canonical correlations $\rho_1, \dots, \rho_{\min(p,q)}$, MI has an exact value:
#
# $$
# I(X;Y) = -\tfrac{1}{2}\sum_i \log_2\!\big(1 - \rho_i^2\big)
# $$
#
# We simulate data with known canonical correlations and check that the estimator recovers it.
# The per-feature function defined further down is checked against the same values.
#
# KSG is biased **downward** when the MI or the dimension is large relative to $N$. You'll see this in the multi-dimensional case below.
# So absolute values are lower bounds, and comparisons are fairest between estimates made with the same $c$ and similar $N$.

# In[ ]:


def simulate_gaussian(rhos: List[float], n: int, rng: np.random.Generator):
    # X, Y with canonical correlations `rhos`, then mixed by random linear maps
    # (invertible, so MI is unchanged)
    d = len(rhos)
    z_x = rng.standard_normal((n, d))
    noise = rng.standard_normal((n, d))
    z_y = z_x * np.array(rhos) + noise * np.sqrt(1 - np.array(rhos) ** 2)
    return z_x @ rng.standard_normal((d, d)), z_y @ rng.standard_normal((d, d))


rng = np.random.default_rng(SEED)
for rhos in ([0.0], [0.5], [0.9], [0.9, 0.5, 0.0]):
    x_sim, y_sim = simulate_gaussian(rhos, n=1000, rng=rng)
    true_mi = -0.5 * np.sum(np.log2(1 - np.array(rhos) ** 2))
    est_mi = ksg_mi(x_sim, y_sim, k=K_NEIGHBORS, rng=rng)
    print(f"rhos={rhos}: true MI = {true_mi:.3f} bits, KSG = {est_mi:.3f} bits")


# ## Analysis 1: whole-profile MI
#
# Profiles have hundreds to thousands of features but only as many paired rows as there are aggregated groups.
# KSG needs the number of samples to be large relative to the dimension, so each side is first reduced to its top $c$ principal components:
#
# $$
# X \in \mathbb{R}^{N \times p_{2D}} \;\to\; X_c \in \mathbb{R}^{N \times c}, \qquad
# Y \in \mathbb{R}^{N \times p_{3D}} \;\to\; Y_c \in \mathbb{R}^{N \times c}
# $$
#
# and we estimate $I(X_c; Y_c)$. Using the same $c$ on both sides also takes care of 2D and 3D having different numbers of features.
# By the data processing inequality, $I(X_c; Y_c) \le I(X; Y)$, so this is a lower bound on the MI of the full profiles.
# We repeat it for several values of $c$ to check that the conclusions don't hinge on that choice.
#
# **Permutation null.** KSG has finite-sample bias, so even independent data give $\hat I$ a little above or below 0.
# We shuffle the 3D rows $B$ times, which breaks the pairing but keeps both marginals, and re-estimate each time:
#
# $$
# p = \frac{1 + \#\{b : \hat I_b^{\text{null}} \ge \hat I\}}{B + 1}, \qquad
# \hat I_{\text{corrected}} = \max\!\big(\hat I - \overline{\hat I^{\text{null}}},\, 0\big)
# $$
#
# $r_I$ is computed from $\hat I_{\text{corrected}}$.

# In[ ]:


def profile_mi(
    x_2d: pd.DataFrame,
    x_3d: pd.DataFrame,
    n_components: int,
    k: int = 3,
    n_permutations: int = 200,
    rng: np.random.Generator = None,
) -> dict:
    # MI between the top-n_components PCs of the 2D and 3D profiles
    rng = rng if rng is not None else np.random.default_rng(SEED)
    n = len(x_2d)
    c_2d = min(n_components, x_2d.shape[1], n - 2)
    c_3d = min(n_components, x_3d.shape[1], n - 2)

    def to_pcs(df, c):
        z = (df.values - df.values.mean(axis=0)) / df.values.std(axis=0)
        pca = PCA(n_components=c, random_state=SEED).fit(z)
        return pca.transform(z), pca.explained_variance_ratio_.sum()

    pcs_2d, var_2d = to_pcs(x_2d, c_2d)
    pcs_3d, var_3d = to_pcs(x_3d, c_3d)

    mi = ksg_mi(pcs_2d, pcs_3d, k=k, rng=rng)
    null = np.array(
        [
            ksg_mi(pcs_2d, pcs_3d[rng.permutation(n)], k=k, rng=rng)
            for _ in tqdm(
                range(n_permutations),
                desc=f"Profile MI null ({n_components} PCs)",
                leave=False,
            )
        ]
    )
    mi_corrected = max(mi - null.mean(), 0.0)
    return {
        "n_components": n_components,
        "n_components_2d": c_2d,
        "n_components_3d": c_3d,
        "explained_variance_2d": var_2d,
        "explained_variance_3d": var_3d,
        "mi_bits": mi,
        "null_mean_bits": null.mean(),
        "null_sd_bits": null.std(),
        "null_p95_bits": np.quantile(null, 0.95),
        "p_value": (1 + np.sum(null >= mi)) / (n_permutations + 1),
        "mi_corrected_bits": mi_corrected,
        "info_correlation": info_correlation(mi_corrected),
    }


# ## Analysis 2: per-feature MI
#
# For every 2D feature $a$ and 3D feature $b$, we estimate $M_{ab} = \hat I(X_a; Y_b)$ with the same KSG estimator in one dimension per side.
# That's $p_{2D} \times p_{3D}$ estimates (millions for the larger comparisons), so the pairs are split across `N_JOBS` worker processes.
# Each pair uses KD-trees, as `ksg_mi` does. The 1D tree for each feature is built once and reused across pairs.
# A vectorized version that builds $N \times N$ distance matrices was tried first. It is limited by memory bandwidth and does not speed up with more cores, but it gives the same result.
# With $D^{(a)}_{ij} = |x_{ia} - x_{ja}|$ and $D^{(b)}_{ij} = |y_{ib} - y_{jb}|$:
#
# $$
# \varepsilon_i^{(ab)} = k\text{-th smallest}_{j \ne i} \; \max\!\big(D^{(a)}_{ij}, D^{(b)}_{ij}\big),
# \qquad
# n_x(i) = \#\{j \ne i : D^{(a)}_{ij} < \varepsilon_i^{(ab)}\}
# $$
#
# This is the same formula as above, applied to one feature per side.
#
# **Null threshold.** The whole matrix is recomputed with the 3D rows shuffled. The 95th percentile of those null MIs, $\tau_{95}$, is the threshold for "more shared information than chance".
#
# **Summary.** For each feature we record its best-matching feature in the other modality, $\max_a M_{ab}$ for a 3D feature $b$ (and $\max_b M_{ab}$ for a 2D feature $a$).
# A 3D feature is labelled **redundant with 2D** if $\max_a M_{ab} > \tau_{95}$.
# The fraction of redundant 3D features is a feature-level answer to the question of how much of what 3D measures 2D also measures.
# This is pairwise only: a 3D feature that is predictable from a *combination* of 2D features but not from any single one won't be flagged.

# In[ ]:


def _feature_mi_rows(z_2d: np.ndarray, z_3d: np.ndarray, k: int) -> np.ndarray:
    # (p_a, p_3d) block of 1D-1D KSG MI in nats for standardized inputs.
    # Same steps as ksg_mi; each feature's 1D tree is built once and reused.
    n = z_2d.shape[0]
    const = digamma(k) + digamma(n)
    trees_3d = [cKDTree(z_3d[:, [b]]) for b in range(z_3d.shape[1])]
    mi = np.empty((z_2d.shape[1], z_3d.shape[1]))
    for a in range(z_2d.shape[1]):
        x = z_2d[:, [a]]
        tree_x = cKDTree(x)
        for b, tree_y in enumerate(trees_3d):
            y = z_3d[:, [b]]
            joint = np.hstack([x, y])
            eps = cKDTree(joint).query(joint, k=k + 1, p=np.inf)[0][:, -1]
            r = np.nextafter(eps, 0)  # strictly closer than eps
            n_x = tree_x.query_ball_point(x, r=r, p=np.inf, return_length=True) - 1
            n_y = tree_y.query_ball_point(y, r=r, p=np.inf, return_length=True) - 1
            mi[a, b] = const - np.mean(digamma(n_x + 1) + digamma(n_y + 1))
    return mi


def pairwise_feature_mi(
    x_2d: np.ndarray,
    x_3d: np.ndarray,
    k: int = 3,
    rng: np.random.Generator = None,
    n_jobs: int = 1,
    desc: str = None,
) -> np.ndarray:
    # (p_2d, p_3d) matrix of 1D-1D KSG MI in bits, computed in parallel over
    # blocks of 2D features. Standardizing (and its jitter) happens here, before
    # the split, so the result does not depend on n_jobs.
    rng = rng if rng is not None else np.random.default_rng(SEED)
    z_2d, z_3d = _standardize(x_2d, rng), _standardize(x_3d, rng)
    p_2d = z_2d.shape[1]

    # ~4 blocks per worker balances load against rebuilding the 3D trees
    block = max(1, -(-p_2d // (4 * n_jobs)))
    starts = list(range(0, p_2d, block))
    tasks = (delayed(_feature_mi_rows)(z_2d[:, s : s + block], z_3d, k) for s in starts)
    mi = np.empty((p_2d, z_3d.shape[1]))
    # one tick per 2D feature; no bar when desc is None
    with tqdm(total=p_2d, desc=desc, leave=False, disable=desc is None) as progress:
        results = Parallel(n_jobs=n_jobs, return_as="generator")(tasks)
        for s, rows in zip(starts, results):
            mi[s : s + len(rows)] = rows
            progress.update(len(rows))
    return mi / LN2


# check against the tree-based estimator and the Gaussian truth
rng = np.random.default_rng(SEED)
x_sim, y_sim = simulate_gaussian([0.9], n=300, rng=rng)
x_ind = rng.standard_normal((300, 1))
m = pairwise_feature_mi(np.hstack([x_sim, x_ind]), y_sim, k=K_NEIGHBORS, rng=rng)
print(f"true MI = {-0.5 * np.log2(1 - 0.81):.3f} bits")
print(f"KSG (tree) = {ksg_mi(x_sim, y_sim, k=K_NEIGHBORS, rng=rng):.3f} bits")
print(
    f"KSG (pairwise) dependent pair = {m[0, 0]:.3f}, independent pair = {m[1, 0]:.3f}"
)

# parallel and serial results must match exactly
x_par = rng.standard_normal((120, 40))
y_par = rng.standard_normal((120, 30))
assert np.array_equal(
    pairwise_feature_mi(x_par, y_par, k=K_NEIGHBORS, n_jobs=1),
    pairwise_feature_mi(x_par, y_par, k=K_NEIGHBORS, n_jobs=min(N_JOBS, 4)),
)


# In[ ]:


def feature_mi_tables(
    x_2d: pd.DataFrame, x_3d: pd.DataFrame, rng: np.random.Generator
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Long-format MI matrix plus a per-feature best-match summary
    mi = pairwise_feature_mi(
        x_2d.values,
        x_3d.values,
        k=K_NEIGHBORS,
        rng=rng,
        n_jobs=N_JOBS,
        desc="Feature MI",
    )
    null = np.concatenate(
        [
            pairwise_feature_mi(
                x_2d.values,
                x_3d.values[rng.permutation(len(x_3d))],
                k=K_NEIGHBORS,
                rng=rng,
                n_jobs=N_JOBS,
                desc=f"Feature MI null {i + 1}/{N_FEATURE_NULL_PERMUTATIONS}",
            ).ravel()
            for i in range(N_FEATURE_NULL_PERMUTATIONS)
        ]
    )
    tau_95 = float(np.quantile(null, 0.95))

    names_2d, names_3d = np.array(x_2d.columns), np.array(x_3d.columns)
    long_df = pd.DataFrame(
        {
            "feature_2d": np.repeat(names_2d, len(names_3d)),
            "feature_3d": np.tile(names_3d, len(names_2d)),
            "mi_bits": mi.ravel(),
        }
    )

    summary_3d = pd.DataFrame(
        {
            "modality": "3D",
            "feature_name": names_3d,
            "best_match": names_2d[mi.argmax(axis=0)],
            "max_mi_bits": mi.max(axis=0),
        }
    )
    summary_2d = pd.DataFrame(
        {
            "modality": "2D",
            "feature_name": names_2d,
            "best_match": names_3d[mi.argmax(axis=1)],
            "max_mi_bits": mi.max(axis=1),
        }
    )
    summary = pd.concat([summary_3d, summary_2d], ignore_index=True)
    summary["null_p95_bits"] = tau_95
    summary["redundant"] = summary["max_mi_bits"] > tau_95
    summary["max_info_correlation"] = [
        info_correlation(v) for v in summary["max_mi_bits"]
    ]
    return long_df, summary


# ## Run all comparisons
#
# For each 2D × 3D comparison, the cell below:
#
# 1. Pairs the profiles at the aggregated level.
# 2. Computes the whole-profile MI for each value in `PCA_COMPONENTS`.
# 3. Computes the per-feature MI matrix and summary.
#
# Results are cached the same way as in notebook 6. Comparisons with fewer than 10 paired groups are skipped, because KSG isn't reliable with so few samples.

# In[ ]:


_mi_dir.mkdir(parents=True, exist_ok=True)
MIN_PAIRED_GROUPS = 10
pairing_report = []


def comparison_meta(comp: dict, x_2d: pd.DataFrame, x_3d: pd.DataFrame) -> dict:
    # Columns identifying the comparison, attached to every output row
    return {
        "comparison": comp["name"],
        "profile_2d": comp["key_2d"],
        "profile_3d": comp["key_3d"],
        "group_label_2d": comp["cfg_2d"]["group_label"],
        "group_label_3d": comp["cfg_3d"]["group_label"],
        "resolution": comp["cfg_2d"]["resolution"],
        "stage": comp["cfg_2d"]["stage"],
        "agg_mode_3d": comp["cfg_3d"]["agg_mode"] or "none",
        "n_paired_groups": len(x_2d),
        "n_features_2d": x_2d.shape[1],
        "n_features_3d": x_3d.shape[1],
    }


# Cache fingerprints: the code and parameters each analysis depends on.
# N_JOBS is left out because it does not change results.
_pairing_code = code_text(
    pair_profiles,
    clean_features,
    load_aggregated,
    resolve_join_keys,
    get_feature_columns,
    comparison_meta,
    _standardize,
    info_correlation,
    JOIN_KEY_CANDIDATES=JOIN_KEY_CANDIDATES,
    REQUIRED_KEYS=REQUIRED_KEYS,
    MAX_NAN_FRACTION=MAX_NAN_FRACTION,
    MIN_PAIRED_GROUPS=MIN_PAIRED_GROUPS,
    K_NEIGHBORS=K_NEIGHBORS,
    SEED=SEED,
)
profile_code = _pairing_code + code_text(
    profile_mi,
    ksg_mi,
    PCA_COMPONENTS=PCA_COMPONENTS,
    N_PERMUTATIONS=N_PERMUTATIONS,
)
feature_code = _pairing_code + code_text(
    feature_mi_tables,
    pairwise_feature_mi,
    _feature_mi_rows,
    N_FEATURE_NULL_PERMUTATIONS=N_FEATURE_NULL_PERMUTATIONS,
)


def is_done(step: str, outputs: List[pathlib.Path], inputs, code: str) -> bool:
    # Whether `outputs` can be reused (see SKIP_EXISTING)
    if os.environ.get(FORCE_ENV_VAR, "") == "1":
        return False
    if SKIP_EXISTING and all(path.exists() for path in outputs):
        return True
    return is_fresh(step, results_root, CODE_PATH, outputs, inputs, code)


for comp in tqdm(comparisons, desc="Comparisons"):
    inputs = [comp["cfg_2d"]["input"], comp["cfg_3d"]["input"]]
    profile_outputs = [comp["profile_output"]]
    feature_outputs = [comp["feature_output"], comp["feature_summary_output"]]
    profile_done = is_done(PROFILE_CACHE_STEP, profile_outputs, inputs, profile_code)
    feature_done = is_done(FEATURE_CACHE_STEP, feature_outputs, inputs, feature_code)
    if profile_done and feature_done:
        tqdm.write(f"Using cached result: {comp['name']}")
        continue

    x_2d, x_3d, keys = pair_profiles(*inputs)
    n_groups = len(x_2d)
    pairing_report.append(
        {
            "comparison": comp["name"],
            "join_keys": ", ".join(keys),
            "n_paired_groups": n_groups,
            "n_features_2d": x_2d.shape[1],
            "n_features_3d": x_3d.shape[1],
        }
    )
    tqdm.write(
        f"{comp['name']}: {n_groups} paired groups on [{', '.join(keys)}], "
        f"{x_2d.shape[1]} 2D x {x_3d.shape[1]} 3D features"
    )
    if n_groups < MIN_PAIRED_GROUPS:
        tqdm.write("  -> skipped: too few paired groups")
        continue

    meta = comparison_meta(comp, x_2d, x_3d)

    # Analysis 1: whole-profile MI (own seed stream, so it reproduces whether
    # or not the feature analysis is recomputed in the same run)
    if not profile_done:
        rng = np.random.default_rng([SEED, 0])
        profile_df = pd.DataFrame(
            [
                {
                    **meta,
                    **profile_mi(
                        x_2d,
                        x_3d,
                        n_components=c,
                        k=K_NEIGHBORS,
                        n_permutations=N_PERMUTATIONS,
                        rng=rng,
                    ),
                }
                for c in PCA_COMPONENTS
            ]
        )
        profile_df.to_parquet(comp["profile_output"], index=False)
        mark_fresh(
            PROFILE_CACHE_STEP,
            results_root,
            CODE_PATH,
            profile_outputs,
            inputs,
            profile_code,
        )

    # Analysis 2: per-feature MI
    if not feature_done:
        rng = np.random.default_rng([SEED, 1])
        feature_df, feature_summary = feature_mi_tables(x_2d, x_3d, rng)
        for col, val in meta.items():
            feature_summary[col] = val
        feature_df.to_parquet(comp["feature_output"], index=False)
        feature_summary.to_parquet(comp["feature_summary_output"], index=False)
        mark_fresh(
            FEATURE_CACHE_STEP,
            results_root,
            CODE_PATH,
            feature_outputs,
            inputs,
            feature_code,
        )

pd.DataFrame(pairing_report)


# In[ ]:


# Combine results across comparisons. Skipped when no per-comparison output
# has changed, so the combined files keep their mtime and notebook 7b's cache
# stays valid.
all_profile_path = _mi_dir / "all_profile_mi.parquet"
all_feature_summary_path = _mi_dir / "all_feature_mi_summary.parquet"
combined_outputs = [all_profile_path, all_feature_summary_path]
profile_inputs = [
    c["profile_output"] for c in comparisons if c["profile_output"].exists()
]
summary_inputs = [
    c["feature_summary_output"]
    for c in comparisons
    if c["feature_summary_output"].exists()
]
combined_inputs = profile_inputs + summary_inputs

if is_fresh(CACHE_STEP, results_root, CODE_PATH, combined_outputs, combined_inputs):
    print("Using cached result: combined MI tables")
    profile_results = pd.read_parquet(all_profile_path)
    feature_summaries = pd.read_parquet(all_feature_summary_path)
else:
    profile_results = pd.concat(
        [
            pd.read_parquet(path)
            for path in tqdm(profile_inputs, desc="Combine profile MI")
        ],
        ignore_index=True,
    )
    feature_summaries = pd.concat(
        [
            pd.read_parquet(path)
            for path in tqdm(summary_inputs, desc="Combine feature MI")
        ],
        ignore_index=True,
    )
    profile_results.to_parquet(all_profile_path, index=False)
    feature_summaries.to_parquet(all_feature_summary_path, index=False)
    mark_fresh(CACHE_STEP, results_root, CODE_PATH, combined_outputs, combined_inputs)

redundancy = (
    feature_summaries.groupby(["comparison", "modality"])["redundant"]
    .mean()
    .unstack("modality")
    .rename(
        columns={"2D": "frac_2d_redundant_with_3d", "3D": "frac_3d_redundant_with_2d"}
    )
)

profile_results.pivot_table(
    index=["resolution", "stage", "group_label_2d", "group_label_3d", "agg_mode_3d"],
    columns="n_components",
    values="info_correlation",
).round(3)


# In[ ]:


redundancy.round(3)


# ## Reading the results
#
# **Whole profile (`all_profile_mi.parquet`)**
#
# - `mi_corrected_bits` is the shared information between the top-$c$ PCs of 2D and 3D, after subtracting the permutation null.
# - `info_correlation` ($r_I$) is the same quantity on a 0–1 scale. Near 0: 2D and 3D measure different things. Near 1: they measure the same thing.
# - `p_value` tests whether there is *any* shared information, not whether it is large.
# - If $r_I$ changes a lot across `n_components`, the result depends on how many PCs you keep. Compare it with `explained_variance_2d/3d`.
#
# **Per feature (`all_feature_mi_summary.parquet`)**
#
# - `frac_3d_redundant_with_2d`: the fraction of 3D features that have at least one 2D feature sharing more information than chance. The two fractions are asymmetric, as they should be.
#   A low value means most 3D features capture something no single 2D feature captures.
# - The full matrix (`*_feature_mi.parquet`) can be clustered or plotted as a heatmap to see *which* feature groups overlap.
#
# **Caveats**
#
# - Aggregation averages out cell-to-cell variation. These results describe how 2D and 3D agree on *group-level* phenotypes, not on individual cells or organoids.
# - With few paired groups, MI estimates are noisy. Trust the permutation-corrected values and the comparisons between stages more than any single number.
# - The normal and fs stages are aggregated here by median, while the agg stage files keep their own upstream aggregation. Differences between stages partly reflect that.
# - If a 2D projection is derived from the same 3D image stack, high MI is partly guaranteed by construction (data processing inequality). The more informative signal is how much of 3D is **not** shared.
