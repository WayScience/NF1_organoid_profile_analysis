#!/usr/bin/env python
# coding: utf-8

# # Predicting viability from image-based profiles using Elastic Net regression

# We train three model types:
# 1. **Leave one patient out (LOPO)**: For each patient, we train the model on all other patients and test on the left-out patient.
# 2. **Leave one treatment out (LOTO)**: For each treatment, we train the model on all other treatments and test on the left-out treatment.
# 3. **random split**: We randomly split the dataset into training and testing sets multiple times to evaluate model performance.
#
# Each model is evaluated using metrics such as R-squared, Mean Squared Error (MSE), and Mean Absolute Error (MAE) to assess the predictive accuracy of the Elastic Net regression model.
# The models are trained on both the sc and the organoid datasets.

# ## Imports, pathing, and constants

# In[ ]:


# MUST be first, before numpy/pandas/sklearn are imported.
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import logging
import pathlib
import time
import warnings
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import threadpoolctl
from joblib import parallel_backend
from notebook_init_utils import bandicoot_check, init_notebook
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Belt-and-suspenders: the OMP/MKL/OPENBLAS env vars above only take effect
# if set before BLAS is first loaded, and some vendored BLAS builds don't
# honor them at all (confirmed via threadpoolctl.threadpool_info() showing
# num_threads=24 despite OPENBLAS_NUM_THREADS=1 being set). This call instead
# directly overrides the ALREADY-LOADED BLAS library's thread count at
# runtime, which works regardless of import order or which env var a given
# BLAS build actually reads.
# set to 1 - was having thread release issues with > 1 thread
# was hitting thread lock issues and notebook stalling.
threadpoolctl.threadpool_limits(1)
root_dir, in_notebook = init_notebook()

if in_notebook:
    import tqdm.notebook as tqdm
else:
    import tqdm


# In[2]:


start_time = time.time()


# In[3]:


# set up logging
LOG_DIR = pathlib.Path("../logs")
LOG_DIR.mkdir(
    parents=True, exist_ok=True
)  # FileHandler errors if this dir doesn't exist

year_month_day_hour_minute_log_name = (
    f"{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M')}_viability_prediction_training.log"
)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / year_month_day_hour_minute_log_name),
    ],
    force=True,  # re-applies config if this cell is re-run in the same kernel session
)
logging.info("Started viability prediction training run")
logging.info(f"Logging to {LOG_DIR / year_month_day_hour_minute_log_name}")


# In[ ]:


"""
Elastic Net viability model:
training + evaluation under three split strategies

  1. "lopo"          - Leave-One-Patient-Out cross-validation
  2. "loto"          - Leave-One-Treatment-Out cross-validation
  3. "random_split"  - a single random 70/30 train/test split (no grouping)

All three reuse the same cleaning / training / metrics / artifact-saving
logic. For "lopo" and "loto", the model is refit once per fold (holding
out all rows for one patient / one treatment as the test set), metrics
are computed per fold, and results are aggregated across folds. For
"random_split", the model is fit once on a random 70% of rows and
evaluated on the held-out 30%, with the same metrics/artifact format so
it can be compared side-by-side with the grouped results.

--- Revision notes (leakage fixes) ---
This version fixes three sources of train/test leakage that were present
in the previous revision:

  * Feature imputation (median fill for NaN/inf) is now fit on each
    fold's TRAINING rows only (fit_feature_cleaning / apply_feature_cleaning),
    instead of being computed once on the full dataset before any split.

  * ElasticNet hyperparameter tuning now runs fresh on each fold's own
    training data by default (TUNE_HYPERPARAMS_PER_FOLD=True), instead of
    being tuned once on fold 0's training data and frozen for every other
    fold - the old approach's comment claimed this "avoided leakage", but
    fold 0's training data contains every other fold's held-out group, so
    every fold except fold 0 was still mildly informed by its own test
    data. A faster, documented-as-imperfect opt-out is still available.

  * The per-patient min-max viability target is no longer computed once,
    globally, before any split. For "lopo", computing it globally was
    actually fine (a held-out patient's own rows never touch training
    either way, since the CV group *is* the patient) - but for "loto" and
    "random_split", a single patient's rows can land in both train and
    test within the same fold, so the old global computation let a
    held-out treatment's value influence the scale applied to that same
    patient's training rows. compute_fold_safe_target() now recomputes
    each patient's min/max from training rows only, for every fold (LOPO
    keeps its safe global-style behavior; LOTO/random_split use
    training-only stats). Patients with an undefined range within a fold
    (e.g. only one training row, or all-identical values) are dropped
    from that fold with a logged warning rather than silently reusing
    test information.

Predicted viability is clipped to PREDICTION_BOUNDS = (0.0, 1.0) before
it's scored or saved, since the target is a per-patient [0, 1]-scaled
quantity (NOT a 0-100 percentage - that was a stale claim in the
previous revision that was never actually implemented).

Every saved artifact (metrics/predictions/importances/summary) carries
Metadata_split_method, Metadata_shuffle_status, Metadata_profile_type,
and Metadata_image_mode columns, so the combined files produced at the
end of the driver loop are self-describing without needing to parse
filenames. Summary files now also include a "pooled" row (out-of-fold
metrics computed over every held-out prediction concatenated together),
alongside the existing per-fold mean/std, since mean-of-folds can be
noisy when individual folds have few or low-variance rows.
"""

# ---------------------------------------------------------------------
# CONFIG - update these to match your data
# ---------------------------------------------------------------------

RETRAIN_EXISTING_MODELS = (
    True  # if False, will load existing models instead of retraining
)

PATIENT_COL = "Metadata_Biology_PatientTumor"  # column identifying each patient
TREATMENT_COL = "Metadata_Experiment_FullTreatment"  # column identifying each treatment
SPLIT_COL = "Metadata_data_split"  # used only for split_method="predefined"
RAW_VIABILITY_COL = "Metadata_Viability_percentage"  # raw (unnormalized) viability -
# used to compute a fold-safe VIABILITY_COL inside each split function below,
# instead of relying on a precomputed global normalization.
VIABILITY_COL = "min_max_viability"  # fold-safe, per-patient min-max target column
# (computed fresh inside each split function via compute_fold_safe_target - see below)

RANDOM_SPLIT_TEST_SIZE = 0.3  # fraction held out for the random_split test set
RANDOM_SPLIT_SEED = 0  # fixed seed so the random split is reproducible

PREDICTION_BOUNDS = (0.0, 1.0)  # min_max_viability is scaled to [0, 1] per patient,
# NOT a 0-100 percentage. Predictions are clipped to this range before being
# scored or saved, since the raw ElasticNet output is unconstrained.

TUNE_HYPERPARAMS_PER_FOLD = True  # Correct-but-slower default: tunes alpha/l1_ratio
# fresh on each fold's own training data (see run_group_cv docstring below). Set
# False only if this is too slow for your data size - it falls back to tuning
# once on fold 0's training data and freezing that for every fold, which is
# faster but mildly leaky for every fold except fold 0.

MAX_ITER = 5000
TOL = 1e-3

MODEL_OUTPUT = pathlib.Path("../trained_models")
RESULTS_OUTPUT = pathlib.Path("../model_results")
MODEL_OUTPUT.mkdir(exist_ok=True)
RESULTS_OUTPUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Diagnostics + fold-safe feature cleaning
# ---------------------------------------------------------------------
def log_feature_diagnostics(df: pd.DataFrame, feature_cols: list) -> None:
    """
    Read-only diagnostics on the feature block, reported once before any
    fold split happens. This is purely informational and never imputes or
    modifies anything - imputation happens per-fold (see
    fit_feature_cleaning / apply_feature_cleaning below) so it can never
    leak a held-out fold's statistics into training.
    """
    X = df[feature_cols]
    has_inf = bool(np.isinf(X.values).any())
    has_nan = bool(np.isnan(X.values).any())
    logging.info(f"Any inf in X: {has_inf}")
    logging.info(f"Any NaN in X: {has_nan}")
    finite_vals = X.values[np.isfinite(X.values)]
    if finite_vals.size:
        logging.info(f"Max abs finite value in X: {np.max(np.abs(finite_vals))}")
    if has_inf:
        inf_mask = np.isinf(X.values).any(axis=0)
        logging.info(f"Columns with inf values: {list(X.columns[inf_mask])}")


def fit_feature_cleaning(train_df: pd.DataFrame, feature_cols: list) -> pd.Series:
    """
    Computes per-feature median values from TRAINING rows only. Call this
    once per fold, then pass the result to apply_feature_cleaning() for
    both that fold's train and test rows. Keeping this fit step scoped to
    train_df only (never the full dataset) is what prevents a held-out
    fold's own values from influencing the imputed value it's later
    evaluated against.
    """
    X_train = train_df[feature_cols].replace([np.inf, -np.inf], np.nan)
    return X_train.median(axis=0, numeric_only=True)


def apply_feature_cleaning(
    df: pd.DataFrame, feature_cols: list, median_values: pd.Series
) -> pd.DataFrame:
    """
    Applies a previously-fit median (from fit_feature_cleaning) to replace
    inf/NaN in feature_cols, then clips to a wide safety bound. The clip
    bound is a fixed constant (not derived from data), and features are
    already clipped to a tighter, domain-specific range upstream (see the
    driver loop) before this ever runs - this clip is just a defensive
    backstop, not a required step, so it carries no leakage risk either way.
    """
    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(median_values)
    # A feature that's all-NaN/inf within this fold's training data has an
    # undefined median (NaN) - fall back to 0 so training can proceed.
    X = X.fillna(0)
    X = X.clip(lower=-1e5, upper=1e5)
    df = df.copy()
    df[feature_cols] = X
    return df


# ---------------------------------------------------------------------
# Fold-safe target normalization
# ---------------------------------------------------------------------
def compute_fold_safe_target(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    raw_col: str,
    patient_col: str,
    normalized_col: str,
    group_is_patient: bool,
) -> tuple:
    """
    Computes the per-patient min-max normalized viability target
    (normalized_col) for this fold's train/test rows, using only
    statistics that are safe for this fold's train/test boundary.

    - group_is_patient=True (LOPO): each patient's rows are either
      ENTIRELY in train_df or ENTIRELY in test_df for this fold (the CV
      group *is* the patient), so computing each patient's min/max from
      "however much of their data is in this fold" never mixes one
      patient's train rows with a *different* patient's test rows, and
      cannot leak information between folds.

    - group_is_patient=False (LOTO / random_split): a single patient can
      contribute rows to BOTH train_df and test_df in the same fold, so
      the patient's min/max must be computed from their TRAINING rows
      only, then applied unchanged to their test rows. If a patient has
      zero training rows in this fold (all of their measurements happen
      to fall under the held-out group), or their training rows have
      zero range (all-identical raw values), a target cannot be safely
      defined for that patient's rows in this fold - those rows are
      dropped, with a logged warning, rather than silently reusing
      information from the test rows themselves.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    if group_is_patient:
        combined = pd.concat([train_df, test_df])
        stats = combined.groupby(patient_col)[raw_col].agg(["min", "max"])
    else:
        stats = train_df.groupby(patient_col)[raw_col].agg(["min", "max"])

    stats["range"] = stats["max"] - stats["min"]
    # A zero (or undefined) range means we cannot safely normalize that
    # patient's rows in this fold - force NaN explicitly so the division
    # below always yields NaN (never +/-inf from a nonzero numerator over
    # a zero denominator), so a single `.isna()` check downstream catches
    # every bad case consistently.
    stats.loc[stats["range"] == 0, "range"] = np.nan

    def _apply(df):
        merged = df.merge(
            stats[["min", "range"]], left_on=patient_col, right_index=True, how="left"
        )
        merged[normalized_col] = (merged[raw_col] - merged["min"]) / merged["range"]
        return merged.drop(columns=["min", "range"])

    train_df = _apply(train_df)
    test_df = _apply(test_df)

    n_dropped_train = int(train_df[normalized_col].isna().sum())
    n_dropped_test = int(test_df[normalized_col].isna().sum())
    if n_dropped_train or n_dropped_test:
        logging.warning(
            f"Dropping {n_dropped_train} train / {n_dropped_test} test row(s) with an "
            f"undefined per-patient viability range (zero training-row range, or a "
            f"patient with zero training rows in this fold)."
        )
        train_df = train_df.dropna(subset=[normalized_col]).reset_index(drop=True)
        test_df = test_df.dropna(subset=[normalized_col]).reset_index(drop=True)

    return train_df, test_df


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics_from_arrays(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {"R2": r2, "MSE": mse, "MAE": mae, "RMSE": rmse}


def compute_metrics(model, X, Y, clip: bool = True) -> dict:
    y_pred = model.predict(X)
    if clip:
        y_pred = np.clip(y_pred, *PREDICTION_BOUNDS)
    return compute_metrics_from_arrays(Y.values.ravel(), y_pred)


# ---------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------
def tune_hyperparameters(X: pd.DataFrame, Y: pd.Series) -> tuple:
    """
    Runs ElasticNetCV once on the given X/Y to select alpha/l1_ratio.
    Callers are responsible for only passing TRAINING data - see
    run_group_cv (tunes per-fold on that fold's training data by default,
    or once on fold 0's training data if TUNE_HYPERPARAMS_PER_FOLD=False)
    and run_random_split (tunes once on the train split).
    """
    tuner = make_pipeline(
        StandardScaler(),
        ElasticNetCV(
            # Trimmed from [0.1, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0] and
            # alphas=100/cv=5 (up to 4,000 sequential fits at n_jobs=1) down
            # to 4 x 20 x 3 = 240 fits. Full grid resolution isn't needed
            # just to locate a decent alpha/l1_ratio, and the cost matters a
            # lot more now that this can run once per fold (see
            # TUNE_HYPERPARAMS_PER_FOLD) on p>>n profiles (e.g. 2593
            # features, 318 samples) where each individual fit is already
            # slow.
            l1_ratio=[0.1, 0.5, 0.9, 1.0],
            alphas=20,
            cv=3,
            random_state=0,
            max_iter=MAX_ITER,
            tol=TOL,
            n_jobs=1,
        ),
    )
    tuner.fit(X, Y.values.ravel())
    enet_cv = tuner.named_steps["elasticnetcv"]
    return enet_cv.alpha_, enet_cv.l1_ratio_


def train_elastic_net(
    X_train: pd.DataFrame,
    Y_train: pd.Series,
    alpha: Optional[float] = None,
    l1_ratio: Optional[float] = None,
) -> Pipeline:
    """
    Fits an elastic net model. If alpha/l1_ratio are provided, fits a plain
    ElasticNet with those fixed values (fast - 1 fit). If either is None,
    falls back to the old behavior of running the full ElasticNetCV grid
    search (slow - up to ~4,000 fits) so this function still works
    standalone if called without pre-tuned hyperparameters.
    """
    if alpha is not None and l1_ratio is not None:
        elastic_net_model = make_pipeline(
            StandardScaler(),
            ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=0,
                max_iter=MAX_ITER,
                tol=TOL,
            ),
        )
    else:
        elastic_net_model = make_pipeline(
            StandardScaler(),
            ElasticNetCV(
                l1_ratio=[0.1, 0.2, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                alphas=100,  # int -> auto-generates 100 alphas along the path
                cv=5,
                random_state=0,
                max_iter=MAX_ITER,
                tol=TOL,
                n_jobs=1,
            ),
        )
    elastic_net_model.fit(X_train, Y_train.values.ravel())
    return elastic_net_model


# ---------------------------------------------------------------------
# Strategy 1 & 2: grouped CV (LOPO / LOTO), reusing the same core logic
# ---------------------------------------------------------------------
def run_group_cv(
    viabilities_df: pd.DataFrame,
    feature_cols: list,
    raw_viability_col: str,
    viability_col: str,
    patient_col: str,
    group_col: str,
    split_name: str,
    **kwargs,
) -> pd.DataFrame:
    """
    Runs Leave-One-Group-Out CV (group_col = patient or treatment column).
    For each fold:
      1. The min-max viability target is recomputed fold-safely (see
         compute_fold_safe_target) so patient-level min/max stats never
         depend on a row this fold's test set contains.
      2. Feature imputation (median fill for NaN/inf) is fit on that
         fold's training rows only, then applied to both train and test.
      3. ElasticNet hyperparameters (alpha/l1_ratio) are tuned fresh on
         that fold's training data (TUNE_HYPERPARAMS_PER_FOLD=True,
         default) or, if you opt out for speed, tuned once on fold 0's
         training data and frozen for every other fold - faster, but NOT
         fully leakage-free (fold 0's training data contains every other
         fold's held-out group, so those folds' hyperparameters are still
         mildly informed by their own test data). Only use the opt-out if
         per-fold tuning is too slow for your data size.
      4. The model is refit and evaluated; predictions are clipped to
         PREDICTION_BOUNDS before being scored or saved, since the target
         is a [0, 1]-scaled quantity and the raw ElasticNet output is
         unconstrained.
    Saves per-fold and aggregated artifacts under RESULTS_OUTPUT /
    MODEL_OUTPUT, including a pooled (out-of-fold) metric alongside the
    usual per-fold mean/std, since mean-of-folds can be noisy when
    individual folds have few or low-variance rows.
    """
    list_of_metadatas = []
    shuffle_status = kwargs.get("shuffle_status", "not_shuffled")
    list_of_metadatas.append(shuffle_status)
    profile_type = kwargs.get("profile_type")
    if profile_type is not None:
        list_of_metadatas.append(profile_type)
    image_mode = kwargs.get("image_mode")
    if image_mode is not None:
        list_of_metadatas.append(image_mode)
    retrain = kwargs.get("retrain")
    tag = "__".join(list_of_metadatas) if list_of_metadatas else split_name

    if viabilities_df.empty:
        raise ValueError(
            f"Input dataframe is empty for split '{split_name}' (profile={profile_type}, shuffle={shuffle_status})."
        )
    if group_col not in viabilities_df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in input dataframe.")
    if patient_col not in viabilities_df.columns:
        raise KeyError(f"Patient column '{patient_col}' not found in input dataframe.")
    if raw_viability_col not in viabilities_df.columns:
        raise KeyError(
            f"Raw viability column '{raw_viability_col}' not found in input dataframe."
        )

    log_feature_diagnostics(viabilities_df, feature_cols)

    missing_group_mask = viabilities_df[group_col].isna()
    if missing_group_mask.any():
        dropped = int(missing_group_mask.sum())
        logging.warning(
            f"Dropping {dropped} row(s) with missing group labels in '{group_col}'."
        )
        viabilities_df = viabilities_df.loc[~missing_group_mask].reset_index(drop=True)

    if viabilities_df.empty:
        raise ValueError(
            f"No rows available after removing missing group labels for '{group_col}' (split={split_name})."
        )

    groups = viabilities_df[group_col].values
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError(
            f"Need at least 2 unique groups for LeaveOneGroupOut on '{group_col}', found {len(unique_groups)}."
        )

    group_is_patient = group_col == patient_col
    logo = LeaveOneGroupOut()
    n_splits = logo.get_n_splits(groups=groups)
    logging.info(
        f"=== {split_name} grouped CV on '{group_col}' ({n_splits} folds) for {shuffle_status} "
        f"(profile={profile_type}) ==="
    )

    all_metrics, all_predictions, all_importances = [], [], []
    tuned_alpha, tuned_l1_ratio = (
        None,
        None,
    )  # only used when TUNE_HYPERPARAMS_PER_FOLD is False

    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(viabilities_df, groups=groups)
    ):
        held_out = np.unique(groups[test_idx])[0]
        train_df = viabilities_df.iloc[train_idx].copy()
        test_df = viabilities_df.iloc[test_idx].copy()

        train_df, test_df = compute_fold_safe_target(
            train_df,
            test_df,
            raw_col=raw_viability_col,
            patient_col=patient_col,
            normalized_col=viability_col,
            group_is_patient=group_is_patient,
        )
        if train_df.empty or test_df.empty:
            logging.warning(
                f"  Fold {fold_idx} (held out={held_out}): skipped - no rows left after "
                f"fold-safe target normalization."
            )
            continue

        median_values = fit_feature_cleaning(train_df, feature_cols)
        train_df = apply_feature_cleaning(train_df, feature_cols, median_values)
        test_df = apply_feature_cleaning(test_df, feature_cols, median_values)

        X_train, X_test = train_df[feature_cols], test_df[feature_cols]
        Y_train, Y_test = train_df[viability_col], test_df[viability_col]

        if TUNE_HYPERPARAMS_PER_FOLD:
            fold_alpha, fold_l1_ratio = tune_hyperparameters(X_train, Y_train)
        else:
            if tuned_alpha is None or tuned_l1_ratio is None:
                tuned_alpha, tuned_l1_ratio = tune_hyperparameters(X_train, Y_train)
                logging.info(
                    f"  Tuned once on fold {fold_idx} training data for "
                    f"{split_name}/{shuffle_status}/{profile_type} and froze (fast, "
                    f"mildly leaky outside this fold): alpha={tuned_alpha:.6g}, "
                    f"l1_ratio={tuned_l1_ratio}"
                )
            fold_alpha, fold_l1_ratio = tuned_alpha, tuned_l1_ratio

        model_output_path = (
            MODEL_OUTPUT / f"{split_name}_model_fold{fold_idx}_{held_out}__{tag}.joblib"
        )
        if model_output_path.exists() and not retrain:
            logging.info(
                f"Model already exists at {model_output_path}, loading it instead of retraining."
            )
            model = joblib.load(model_output_path)
            fold_alpha = getattr(model[-1], "alpha", fold_alpha)
            fold_l1_ratio = getattr(model[-1], "l1_ratio", fold_l1_ratio)
        else:
            model = train_elastic_net(
                X_train, Y_train, alpha=fold_alpha, l1_ratio=fold_l1_ratio
            )
            joblib.dump(model, model_output_path)

        for eval_split, (X_eval, Y_eval) in {
            "train": (X_train, Y_train),
            "test": (X_test, Y_test),
        }.items():
            m = compute_metrics(model, X_eval, Y_eval)
            m.update(
                {
                    "Metadata_fold": fold_idx,
                    "Metadata_held_out_group": held_out,
                    "Metadata_eval_split": eval_split,
                    "Metadata_n_samples": len(X_eval),
                    "Metadata_split_method": split_name,
                    "Metadata_shuffle_status": shuffle_status,
                    "Metadata_profile_type": profile_type,
                    "Metadata_image_mode": image_mode,
                    "Metadata_alpha": fold_alpha,
                    "Metadata_l1_ratio": fold_l1_ratio,
                }
            )
            all_metrics.append(m)
            logging.info(
                f"  Fold {fold_idx} (held out={held_out}, n={len(X_eval)}, {eval_split}): "
                f"R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}"
            )

        # Everything below runs once per FOLD (not once per eval_split) -
        # the model/predictions/coefficients don't change between the
        # train-eval and test-eval passes above.
        fold_preds = test_df.copy()
        fold_preds["Actual_Viability"] = Y_test.values
        fold_preds["Predicted_Viability"] = np.clip(
            model.predict(X_test), *PREDICTION_BOUNDS
        )
        fold_preds["Metadata_fold"] = fold_idx
        fold_preds["Metadata_held_out_group"] = held_out
        fold_preds["Metadata_split_method"] = split_name
        fold_preds["Metadata_shuffle_status"] = shuffle_status
        fold_preds["Metadata_profile_type"] = profile_type
        fold_preds["Metadata_image_mode"] = image_mode
        all_predictions.append(fold_preds)

        fold_importance = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": model[
                    -1
                ].coef_,  # last pipeline step - works for both ElasticNet and ElasticNetCV
                "Metadata_fold": fold_idx,
                "Metadata_held_out_group": held_out,
                "Metadata_split_method": split_name,
                "Metadata_shuffle_status": shuffle_status,
                "Metadata_profile_type": profile_type,
                "Metadata_image_mode": image_mode,
            }
        )
        all_importances.append(fold_importance)

    if not all_metrics:
        raise ValueError(
            f"No folds produced results for split '{split_name}' (profile={profile_type}, "
            f"shuffle={shuffle_status}) - every fold was skipped."
        )

    metrics_df = pd.DataFrame(all_metrics)[
        [
            "Metadata_fold",
            "Metadata_held_out_group",
            "Metadata_eval_split",
            "Metadata_n_samples",
            "Metadata_split_method",
            "Metadata_shuffle_status",
            "Metadata_profile_type",
            "Metadata_image_mode",
            "Metadata_alpha",
            "Metadata_l1_ratio",
            "R2",
            "MSE",
            "MAE",
            "RMSE",
        ]
    ]
    metrics_df.to_parquet(
        RESULTS_OUTPUT / f"{split_name}_model_performance__{tag}.parquet", index=False
    )

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_df.to_parquet(
        RESULTS_OUTPUT / f"{split_name}_predicted_viabilities__{tag}.parquet",
        index=False,
    )

    importances_df = pd.concat(all_importances, ignore_index=True)
    importances_df.to_parquet(
        RESULTS_OUTPUT / f"{split_name}_feature_importances__{tag}.parquet", index=False
    )

    # Aggregate summary (mean/std across folds, test set only), plus a
    # pooled (out-of-fold) metric computed over every held-out prediction
    # concatenated together - less noisy than mean-of-folds when
    # individual folds have few or low-variance rows.
    test_metrics = metrics_df[metrics_df["Metadata_eval_split"] == "test"]
    summary = test_metrics[["R2", "MSE", "MAE", "RMSE"]].agg(["mean", "std"])
    pooled = compute_metrics_from_arrays(
        predictions_df["Actual_Viability"], predictions_df["Predicted_Viability"]
    )
    summary.loc["pooled"] = pooled
    summary["Metadata_split_method"] = split_name
    summary["Metadata_shuffle_status"] = shuffle_status
    summary["Metadata_profile_type"] = profile_type
    summary["Metadata_image_mode"] = image_mode
    logging.info(
        f"--- {split_name} summary (across {n_splits} folds, test set) ---\n{summary}"
    )
    summary.to_parquet(RESULTS_OUTPUT / f"{split_name}_summary_metrics__{tag}.parquet")

    return metrics_df


# ---------------------------------------------------------------------
# Strategy 3: single random 70/30 train/test split (no grouping)
# ---------------------------------------------------------------------
def run_random_split(
    viabilities_df: pd.DataFrame,
    feature_cols: list,
    raw_viability_col: str,
    viability_col: str,
    patient_col: str,
    split_name: str = "random_split",
    test_size: float = RANDOM_SPLIT_TEST_SIZE,
    random_state: int = RANDOM_SPLIT_SEED,
    **kwargs,
) -> pd.DataFrame:
    """
    Trains once on a random (test_size fraction) held-out split rather than
    grouping by patient or treatment. Rows are shuffled and split
    independently of any Metadata_* grouping column, so the same
    patient/treatment can appear in both train and test - this is meant as
    a rough, optimistic ceiling to compare against the stricter LOPO/LOTO
    splits, NOT a substitute for them: a model can partly "cheat" here by
    learning a patient's baseline morphology from their other treatments in
    train, rather than a genuine treatment-response relationship.

    Fold-safe target normalization and feature imputation are still fit on
    the training partition only and applied to test (same as run_group_cv),
    and predictions are clipped to PREDICTION_BOUNDS.

    Saves the same artifact shapes (metrics/predictions/importances/model)
    as run_group_cv, using "Metadata_fold" = 0 and a fixed
    "Metadata_held_out_group" label, so results from both strategies can be
    concatenated and compared directly.
    """
    list_of_metadatas = []
    shuffle_status = kwargs.get("shuffle_status", "not_shuffled")
    list_of_metadatas.append(shuffle_status)
    profile_type = kwargs.get("profile_type")
    if profile_type is not None:
        list_of_metadatas.append(profile_type)
    image_mode = kwargs.get("image_mode")
    if image_mode is not None:
        list_of_metadatas.append(image_mode)
    retrain = kwargs.get("retrain")
    tag = "__".join(list_of_metadatas) if list_of_metadatas else split_name

    if viabilities_df.empty:
        raise ValueError(
            f"Input dataframe is empty for split '{split_name}' (profile={profile_type}, shuffle={shuffle_status})."
        )
    if raw_viability_col not in viabilities_df.columns:
        raise KeyError(
            f"Raw viability column '{raw_viability_col}' not found in input dataframe."
        )
    if len(viabilities_df) < 2:
        raise ValueError(
            f"Need at least 2 rows for train/test split, found {len(viabilities_df)} (split={split_name})."
        )

    log_feature_diagnostics(viabilities_df, feature_cols)

    held_out_label = f"random_{int(round(test_size * 100))}pct_holdout"

    train_df, test_df = train_test_split(
        viabilities_df, test_size=test_size, random_state=random_state
    )
    train_df, test_df = train_df.copy(), test_df.copy()

    train_df, test_df = compute_fold_safe_target(
        train_df,
        test_df,
        raw_col=raw_viability_col,
        patient_col=patient_col,
        normalized_col=viability_col,
        group_is_patient=False,  # a patient can straddle train/test here, so
        # per-patient stats must come from training rows only (see
        # compute_fold_safe_target's docstring).
    )
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"No rows left for split '{split_name}' after fold-safe target normalization "
            f"(profile={profile_type})."
        )

    logging.info(
        f"=== {split_name} ({int(round((1 - test_size) * 100))}/{int(round(test_size * 100))} "
        f"train/test, n_train={len(train_df)}, n_test={len(test_df)}) for {shuffle_status} "
        f"(profile={profile_type}) ==="
    )

    median_values = fit_feature_cleaning(train_df, feature_cols)
    train_df = apply_feature_cleaning(train_df, feature_cols, median_values)
    test_df = apply_feature_cleaning(test_df, feature_cols, median_values)

    X_train, X_test = train_df[feature_cols], test_df[feature_cols]
    Y_train, Y_test = train_df[viability_col], test_df[viability_col]

    model_output_path = MODEL_OUTPUT / f"{split_name}_model__{tag}.joblib"
    if model_output_path.exists() and not retrain:
        logging.info(
            f"Model already exists at {model_output_path}, loading it instead of retraining."
        )
        model = joblib.load(model_output_path)
        tuned_alpha = getattr(model[-1], "alpha", None)
        tuned_l1_ratio = getattr(model[-1], "l1_ratio", None)
    else:
        # Tune once on the training split, then fit a plain ElasticNet with
        # those fixed values instead of re-running the full grid search.
        tuned_alpha, tuned_l1_ratio = tune_hyperparameters(X_train, Y_train)
        logging.info(
            f"  Tuned once for {split_name}/{shuffle_status}/{profile_type}: "
            f"alpha={tuned_alpha:.6g}, l1_ratio={tuned_l1_ratio}"
        )
        model = train_elastic_net(
            X_train, Y_train, alpha=tuned_alpha, l1_ratio=tuned_l1_ratio
        )
        joblib.dump(model, model_output_path)

    all_metrics = []
    for eval_split, (X_eval, Y_eval) in {
        "train": (X_train, Y_train),
        "test": (X_test, Y_test),
    }.items():
        m = compute_metrics(model, X_eval, Y_eval)
        m.update(
            {
                "Metadata_fold": 0,
                "Metadata_held_out_group": held_out_label,
                "Metadata_eval_split": eval_split,
                "Metadata_n_samples": len(X_eval),
                "Metadata_split_method": split_name,
                "Metadata_shuffle_status": shuffle_status,
                "Metadata_profile_type": profile_type,
                "Metadata_image_mode": image_mode,
                "Metadata_alpha": tuned_alpha,
                "Metadata_l1_ratio": tuned_l1_ratio,
            }
        )
        all_metrics.append(m)
        logging.info(
            f"  {eval_split} (n={len(X_eval)}): R2={m['R2']:.4f}, RMSE={m['RMSE']:.4f}"
        )

    # Held-out predictions (clipped to PREDICTION_BOUNDS)
    fold_preds = test_df.copy()
    fold_preds["Actual_Viability"] = Y_test.values
    fold_preds["Predicted_Viability"] = np.clip(
        model.predict(X_test), *PREDICTION_BOUNDS
    )
    fold_preds["Metadata_fold"] = 0
    fold_preds["Metadata_held_out_group"] = held_out_label
    fold_preds["Metadata_split_method"] = split_name
    fold_preds["Metadata_shuffle_status"] = shuffle_status
    fold_preds["Metadata_profile_type"] = profile_type
    fold_preds["Metadata_image_mode"] = image_mode

    fold_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model[
                -1
            ].coef_,  # last pipeline step - works for both ElasticNet and ElasticNetCV
            "Metadata_fold": 0,
            "Metadata_held_out_group": held_out_label,
            "Metadata_split_method": split_name,
            "Metadata_shuffle_status": shuffle_status,
            "Metadata_profile_type": profile_type,
            "Metadata_image_mode": image_mode,
        }
    )

    metrics_df = pd.DataFrame(all_metrics)[
        [
            "Metadata_fold",
            "Metadata_held_out_group",
            "Metadata_eval_split",
            "Metadata_n_samples",
            "Metadata_split_method",
            "Metadata_shuffle_status",
            "Metadata_profile_type",
            "Metadata_image_mode",
            "Metadata_alpha",
            "Metadata_l1_ratio",
            "R2",
            "MSE",
            "MAE",
            "RMSE",
        ]
    ]
    metrics_df.to_parquet(
        RESULTS_OUTPUT / f"{split_name}_model_performance__{tag}.parquet", index=False
    )
    fold_preds.to_parquet(
        RESULTS_OUTPUT / f"{split_name}_predicted_viabilities__{tag}.parquet",
        index=False,
    )
    fold_importance.to_parquet(
        RESULTS_OUTPUT / f"{split_name}_feature_importances__{tag}.parquet", index=False
    )

    # Single-split summary, plus a "pooled" row for schema consistency with
    # run_group_cv - with only one split, pooled and mean are identical by
    # construction.
    test_metrics = metrics_df[metrics_df["Metadata_eval_split"] == "test"]
    summary = test_metrics[["R2", "MSE", "MAE", "RMSE"]].agg(["mean", "std"])
    summary.loc["pooled"] = compute_metrics_from_arrays(
        fold_preds["Actual_Viability"], fold_preds["Predicted_Viability"]
    )
    summary["Metadata_split_method"] = split_name
    summary["Metadata_shuffle_status"] = shuffle_status
    summary["Metadata_profile_type"] = profile_type
    summary["Metadata_image_mode"] = image_mode
    logging.info(f"--- {split_name} summary (single split, test set) ---\n{summary}")
    summary.to_parquet(RESULTS_OUTPUT / f"{split_name}_summary_metrics__{tag}.parquet")

    return metrics_df


# In[5]:


patient_ids = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    sep="\t",
    names=["patient_id"],
)["patient_id"].to_list()

viabilities_path = pathlib.Path(f"{root_dir}/data/viabilities/").resolve(strict=True)


# ## Combine the profiles, viabilities, and platemap information

# In[6]:


platemap_df_list = []
for patient in patient_ids:
    platemap_file_path = pathlib.Path(
        f"{root_dir}/config/platemaps/{patient}_platemap.csv"
    ).resolve(strict=True)
    tmp_df = pd.read_csv(platemap_file_path, index_col=0)
    tmp_df["patient_id"] = patient
    platemap_df_list.append(tmp_df)
platemap_df = pd.concat(platemap_df_list, axis=0)


# In[7]:


viabilities_df_list = []
for patient in patient_ids:
    viabilities_file_path = pathlib.Path(
        f"{root_dir}/data/viabilities/{patient}_Viabilities.csv"
    ).resolve()
    if not viabilities_file_path.exists():
        continue
    viabilities_df = pd.read_csv(viabilities_file_path)
    # change DMSO dose to 1
    viabilities_df.loc[viabilities_df["Drug"] == "DMSO", "Concentration_uM"] = 1
    viabilities_df.loc[viabilities_df["Drug"] == "PD0325901", "Drug"] = "Mirdametinib"
    viabilities_df["patient_id"] = patient

    viabilities_df_list.append(viabilities_df)
viabilities_df = pd.concat(viabilities_df_list, axis=0)


# In[8]:


# merge the viabilities with the platemap
platemap_viability_df = pd.merge(
    platemap_df,
    viabilities_df,
    how="left",
    left_on=["Treatment", "Dose", "patient_id"],
    right_on=["Drug", "Concentration_uM", "patient_id"],
)

# Check (not just assume) that unmatched rows really are the expected
# empty/control wells, rather than a silent Treatment/Dose/patient_id
# naming mismatch that would disproportionately and invisibly drop real
# data. Done BEFORE dropping WellCol/WellPosition so the breakdown below
# can still see well position.
nan_rows = platemap_viability_df[platemap_viability_df.isna().any(axis=1)]
n_nan = len(nan_rows)
if n_nan > 0:
    breakdown = (
        nan_rows.groupby(["patient_id", "Treatment", "Dose"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    logging.warning(
        f"{n_nan} of {len(platemap_viability_df)} row(s) "
        f"({n_nan / len(platemap_viability_df):.1%}) failed to merge platemap <-> "
        f"viability data and will be dropped. Breakdown by (patient_id, Treatment, Dose):\n"
        f"{breakdown.head(20)}"
    )
    if "WellPosition" in nan_rows.columns:
        non_b_well_nans = nan_rows[
            ~nan_rows["WellPosition"].astype(str).str.startswith("B")
        ]
        if len(non_b_well_nans) > 0:
            logging.warning(
                f"{len(non_b_well_nans)} of the {n_nan} dropped row(s) are NOT from a 'B' "
                f"well - this may indicate a real Treatment/Dose/patient_id naming "
                f"mismatch rather than the expected empty control wells. Inspect "
                f"`nan_rows` before trusting the drop below."
            )
        else:
            logging.info(
                "All dropped rows are from 'B' wells, matching the expected "
                "empty-control assumption."
            )

# now safe to drop these columns and the unmatched rows
platemap_viability_df = platemap_viability_df.drop(columns=["WellCol", "WellPosition"])
platemap_viability_df = platemap_viability_df.dropna().reset_index(drop=True)
platemap_viability_df.rename(
    columns={"Viability_percentage": "Metadata_Viability_percentage"}, inplace=True
)
# NOTE: this global, per-patient min-max column is kept here only for
# backward-compatibility with other notebooks that read
# combined_platemaps.parquet directly. The model-training pipeline further
# below does NOT use this column - it recomputes a fold-safe version from
# Metadata_Viability_percentage inside run_group_cv / run_random_split
# (see compute_fold_safe_target), since this global version is safe to use
# as-is for LOPO but leaks a held-out treatment's value into training
# targets for LOTO / random_split.
platemap_viability_df["min_max_viability"] = platemap_viability_df.groupby(
    "patient_id"
)["Metadata_Viability_percentage"].transform(
    lambda x: (x - x.min()) / (x.max() - x.min())
)
# save the combined platemaps
combined_platemaps_path = pathlib.Path(
    f"{root_dir}/data/viabilities/combined_platemaps.parquet"
).resolve()
# save this for use in later notebooks where viability is needed - no need to recombine the platemaps and viabilities each time
platemap_viability_df.to_parquet(combined_platemaps_path, index=False)


# ## Get all of the morphology profiles to work with

# In[9]:


consensus_profiles_3D_path = pathlib.Path(
    f"{root_dir}/data/profiles_3D/all_patients/3.consensus_profiles/"
).resolve(strict=True)

# Only fit on the "combined" (cross-patient normalized) consensus profiles -
# one single-cell profile and one organoid profile - not the plain per-patient
# consensus profiles (2D tree: organoid_consensus_profiles, sc_consensus_profiles)
# or the DL embedding variants (sammed_*, nucleocentric_*) that also live under
# this same 3D directory.
COMBINED_PROFILE_STEMS = {
    "organoid_norm_sc_consensus_profiles",
    "sc_norm_sc_consensus_profiles",
}

censensus_profiles_all = [
    x
    for x in consensus_profiles_3D_path.glob("*.parquet")
    if x.is_file() and x.suffix == ".parquet" and x.stem in COMBINED_PROFILE_STEMS
]


# ## Train the models:

# In[10]:


# Initialized ONCE, outside the per-profile loop, so results from every
# profile accumulate instead of being overwritten each iteration.
results = {
    "profile_type": [],
    "split_method": [],
    "shuffle_status": [],
    "results": [],
}

# Fixed seed so the "shuffled" permutation control is reproducible run-to-run.
rng = np.random.default_rng(0)

logging.info(f"Found {len(censensus_profiles_all)} consensus profile(s) to process")

for consensus_path in tqdm.tqdm(
    censensus_profiles_all,
    total=len(censensus_profiles_all),
    desc="Processing consensus profiles",
    leave=True,
):
    consensus_profile_name = consensus_path.stem
    logging.info(f"Processing and training model for {consensus_profile_name}...")
    consensus_df = pd.read_parquet(consensus_path)
    if "2D" in str(consensus_path):
        image_mode = "2D"
        # wrangle the metadata column names to match the 3D consensus profile format
        consensus_df = consensus_df.rename(
            columns={
                "Metadata_patient_tumor": "Metadata_Biology_PatientTumor",
                "Metadata_treatment": "Metadata_Experiment_Treatment",
                "Metadata_dose": "Metadata_Experiment_Dose",
            }
        )
    elif "3D" in str(consensus_path):
        image_mode = "3D"
    else:
        raise ValueError(f"Unexpected consensus profile path: {consensus_path}")

    viabilities_df = (
        pd.merge(
            consensus_df,
            platemap_viability_df,
            how="left",
            left_on=[
                "Metadata_Biology_PatientTumor",
                "Metadata_Experiment_Treatment",
                "Metadata_Experiment_Dose",
            ],
            right_on=["patient_id", "Treatment", "Dose"],
        )
        .drop(
            columns=[
                "Unit",
                "patient_id",
                "Drug",
                "Concentration_uM",
                "Treatment",
                "Dose",
            ]
        )
        .rename(
            columns={
                x: f"Metadata_{x}"
                for x in platemap_viability_df.columns
                if VIABILITY_COL not in x
                # BUGFIX: columns that already start with "Metadata_" (e.g.
                # Metadata_Viability_percentage, already prefixed back in the
                # platemap/viability-merge cell) must NOT be re-prefixed, or
                # they end up named "Metadata_Metadata_..." and become
                # unreachable under their expected name.
                and not x.startswith("Metadata_")
                and x not in ["WellCol", "WellPosition", "Drug", "Concentration_uM"]
            }
        )
    )

    # The precomputed global min_max_viability column (if it rode along from
    # platemap_viability_df) is intentionally dropped here - it's only safe
    # to use as-is for LOPO and leaks for LOTO/random_split (see the
    # revision notes in the training-functions cell above). Each split
    # function below recomputes it fold-safely from RAW_VIABILITY_COL.
    if VIABILITY_COL in viabilities_df.columns:
        viabilities_df = viabilities_df.drop(columns=[VIABILITY_COL])

    # drop rows with no raw viability measurement at all (patients/treatments
    # that never matched during the platemap<->viability merge)
    viabilities_df = viabilities_df.dropna(subset=[RAW_VIABILITY_COL]).reset_index(
        drop=True
    )
    # combine the two stratification columns into a single key
    viabilities_df["Metadata_Experiment_FullTreatment"] = (
        viabilities_df["Metadata_Experiment_Treatment"].astype(str)
        + "_"
        + viabilities_df["Metadata_Experiment_Dose"].astype(str)
    )
    metadata_cols = [
        col for col in viabilities_df.columns if col.startswith("Metadata_")
    ]
    feature_cols = [
        col
        for col in viabilities_df.columns
        if col not in metadata_cols and col not in [VIABILITY_COL]
    ]
    logging.info(
        f"  {consensus_profile_name}: {len(viabilities_df)} rows, "
        f"{len(feature_cols)} feature columns"
    )
    viabilities_df[feature_cols] = viabilities_df[feature_cols].clip(
        lower=-1e1, upper=1e1
    )

    for shuffle_status in tqdm.tqdm(
        ["not_shuffled", "shuffled"], desc="Processing shuffle statuses", leave=False
    ):
        if shuffle_status == "shuffled":
            # permute the values in every column (fixed seed via rng, defined above)
            viabilities_df[feature_cols] = viabilities_df[feature_cols].apply(
                lambda col: rng.permutation(col.values)
            )
        for split_method in tqdm.tqdm(
            ["lopo", "loto", "random_split"],
            desc="Processing split methods",
            leave=False,
        ):
            logging.info(
                f"Running split_method={split_method}, shuffle_status={shuffle_status}, "
                f"profile={consensus_profile_name}"
            )

            if split_method == "lopo":
                group_col = PATIENT_COL
            elif split_method == "loto":
                group_col = TREATMENT_COL
            else:
                group_col = None  # not used for "random_split"

            if split_method == "random_split":
                fold_result = run_random_split(
                    viabilities_df,
                    feature_cols,
                    RAW_VIABILITY_COL,
                    VIABILITY_COL,
                    PATIENT_COL,
                    split_name=split_method,
                    test_size=RANDOM_SPLIT_TEST_SIZE,
                    random_state=RANDOM_SPLIT_SEED,
                    shuffle_status=shuffle_status,
                    profile_type=consensus_profile_name,
                    retrain=RETRAIN_EXISTING_MODELS,
                    image_mode=image_mode,
                )
            else:
                fold_result = run_group_cv(
                    viabilities_df,
                    feature_cols,
                    RAW_VIABILITY_COL,
                    VIABILITY_COL,
                    PATIENT_COL,
                    group_col=group_col,
                    split_name=split_method,
                    shuffle_status=shuffle_status,
                    profile_type=consensus_profile_name,
                    retrain=RETRAIN_EXISTING_MODELS,
                    image_mode=image_mode,
                )

            results["results"].append(fold_result)
            results["profile_type"].append(consensus_profile_name)
            results["split_method"].append(split_method)
            results["shuffle_status"].append(shuffle_status)

            logging.info(
                f"Finished split_method={split_method}, shuffle_status={shuffle_status}, "
                f"profile={consensus_profile_name}"
            )

    # Free the (potentially large, high-dimensional) per-profile dataframe
    # before moving to the next consensus profile - helps avoid memory
    # bloat across 12 profiles x 2 shuffle statuses x 3 split methods x N
    # folds worth of accumulated artifacts.
    del viabilities_df
    gc.collect()

logging.info(f"Finished processing all {len(censensus_profiles_all)} profile(s)")


# In[11]:


# ---------------------------------------------------------------------
# Concatenate every per-run output into a single combined file per metric,
# now that all profiles / split methods / shuffle statuses have finished.
# ---------------------------------------------------------------------
logging.info(
    "All profiles processed. Concatenating per-run outputs into combined files..."
)

METRIC_FILE_PATTERNS = {
    "model_performance": "*_model_performance__*.parquet",
    "predicted_viabilities": "*_predicted_viabilities__*.parquet",
    "feature_importances": "*_feature_importances__*.parquet",
    "summary_metrics": "*_summary_metrics__*.parquet",
}

combined_paths = {}
for metric_name, pattern in METRIC_FILE_PATTERNS.items():
    matching_files = sorted(RESULTS_OUTPUT.glob(pattern))
    # exclude any combined file from a previous run of this cell
    matching_files = [f for f in matching_files if not f.stem.startswith("combined_")]

    if not matching_files:
        logging.warning(
            f"No files found for pattern '{pattern}' - skipping {metric_name}"
        )
        continue

    dfs = []
    for f in matching_files:
        df = pd.read_parquet(f)
        if metric_name == "summary_metrics":
            # mean/std are stored as the index on these files - promote to a
            # column before concatenating with ignore_index=True, or the
            # mean/std label is lost.
            df = df.reset_index().rename(columns={"index": "Metadata_stat"})
        df["Metadata_source_file"] = f.name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_path = RESULTS_OUTPUT / f"combined_{metric_name}.parquet"
    combined_df.to_parquet(combined_path, index=False)
    combined_paths[metric_name] = combined_path

    logging.info(
        f"Wrote {len(combined_df)} rows from {len(matching_files)} file(s) to {combined_path}"
    )

logging.info(
    f"Finished concatenating results into: {[str(p) for p in combined_paths.values()]}"
)
combined_paths


# In[12]:


final_time = time.time()
elapsed_time = final_time - start_time
seconds_gate = 60
minutes_gate = 3600
hours_gate = 3600 * 24
if elapsed_time < seconds_gate:
    logging.info(f"Total elapsed time: {elapsed_time:.2f} seconds")
elif elapsed_time < minutes_gate:
    logging.info(f"Total elapsed time: {elapsed_time / 60:.2f} minutes")
elif elapsed_time < hours_gate:
    logging.info(f"Total elapsed time: {elapsed_time / 3600:.2f} hours")
else:
    logging.info(f"Total elapsed time: {elapsed_time / 86400:.2f} days")
