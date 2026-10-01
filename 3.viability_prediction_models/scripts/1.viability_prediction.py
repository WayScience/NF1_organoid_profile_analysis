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

# In[1]:


# MUST be first, before numpy/pandas/sklearn are imported.
import gc
import logging
import multiprocessing
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

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"


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
LOG_DIR = pathlib.Path(f"{root_dir}/3.viability_prediction_models/logs")
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


# ## Functions and helpers

# In[4]:


"""
Elastic Net viability model:
training + evaluation under three split strategies

  1. "lopo"          - Leave-One-Patient-Out cross-validation
  2. "loto"          - Leave-One-Treatment-Out cross-validation
  3. "random_split"  - a single random 70/30 train/test split (no grouping)

All three reuse the same training / metrics / artifact-saving
logic. For "lopo" and "loto", the model is refit once per fold (holding
out all rows for one patient / one treatment as the test set), metrics
are computed per fold, and results are aggregated across folds. For
"random_split", the model is fit once on a random 70% of rows and
evaluated on the held-out 30%, with the same metrics/artifact format so
it can be compared side-by-side with the grouped results.

--- Revision notes (leakage fixes) ---
This version fixes three sources of train/test leakage that were present
in the previous revision:

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
    False  # if False, will load existing models instead of retraining
)

PATIENT_COL = "Metadata_Biology_PatientTumor"  # column identifying each patient
TREATMENT_COL = "Metadata_Experiment_FullTreatment"  # column identifying each treatment
SPLIT_COL = "Metadata_data_split"  # used only for split_method="predefined"
RAW_VIABILITY_COL = (
    "Metadata_Experiment_ViabilityPercentage"  # raw (unnormalized) viability -
)
# used to compute a fold-safe VIABILITY_COL inside each split function below,
# instead of relying on a precomputed global normalization.
VIABILITY_COL = "min_max_viability"  # fold-safe, per-patient min-max target column
# (computed fresh inside each split function via compute_fold_safe_target - see below)

RANDOM_SPLIT_TEST_SIZE = 0.3  # fraction held out for the random_split test set
RANDOM_SPLIT_SEED = 0  # fixed seed so the random split is reproducible

TUNE_HYPERPARAMS_PER_FOLD = True  # Correct-but-slower default: tunes alpha/l1_ratio
# fresh on each fold's own training data (see run_group_cv docstring below). Set
# False only if this is too slow for your data size - it falls back to tuning
# once on fold 0's training data and freezing that for every fold, which is
# faster but mildly leaky for every fold except fold 0.

MAX_ITER = 1000
TOL = 1e-3
N_WORKERS = multiprocessing.cpu_count() - 2  # for parallelized ElasticNetCV fits

MODEL_OUTPUT = pathlib.Path(f"{root_dir}/3.viability_prediction_models/trained_models")
RESULTS_OUTPUT = pathlib.Path(f"{root_dir}/3.viability_prediction_models/model_results")
MODEL_OUTPUT.mkdir(exist_ok=True)
RESULTS_OUTPUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# Diagnostics + fold-safe feature cleaning
# ---------------------------------------------------------------------
def log_feature_diagnostics(df: pd.DataFrame, feature_cols: list) -> None:
    """
    Read-only diagnostics on the feature block, reported once before any
    fold split happens. This is purely informational and never imputes or
    modifies anything - imputation happens per-fold
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


def compute_metrics(model, X, Y, clip: bool = False) -> dict:
    y_pred = model.predict(X)
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
            # alphas=100/cv=5 (up to 4,000 sequential fits at n_jobs=N_WORKERS) down
            # to 4 x 20 x 3 = 240 fits. Full grid resolution isn't needed
            # just to locate a decent alpha/l1_ratio, and the cost matters a
            # lot more now that this can run once per fold (see
            # TUNE_HYPERPARAMS_PER_FOLD) on p>>n profiles (e.g. 2593
            # features, 318 samples) where each individual fit is already
            # slow.
            l1_ratio=[0.1, 0.5, 0.9, 0.95, 0.99, 1.0],
            alphas=20,
            cv=3,
            random_state=0,
            max_iter=MAX_ITER,
            tol=TOL,
            n_jobs=N_WORKERS,
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
                l1_ratio=[0.1, 0.5, 0.9, 0.95, 0.99, 1.0],
                alphas=20,
                cv=3,
                random_state=0,
                max_iter=MAX_ITER,
                tol=TOL,
                n_jobs=N_WORKERS,
            ),
        )
    elastic_net_model.fit(X_train, Y_train.values.ravel())
    return elastic_net_model


# ---------------------------------------------------------------------
# Strategy 1 & 2: grouped CV (LOPO / LOTO), reusing the same core logic
# ---------------------------------------------------------------------
def run_group_cv(
    consensus_df: pd.DataFrame,
    feature_cols: list,
    raw_viability_col: str,
    viability_col: str,
    patient_col: str,
    group_col: str,
    split_name: str,
    **kwargs,
) -> pd.DataFrame | None:
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
    Saves per-fold and aggregated artifacts under RESULTS_OUTPUT /
    MODEL_OUTPUT, including a pooled (out-of-fold) metric alongside the
    usual per-fold mean/std, since mean-of-folds can be noisy when
    individual folds have few or low-variance rows.

    Returns None if all output files already exist (no need to recompute).
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

    if consensus_df.empty:
        raise ValueError(
            f"Input dataframe is empty for split '{split_name}' (profile={profile_type}, shuffle={shuffle_status})."
        )
    if group_col not in consensus_df.columns:
        raise KeyError(f"Grouping column '{group_col}' not found in input dataframe.")
    if patient_col not in consensus_df.columns:
        raise KeyError(f"Patient column '{patient_col}' not found in input dataframe.")
    if raw_viability_col not in consensus_df.columns:
        raise KeyError(
            f"Raw viability column '{raw_viability_col}' not found in input dataframe."
        )

    log_feature_diagnostics(consensus_df, feature_cols)

    missing_group_mask = consensus_df[group_col].isna()
    if missing_group_mask.any():
        dropped = int(missing_group_mask.sum())
        logging.warning(
            f"Dropping {dropped} row(s) with missing group labels in '{group_col}'."
        )
        consensus_df = consensus_df.loc[~missing_group_mask].reset_index(drop=True)

    if consensus_df.empty:
        raise ValueError(
            f"No rows available after removing missing group labels for '{group_col}' (split={split_name})."
        )

    groups = consensus_df[group_col].values
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
    split_output_dir = RESULTS_OUTPUT / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_path = (
        split_output_dir / f"{split_name}_summary_metrics__{tag}.parquet"
    )
    importances_output_path = (
        split_output_dir / f"{split_name}_feature_importances__{tag}.parquet"
    )
    predictions_output_path = (
        split_output_dir / f"{split_name}_predicted_viabilities__{tag}.parquet"
    )
    metrics_output_path = split_output_dir / f"{split_name}_fold_metrics__{tag}.parquet"
    outputs_to_check_for = [
        summary_output_path,
        importances_output_path,
        predictions_output_path,
        metrics_output_path,
    ]
    if not retrain and all(path.exists() for path in outputs_to_check_for):
        # All output files for this split/profile/shuffle combination
        # already exist - skip the whole run. This must happen before the
        # fold loop below, so fold_idx/test_idx are never referenced here.
        logging.info(
            f"All output files already exist for {split_name}/{shuffle_status}/"
            f"{profile_type} (tag={tag}) - skipping this run."
        )
        return None
    for fold_idx, (train_idx, test_idx) in enumerate(
        logo.split(consensus_df, groups=groups)
    ):
        held_out = np.unique(groups[test_idx])[0]
        train_df = consensus_df.iloc[train_idx].copy()
        test_df = consensus_df.iloc[test_idx].copy()

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
        fold_preds["Predicted_Viability"] = model.predict(X_test)
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

    metrics_df.to_parquet(metrics_output_path, index=False)

    predictions_df = pd.concat(all_predictions, ignore_index=True)

    predictions_df.to_parquet(
        predictions_output_path,
        index=False,
    )

    importances_df = pd.concat(all_importances, ignore_index=True)
    importances_df.to_parquet(importances_output_path, index=False)

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
    summary.to_parquet(summary_output_path, index=False)

    return metrics_df


# ---------------------------------------------------------------------
# Strategy 3: single random 70/30 train/test split (no grouping)
# ---------------------------------------------------------------------
def run_random_split(
    consensus_df: pd.DataFrame,
    feature_cols: list,
    raw_viability_col: str,
    viability_col: str,
    patient_col: str,
    split_name: str = "random_split",
    test_size: float = RANDOM_SPLIT_TEST_SIZE,
    random_state: int = RANDOM_SPLIT_SEED,
    **kwargs,
) -> pd.DataFrame | None:
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
    the training partition only and applied to test (same as run_group_cv).

    Saves the same artifact shapes (metrics/predictions/importances/model)
    as run_group_cv, using "Metadata_fold" = 0 and a fixed
    "Metadata_held_out_group" label, so results from both strategies can be
    concatenated and compared directly.

    Returns None if all output files already exist (no need to recompute).
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

    split_output_dir = RESULTS_OUTPUT / split_name
    split_output_dir.mkdir(parents=True, exist_ok=True)
    summary_output_path = (
        split_output_dir / f"{split_name}_summary_metrics__{tag}.parquet"
    )
    metrics_output_path = split_output_dir / f"{split_name}_fold_metrics__{tag}.parquet"
    fold_preds_output_path = (
        split_output_dir / f"{split_name}_predicted_viabilities__{tag}.parquet"
    )
    fold_importance_output_path = (
        split_output_dir / f"{split_name}_feature_importances__{tag}.parquet"
    )
    output_files = [
        summary_output_path,
        metrics_output_path,
        fold_preds_output_path,
        fold_importance_output_path,
    ]
    if not retrain and all(path.exists() for path in output_files):
        logging.info(
            f"All output files already exist for {split_name} (profile={profile_type}, shuffle={shuffle_status}), skipping."
        )
        return None

    if consensus_df.empty:
        raise ValueError(
            f"Input dataframe is empty for split '{split_name}' (profile={profile_type}, shuffle={shuffle_status})."
        )
    if raw_viability_col not in consensus_df.columns:
        raise KeyError(
            f"Raw viability column '{raw_viability_col}' not found in input dataframe."
        )
    if len(consensus_df) < 2:
        raise ValueError(
            f"Need at least 2 rows for train/test split, found {len(consensus_df)} (split={split_name})."
        )

    log_feature_diagnostics(consensus_df, feature_cols)

    held_out_label = f"random_{int(round(test_size * 100))}pct_holdout"

    train_df, test_df = train_test_split(
        consensus_df, test_size=test_size, random_state=random_state
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

    fold_preds = test_df.copy()
    fold_preds["Actual_Viability"] = Y_test.values
    fold_preds["Predicted_Viability"] = model.predict(X_test)
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

    metrics_df.to_parquet(metrics_output_path, index=False)
    fold_preds.to_parquet(
        fold_preds_output_path,
        index=False,
    )
    fold_importance.to_parquet(fold_importance_output_path, index=False)

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
    summary.to_parquet(summary_output_path, index=False)

    return metrics_df


# In[5]:


patient_ids = pd.read_csv(
    pathlib.Path(f"{root_dir}/data/patient_IDs.txt").resolve(strict=True),
    header=None,
    sep="\t",
    names=["patient_id"],
)["patient_id"].to_list()

viabilities_path = pathlib.Path(f"{root_dir}/data/viabilities/").resolve(strict=True)


# ## Get all of the morphology profiles to work with

# In[6]:


consensus_profiles_3D_paths = list(
    pathlib.Path(f"{root_dir}/3.viability_prediction_models/data/processed_profiles_3D")
    .resolve(strict=True)
    .glob("*.parquet")
)
consensus_profiles_2D_paths = list(
    pathlib.Path(f"{root_dir}/3.viability_prediction_models/data/processed_profiles_2D")
    .resolve(strict=True)
    .glob("*.parquet")
)
censensus_profiles_all_dict = {
    "3D": consensus_profiles_3D_paths,
    "2D": consensus_profiles_2D_paths,
}


# ## Train the models:

# In[7]:


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

total_profiles = sum(len(paths) for paths in censensus_profiles_all_dict.values())
logging.info(f"Found {total_profiles} consensus profile(s) to process")

for image_mode, consensus_paths in censensus_profiles_all_dict.items():
    for consensus_path in tqdm.tqdm(
        consensus_paths,
        total=len(consensus_paths),
        desc=f"Processing {image_mode} consensus profiles",
        leave=True,
    ):
        consensus_profile_name = consensus_path.stem
        logging.info(f"Processing and training model for {consensus_profile_name}...")
        consensus_df = pd.read_parquet(consensus_path)
        if image_mode == "2D":
            # wrangle the metadata column names to match the 3D consensus profile format
            consensus_df = consensus_df.rename(
                columns={
                    "Metadata_patient_tumor": "Metadata_Biology_PatientTumor",
                    "Metadata_treatment": "Metadata_Experiment_Treatment",
                    "Metadata_dose": "Metadata_Experiment_Dose",
                }
            )

        # The precomputed global min_max_viability column (if it rode along from
        # platemap_viability_df) is intentionally dropped here - it's only safe
        # to use as-is for LOPO and leaks for LOTO/random_split (see the
        # revision notes in the training-functions cell above). Each split
        # function below recomputes it fold-safely from RAW_VIABILITY_COL.

        # drop rows with no raw viability measurement at all (patients/treatments
        # that never matched during the platemap<->viability merge)
        consensus_df = consensus_df.dropna(subset=[VIABILITY_COL]).reset_index(
            drop=True
        )
        # combine the two stratification columns into a single key
        consensus_df["Metadata_Experiment_FullTreatment"] = (
            consensus_df["Metadata_Experiment_Treatment"].astype(str)
            + "_"
            + consensus_df["Metadata_Experiment_Dose"].astype(str)
        )
        metadata_cols = [
            col for col in consensus_df.columns if col.startswith("Metadata_")
        ]
        feature_cols = [
            col
            for col in consensus_df.columns
            if col not in metadata_cols and col not in [VIABILITY_COL]
        ]
        logging.info(
            f"  {consensus_profile_name}: {len(consensus_df)} rows, "
            f"{len(feature_cols)} feature columns"
        )

        for shuffle_status in tqdm.tqdm(
            ["not_shuffled", "shuffled"],
            desc="Processing shuffle statuses",
            leave=False,
        ):
            if shuffle_status == "shuffled":
                # permute the values in every column (fixed seed via rng, defined above)
                consensus_df[feature_cols] = consensus_df[feature_cols].apply(
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
                        consensus_df,
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
                        consensus_df,
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
        # bloat across profiles x 2 shuffle statuses x 3 split methods x N
        # folds worth of accumulated artifacts.
        del consensus_df
        gc.collect()

logging.info(f"Finished processing all {total_profiles} profile(s)")


# In[8]:


# ---------------------------------------------------------------------
# Concatenate every per-run output into a single combined file per metric,
# now that all profiles / split methods / shuffle statuses have finished.
# ---------------------------------------------------------------------
logging.info(
    "All profiles processed. Concatenating per-run outputs into combined files..."
)

METRIC_FILE_PATTERNS = {
    "fold_metrics": "*_fold_metrics__*.parquet",
    "predicted_viabilities": "*_predicted_viabilities__*.parquet",
    "feature_importances": "*_feature_importances__*.parquet",
    "summary_metrics": "*_summary_metrics__*.parquet",
}

combined_paths = {}
for metric_name, pattern in METRIC_FILE_PATTERNS.items():
    matching_files = sorted(RESULTS_OUTPUT.glob(f"*/{pattern}"))
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


# In[9]:


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
