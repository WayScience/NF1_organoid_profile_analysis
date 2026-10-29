import pathlib
import sys

import pandas as pd

cwd = pathlib.Path.cwd()
if (cwd / ".git").is_dir():
    root_dir = cwd
else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break
sys.path.append(str(root_dir / "utils"))
from notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

module_dir = pathlib.Path(f"{root_dir}/figures/clinical_trial_data_plots").resolve()
data_dir = pathlib.Path(f"{module_dir}/data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)
trial_summary_path = pathlib.Path(
    f"{data_dir}/trial_efficacy_safety_summary.parquet"
).resolve()
response_distribution_path = pathlib.Path(
    f"{data_dir}/trial_response_heterogeneity.parquet"
).resolve()

trial_summary_records = [
    {
        "trial": "SPRINT Phase I",
        "nct_id": "NCT01362803",
        "year": 2016,
        "drug": "Selumetinib",
        "population": "Children",
        "arm": "Selumetinib",
        "design": "Single-arm, phase I",
        "n": 24,
        "orr_pct": 71.0,
        "orr_ci_low": None,
        "orr_ci_high": None,
        "median_best_pct_change": -31.0,
        "min_pct_change": -47.0,
        "max_pct_change": -6.0,
        "durable_response_pct": None,
        "deep_response_gt50pct_of_responders": None,
        "median_time_to_response_months": None,
        "any_ae_pct": None,
        "grade3plus_ae_pct": None,
        "discontinued_for_ae_pct": None,
    },
    {
        "trial": "SPRINT Phase II",
        "nct_id": "NCT01362803",
        "year": 2020,
        "drug": "Selumetinib",
        "population": "Children",
        "arm": "Selumetinib",
        "design": "Single-arm, phase II",
        "n": 50,
        "orr_pct": 70.0,
        "orr_ci_low": None,
        "orr_ci_high": None,
        "median_best_pct_change": -27.9,
        "min_pct_change": -55.1,
        "max_pct_change": 2.2,
        "durable_response_pct": 56.0,
        "deep_response_gt50pct_of_responders": None,
        "median_time_to_response_months": None,
        "any_ae_pct": None,
        "grade3plus_ae_pct": None,
        "discontinued_for_ae_pct": 10.0,
    },
    {
        "trial": "ReNeu",
        "nct_id": "NCT03962543",
        "year": 2024,
        "drug": "Mirdametinib",
        "population": "Adults",
        "arm": "Mirdametinib",
        "design": "Single-arm, phase IIb",
        "n": 58,
        "orr_pct": 41.0,
        "orr_ci_low": 29.0,
        "orr_ci_high": 55.0,
        "median_best_pct_change": -41.0,
        "min_pct_change": -90.0,
        "max_pct_change": 13.0,
        "durable_response_pct": 96.0,
        "deep_response_gt50pct_of_responders": 62.0,
        "median_time_to_response_months": 7.8,
        "any_ae_pct": 100.0,
        "grade3plus_ae_pct": 16.0,
        "discontinued_for_ae_pct": 22.0,
    },
    {
        "trial": "ReNeu",
        "nct_id": "NCT03962543",
        "year": 2024,
        "drug": "Mirdametinib",
        "population": "Children",
        "arm": "Mirdametinib",
        "design": "Single-arm, phase IIb",
        "n": 56,
        "orr_pct": 52.0,
        "orr_ci_low": 38.0,
        "orr_ci_high": 65.0,
        "median_best_pct_change": -42.0,
        "min_pct_change": -91.0,
        "max_pct_change": 48.0,
        "durable_response_pct": 100.0,
        "deep_response_gt50pct_of_responders": 52.0,
        "median_time_to_response_months": 7.9,
        "any_ae_pct": 100.0,
        "grade3plus_ae_pct": 25.0,
        "discontinued_for_ae_pct": 9.0,
    },
    {
        "trial": "KOMET",
        "nct_id": "NCT04924608",
        "year": 2025,
        "drug": "Selumetinib",
        "population": "Adults",
        "arm": "Selumetinib",
        "design": "Randomized, placebo-controlled, phase III",
        "n": 71,
        "orr_pct": 20.0,
        "orr_ci_low": 11.2,
        "orr_ci_high": 30.9,
        "median_best_pct_change": None,
        "min_pct_change": None,
        "max_pct_change": None,
        "durable_response_pct": 86.0,
        "deep_response_gt50pct_of_responders": None,
        "median_time_to_response_months": 3.7,
        "any_ae_pct": 100.0,
        "grade3plus_ae_pct": 32.0,
        "discontinued_for_ae_pct": 13.0,
    },
    {
        "trial": "KOMET",
        "nct_id": "NCT04924608",
        "year": 2025,
        "drug": "Placebo",
        "population": "Adults",
        "arm": "Placebo",
        "design": "Randomized, placebo-controlled, phase III",
        "n": 74,
        "orr_pct": 5.0,
        "orr_ci_low": 1.5,
        "orr_ci_high": 13.3,
        "median_best_pct_change": None,
        "min_pct_change": None,
        "max_pct_change": None,
        "durable_response_pct": None,
        "deep_response_gt50pct_of_responders": None,
        "median_time_to_response_months": None,
        "any_ae_pct": 92.0,
        "grade3plus_ae_pct": 18.0,
        "discontinued_for_ae_pct": 7.0,
    },
]
trial_summary_df = pd.DataFrame.from_records(trial_summary_records)
trial_summary_df.to_parquet(trial_summary_path, index=False)
trial_summary_df

response_distribution_records = [
    {
        "trial": "SPRINT Phase II",
        "drug": "Selumetinib",
        "population": "Children",
        "subgroup": "All patients",
        "n": 50,
        "median_pct_change": -27.9,
        "min_pct_change": -55.1,
        "max_pct_change": 2.2,
    },
    {
        "trial": "ReNeu",
        "drug": "Mirdametinib",
        "population": "Adults",
        "subgroup": "All patients",
        "n": 58,
        "median_pct_change": -41.0,
        "min_pct_change": -90.0,
        "max_pct_change": 13.0,
    },
    {
        "trial": "ReNeu",
        "drug": "Mirdametinib",
        "population": "Children",
        "subgroup": "All patients",
        "n": 56,
        "median_pct_change": -42.0,
        "min_pct_change": -91.0,
        "max_pct_change": 48.0,
    },
    {
        "trial": "KOMET",
        "drug": "Selumetinib",
        "population": "Adults",
        "subgroup": "Objective responders",
        "n": 14,
        "median_pct_change": -33.9,
        "min_pct_change": -58.1,
        "max_pct_change": -22.6,
    },
    {
        "trial": "KOMET",
        "drug": "Selumetinib",
        "population": "Adults",
        "subgroup": "Non-responders",
        "n": 51,
        "median_pct_change": -11.4,
        "min_pct_change": -26.7,
        "max_pct_change": 24.0,
    },
]
response_distribution_df = pd.DataFrame.from_records(response_distribution_records)
response_distribution_df.to_parquet(response_distribution_path, index=False)
response_distribution_df
