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
dropout_summary_path = pathlib.Path(
    f"{data_dir}/trial_dropout_summary.parquet"
).resolve()
adverse_events_detail_path = pathlib.Path(
    f"{data_dir}/trial_adverse_events_detail.parquet"
).resolve()

dropout_records = [
    {
        "trial": "SPRINT Phase II",
        "drug": "Selumetinib",
        "population": "Children",
        "arm": "Selumetinib",
        "n": 50,
        "discontinued_any_reason_pct": 42.0,
        "discontinued_any_reason_n": 21,
        "discontinued_for_ae_pct": 10.0,
        "discontinued_for_ae_n": 5,
        "dose_reduction_pct": 28.0,
        "dose_interruption_pct": None,
        "notes": "Discontinuation reasons: progressive disease 5, stable (non-progressive) disease 2, toxic effects 5, PI discretion 4, patient declined 2, nonadherence 1, intercurrent illness (MPNST) 2.",
    },
    {
        "trial": "ReNeu",
        "drug": "Mirdametinib",
        "population": "Adults",
        "arm": "Mirdametinib",
        "n": 58,
        "discontinued_any_reason_pct": None,
        "discontinued_any_reason_n": None,
        "discontinued_for_ae_pct": 22.0,
        "discontinued_for_ae_n": 13,
        "dose_reduction_pct": 17.0,
        "dose_interruption_pct": 31.0,
        "notes": "Overall (any-reason) discontinuation count not reported separately from AE-related discontinuation in the results text.",
    },
    {
        "trial": "ReNeu",
        "drug": "Mirdametinib",
        "population": "Children",
        "arm": "Mirdametinib",
        "n": 56,
        "discontinued_any_reason_pct": None,
        "discontinued_any_reason_n": None,
        "discontinued_for_ae_pct": 9.0,
        "discontinued_for_ae_n": 5,
        "dose_reduction_pct": 12.0,
        "dose_interruption_pct": 30.0,
        "notes": "Overall (any-reason) discontinuation count not reported separately from AE-related discontinuation in the results text.",
    },
    {
        "trial": "KOMET",
        "drug": "Selumetinib",
        "population": "Adults",
        "arm": "Selumetinib",
        "n": 71,
        "discontinued_any_reason_pct": 25.0,
        "discontinued_any_reason_n": 18,
        "discontinued_for_ae_pct": 13.0,
        "discontinued_for_ae_n": 9,
        "dose_reduction_pct": 14.0,
        "dose_interruption_pct": 27.0,
        "notes": "Discontinuation reasons (Fig 1B): adverse event 10, withdrawal by participant 5, progressive disease 2, lost to follow-up 1.",
    },
    {
        "trial": "KOMET",
        "drug": "Placebo",
        "population": "Adults",
        "arm": "Placebo",
        "n": 74,
        "discontinued_any_reason_pct": 20.0,
        "discontinued_any_reason_n": 15,
        "discontinued_for_ae_pct": 7.0,
        "discontinued_for_ae_n": 5,
        "dose_reduction_pct": 4.0,
        "dose_interruption_pct": 11.0,
        "notes": "Discontinuation reasons (Fig 1B, before protocol-specified crossover): adverse event 6, withdrawal by participant 9. 66 of 74 placebo participants crossed over to selumetinib.",
    },
]
dropout_df = pd.DataFrame.from_records(dropout_records)
dropout_df.to_parquet(dropout_summary_path, index=False)
dropout_df

adverse_event_records = [
    (
        "ReNeu",
        "Mirdametinib",
        "Adults",
        "Mirdametinib",
        58,
        "Dermatitis acneiform",
        78.0,
        9.0,
    ),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Diarrhea", 48.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Nausea", 36.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Vomiting", 28.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Fatigue", 21.0, 2.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Alopecia", 12.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Dry skin", 14.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Rash", 7.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Paronychia", 2.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Pruritus", 9.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Headache", 0.0, 0.0),
    ("ReNeu", "Mirdametinib", "Adults", "Mirdametinib", 58, "Constipation", 7.0, 0.0),
    (
        "ReNeu",
        "Mirdametinib",
        "Adults",
        "Mirdametinib",
        58,
        "Blood creatine phosphokinase increased",
        10.0,
        2.0,
    ),
    (
        "ReNeu",
        "Mirdametinib",
        "Children",
        "Mirdametinib",
        56,
        "Dermatitis acneiform",
        43.0,
        2.0,
    ),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Diarrhea", 38.0, 2.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Nausea", 21.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Vomiting", 14.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Fatigue", 9.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Alopecia", 12.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Dry skin", 14.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Rash", 12.0, 2.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Paronychia", 30.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Pruritus", 5.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Headache", 11.0, 0.0),
    ("ReNeu", "Mirdametinib", "Children", "Mirdametinib", 56, "Constipation", 5.0, 0.0),
    (
        "ReNeu",
        "Mirdametinib",
        "Children",
        "Mirdametinib",
        56,
        "Blood creatine phosphokinase increased",
        20.0,
        7.0,
    ),
    (
        "KOMET",
        "Selumetinib",
        "Adults",
        "Selumetinib",
        71,
        "Dermatitis acneiform",
        59.0,
        None,
    ),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Diarrhea", 42.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Nausea", 25.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Vomiting", 25.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Fatigue", 20.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Alopecia", 18.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Dry skin", 18.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Rash", 15.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Paronychia", 13.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Pruritus", 10.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Headache", 11.0, None),
    ("KOMET", "Selumetinib", "Adults", "Selumetinib", 71, "Constipation", 10.0, None),
    (
        "KOMET",
        "Selumetinib",
        "Adults",
        "Selumetinib",
        71,
        "Blood creatine phosphokinase increased",
        45.0,
        None,
    ),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Dermatitis acneiform", 11.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Diarrhea", 12.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Nausea", 16.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Vomiting", 8.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Fatigue", 14.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Alopecia", 11.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Dry skin", 5.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Rash", 4.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Paronychia", 12.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Pruritus", 7.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Headache", 12.0, None),
    ("KOMET", "Placebo", "Adults", "Placebo", 74, "Constipation", 1.0, None),
    (
        "KOMET",
        "Placebo",
        "Adults",
        "Placebo",
        74,
        "Blood creatine phosphokinase increased",
        5.0,
        None,
    ),
]
adverse_events_df = pd.DataFrame(
    adverse_event_records,
    columns=[
        "trial",
        "drug",
        "population",
        "arm",
        "n",
        "ae_term",
        "pct_all_grades",
        "pct_grade3plus",
    ],
)
adverse_events_df.to_parquet(adverse_events_detail_path, index=False)
adverse_events_df
