#!/usr/bin/env python
# coding: utf-8

# # Extract clinical trial stats
# Manually transcribed efficacy, safety, dropout/discontinuation, adverse-event, and response-heterogeneity statistics from the three NF1-plexiform-neurofibroma MEK-inhibitor trial papers in `../papers/`:
# 1. Gross et al. 2020, *N Engl J Med* (`nihms-1590456.pdf`) - SPRINT Phase II, selumetinib, children.
# 2. Moertel et al. 2024, *J Clin Oncol* (`Moertel-2025-Reneu-a-pivotal-phase-iib-trial-of-.pdf`) - ReNeu, mirdametinib, adults and children.
# 3. Chen et al. 2025, *Lancet* (`1-s2.0-S0140673625009869.pdf`) - KOMET, selumetinib vs placebo, adults.
#
# Every value below is quoted directly from the published abstract, results text, tables, or figures of these three papers; no values are simulated or inferred from pixel measurements of the waterfall plots. See `../README.md` for the full per-field provenance (which table/figure/paragraph each column was transcribed from).

# In[1]:


import pathlib

import pandas as pd
from notebook_init_utils.notebook_init_utils import init_notebook

root_dir, in_notebook = init_notebook()

module_dir = pathlib.Path(f"{root_dir}/figures/clinical_trial_data_plots").resolve()
data_dir = pathlib.Path(f"{module_dir}/data").resolve()
data_dir.mkdir(parents=True, exist_ok=True)
trial_summary_path = pathlib.Path(
    f"{data_dir}/trial_efficacy_safety_summary.csv"
).resolve()
response_distribution_path = pathlib.Path(
    f"{data_dir}/trial_response_heterogeneity.csv"
).resolve()
dropout_summary_path = pathlib.Path(f"{data_dir}/trial_dropout_summary.csv").resolve()
adverse_events_detail_path = pathlib.Path(
    f"{data_dir}/trial_adverse_events_detail.csv"
).resolve()


# ## Efficacy and safety summary, one row per trial x cohort (x arm)

# In[2]:


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
        "grade3plus_ae_basis": None,
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
        "grade3plus_ae_basis": None,
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
        "grade3plus_ae_basis": "treatment-related",
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
        "grade3plus_ae_basis": "treatment-related",
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
        "grade3plus_ae_basis": "all-cause",
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
        "grade3plus_ae_basis": "all-cause",
        "discontinued_for_ae_pct": 7.0,
    },
]
trial_summary_df = pd.DataFrame.from_records(trial_summary_records)
trial_summary_df.to_csv(trial_summary_path, index=False)
trial_summary_df


# ## Best percent change in target plexiform neurofibroma volume, one row per trial x cohort x response subgroup
# Captures the per-patient heterogeneity in response reported as median/min/max (or, for KOMET, the responder vs non-responder min/max) in each paper; values are not available at the single-patient level outside of the published waterfall plots, so no attempt is made to digitize individual patients from those figures.

# In[3]:


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
response_distribution_df.to_csv(response_distribution_path, index=False)
response_distribution_df


# ## Dropout / discontinuation summary, one row per trial x cohort x arm
# `discontinued_any_reason_pct` is only populated where the paper reports an overall (any-cause) discontinuation count (SPRINT II text; KOMET Figure 1B CONSORT diagram). ReNeu's results text only reports discontinuation attributed to adverse events, not an overall figure, so `discontinued_any_reason_pct` is left missing (`None`) rather than inferred.

# In[4]:


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
        "discontinued_for_ae_pct": 14.0,
        "discontinued_for_ae_n": 10,
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
        "discontinued_for_ae_pct": 8.0,
        "discontinued_for_ae_n": 6,
        "dose_reduction_pct": 4.0,
        "dose_interruption_pct": 11.0,
        "notes": "Discontinuation reasons (Fig 1B, before protocol-specified crossover): adverse event 6, withdrawal by participant 9. 66 of 74 placebo participants crossed over to selumetinib.",
    },
]
dropout_df = pd.DataFrame.from_records(dropout_records)
dropout_df.to_csv(dropout_summary_path, index=False)
dropout_df


# ## Most common adverse events, one row per trial x arm x adverse event term
# Limited to ReNeu and KOMET, the two papers whose main text includes a per-term adverse-event frequency table (ReNeu Table 2: treatment-related AEs occurring in >=5% of either cohort; KOMET Table 3: all adverse events occurring in >=10% of either arm). SPRINT II's main text reports toxicity only narratively (most frequent: grade 1-2 GI symptoms, asymptomatic creatine phosphokinase increase, acneiform rash, paronychia), without a frequency table, so it is excluded here rather than assigning it fabricated percentages. `pct_grade3plus` is only available for ReNeu, which reports a per-term grade >=3 column; KOMET's Table 3 reports grade >=3 only as a trial-level aggregate (selumetinib 32%, placebo 18%), not per adverse-event term. `ae_basis` records the causality basis of each row's rates (`"treatment-related"` for ReNeu, `"all-cause"` for KOMET) so the two trials' per-term rates are never compared as if on the same basis.

# In[ ]:


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
# ReNeu's Table 2 reports treatment-related adverse events only; KOMET's Table 3
# reports all adverse events regardless of attribution (see grade3plus_ae_basis
# in trial_efficacy_safety_summary.csv for the same distinction at the trial level).
adverse_events_df["ae_basis"] = adverse_events_df["trial"].map(
    {"ReNeu": "treatment-related", "KOMET": "all-cause"}
)
adverse_events_df.to_csv(adverse_events_detail_path, index=False)
adverse_events_df
