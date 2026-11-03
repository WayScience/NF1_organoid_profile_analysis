# Clinical trial data plots

Figures contextualizing this repository's organoid MEK-inhibitor results against published human clinical-trial outcomes for NF1 plexiform neurofibroma (PN). Every number in `data/*.csv` is a manual transcription from the published, peer-reviewed report of one of the four trials below — no values are simulated, digitized from waterfall-plot pixels, or inferred. This document is the provenance record: it traces every field back to the publication, and in most cases the table/figure, it was transcribed from.

This module stores its intermediate data as CSV rather than parquet, as a deliberate one-off exception to the repo-wide parquet convention in `AGENTS.md` — kept this way for easy diffing/inspection of these small, hand-transcribed tables.

## Source publications

| Trial | Drug | Population | Citation | Identifiers |
|---|---|---|---|---|
| SPRINT Phase I | Selumetinib | Children | Dombi E, Baldwin A, Marcus LJ, et al. Activity of Selumetinib in Neurofibromatosis Type 1–Related Plexiform Neurofibromas. *N Engl J Med*. 2016;375(26):2550–2560. | doi:[10.1056/NEJMoa1605943](https://doi.org/10.1056/NEJMoa1605943) · PMID:28029918 · [NCT01362803](https://clinicaltrials.gov/study/NCT01362803) |
| SPRINT Phase II | Selumetinib | Children | Gross AM, Wolters PL, Dombi E, et al. Selumetinib in Children with Inoperable Plexiform Neurofibromas. *N Engl J Med*. 2020;382(15):1430–1442. | doi:[10.1056/NEJMoa1912735](https://doi.org/10.1056/NEJMoa1912735) · PMID:32187457 · [NCT01362803](https://clinicaltrials.gov/study/NCT01362803) |
| ReNeu | Mirdametinib | Adults and children | Moertel CL, Hirbe AC, Shuhaiber HH, et al. ReNeu: A Pivotal, Phase IIb Trial of Mirdametinib in Adults and Children with Symptomatic Neurofibromatosis Type 1-Associated Plexiform Neurofibroma. *J Clin Oncol*. 2025;43(6):716–729. | doi:[10.1200/JCO.24.01034](https://doi.org/10.1200/JCO.24.01034) · PMID:39514826 · PMCID:PMC11825507 · [NCT03962543](https://clinicaltrials.gov/study/NCT03962543) |
| KOMET | Selumetinib vs. placebo | Adults | KOMET study investigators (Chen AP, et al). Efficacy and safety of selumetinib in adults with neurofibromatosis type 1-related plexiform neurofibromas (KOMET): a randomised, double-blind, placebo-controlled, phase 3 trial. *Lancet*. 2025;405(10496):2217–2230. | doi:[10.1016/S0140-6736(25)00986-9](https://doi.org/10.1016/S0140-6736(25)00986-9) · [NCT04924608](https://clinicaltrials.gov/study/NCT04924608) |

Cross-checks against independently published secondary sources (ASCO/Lancet press summaries, FDA approval materials) confirm the key headline figures transcribed here: ReNeu's BICR-confirmed ORR of 41% (24/58 adults) and 52% (29/56 children), and KOMET's ORR of 20% (14/71, 95% CI 11.2–30.9) with selumetinib vs. 5% (4/74, 95% CI 1.5–13.3) with placebo and grade ≥3 adverse events in 32% of the selumetinib arm.

**A note on the SPRINT Phase II ORR (70%, not 68%, 66%, or 74%):** several different response figures circulate for this trial, because the paper itself reports more than one. Verbatim from the Results of Gross et al. 2020 (PMCID PMC7305659): *"A total of 37 of 50 patients (74%; 95% confidence interval [CI], 60 to 85) had a partial response, 35 (70%) had a confirmed partial response, and 28 (56%) had a durable response."* The paper's own Methods section defines the primary objective response rate as the *confirmed* partial response rate, so **70% (35/50) is the correct ORR to use** — the value transcribed here. The 74% (37/50) figure counts unconfirmed partial responses and is not the ORR; 66% (33/50) and 44% are from earlier/different analyses (NCI unblinded assessment vs. independent central review — see ReNeu's citation of SPRINT below) and should not be substituted for either.
>
> This article has a published correction (*N Engl J Med* 2020;383(13):1290, PMID 32966737) whose content could not be read during this review (NEJM, PubMed, and PMC all blocked automated access). If that correction changes the 35/50 confirmed-response count, it would need to be re-verified directly from that notice before trusting it over the verbatim Results text above.

## Pipeline

```
0.extract_clinical_trial_stats.{ipynb,py}               -> trial_efficacy_safety_summary.csv, trial_response_heterogeneity.csv,
                                                             trial_dropout_summary.csv, trial_adverse_events_detail.csv
1.plot_clinical_trial_efficacy_heterogeneity.{ipynb,r}  -> clinical_trial_orr_efficacy.png, clinical_trial_response_heterogeneity.png
2.plot_clinical_trial_safety.{ipynb,r}                  -> clinical_trial_dropout_rates.png, clinical_trial_adverse_events.png
```

Step 0 extracts all four tables (efficacy/safety summary, response heterogeneity, dropout, and adverse-event detail) from the source papers in one pass; steps 1 and 2 each plot a subset of those tables.

Run the full pipeline with `./run_clinical_trial_figures_local.sh` from this directory. It regenerates `scripts/*.py`/`*.r` from the notebooks (the notebooks are the source of truth; the scripts are a mirrored, non-interactive copy kept in sync automatically by the runner) and then executes all three steps in order.

## Field-level provenance

### `trial_efficacy_safety_summary.csv` — one row per trial x cohort x arm

Identifying columns (`trial`, `nct_id`, `year`, `drug`, `population`, `arm`, `design`, `n`) are taken from each paper's title, abstract, and methods/CONSORT description; `nct_id` is independently verifiable against the ClinicalTrials.gov links above.

| Column | Meaning | Source location |
|---|---|---|
| `orr_pct`, `orr_ci_low`, `orr_ci_high` | Confirmed objective response rate (and 95% CI, where reported) | Abstract / primary endpoint results. SPRINT I/II report ORR without CI; ReNeu and KOMET report ORR with 95% CI. |
| `median_best_pct_change`, `min_pct_change`, `max_pct_change` | Median and range of best on-treatment percent change in target PN volume | Results text / waterfall-plot summary statistics (not digitized from the plot itself). Not reported for KOMET at the trial-summary level (only by responder subgroup — see `trial_response_heterogeneity.csv`). |
| `durable_response_pct` | Percent of responders with a durable response (≥1 year, trial-specific definition) | Results text |
| `deep_response_gt50pct_of_responders` | Percent of responders with ≥50% volume reduction | Results text; ReNeu only |
| `median_time_to_response_months` | Median time to first confirmed response | Results text |
| `any_ae_pct` | Percent of patients with any adverse event | Safety results text/table |
| `grade3plus_ae_pct` | Percent of patients with a grade ≥3 adverse event | Safety results text/table |
| `grade3plus_ae_basis` | Causality basis of `grade3plus_ae_pct`: `"treatment-related"` for ReNeu (its AE table reports treatment-related events only), `"all-cause"` for KOMET (its AE table reports all adverse events regardless of attribution), `None` where `grade3plus_ae_pct` itself is not reported (SPRINT I/II) | Table legend / methods describing the adverse-event table in each paper |
| `discontinued_for_ae_pct` | Percent of patients who discontinued treatment due to an adverse event | Results text; mirrors `discontinued_for_ae_pct` in `trial_dropout_summary.csv` |

### `trial_response_heterogeneity.csv` — one row per trial x cohort x subgroup

Median/min/max best percent change in target PN volume, reported per paper as these summary statistics rather than digitized from individual waterfall-plot bars. For KOMET, the paper reports this only split by `subgroup` ("Objective responders" vs. "Non-responders"), which is why KOMET has no trial-level entry in `trial_efficacy_safety_summary.csv` for this quantity. KOMET's two subgroup rows (`n` = 14 and 51) sum to 65, not the full 71-patient selumetinib arm — this is a known, deliberate post hoc subset (the paper excludes patients who were not evaluable for this analysis), not a transcription gap.

### `trial_dropout_summary.csv` — one row per trial x cohort x arm

| Column | Meaning | Source location |
|---|---|---|
| `discontinued_any_reason_pct`/`_n` | Overall discontinuation count/percent, any cause | SPRINT II results text; KOMET Figure 1B CONSORT diagram. Left `None` for ReNeu, whose results text reports only AE-attributed discontinuation, not an overall figure — not inferred. |
| `discontinued_for_ae_pct`/`_n` | Discontinuation attributed to an adverse event | Same locations as above |
| `dose_reduction_pct`, `dose_interruption_pct` | Percent of patients with a dose reduction / interruption | Results text |
| `notes` | Verbatim breakdown of discontinuation reasons as reported in the paper (e.g., KOMET Figure 1B's reason-by-reason counts) | Source of truth used to cross-check `discontinued_*_n`/`_pct` for internal consistency |

The KOMET `discontinued_for_ae_n`/`_pct` values (10/71, 14% for selumetinib; 6/74, 8% for placebo) were reconciled against the reason-by-reason breakdown in `notes` ("adverse event 10", "adverse event 6"), which independently sums to the already-verified `discontinued_any_reason_n` totals (18 and 15, respectively).

**Known conflict with `trial_efficacy_safety_summary.csv`:** that file's `discontinued_for_ae_pct` for KOMET is 13% (selumetinib) / 7% (placebo) — lower than this file's 14%/8%. Both figures are plausible as reported in the paper, and the most likely explanation is that they come from different follow-up windows (the randomized-treatment-period table vs. the through-data-cutoff CONSORT figure that this file's Figure 1B-derived counts come from), but this has not been independently confirmed against the source text. Until confirmed, treat the two files' KOMET discontinuation-for-AE figures as non-interchangeable rather than assuming one is simply wrong.

### `trial_adverse_events_detail.csv` — one row per trial x arm x adverse-event term

Limited to ReNeu and KOMET, the two papers whose main text includes a per-term adverse-event frequency table. SPRINT II's toxicity is reported only narratively (most frequent terms named, no frequency table) and is excluded here rather than assigning it fabricated percentages.

| Column | Meaning | Source location |
|---|---|---|
| `pct_all_grades` | All-grade incidence of the named adverse-event term | ReNeu Table 2 (treatment-related AEs occurring in ≥5% of either cohort); KOMET Table 3 (all adverse events occurring in ≥10% of either arm) |
| `pct_grade3plus` | Grade ≥3 incidence of the named term | Only available per-term for ReNeu (Table 2's grade ≥3 column). `None` for KOMET, which reports grade ≥3 only as a trial-level aggregate (see `grade3plus_ae_pct`/`grade3plus_ae_basis` in `trial_efficacy_safety_summary.csv`), not per term. |
| `ae_basis` | Causality basis of this row's rates: `"treatment-related"` for every ReNeu row, `"all-cause"` for every KOMET row | Derived from each trial's AE-table definition (same distinction as `grade3plus_ae_basis` in `trial_efficacy_safety_summary.csv`), not an independently transcribed value |

## Transcription methodology and limitations

- **Manual transcription only.** Every value is typed in directly from the published text, table, or reported summary statistic. No OCR, no digitizing of plot pixel positions (e.g., waterfall-plot bar heights), no interpolation.
- **`None` is deliberate, not missing data.** A `None` means the source paper does not report that quantity for that cohort at all (e.g., ReNeu's overall discontinuation rate, KOMET's per-term grade ≥3 breakdown) — it is never used to mean "not yet transcribed."
- **Causality basis differs by trial, and sometimes within a trial's own row.** `grade3plus_ae_basis` and `ae_basis` record the treatment-related-vs.-all-cause distinction between ReNeu and KOMET (see those columns above). Within ReNeu's own `trial_efficacy_safety_summary.csv` rows, `any_ae_pct` (100%) is all-cause while the adjacent `grade3plus_ae_pct` is treatment-related — the two columns are not on the same basis even within one row. Comparing any AE rate across trials, or across columns within a row, without checking the relevant basis column first is not a like-for-like comparison.
- **"Durable response" means a different thing in each trial**, so `durable_response_pct` should never be compared across trials at face value:

  | Trial | Definition | Denominator |
  |---|---|---|
  | SPRINT Phase II (56%) | Confirmed response lasting ≥12 cycles | All patients |
  | ReNeu (96% adults, 100% children) | Response still ongoing/durable at data cutoff | Responders only |
  | KOMET (86%) | Response lasting ≥6 months | Responders only |

- **ORR assessment method and timing differ across trials** and are not harmonized in this data: SPRINT's reported ORR used unblinded NCI investigator assessment (an independent central review of SPRINT has been cited elsewhere giving a different, lower figure); KOMET's primary ORR window runs through cycle 16; ReNeu's through cycle 24.
- **KOMET's 20% vs. 5% ORR and 32%/18% grade ≥3 AE rates, and ReNeu's 41%/52% ORR, were independently cross-checked** against secondary published summaries during this review (see citation table above) and match exactly.

## Open items from an external review (not yet applied)

A separate review pass (not run by, or independently verifiable from, this session — the source PDFs it used are not in this repo) flagged the items below. Each is a specific, falsifiable claim about the source papers that I could not confirm against primary text during this session (NEJM/PubMed/PMC blocked automated access to the one document I could try, the SPRINT II correction notice; the rest would need the actual paper PDFs). None of these have been applied to the data. Treat them as a punch list for whoever has direct access to the source PDFs, not as settled fixes:

- **SPRINT II `orr_pct`:** possibly corrected to 68% (34/50) by NEJM's published correction notice (2020;383(13):1290) — see the note under Source publications above. Unresolved; current value (70%) is backed by the verbatim, pre-correction Results text.
- **KOMET placebo paronychia `pct_all_grades`:** claimed to be 4% (3/74) per Table 3, not the 12% currently transcribed.
- **KOMET placebo dropout `notes`:** claimed that "before protocol-specified crossover" mischaracterizes what Figure 1B covers (claim: only 8/74 placebo patients never crossed over, so most of the 15 discontinuations likely happened post-crossover).
- **SPRINT Phase I has no dropout row** in `trial_dropout_summary.csv`. Claimed figures: 5/24 discontinued (1 for dose-limiting toxicity), 11/24 had a dose reduction.
- **SPRINT II `median_time_to_response_months`** is blank; claimed the paper reports 8 cycles to initial response (not directly comparable to other trials' month-denominated values without a cycle-length conversion).
- **KOMET selumetinib grade ≥3 CPK increase:** claimed derivable per-term as ~10% (7/71: 5 grade 3, 2 grade 4) despite KOMET otherwise lacking per-term grade ≥3 data.
- **ReNeu ejection-fraction decrease** adverse-event term claimed missing from `trial_adverse_events_detail.csv` (claimed 12% adults / 20% children, treatment-related).
- **KOMET placebo-only ORR:** claimed that 2 of the 4 placebo "responders" were confirmed only after crossing over to selumetinib, making the true placebo-only ORR 2/74 (2.7%) rather than 5%.
- Minor, data-unaffected items: a possible typo between KOMET's Discussion text and Figure 2 footnote for an IQR value; a possible mismatch between the SPRINT Phase I `min_pct_change` (-6, sourced via the Phase II paper's citation) and a -5.8 figure in the Phase I paper's own text for a broader patient set.

If you have access to the primary PDFs (or the NEJM correction notice) and can confirm any of these, they should be applied with the same per-field provenance documentation as the rest of this file.
