# pni-aki-analysis

Analysis code for: *Incremental predictive value of the preoperative prognostic
nutritional index over albumin for creatinine-defined acute kidney injury after
major abdominal surgery: a dual-cohort study with testing-window sensitivity
analysis.*

Two de-identified cohorts:

- **INSPIRE (derivation)** — Seoul National University Hospital, Republic of Korea.
  Publicly available via PhysioNet (doi:10.13026/1eay-yc85).
- **MOVER (replication)** — University of California, Irvine. Restricted access;
  download by signing the UCI data use agreement (doi:10.24432/C5VS5G).

This repository contains **code only**. No patient data, cohort extracts, or
output files are committed, in accordance with both data use agreements.

## Data access

| Cohort | Source | Access |
|--------|--------|--------|
| INSPIRE v1.4.2 | https://doi.org/10.13026/1eay-yc85 (PhysioNet) | open to credentialed users |
| MOVER | https://doi.org/10.24432/C5VS5G | UCI DUA + data-request form |

Obtain the raw files before running; the scripts expect the paths configured
below.

## Configuration (environment variables)

Paths are read from the environment so that no machine-specific location is
hard-coded. The values shown are defaults; override them for your install.

| Variable | Meaning | Default (placeholder) |
|----------|---------|-----------------------|
| `INSPIRE_DATA` | INSPIRE data directory (`*.csv.gz`) | `./INSPIRE-1.4.2` |
| `INSPIRE_OUT`  | results/output directory | `output` |
| `INSPIRE_WORK` | R working directory | `.` |
| `MOVER_LABS`   | MOVER patient_labs.csv | `./MOVER/patient_labs.csv` |
| `RLIB`         | R library path | `./Rlib` |

`paths.py` and the R scripts `Sys.getenv(...)` / `os.environ.get(...)` these
values. Set the variables on your OS before running (or edit the defaults).

## Dependencies

- Python 3.13 (pandas, numpy, scipy, statsmodels)
- R 4.6, with `lcmm`, `survival`, `rms`, `MatchIt`, `tableone`
- For figure/table rendering: `matplotlib`, `python-docx`

## Run order

The scripts are numbered by stage; run top to bottom. Each reads its inputs
from `INSPIRE_OUT` / the configured data directories and writes results back
there.

1. `p0_aki_harmonize_inspire.py` — build the INSPIRE analysis cohort; rolling
   KDIGO (creatinine-only) AKI definitions.
2. `p0_aki_harmonize_mover.py` / `p0_mover_rebuild_unified.py` — build the MOVER
   cohort and harmonised outcome.
3. `p0_final_two_cohort.py` — Table 2 models (derivation + replication),
   complete-case and multiple-imputation estimates.
4. `p0_rolling_compare.py` — comparison of the three reference definitions
   (rolling minimum / most-recent / fixed preoperative baseline).
5. `10c_preop_density_analysis.py` — pre- vs postoperative testing-density
   attenuation.
6. `11b_pni_aki_timing_v2.py` — trajectory timing (PNI measurements vs. first
   qualifying creatinine).
7. `15_alb_vs_pni_dauc.py` — paired ΔAUC, NRI, decision-curve analysis.
8. `07_prediction_increment.py` / `08b_multimarker_compare.py` — predictive
   increments and marker comparison.
9. `09_make_figures.py`, `09b_fig2_v2.py`, `13_make_flow_figure.py` — figures.
10. `14_make_tables_docx.py` — Word tables.
11. `r/05b_lcmm_grid.R`, `r/06b_lcmm_adjusted.R`, `r/10b_bic_model.R` — latent
    class trajectory models and the outcome-blind BIC comparator.

`test_aki_rolling.py` exercises `aki_rolling.py` (the AKI criterion function)
against fixed edge cases; run it after cloning to confirm the outcome
implementation is intact.

## AKI criterion

`aki_rolling.py` defines the creatinine-based rolling AKI rules (absolute and
relative pathways, staging, first qualifying time point). This single function
is the outcome implementation the manuscript's "code audit" statement refers
to; its behaviour is pinned by the test file.

## Data availability

INSPIRE: https://doi.org/10.13026/1eay-yc85
MOVER: https://doi.org/10.24432/C5VS5G
