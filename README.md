# Urban flash-drought-to-pluvial analysis

Code for the principal statistical analyses in the global-city
flash-drought and post-drought pluvial-transition study.

## Scope

The repository contains five analysis programs:

1. `01_event_frequency_summary.py` — annual city-level event summaries.
2. `02_event_response_and_metrics.py` — lagged responses and event-level metrics.
3. `03_event_matching_and_effects.py` — matched-event comparisons.
4. `04_gamm_response_surfaces.py` — GAMM response surfaces, event-type contrasts and marginal effects.
5. `05_city_typology.py` — robust k-medoids typology, stability analysis and PCA.

The implementation follows the manuscript analysis: matching requires all five
pre-event covariates; GAMMs use median city-by-event-type inputs with at least
two events, joint event-type-specific smooths, a city random intercept, common
30-nearest-neighbour support and 2,000 conditional posterior draws; clustering
uses 18 k-medoids initializations and 1,000 independent 80% subsamples by
default. The programs export numerical analysis results only and do not contain
the manuscript plotting layer.


## Data policy

The files in `example_data/` are fully synthetic and are supplied only to verify that the
analysis workflow executes. Required input fields are documented in [`INPUT_SCHEMA.md`](INPUT_SCHEMA.md).

## Software

- Python 3.11 or later
- Python packages in `requirements.txt`

```bash
python -m pip install -r requirements.txt
```

## Example run

Run commands from the repository root.

```bash
mkdir results

python 01_event_frequency_summary.py \
  --events example_data/city_events.csv \
  --city-panel example_data/city_year.csv \
  --output results/city_year_event_summary.csv

python 02_event_response_and_metrics.py \
  --responses example_data/event_response_long.parquet \
  --events example_data/city_events.csv \
  --city-year example_data/city_year.csv \
  --lag-summary results/event_lag_summary.csv \
  --metrics results/event_response_metrics.parquet

python 03_event_matching_and_effects.py \
  --metrics results/event_response_metrics.parquet \
  --output-dir results/matching

python 04_gamm_response_surfaces.py \
  --metrics results/event_response_metrics.parquet \
  --output-dir results/gamm \
  --posterior-draws 2000 \
  --support-k 30

python 05_city_typology.py \
  --metrics results/event_response_metrics.parquet \
  --city-year example_data/city_year.csv \
  --output-dir results/clustering \
  --clusters 0 \
  --stability-runs 1000
```

The clustering program evaluates three to six groups and uses the published
selection score when `--clusters 0`. The synthetic inputs are for execution
checks only and are not the study data. All randomised analyses use fixed seeds.
