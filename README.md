# Urban flash-drought-to-pluvial analysis

Reference implementation of the principal analyses in the global urban
flash-drought-to-heavy-rainfall study. Five programs operate on analysis-ready
city and event tables. They export numerical results, not figures.

## Scope

| Program | Analysis |
|---|---|
| `01_event_frequency_summary.py` | Annual city-level counts and FD_P proportions, including zero-event years |
| `02_event_response_and_metrics.py` | Lag trajectories, loss, nadir timing, threshold-based recovery, legacy and thermal metrics |
| `03_event_matching_and_effects.py` | Strict five-covariate matching, balance and the six principal paired comparisons |
| `04_gamm_response_surfaces.py` | Joint event-type-specific Gaussian GAMMs with city random intercepts, surfaces and marginal effects |
| `05_city_typology.py` | Seven-indicator robust k-medoids typology, stability assessment and PCA |

Raster preprocessing, event identification, temporal-trend estimation,
threshold detection and plotting are outside this repository. The supplied
event catalogue is assumed to use the study's reference definition: local
monthly P95 of rolling five-day precipitation, a 10-mm floor and a 1–20-day
transition window. The programs do not re-identify or validate these events.

## Data and reproducibility

The files in `example_data/` are **synthetic execution examples**, not study
data. They demonstrate interfaces and algorithms; their outputs must not be
interpreted as manuscript estimates or used to verify published numerical
results. The full analysis-ready observational inputs are not included here.
The figure Source Data workbooks and these synthetic inputs are different
resources; a figure workbook is not automatically an input to this pipeline.

See [INPUT_SCHEMA.md](INPUT_SCHEMA.md) for the three input tables and
[ANALYSIS_SPEC.md](ANALYSIS_SPEC.md) for definitions, units and statistical
scope. No figure-specific intermediate table is required by the implementation.
Matching estimates conditional associations, not randomized causal effects.

## Installation

Use Python 3.11 or later in a dedicated environment:

```bash
python -m pip install -r requirements.txt
```

`requirements-tested.txt` records the package versions used for the local
Windows/Python 3.11 verification. The general requirements allow compatible
versions. GAMMs require `mssm`; no alternative model is silently substituted.
Conda DLL-directory handling is included for direct Windows/PyCharm execution.

## Run

Run from the repository root. Each command below is a single line and works in
PowerShell, a terminal, or a PyCharm run configuration with these arguments:

```bash
python 01_event_frequency_summary.py --events example_data/city_events.csv --city-panel example_data/city_year.csv --output results/city_year_event_summary.csv
python 02_event_response_and_metrics.py --responses example_data/event_response_long.parquet --events example_data/city_events.csv --city-year example_data/city_year.csv --lag-summary results/event_lag_summary.csv --metrics results/event_response_metrics.parquet
python 03_event_matching_and_effects.py --metrics results/event_response_metrics.parquet --output-dir results/matching
python 04_gamm_response_surfaces.py --metrics results/event_response_metrics.parquet --output-dir results/gamm
python 05_city_typology.py --metrics results/event_response_metrics.parquet --events example_data/city_events.csv --city-year example_data/city_year.csv --output-dir results/clustering
```

Output directories are created automatically. Default settings include 2,000
paired bootstrap resamples, 2,000 conditional coefficient draws, 30-neighbour
common support, 18 full-sample clustering initializations and 1,000 independent
80% stability subsamples. `--clusters 0` evaluates 3–6 clusters; the synthetic
example need not select the manuscript's three-cluster solution.

For a quick execution check only, use `--bootstrap-draws 200`,
`--posterior-draws 500 --support-k 5`, and
`--clusters 3 --stability-runs 10` for programs 03, 04 and 05, respectively.
These reduced settings are not the manuscript settings.

## Tests and versioning

```bash
python -m unittest discover -s tests -v
```

Continuous integration runs the definition tests and the synthetic analysis
chain. Passing these tests establishes implementation behaviour, not
reproduction of the observational study. Cite an immutable commit or an
archived release rather than an unspecified moving branch.

## Licence

MIT; see [LICENSE](LICENSE). No dataset licence or access permission is granted
by the software licence.
