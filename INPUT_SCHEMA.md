# Input schema

CSV or Parquet is accepted for city/event tables; the response long table must
be Parquet. Column names are case-sensitive. Keep identifiers as strings,
missing values as missing (not zero), and Boolean flags as true/false or 1/0.
Extra columns are allowed. The tables must describe the same study period.

## `city_events`

One row per detected flash drought, including events without usable ecological
responses. `event_id` is globally unique. Programs 01, 02 and 05 use this table.

| Field | Definition / unit |
|---|---|
| `event_id`, `city_id` | Event and city identifiers |
| `event_type` | `FD_only` or `FD_P` |
| `fd_onset` | Catalogue onset date used to assign the FD alignment window; ISO date |
| `sm_min_pct_weighted` | Area-weighted minimum soil-moisture percentile, 0–100 |
| `decline_rate_weighted` | Area-weighted decline rate, percentile points per pentad |
| `fd_duration_days_weighted` | Area-weighted drought duration, days |
| `vpd_mean_drought` | Mean VPD during drought, kPa |

The first four fields are required for catalogue joins and counts; the remaining
fields supply matching and model covariates. Onset/formation and rainfall-trigger
dates must be defined by the upstream catalogue; no event-detection convention
is inferred from a file name.

## `city_year`

One row per city and year, **including zero-event years**. The panel defines
frequency denominators; missing years must not be silently interpreted as zero
events. Programs 01, 02 and 05 use this table.

| Field | Definition / unit |
|---|---|
| `city_id`, `year` | Unique city/year key |
| `city_name` | Optional display name |
| `longterm_aridity_index` | Long-term total precipitation / total PET |
| `vegetation_fraction` | Urban vegetation fraction, 0–1 |
| `built_up_fraction` | Built-up fraction, 0–1 |
| `dominant_climate_code` | Dominant Köppen–Geiger class, needed for matching fallback |
| `dominant_climate_name` | Optional alternative climate label |

`built_fraction` is accepted when `built_up_fraction` is absent. If both are
provided, the canonical `built_up_fraction` is used. Program 05 also accepts
`vegetation_fraction_mean` and `built_fraction_mean` as fallback names.
An average of annual P/PET ratios is **not** substituted for long-term P/PET.

## `event_response_long`

One row per event × variable × alignment × lag. Only response events within
the ecological analysis domain (including the upstream growing-season rule)
should be supplied. Programs do not perform phenology or raster preprocessing.

| Field | Definition / unit |
|---|---|
| `event_id`, `city_id`, `event_type` | Must agree with `city_events` |
| `alignment` | `fd` or `transition` (the latter exists only for FD_P) |
| `lag` | Integer composite-window offset: FD −4…+8; transition −3…+6 |
| `variable` | `evi_vegetation`, `gpp_primary`, `vpd_mean`, `thermal_excess_day_all`, etc. |
| `eightday_start` | Actual composite start date, including short year-end intervals |
| `detrended_standardized_anomaly` | Deseasonalized, detrended robust standardized response |
| `detrended_anomaly` | Physical-unit detrended anomaly; °C for thermal excess |

For matching, include FD-aligned EVI and VPD responses. For the principal
comparisons/models, include FD-aligned EVI, GPP and daytime thermal excess.
Transition alignment is optional for the principal paired comparisons.
Do not substitute physical-unit anomalies for standardized EVI/GPP responses.

## Program 02 output

`event_response_metrics.parquet` contains one row per event and variable, with
event/city-year context attached. It supplies all inputs to programs 03 and 04
and the ecological indicators in program 05. Important fields include:

- `event_response_eligible`: at least three finite FD pre-windows (−4…−1)
  and four finite early post-windows (0…4). Programs 03–05 apply this flag.
- `pre_event_evi_anomaly`, `pre_event_vpd_anomaly`: means over valid pre-windows;
  at least three valid pre-windows are required to supply each covariate.
- `time_to_nadir_days`, `recovery_time_days_after_nadir`: elapsed days between
  actual composite start dates, not daily physiological observations.
- `recovery_observed`, `recovery_censored`: threshold-attainment status; a
  recovery time is used in a paired difference only if both events have an
  observed time.
- `nadir_below_recovery_threshold`: diagnostic flag, not an additional filter
  silently imposed on the manuscript's recovery definition.
- `post_peak_value_lag_0_4`: the early EVI peak used by the paper's rebound
  comparison. `overshoot_transition_lag_1_4` is a separate FD_P-only descriptor,
  **not** its paired replacement.

Programs reject missing required fields and duplicate keys rather than silently
reducing the matching covariates or treating missing responses as zeros.
