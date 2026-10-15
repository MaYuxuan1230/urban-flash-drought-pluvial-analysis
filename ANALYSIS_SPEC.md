# Analysis specification

This document states what the five programs calculate. It does not claim that
synthetic examples reproduce the manuscript's sample sizes or numerical results.

## 01 — Frequency

Counts come from the full detected-event catalogue and are joined to the supplied
city/year panel. FD_P proportion is FD_P / (FD_P + FD_only), undefined when the
denominator is zero. `has_fd_p_in_study_period` permits the at-least-one-FD_P
city subset to be selected without changing the global annual denominator.
No trend estimator or count model is fitted.

## 02 — Responses and metrics

Inputs already contain the study's seasonal and trend removal. For each variable,
FD lag −4…−1 defines the pre-event baseline and lag 0…4 defines the early window.
Eligible responses require at least three finite pre-windows and four finite
early windows. Trajectory summaries apply the same event-variable eligibility.
The trajectory interval is mean ± 1.96 × sample SD / sqrt(number of events);
it is not a city-block or spatially adjusted interval.

- Loss: negative of the early-window minimum standardized anomaly; no clipping
  is applied. Positive anomalies can therefore produce a negative signed loss.
- Nadir timing: elapsed time from the lag-zero composite start to the minimum's
  composite start. The window count is also retained. This does not resolve the
  exact within-composite time of the physiological minimum.
- Recovery: from the window **after** the early minimum, find the first two
  consecutive windows at or above −0.25 standardized units, through lag +8.
  Recovery time is their first start date minus the nadir window start date.
  No eligible attainment by the endpoint is retained as censored, not set to zero.
- Legacy: mean standardized anomaly at lags 5…8 relative to the seasonal/trend
  baseline, not an automatic estimate of damage newly caused by this event.
- Early EVI peak (called rebound in the manuscript): maximum anomaly at lags
  0…4. It is not peak minus nadir and need not occur after rainfall.
- Thermal peak: maximum physical-unit daytime (LST − air-temperature) anomaly
  at lags 0…4, in °C; the standardized thermal peak is a separate field.

The manuscript's recovery definition does not first require the nadir to be
below −0.25. This operational rule is retained for consistency, and
`nadir_below_recovery_threshold` exposes the distinction. Threshold-attainment
time must not automatically be interpreted as damage recovery in events that
never entered the below-threshold state. This repository does not silently
replace it with a different risk set or a survival model.

## 03 — Matching and paired comparisons

FD_P targets are processed in event-identifier order and assigned one unused
FD_only control, preferentially in the same city; an acceptable same-climate
fallback is allowed. All five covariates are required: minimum soil-moisture
percentile, decline rate, duration, pre-event EVI and pre-event VPD anomalies.
Distance is Euclidean after division by each candidate pool's sample SD, plus
the circular day-of-year distance / 30. A pool with an undefined or zero SD
for a required dimension is infeasible rather than reduced to fewer covariates.
The maximum distance is 4; matching is without replacement.

The six principal comparisons are GPP loss, GPP nadir time, EVI observed recovery
time, EVI legacy, early EVI peak and physical thermal peak. Differences are
FD_P minus FD_only. Variable-specific eligibility is applied after pairing;
recovery comparisons require observed recovery in both events. Balance is
reported for all five covariates.

Intervals use 2,000 resamples of complete matched pairs. The standardized paired
effect is the mean difference divided by the sample SD of differences; its
interval is the raw mean interval divided by the same observed SD. Wilcoxon
tests and Benjamini–Hochberg q values are reported for the six principal
comparisons. Pair resampling is retained as the manuscript procedure; it is
not represented as a correction for repeated events within cities.

## 04 — GAMMs

Use eligible events, aggregate responses and covariates to city × event-type
medians, and require at least two valid source events. Recovery uses observed
threshold attainments and ln(1 + days). Each joint Gaussian `mssm` GAMM includes
an event-type intercept, four event-type-specific univariate smooths (AI,
vegetation, VPD, built-up fraction), **one outcome-specific bivariate smooth**,
and a city random intercept:

| Outcome | Bivariate smooth |
|---|---|
| GPP loss | AI × vegetation |
| EVI recovery time | AI × vegetation |
| Physical thermal peak | VPD × built-up fraction |

These are the model structures used for the current response surfaces; adding
both bivariate terms to every model would be a different specification.
Smoothing penalties are estimated by `mssm`; no fixed-penalty surrogate is used.
Predictions exclude the random-intercept contribution. Common plotting ranges
are pooled 5th–95th percentiles; other covariates use pooled median references.
Surface support is the intersection of both event types' two-dimensional kNN
support, with k = 30 and the 95th percentile of observed neighbour distances.

Urban effects are centered finite contrasts over exactly 10 percentage points.
Intervals are pointwise, conditional on the fitted smoothing parameters, from
2,000 multivariate-normal coefficient draws. Recovery predictions are inverse
transforms of the fitted log-scale response (bounded below by zero), not
lognormal arithmetic-mean or survival-time estimates. No threshold detection
is included.

## 05 — City typology

Indicators are long-term P/PET, annual FD_P frequency, mean FD_P GPP loss,
EVI recovery burden, mean physical thermal peak, vegetation and built-up
fractions. Ecological summaries use variable-specific eligible FD_P events;
frequency uses the **full catalogue** and the number of panel years.
At least three eligible FD_P–EVI events and all seven finite indicators are
required. Background indicators are retained as in the manuscript.

Recovery burden is `p_observed × mean_observed_days + (1 − p_observed) × 56`.
The 56-day value is a fixed penalty score, not actual follow-up for every event
or a survival-based censoring correction.

Indicators are winsorized at P1/P99 and scaled by median/IQR. K-medoids uses
Manhattan distance, 18 full-sample initializations and k = 3…6. Stability uses
1,000 independent 80% samples without replacement, six initializations per
subsample, and ARI against the full solution after assigning all cities to the
subsample medoids. Selection combines min–max scores: 0.45 silhouette + 0.35
ARI + 0.15 reversed Davies–Bouldin + 0.05 minimum-cluster fraction; subtract
0.35 when the minimum fraction is below 0.08. No k is forced by published counts.

PCA uses centered robust scores; arbitrary component signs are fixed by their
largest absolute loading. Outputs include medoids, profile medians/IQRs and
city assignments. `membership_strength` is one minus nearest-medoid distance
divided by the sum of the nearest two distances; it is a descriptive closeness
score, not a posterior membership probability. The three descriptive names
follow the manuscript; other k values receive neutral numeric labels.
