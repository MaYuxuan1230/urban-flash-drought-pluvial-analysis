#!/usr/bin/env python3
"""Summarise lagged responses and calculate event-level response metrics.

Input is the processed event-by-lag table. Responses are based on detrended,
seasonally standardised anomalies. The script retains right-censored recovery
events and processes one variable at a time to limit memory use.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

try:
    import duckdb
except ImportError:  # Small datasets can use the pandas fallback below.
    duckdb = None


RECOVERY_THRESHOLD = -0.25
RECOVERY_CONSECUTIVE_WINDOWS = 2
THERMAL_VARIABLES = {
    "thermal_excess_day_all",
    "thermal_excess_day_vegetation",
    "thermal_excess_day_built",
    "thermal_excess_night_all",
}


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''").replace("\\", "/")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_table(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".parquet", ".pq"}:
        data.to_parquet(path, index=False)
    else:
        data.to_csv(path, index=False)


def require_columns(data: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{name} is missing: " + ", ".join(missing))


def first_consecutive_lag(
    lag: pd.Series,
    value: pd.Series,
    start_lag: int,
    mode: str,
) -> tuple[float, bool]:
    data = pd.DataFrame({"lag": lag, "value": value}).dropna()
    data = data.loc[data["lag"] >= start_lag].sort_values("lag")
    if data.empty:
        return np.nan, True

    lags = data["lag"].to_numpy(dtype=int)
    values = data["value"].to_numpy(dtype=float)
    for index in range(len(data) - RECOVERY_CONSECUTIVE_WINDOWS + 1):
        window_lags = lags[index : index + RECOVERY_CONSECUTIVE_WINDOWS]
        if not np.all(np.diff(window_lags) == 1):
            continue
        window = values[index : index + RECOVERY_CONSECUTIVE_WINDOWS]
        valid = (
            np.all(window >= RECOVERY_THRESHOLD)
            if mode == "ecological"
            else np.all(window <= 0)
        )
        if valid:
            return float(window_lags[0]), False
    return np.nan, True


def lag_summary(source: Path) -> pd.DataFrame:
    if duckdb is None:
        columns = [
            "event_type", "alignment", "variable", "lag",
            "detrended_standardized_anomaly", "standardized_anomaly",
        ]
        data = pd.read_parquet(source, columns=columns)
        data["response"] = pd.to_numeric(
            data["detrended_standardized_anomaly"], errors="coerce"
        ).combine_first(pd.to_numeric(data["standardized_anomaly"], errors="coerce"))
        data = data.dropna(subset=["response"])
        grouped = data.groupby(
            ["event_type", "alignment", "variable", "lag"], observed=True
        )["response"]
        result = grouped.agg(
            n="count", mean="mean", median="median", sd="std",
            q25=lambda values: values.quantile(0.25),
            q75=lambda values: values.quantile(0.75),
        ).reset_index()
        standard_error = result["sd"] / np.sqrt(result["n"])
        result["ci_low"] = result["mean"] - 1.96 * standard_error
        result["ci_high"] = result["mean"] + 1.96 * standard_error
        return result.sort_values(["variable", "alignment", "event_type", "lag"])

    connection = duckdb.connect()
    table = f"read_parquet('{sql_path(source)}')"
    query = f"""
        WITH x AS (
            SELECT
                event_type,
                alignment,
                variable,
                CAST(lag AS INTEGER) AS lag,
                COALESCE(detrended_standardized_anomaly, standardized_anomaly) AS response
            FROM {table}
            WHERE event_type IN ('FD_only', 'FD_P')
        )
        SELECT
            event_type, alignment, variable, lag,
            COUNT(response) AS n,
            AVG(response) AS mean,
            MEDIAN(response) AS median,
            STDDEV_SAMP(response) AS sd,
            QUANTILE_CONT(response, 0.25) AS q25,
            QUANTILE_CONT(response, 0.75) AS q75,
            AVG(response) - 1.96 * STDDEV_SAMP(response) / SQRT(COUNT(response)) AS ci_low,
            AVG(response) + 1.96 * STDDEV_SAMP(response) / SQRT(COUNT(response)) AS ci_high
        FROM x
        WHERE response IS NOT NULL
        GROUP BY event_type, alignment, variable, lag
        ORDER BY variable, alignment, event_type, lag
    """
    result = connection.execute(query).df()
    connection.close()
    return result


def metric_record(group: pd.DataFrame, transition: pd.DataFrame | None) -> dict:
    group = group.sort_values("lag")
    pre = group.loc[group["lag"].between(-4, -1)]
    post = group.loc[group["lag"].between(0, 4)].dropna(subset=["response_value"])
    legacy = group.loc[group["lag"].between(5, 8)]
    variable = str(group.iloc[0]["variable"])

    if post.empty:
        nadir_value = nadir_lag = peak_value = peak_lag = np.nan
    else:
        nadir = post.loc[post["response_value"].idxmin()]
        peak = post.loc[post["response_value"].idxmax()]
        nadir_value, nadir_lag = float(nadir["response_value"]), int(nadir["lag"])
        peak_value, peak_lag = float(peak["response_value"]), int(peak["lag"])

    recovery_lag, recovery_censored = first_consecutive_lag(
        group["lag"],
        group["response_value"],
        int(nadir_lag) + 1 if np.isfinite(nadir_lag) else 0,
        "ecological",
    )
    recovery_windows = (
        recovery_lag - nadir_lag
        if np.isfinite(recovery_lag) and np.isfinite(nadir_lag)
        else np.nan
    )

    recovery_days = np.nan
    censor_windows = np.nan
    censor_days = np.nan
    if np.isfinite(nadir_lag):
        nadir_dates = pd.to_datetime(
            group.loc[group["lag"] == int(nadir_lag), "eightday_start"],
            errors="coerce",
        ).dropna()
        follow = group.loc[(group["lag"] >= nadir_lag) & group["response_value"].notna()]
        if not follow.empty:
            last_lag = int(follow["lag"].max())
            censor_windows = last_lag - nadir_lag
            last_dates = pd.to_datetime(
                follow.loc[follow["lag"] == last_lag, "eightday_start"], errors="coerce"
            ).dropna()
            if len(nadir_dates) and len(last_dates):
                censor_days = float((last_dates.iloc[0] - nadir_dates.iloc[0]).days)
        if np.isfinite(recovery_lag):
            recovery_dates = pd.to_datetime(
                group.loc[group["lag"] == int(recovery_lag), "eightday_start"],
                errors="coerce",
            ).dropna()
            if len(nadir_dates) and len(recovery_dates):
                recovery_days = float((recovery_dates.iloc[0] - nadir_dates.iloc[0]).days)

    overshoot_value = overshoot_lag = np.nan
    if transition is not None:
        window = transition.loc[transition["lag"].between(1, 4)].dropna(
            subset=["response_value"]
        )
        if not window.empty:
            overshoot = window.loc[window["response_value"].idxmax()]
            overshoot_value = float(overshoot["response_value"])
            overshoot_lag = int(overshoot["lag"])

    thermal_recovery_lag = thermal_recovery_censored = np.nan
    thermal_physical = thermal_standardised = np.nan
    if variable in THERMAL_VARIABLES:
        thermal_recovery_lag, thermal_recovery_censored = first_consecutive_lag(
            group["lag"],
            group["response_value"],
            int(peak_lag) + 1 if np.isfinite(peak_lag) else 0,
            "thermal",
        )
        physical = pd.to_numeric(post["response_value_physical"], errors="coerce")
        physical = physical[np.isfinite(physical)]
        thermal_physical = float(physical.max()) if len(physical) else np.nan
        thermal_standardised = peak_value

    valid_pre = int(pre["response_value"].notna().sum())
    valid_post = int(post["response_value"].notna().sum())
    valid_legacy = int(legacy["response_value"].notna().sum())
    first = group.iloc[0]
    return {
        "event_id": first["event_id"],
        "city_id": str(first["city_id"]),
        "event_type": first["event_type"],
        "variable": variable,
        "response_scale": "detrended_standardized_anomaly_SD",
        "baseline_mean_lag_m4_m1": pre["response_value"].mean(),
        "baseline_median_lag_m4_m1": pre["response_value"].median(),
        "nadir_value_lag_0_4": nadir_value,
        "loss_magnitude_lag_0_4": -nadir_value if np.isfinite(nadir_value) else np.nan,
        "time_to_nadir_windows": nadir_lag,
        "recovery_lag_windows": recovery_lag,
        "recovery_time_windows_after_nadir": recovery_windows,
        "recovery_time_days_after_nadir": recovery_days,
        "recovery_observed": not recovery_censored,
        "recovery_censored": recovery_censored,
        "recovery_censor_time_windows": censor_windows,
        "recovery_censor_time_days": censor_days,
        "legacy_mean_lag_5_8": legacy["response_value"].mean(),
        "post_peak_value_lag_0_4": peak_value,
        "post_peak_lag": peak_lag,
        "overshoot_transition_lag_1_4": overshoot_value,
        "overshoot_transition_lag": overshoot_lag,
        "thermal_amplification_lag_0_4": thermal_physical,
        "thermal_amplification_sd_lag_0_4": thermal_standardised,
        "thermal_recovery_lag_windows": thermal_recovery_lag,
        "thermal_recovery_censored": thermal_recovery_censored,
        "valid_pre_windows": valid_pre,
        "valid_post_0_4_windows": valid_post,
        "valid_legacy_5_8_windows": valid_legacy,
        "event_response_eligible": valid_pre >= 3 and valid_post >= 4,
    }


def calculate_metrics(response_file: Path) -> pd.DataFrame:
    dataset = pads.dataset(response_file, format="parquet")
    variables = sorted(
        value.as_py()
        for value in dataset.to_table(columns=["variable"])["variable"].unique()
    )
    parts: list[pd.DataFrame] = []

    columns = [
        "event_id", "city_id", "event_type", "alignment", "lag", "variable",
        "eightday_start", "detrended_standardized_anomaly", "standardized_anomaly",
        "detrended_anomaly",
    ]
    for variable in variables:
        condition = (
            (pads.field("variable") == variable)
            & pads.field("alignment").isin(["fd", "transition"])
        )
        data = dataset.to_table(columns=columns, filter=condition).to_pandas()
        if data.empty:
            continue
        data["lag"] = pd.to_numeric(data["lag"], errors="coerce")
        data["response_value"] = pd.to_numeric(
            data["detrended_standardized_anomaly"], errors="coerce"
        ).combine_first(pd.to_numeric(data["standardized_anomaly"], errors="coerce"))
        data["response_value_physical"] = pd.to_numeric(
            data["detrended_anomaly"], errors="coerce"
        )
        data = data.sort_values(["event_id", "alignment", "lag"], kind="stable")

        transition_lookup = {
            event_id: group
            for event_id, group in data.loc[data["alignment"] == "transition"].groupby(
                "event_id", sort=False
            )
        }
        records = []
        for event_id, group in data.loc[data["alignment"] == "fd"].groupby(
            "event_id", sort=False
        ):
            records.append(metric_record(group, transition_lookup.get(event_id)))
        parts.append(pd.DataFrame.from_records(records))
        print(f"Calculated {variable}: {len(records):,} events")

    if not parts:
        raise ValueError("No event metrics could be calculated.")
    metrics = pd.concat(parts, ignore_index=True)

    context = metrics.loc[
        metrics["variable"].isin(["evi_vegetation", "vpd_mean"]),
        ["event_id", "variable", "baseline_mean_lag_m4_m1"],
    ].pivot_table(
        index="event_id", columns="variable", values="baseline_mean_lag_m4_m1", aggfunc="first"
    )
    context = context.rename(
        columns={
            "evi_vegetation": "pre_event_evi_anomaly",
            "vpd_mean": "pre_event_vpd_anomaly",
        }
    ).reset_index()
    return metrics.merge(context, on="event_id", how="left")


def add_event_context(
    metrics: pd.DataFrame,
    events_file: Path,
    city_year_file: Path,
) -> pd.DataFrame:
    events = read_table(events_file).copy()
    require_columns(events, {"event_id", "city_id", "event_type", "fd_onset"}, "events")
    events["city_id"] = events["city_id"].astype(str)
    events["fd_onset"] = pd.to_datetime(events["fd_onset"], errors="coerce")
    events["event_year"] = events["fd_onset"].dt.year
    events["event_doy"] = events["fd_onset"].dt.dayofyear
    events = events.drop_duplicates("event_id")

    keys = {"event_id", "city_id", "event_type"}
    event_columns = [column for column in events.columns if column not in keys]
    result = metrics.merge(
        events[[*keys, *event_columns]],
        on=["event_id", "city_id", "event_type"],
        how="left",
        validate="many_to_one",
    )

    city_year = read_table(city_year_file).copy()
    require_columns(city_year, {"city_id", "year"}, "city-year table")
    city_year["city_id"] = city_year["city_id"].astype(str)
    city_year = city_year.rename(columns={"year": "event_year"})
    city_year = city_year.drop_duplicates(["city_id", "event_year"])
    context_columns = [
        column
        for column in (
            "dominant_climate_code",
            "dominant_climate_name",
            "longterm_aridity_index",
            "annual_aridity_index",
            "vegetation_fraction",
            "built_up_fraction",
            "built_fraction",
            "annual_vpd_mean",
            "population",
            "population_density_km2",
        )
        if column in city_year.columns
    ]
    return result.merge(
        city_year[["city_id", "event_year", *context_columns]],
        on=["city_id", "event_year"],
        how="left",
        validate="many_to_one",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True, help="Event-response Parquet file")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--city-year", type=Path, required=True)
    parser.add_argument("--lag-summary", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.responses.suffix.lower() not in {".parquet", ".pq"}:
        raise ValueError("--responses must be a Parquet file for memory-efficient processing.")
    summary = lag_summary(args.responses)
    write_table(summary, args.lag_summary)

    metrics = calculate_metrics(args.responses)
    metrics = add_event_context(metrics, args.events, args.city_year)
    write_table(metrics, args.metrics)
    print(f"Wrote {len(metrics):,} event-variable metric records to {args.metrics}")


if __name__ == "__main__":
    main()
