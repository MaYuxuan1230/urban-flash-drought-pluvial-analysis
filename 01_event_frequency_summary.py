#!/usr/bin/env python3
"""Summarise annual flash-drought event frequencies at the city level.

This script starts from the processed event catalogue. It does not identify
events and does not estimate temporal trends or fit frequency models.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_TYPES = ("FD_only", "FD_P")


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


def require_columns(data: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))


def summarise(events: pd.DataFrame, city_panel: pd.DataFrame) -> pd.DataFrame:
    require_columns(events, {"event_id", "city_id", "event_type", "fd_onset"})
    data = events.loc[events["event_type"].isin(EVENT_TYPES)].copy()
    data["fd_onset"] = pd.to_datetime(data["fd_onset"], errors="coerce")
    data = data.dropna(subset=["fd_onset"])
    data["year"] = data["fd_onset"].dt.year.astype(int)
    data["city_id"] = data["city_id"].astype(str)

    data["fd_count"] = 1
    data["fd_p_count"] = (data["event_type"] == "FD_P").astype(int)
    data["fd_only_count"] = (data["event_type"] == "FD_only").astype(int)

    optional_means = {
        "transition_interval_days": "mean_transition_interval_days",
        "sm_min_pct_weighted": "mean_minimum_soil_moisture_percentile",
        "decline_rate_weighted": "mean_soil_moisture_decline_rate",
        "fd_duration_days_weighted": "mean_flash_drought_duration_days",
        "max_fd_coverage": "mean_maximum_flash_drought_coverage",
        "wet_p5max_20d": "mean_post_drought_maximum_5day_precipitation",
        "wet_threshold_ratio_max_20d": "mean_post_drought_precipitation_ratio",
    }
    aggregations: dict[str, tuple[str, str]] = {
        "fd_count": ("fd_count", "sum"),
        "fd_p_count": ("fd_p_count", "sum"),
        "fd_only_count": ("fd_only_count", "sum"),
    }
    for source, output in optional_means.items():
        if source in data.columns:
            data[source] = pd.to_numeric(data[source], errors="coerce")
            aggregations[output] = (source, "mean")

    event_summary = (
        data.groupby(["city_id", "year"], observed=True)
        .agg(**aggregations)
        .reset_index()
        .sort_values(["city_id", "year"], kind="stable")
    )

    require_columns(city_panel, {"city_id", "year"})
    panel = city_panel.copy()
    panel["city_id"] = panel["city_id"].astype(str)
    panel["year"] = pd.to_numeric(panel["year"], errors="raise").astype(int)
    panel = panel.drop_duplicates(["city_id", "year"])
    overlapping = [
        column for column in event_summary.columns
        if column in panel.columns and column not in {"city_id", "year"}
    ]
    panel = panel.drop(columns=overlapping)
    result = panel.merge(event_summary, on=["city_id", "year"], how="left")
    for column in ("fd_count", "fd_p_count", "fd_only_count"):
        result[column] = result[column].fillna(0).astype(int)
    result["fd_p_fraction"] = np.divide(
        result["fd_p_count"], result["fd_count"],
        out=np.full(len(result), np.nan), where=result["fd_count"].to_numpy() > 0,
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--city-panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = summarise(read_table(args.events), read_table(args.city_panel))
    write_table(result, args.output)
    print(f"Wrote {len(result):,} city-year records to {args.output}")


if __name__ == "__main__":
    main()
