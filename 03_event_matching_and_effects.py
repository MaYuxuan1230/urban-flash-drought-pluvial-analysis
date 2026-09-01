#!/usr/bin/env python3
"""Match FD-to-pluvial events to FD-only events and estimate paired effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MATCH_FEATURES = (
    "sm_min_pct_weighted",
    "decline_rate_weighted",
    "fd_duration_days_weighted",
    "pre_event_evi_anomaly",
    "pre_event_vpd_anomaly",
)
METRICS = (
    "loss_magnitude_lag_0_4",
    "time_to_nadir_windows",
    "recovery_time_windows_after_nadir",
    "recovery_time_days_after_nadir",
    "legacy_mean_lag_5_8",
    "overshoot_transition_lag_1_4",
    "thermal_amplification_lag_0_4",
    "thermal_amplification_sd_lag_0_4",
    "thermal_recovery_lag_windows",
)


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


def circular_day_distance(day_a: float, day_b: pd.Series) -> pd.Series:
    difference = (pd.to_numeric(day_b, errors="coerce") - day_a).abs()
    return np.minimum(difference, 365.25 - difference)


def standardised_distance(target: pd.Series, candidates: pd.DataFrame) -> pd.Series:
    distance = pd.Series(0.0, index=candidates.index)
    valid_count = pd.Series(0, index=candidates.index, dtype=int)
    for feature in MATCH_FEATURES:
        target_value = pd.to_numeric(pd.Series([target.get(feature)]), errors="coerce").iloc[0]
        values = pd.to_numeric(candidates[feature], errors="coerce")
        scale = values.std(ddof=1)
        if not np.isfinite(target_value) or not np.isfinite(scale) or scale <= 0:
            continue
        valid = values.notna()
        distance.loc[valid] += ((values.loc[valid] - target_value) / scale) ** 2
        valid_count.loc[valid] += 1
    distance = np.sqrt(distance)
    distance.loc[valid_count < 3] = np.inf
    return distance


def select_event_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        metrics,
        {"event_id", "city_id", "event_type", "event_year", "event_doy", "variable"},
    )
    event_rows = metrics.loc[metrics["variable"] == "evi_vegetation"].copy()
    if "event_response_eligible" in event_rows.columns:
        event_rows = event_rows.loc[event_rows["event_response_eligible"].fillna(False)]
    event_rows = event_rows.loc[event_rows["event_type"].isin(["FD_P", "FD_only"])]
    event_rows = event_rows.drop_duplicates("event_id").sort_values("event_id")
    for feature in MATCH_FEATURES:
        if feature not in event_rows.columns:
            event_rows[feature] = np.nan
    return event_rows


def match_events(metrics: pd.DataFrame, max_distance: float) -> pd.DataFrame:
    event_rows = select_event_rows(metrics)
    targets = event_rows.loc[event_rows["event_type"] == "FD_P"].copy()
    controls = event_rows.loc[event_rows["event_type"] == "FD_only"].copy()
    used_controls: set[str] = set()
    pairs: list[dict] = []

    climate_column = next(
        (
            column
            for column in ("dominant_climate_code", "dominant_climate_name")
            if column in event_rows.columns and event_rows[column].notna().any()
        ),
        None,
    )

    for target in targets.itertuples(index=False):
        target_row = pd.Series(target._asdict())
        candidate_sets = [("same_city", controls.loc[controls["city_id"] == target.city_id])]
        if climate_column and pd.notna(target_row.get(climate_column)):
            candidate_sets.append(
                (
                    "same_climate",
                    controls.loc[controls[climate_column] == target_row[climate_column]],
                )
            )

        chosen = None
        chosen_scope = None
        chosen_distance = np.inf
        for scope, candidates in candidate_sets:
            candidates = candidates.loc[~candidates["event_id"].astype(str).isin(used_controls)]
            if candidates.empty:
                continue
            distance = standardised_distance(target_row, candidates)
            distance += circular_day_distance(
                float(target.event_doy), candidates["event_doy"]
            ) / 30.0
            if np.isfinite(distance).any():
                index = distance.idxmin()
                chosen = candidates.loc[index]
                chosen_scope = scope
                chosen_distance = float(distance.loc[index])
                break

        if chosen is None or chosen_distance > max_distance:
            continue
        used_controls.add(str(chosen["event_id"]))
        pairs.append(
            {
                "pair_id": f"PAIR_{len(pairs) + 1:06d}",
                "fd_p_event_id": target.event_id,
                "fd_only_event_id": chosen["event_id"],
                "fd_p_city_id": str(target.city_id),
                "fd_only_city_id": str(chosen["city_id"]),
                "match_scope": chosen_scope,
                "match_distance": chosen_distance,
                **{f"fd_p_{name}": target_row.get(name) for name in MATCH_FEATURES},
                **{f"fd_only_{name}": chosen.get(name) for name in MATCH_FEATURES},
            }
        )
    return pd.DataFrame.from_records(pairs)


def matched_differences(metrics: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    available_metrics = [name for name in METRICS if name in metrics.columns]
    values = metrics[["event_id", "variable", *available_metrics]].copy()
    fd_p = pairs[["pair_id", "fd_p_event_id"]].merge(
        values.rename(columns={"event_id": "fd_p_event_id"}),
        on="fd_p_event_id",
        how="left",
    )
    fd_only = pairs[["pair_id", "fd_only_event_id"]].merge(
        values.rename(columns={"event_id": "fd_only_event_id"}),
        on="fd_only_event_id",
        how="left",
    )
    result = fd_p.merge(
        fd_only,
        on=["pair_id", "variable"],
        how="inner",
        suffixes=("_fd_p", "_fd_only"),
    )
    for metric in available_metrics:
        result[f"{metric}_difference_fd_p_minus_fd_only"] = (
            pd.to_numeric(result[f"{metric}_fd_p"], errors="coerce")
            - pd.to_numeric(result[f"{metric}_fd_only"], errors="coerce")
        )
    return result


def bootstrap_interval(values: np.ndarray, draws: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def false_discovery_rate(p_values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna().sort_values()
    if valid.empty:
        return result
    ranks = np.arange(1, len(valid) + 1)
    adjusted = np.minimum.accumulate((valid.to_numpy() * len(valid) / ranks)[::-1])[::-1]
    result.loc[valid.index] = np.clip(adjusted, 0, 1)
    return result


def effect_summary(
    differences: pd.DataFrame,
    bootstrap_draws: int,
    seed: int,
) -> pd.DataFrame:
    columns = [column for column in differences if column.endswith("_difference_fd_p_minus_fd_only")]
    records = []
    for variable in sorted(differences["variable"].dropna().unique()):
        subset = differences.loc[differences["variable"] == variable]
        for column in columns:
            values = pd.to_numeric(subset[column], errors="coerce").dropna().to_numpy()
            if len(values) < 2:
                continue
            ci_low, ci_high = bootstrap_interval(values, bootstrap_draws, seed)
            try:
                p_value = float(wilcoxon(values, zero_method="wilcox", alternative="two-sided").pvalue)
            except ValueError:
                p_value = np.nan
            records.append(
                {
                    "variable": variable,
                    "metric": column.removesuffix("_difference_fd_p_minus_fd_only"),
                    "n_pairs": len(values),
                    "mean_difference": float(values.mean()),
                    "median_difference": float(np.median(values)),
                    "sd_difference": float(values.std(ddof=1)),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "wilcoxon_p": p_value,
                }
            )
    summary = pd.DataFrame.from_records(records)
    if not summary.empty:
        summary["fdr_q"] = false_discovery_rate(summary["wilcoxon_p"])
    return summary


def standardised_mean_difference(group_a: pd.Series, group_b: pd.Series) -> float:
    a = pd.to_numeric(group_a, errors="coerce").dropna().to_numpy()
    b = pd.to_numeric(group_b, errors="coerce").dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else np.nan


def balance_summary(metrics: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    events = select_event_rows(metrics)
    target = events.loc[events["event_type"] == "FD_P"]
    control = events.loc[events["event_type"] == "FD_only"]
    records = []
    for feature in MATCH_FEATURES:
        records.append(
            {
                "feature": feature,
                "smd_before_matching": standardised_mean_difference(target[feature], control[feature]),
                "smd_after_matching": standardised_mean_difference(
                    pairs[f"fd_p_{feature}"], pairs[f"fd_only_{feature}"]
                ),
            }
        )
    return pd.DataFrame.from_records(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=4.0)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260825)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = read_table(args.metrics)
    pairs = match_events(metrics, args.max_distance)
    if pairs.empty:
        raise RuntimeError("No matched pairs were found.")
    differences = matched_differences(metrics, pairs)
    effects = effect_summary(differences, args.bootstrap_draws, args.seed)
    balance = balance_summary(metrics, pairs)

    output = args.output_dir
    write_table(pairs, output / "matched_event_pairs.parquet")
    write_table(differences, output / "matched_event_differences.parquet")
    write_table(effects, output / "matched_effects_summary.csv")
    write_table(balance, output / "matching_balance.csv")
    print(f"Matched {len(pairs):,} FD_P events to FD_only controls.")


if __name__ == "__main__":
    main()
