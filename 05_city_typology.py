#!/usr/bin/env python3
"""Build and evaluate the global-city typology from published analysis outputs."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Avoid Windows/Conda DLL lookup failures during NumPy linear algebra when the
# script is launched directly from PyCharm.
_DLL_HANDLE = None
if os.name == "nt":
    _conda_dll = Path(sys.prefix) / "Library" / "bin"
    if _conda_dll.is_dir():
        os.environ["PATH"] = str(_conda_dll) + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _DLL_HANDLE = os.add_dll_directory(str(_conda_dll))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd


RANDOM_SEED = 20260825
RECOVERY_CENSOR_LIMIT_DAYS = 56.0
MIN_EVI_EVENTS = 3
N_INIT = 18
MAX_ITER = 60
STABILITY_SAMPLE_FRACTION = 0.80
FEATURES = (
    "aridity_index", "fd_p_frequency", "gpp_loss",
    "evi_recovery_burden_days", "peak_thermal_excess",
    "vegetation_fraction", "built_fraction",
)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def write_csv(data: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)


def require_columns(data: pd.DataFrame, columns: set[str], table: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {table}: " + ", ".join(missing))


def first_available(data: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    for name in names:
        if name in data.columns:
            return name
    raise ValueError(f"No column available for {label}: {', '.join(names)}")


def build_city_features(metrics: pd.DataFrame, city_year: pd.DataFrame) -> pd.DataFrame:
    """Construct clustering indicators without a separate processed cluster table."""
    require_columns(
        metrics,
        {
            "event_id", "city_id", "event_type", "variable",
            "loss_magnitude_lag_0_4", "recovery_time_days_after_nadir",
            "recovery_observed", "thermal_amplification_lag_0_4",
        },
        "event metrics",
    )
    require_columns(city_year, {"city_id", "year"}, "city-year table")
    metrics = metrics.copy()
    city_year = city_year.copy()
    metrics["city_id"] = metrics["city_id"].astype(str)
    city_year["city_id"] = city_year["city_id"].astype(str)
    fd_p = metrics.loc[metrics["event_type"].eq("FD_P")].copy()

    ai_col = first_available(
        city_year, ("longterm_aridity_index", "annual_aridity_index"), "aridity index"
    )
    green_col = first_available(
        city_year, ("vegetation_fraction", "vegetation_fraction_mean"), "vegetation fraction"
    )
    built_col = first_available(
        city_year, ("built_up_fraction", "built_fraction", "built_fraction_mean"),
        "built-up fraction",
    )
    for column in (ai_col, green_col, built_col):
        city_year[column] = pd.to_numeric(city_year[column], errors="coerce")
    city_year["year"] = pd.to_numeric(city_year["year"], errors="coerce")

    aggregation = {
        "aridity_index": (ai_col, "mean"),
        "vegetation_fraction": (green_col, "mean"),
        "built_fraction": (built_col, "mean"),
        "n_years": ("year", "nunique"),
    }
    if "city_name" not in city_year.columns:
        city_year["city_name"] = city_year["city_id"]
    aggregation["city_name"] = ("city_name", "first")
    city = city_year.groupby("city_id", observed=True).agg(**aggregation).reset_index()

    event_counts = (
        fd_p[["city_id", "event_id"]].drop_duplicates()
        .groupby("city_id", observed=True).size().rename("fd_p_events").reset_index()
    )
    city = city.merge(event_counts, on="city_id", how="left")
    city["fd_p_events"] = city["fd_p_events"].fillna(0).astype(int)
    city["fd_p_frequency"] = city["fd_p_events"] / city["n_years"].replace(0, np.nan)

    gpp = fd_p.loc[fd_p["variable"].eq("gpp_primary")].copy()
    gpp["loss_magnitude_lag_0_4"] = pd.to_numeric(
        gpp["loss_magnitude_lag_0_4"], errors="coerce"
    )
    gpp_city = gpp.groupby("city_id", observed=True).agg(
        gpp_loss=("loss_magnitude_lag_0_4", "mean")
    ).reset_index()

    evi = fd_p.loc[fd_p["variable"].eq("evi_vegetation")].copy()
    evi["recovery_time_days_after_nadir"] = pd.to_numeric(
        evi["recovery_time_days_after_nadir"], errors="coerce"
    )
    evi["recovery_observed"] = evi["recovery_observed"].astype("boolean")
    evi["observed_days"] = evi["recovery_time_days_after_nadir"].where(
        evi["recovery_observed"].fillna(False)
    )
    evi_city = evi.groupby("city_id", observed=True).agg(
        evi_fd_p_events=("event_id", "nunique"),
        evi_recovery_observed_fraction=("recovery_observed", "mean"),
        evi_recovery_time_observed_days=("observed_days", "mean"),
    ).reset_index()
    fraction = evi_city["evi_recovery_observed_fraction"].clip(0, 1)
    observed_days = evi_city["evi_recovery_time_observed_days"].fillna(
        RECOVERY_CENSOR_LIMIT_DAYS
    )
    evi_city["evi_recovery_burden_days"] = (
        fraction * observed_days + (1 - fraction) * RECOVERY_CENSOR_LIMIT_DAYS
    )

    thermal = fd_p.loc[fd_p["variable"].eq("thermal_excess_day_all")].copy()
    thermal["thermal_amplification_lag_0_4"] = pd.to_numeric(
        thermal["thermal_amplification_lag_0_4"], errors="coerce"
    )
    thermal_city = thermal.groupby("city_id", observed=True).agg(
        peak_thermal_excess=("thermal_amplification_lag_0_4", "mean")
    ).reset_index()

    return city.merge(gpp_city, on="city_id", how="left").merge(
        evi_city, on="city_id", how="left"
    ).merge(thermal_city, on="city_id", how="left")


def select_cities(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete = np.isfinite(data[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)
    enough = data["evi_fd_p_events"].ge(MIN_EVI_EVENTS).fillna(False)
    included = data.loc[complete & enough].copy()
    excluded = data.loc[~(complete & enough)].copy()
    excluded["exclusion_reason"] = np.where(
        ~enough.loc[excluded.index],
        f"fewer_than_{MIN_EVI_EVENTS}_evi_fd_p_events",
        "missing_clustering_indicator",
    )
    return included, excluded


def robust_scale(data: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = np.empty((len(data), len(FEATURES)), dtype=float)
    records = []
    for index, feature in enumerate(FEATURES):
        raw = data[feature].to_numpy(dtype=float)
        lower, upper = np.quantile(raw, [0.01, 0.99])
        clipped = np.clip(raw, lower, upper)
        median = float(np.median(clipped))
        q25, q75 = np.quantile(clipped, [0.25, 0.75])
        iqr = float(q75 - q25)
        if not np.isfinite(iqr) or iqr <= 0:
            raise ValueError(f"Invalid IQR for {feature}")
        matrix[:, index] = (clipped - median) / iqr
        records.append({
            "indicator": feature, "winsor_p01": lower, "winsor_p99": upper,
            "median": median, "iqr": iqr,
        })
    return matrix, pd.DataFrame.from_records(records)


def pairwise_manhattan(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    b = a if b is None else b
    return np.abs(a[:, None, :] - b[None, :, :]).sum(axis=2)


def initialise_medoids(matrix: np.ndarray, clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    medoids = [int(rng.integers(len(matrix)))]
    while len(medoids) < clusters:
        distance = pairwise_manhattan(matrix, matrix[np.asarray(medoids)]).min(axis=1)
        distance[np.asarray(medoids)] = 0
        probability = distance**2
        candidates = np.setdiff1d(np.arange(len(matrix)), medoids)
        medoids.append(int(
            rng.choice(candidates) if probability.sum() == 0
            else rng.choice(len(matrix), p=probability / probability.sum())
        ))
    return np.asarray(medoids, dtype=int)


def fit_kmedoids(
    matrix: np.ndarray,
    clusters: int,
    seed: int,
    n_init: int = N_INIT,
    max_iter: int = MAX_ITER,
) -> tuple[np.ndarray, np.ndarray]:
    distance = pairwise_manhattan(matrix)
    master = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        medoids = initialise_medoids(
            matrix, clusters, int(master.integers(0, 2**31 - 1))
        )
        for _iteration in range(max_iter):
            labels = distance[:, medoids].argmin(axis=1)
            updated = medoids.copy()
            for cluster in range(clusters):
                members = np.flatnonzero(labels == cluster)
                if len(members):
                    costs = distance[np.ix_(members, members)].sum(axis=1)
                    updated[cluster] = members[np.argmin(costs)]
                else:
                    updated[cluster] = int(np.argmax(distance[:, medoids].min(axis=1)))
            updated = np.unique(updated)
            if len(updated) < clusters:
                candidates = np.argsort(distance[:, medoids].min(axis=1))[::-1]
                fill = [index for index in candidates if index not in set(updated)]
                updated = np.r_[updated, fill[: clusters - len(updated)]]
            updated = np.asarray(updated[:clusters], dtype=int)
            if np.array_equal(np.sort(updated), np.sort(medoids)):
                medoids = updated
                break
            medoids = updated
        labels = distance[:, medoids].argmin(axis=1)
        inertia = float(distance[np.arange(len(matrix)), medoids[labels]].sum())
        candidate = (inertia, medoids.copy(), labels.copy())
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best[1], best[2]


def silhouette_manhattan(matrix: np.ndarray, labels: np.ndarray) -> float:
    distance = pairwise_manhattan(matrix)
    values = np.zeros(len(matrix), dtype=float)
    clusters = np.unique(labels)
    for index in range(len(matrix)):
        own = labels[index]
        same = np.flatnonzero(labels == own)
        a = distance[index, same[same != index]].mean() if len(same) > 1 else 0.0
        b = min(distance[index, labels == other].mean() for other in clusters if other != own)
        values[index] = (b - a) / max(a, b) if max(a, b) else 0.0
    return float(values.mean())


def davies_bouldin_manhattan(
    matrix: np.ndarray, labels: np.ndarray, medoids: np.ndarray
) -> float:
    distance = pairwise_manhattan(matrix)
    scatter = np.zeros(len(medoids), dtype=float)
    for cluster in range(len(medoids)):
        members = np.flatnonzero(labels == cluster)
        scatter[cluster] = float(distance[members, medoids[cluster]].mean())
    ratios = []
    for first in range(len(medoids)):
        candidates = []
        for second in range(len(medoids)):
            if first == second:
                continue
            separation = float(distance[medoids[first], medoids[second]])
            candidates.append(
                (scatter[first] + scatter[second]) / max(separation, 1e-12)
            )
        ratios.append(max(candidates))
    return float(np.mean(ratios))


def adjusted_rand(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    table = pd.crosstab(labels_a, labels_b).to_numpy(dtype=np.int64)
    comb2 = lambda x: x * (x - 1) / 2
    observed = comb2(table).sum()
    row_pairs = comb2(table.sum(axis=1)).sum()
    col_pairs = comb2(table.sum(axis=0)).sum()
    total_pairs = comb2(table.sum())
    expected = row_pairs * col_pairs / total_pairs if total_pairs else 0.0
    maximum = 0.5 * (row_pairs + col_pairs)
    return float((observed - expected) / (maximum - expected)) if maximum != expected else 1.0


def stability(
    matrix: np.ndarray, reference: np.ndarray, clusters: int, runs: int, seed: int
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    scores = []
    size = min(
        len(matrix),
        max(clusters * 10, int(round(STABILITY_SAMPLE_FRACTION * len(matrix)))),
    )
    for run in range(runs):
        indices = np.sort(rng.choice(len(matrix), size=size, replace=False))
        medoids, _ = fit_kmedoids(
            matrix[indices],
            clusters,
            seed + 1000 + run,
            n_init=max(5, N_INIT // 3),
            max_iter=MAX_ITER,
        )
        predicted = pairwise_manhattan(matrix, matrix[indices][medoids]).argmin(axis=1)
        scores.append(adjusted_rand(reference, predicted))
    values = np.asarray(scores, dtype=float)
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
        "q025": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "q975": float(np.quantile(values, 0.975)),
    }


def evaluate_cluster_numbers(matrix: np.ndarray, runs: int, seed: int) -> pd.DataFrame:
    records = []
    for clusters in range(3, 7):
        medoids, labels = fit_kmedoids(matrix, clusters, seed + clusters)
        stability_result = stability(
            matrix, labels, clusters, runs, seed + clusters * 100
        )
        counts = np.bincount(labels, minlength=clusters)
        records.append({
            "n_clusters": clusters,
            "silhouette_manhattan": silhouette_manhattan(matrix, labels),
            "davies_bouldin_manhattan": davies_bouldin_manhattan(
                matrix, labels, medoids
            ),
            "stability_adjusted_rand_mean": stability_result["mean"],
            "stability_adjusted_rand_sd": stability_result["sd"],
            "stability_adjusted_rand_q025": stability_result["q025"],
            "stability_adjusted_rand_median": stability_result["median"],
            "stability_adjusted_rand_q975": stability_result["q975"],
            "minimum_cluster_size": int(counts.min()),
            "minimum_cluster_fraction": float(counts.min() / len(labels)),
        })
    result = pd.DataFrame.from_records(records)
    def minmax(column: str, higher: bool) -> pd.Series:
        values = result[column].astype(float)
        span = values.max() - values.min()
        scaled = pd.Series(np.ones(len(values)), index=values.index) if span <= 0 else (
            values - values.min()
        ) / span
        return scaled if higher else 1.0 - scaled

    result["silhouette_scaled"] = minmax("silhouette_manhattan", True)
    result["stability_scaled"] = minmax("stability_adjusted_rand_mean", True)
    result["davies_bouldin_scaled"] = minmax("davies_bouldin_manhattan", False)
    result["size_balance_scaled"] = minmax("minimum_cluster_fraction", True)
    result["selection_score"] = (
        0.45 * result["silhouette_scaled"]
        + 0.35 * result["stability_scaled"]
        + 0.15 * result["davies_bouldin_scaled"]
        + 0.05 * result["size_balance_scaled"]
    )
    result.loc[result["minimum_cluster_fraction"].lt(0.08), "selection_score"] -= 0.35
    return result


def pca_scores(matrix: np.ndarray) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    centred = matrix - matrix.mean(axis=0)
    _, singular, vectors = np.linalg.svd(centred, full_matrices=False)
    loadings = vectors[:2].T
    scores = centred @ loadings
    variance = singular**2 / max(len(matrix) - 1, 1)
    explained = variance[:2] / variance.sum()
    for component in range(2):
        anchor = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[anchor, component] < 0:
            loadings[:, component] *= -1
            scores[:, component] *= -1
    table = pd.DataFrame({
        "indicator": FEATURES,
        "PC1_loading": loadings[:, 0], "PC2_loading": loadings[:, 1],
    })
    return scores, table, explained


def descriptive_names(profile: pd.DataFrame) -> dict[int, str]:
    ids = set(profile.index.astype(int))
    if len(ids) == 3:
        green = int((
            profile["vegetation_fraction"]
            - profile["gpp_loss"]
            - profile["evi_recovery_burden_days"]
        ).idxmax())
        remaining = ids - {green}
        vulnerable = int((
            profile.loc[list(remaining), "gpp_loss"]
            + profile.loc[list(remaining), "evi_recovery_burden_days"]
        ).idxmax())
        built = int(next(iter(remaining - {vulnerable})))
        return {
            green: "Green-buffered",
            built: "Built-up intermediate",
            vulnerable: "High-loss / slow-recovery",
        }
    if len(ids) != 4:
        return {cluster: f"Cluster {cluster + 1}" for cluster in ids}
    names: dict[int, str] = {}
    dryland = int(profile["aridity_index"].idxmin())
    names[dryland] = "Dryland water-limited"
    remaining = ids - {dryland}
    built = int((profile.loc[list(remaining), "built_fraction"] + profile.loc[
        list(remaining), "peak_thermal_excess"
    ]).idxmax())
    names[built] = "High-built thermal-sensitive"
    remaining.remove(built)
    vulnerability = (
        profile.loc[list(remaining), "fd_p_frequency"]
        + profile.loc[list(remaining), "gpp_loss"]
        + profile.loc[list(remaining), "evi_recovery_burden_days"]
    )
    vulnerable = int(vulnerability.idxmax())
    names[vulnerable] = "High-whiplash vulnerable"
    remaining.remove(vulnerable)
    names[int(next(iter(remaining)))] = "Humid buffered"
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--city-year", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--clusters", type=int, default=0, help="Use 0 for automatic selection")
    parser.add_argument("--stability-runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_city_features(read_table(args.metrics), read_table(args.city_year))
    included, excluded = select_cities(data)
    if len(included) < 12:
        raise ValueError("At least 12 complete cities are required for clustering.")
    matrix, scaling = robust_scale(included)
    evaluation = evaluate_cluster_numbers(matrix, args.stability_runs, args.seed)
    clusters = args.clusters
    if clusters == 0:
        eligible = evaluation.dropna(subset=["selection_score"])
        clusters = int(eligible.loc[eligible["selection_score"].idxmax(), "n_clusters"])

    _, labels = fit_kmedoids(matrix, clusters, args.seed + clusters)
    scores, loadings, explained = pca_scores(matrix)
    scaled = pd.DataFrame(matrix, columns=FEATURES, index=included.index)
    scaled["cluster_id"] = labels
    scaled_profile = scaled.groupby("cluster_id")[list(FEATURES)].median()
    names = descriptive_names(scaled_profile)

    assignments = included.copy()
    assignments["cluster_id"] = labels
    assignments["city_type"] = assignments["cluster_id"].map(names)
    assignments["PC1"] = scores[:, 0]
    assignments["PC2"] = scores[:, 1]
    assignments["PC1_explained_variance"] = explained[0]
    assignments["PC2_explained_variance"] = explained[1]

    raw_profile = assignments.groupby(["cluster_id", "city_type"])[list(FEATURES)].median().reset_index()
    scaled_profile = scaled_profile.reset_index()
    scaled_profile["city_type"] = scaled_profile["cluster_id"].map(names)

    output = args.output_dir
    write_csv(data, output / "city_typology_input.csv")
    write_csv(assignments, output / "city_cluster_assignments.csv")
    write_csv(excluded, output / "excluded_cities.csv")
    write_csv(raw_profile, output / "cluster_profiles_raw.csv")
    write_csv(scaled_profile, output / "cluster_profiles_robust_z.csv")
    write_csv(evaluation, output / "cluster_number_evaluation.csv")
    write_csv(scaling, output / "robust_scaling_parameters.csv")
    write_csv(loadings, output / "pca_loadings.csv")
    print(f"Clustered {len(assignments):,} cities into {clusters} groups.")


if __name__ == "__main__":
    main()
