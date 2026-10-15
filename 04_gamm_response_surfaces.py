#!/usr/bin/env python3
"""Fit the manuscript GAMMs and export response surfaces and effects.

The analysis uses one median observation per city and event type (at least two
source events), joint FD-only/FD-to-pluvial Gaussian GAMMs, event-specific
smooths, a city random intercept and conditional posterior intervals. EVI
recovery time is analysed among observed recoveries on a log1p scale.
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path


# Make direct Windows/Conda and PyCharm execution robust to DLL lookup issues.
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
from scipy.spatial import cKDTree

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", message="Multi-processing hsmm computations")
    from mssm.models import GAMM
    from mssm.src.python.exp_fam import Gaussian
    from mssm.src.python.formula import Formula, lhs
    from mssm.src.python.terms import f, i, l, ri


EVENT_ORDER = ["FD_only", "FD_P"]
EVENT_CODEBOOK = {"event_type": {"FD_only": 0, "FD_P": 1}}
RANDOM_SEED = 20260817
MIN_EVENTS = 2
POSTERIOR_DRAWS = 2000
SUPPORT_K = 30
SUPPORT_REFERENCE_QUANTILE = 0.95
SURFACE_N = 35
CURVE_N = 91

OUTCOMES = {
    "gpp_loss": {
        "variable": "gpp_primary",
        "response": "loss_magnitude_lag_0_4",
        "transform": "identity",
        "plane": ("ai", "green"),
    },
    "evi_recovery_time": {
        "variable": "evi_vegetation",
        "response": "recovery_time_days_after_nadir",
        "transform": "log1p",
        "plane": ("ai", "green"),
    },
    "peak_thermal_excess": {
        "variable": "thermal_excess_day_all",
        "response": "thermal_amplification_lag_0_4",
        "transform": "identity",
        "plane": ("vpd", "built"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--posterior-draws", type=int, default=POSTERIOR_DRAWS)
    parser.add_argument("--support-k", type=int, default=SUPPORT_K)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False, dtype={"city_id": "string", "event_id": "string"})


def eligible_rows(data: pd.DataFrame) -> pd.DataFrame:
    require_columns(data, {"event_response_eligible"}, "event metrics")
    flag = data["event_response_eligible"].map({True: True, False: False, "True": True, "False": False, "true": True, "false": False, "1": True, "0": False})
    if (data["event_response_eligible"].notna() & flag.isna()).any():
        raise ValueError("event_response_eligible must be a Boolean flag.")
    return data.loc[flag.fillna(False)].copy()


def require_columns(data: pd.DataFrame, columns: set[str], table: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"Missing columns in {table}: " + ", ".join(missing))


def validate_source(source: pd.DataFrame) -> str:
    built = "built_up_fraction" if "built_up_fraction" in source else "built_fraction"
    required = {
        "event_id", "city_id", "event_type", "variable",
        "longterm_aridity_index", "vegetation_fraction", "vpd_mean_drought", built,
        "event_response_eligible",
    }
    required.update(spec["response"] for spec in OUTCOMES.values())
    require_columns(source, required, "event metrics")
    unknown = sorted(set(source["event_type"].dropna().astype(str)) - set(EVENT_ORDER))
    if unknown:
        raise ValueError(f"Unknown event types: {unknown}")
    return built


def prepare_city_data(source: pd.DataFrame, key: str, built_col: str) -> pd.DataFrame:
    spec = OUTCOMES[key]
    columns = [
        spec["response"], "longterm_aridity_index", "vegetation_fraction",
        "vpd_mean_drought", built_col,
    ]
    if key == "evi_recovery_time":
        require_columns(source, {"recovery_observed"}, "event metrics")
    data = eligible_rows(source.loc[source["variable"].eq(spec["variable"])])
    if key == "evi_recovery_time":
        observed = data["recovery_observed"].map({True: True, False: False, "True": True, "False": False, "true": True, "false": False})
        data = data.loc[observed.fillna(False)].copy()
    data["city_id"] = data["city_id"].astype(str)
    data["event_type"] = pd.Categorical(
        data["event_type"], categories=EVENT_ORDER, ordered=True
    )
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[~np.isfinite(data[column]), column] = np.nan
    data = data.dropna(subset=columns + ["city_id", "event_type"])
    if data.duplicated("event_id").any():
        raise ValueError(f"Duplicate event metrics for {key}")
    if key == "evi_recovery_time" and data[spec["response"]].lt(0).any():
        raise ValueError("Observed recovery time must be non-negative; do not silently clip inputs.")

    group = ["city_id", "event_type"]
    counts = data.groupby(group, observed=True).size().rename("n_events")
    city = data.groupby(group, observed=True, as_index=False)[columns].median()
    city = city.merge(counts.reset_index(), on=group, how="inner")
    city = city.loc[city["n_events"].ge(MIN_EVENTS)].copy()
    city = city.rename(columns={
        spec["response"]: "response",
        "longterm_aridity_index": "ai",
        "vegetation_fraction": "green",
        "vpd_mean_drought": "vpd",
        built_col: "built",
    })
    city["event_type"] = pd.Categorical(
        city["event_type"], categories=EVENT_ORDER, ordered=True
    )
    city["response_model"] = (
        np.log1p(city["response"].to_numpy(float))
        if spec["transform"] == "log1p"
        else city["response"].to_numpy(float)
    )
    if city.empty or city["event_type"].nunique() < 2:
        raise ValueError(f"Insufficient complete data for {key}")
    return city


def reference_table(source: pd.DataFrame, built_col: str) -> pd.DataFrame:
    columns = [
        "longterm_aridity_index", "vegetation_fraction", "vpd_mean_drought", built_col
    ]
    data = eligible_rows(source)[["city_id", "event_type", *columns]].copy()
    data["city_id"] = data["city_id"].astype(str)
    data["event_type"] = pd.Categorical(
        data["event_type"], categories=EVENT_ORDER, ordered=True
    )
    for column in columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[~np.isfinite(data[column]), column] = np.nan
    data = data.dropna(subset=columns + ["city_id", "event_type"])
    data = data.groupby(["city_id", "event_type"], observed=True, as_index=False)[
        columns
    ].median()
    return data.rename(columns={
        "longterm_aridity_index": "ai", "vegetation_fraction": "green",
        "vpd_mean_drought": "vpd", built_col: "built",
    })


def reference_and_ranges(data: pd.DataFrame) -> tuple[dict, dict]:
    keys = ["ai", "green", "vpd", "built"]
    reference = {key: float(data[key].median()) for key in keys}
    ranges = {
        key: (float(data[key].quantile(0.05)), float(data[key].quantile(0.95)))
        for key in keys
    }
    return reference, ranges


def fit_gamm(city: pd.DataFrame, key: str):
    x_name, y_name = OUTCOMES[key]["plane"]
    terms = [
        i(),
        l(["event_type"]),
        f(["ai"], by="event_type", nk=8, penalize_null=True),
        f(["green"], by="event_type", nk=8, penalize_null=True),
        f([x_name, y_name], by="event_type", nk=[6, 6], te=False, penalize_null=True),
        f(["vpd"], by="event_type", nk=8, penalize_null=True),
        f(["built"], by="event_type", nk=8, penalize_null=True),
        ri("city_id"),
    ]
    formula = Formula(
        lhs("response_model"), terms, data=city,
        codebook=EVENT_CODEBOOK, print_warn=False,
    )
    model = GAMM(formula, Gaussian())
    model.fit(
        method="Chol", progress_bar=False, n_cores=1, max_outer=500,
        conv_tol=1e-7, check_cond=1,
    )
    return model, list(range(len(terms) - 1))


def backtransform(key: str, eta):
    eta = np.asarray(eta, dtype=float)
    if OUTCOMES[key]["transform"] == "log1p":
        return np.maximum(np.expm1(eta), 0.0)
    return eta


def prediction_data(ai, green, vpd, built, event_type, city_token: str) -> pd.DataFrame:
    arrays = np.broadcast_arrays(ai, green, vpd, built, event_type)
    return pd.DataFrame({
        "ai": np.asarray(arrays[0]).ravel(),
        "green": np.asarray(arrays[1]).ravel(),
        "vpd": np.asarray(arrays[2]).ravel(),
        "built": np.asarray(arrays[3]).ravel(),
        "event_type": pd.Categorical(
            np.asarray(arrays[4]).ravel(), categories=EVENT_ORDER, ordered=True
        ),
        "city_id": [city_token] * arrays[0].size,
    })


def predict_matrix(model, fixed_terms, new_data: pd.DataFrame):
    _, matrix, _ = model.predict(fixed_terms, new_data, ci=False)
    return np.asarray(matrix @ model.coef).reshape(-1), matrix


def robust_xy(points: np.ndarray, grid: np.ndarray):
    median = np.nanmedian(points, axis=0)
    scale = np.nanquantile(points, 0.75, axis=0) - np.nanquantile(points, 0.25, axis=0)
    scale[~np.isfinite(scale) | (scale <= 0)] = 1.0
    return (points - median) / scale, (grid - median) / scale


def common_knn_support(
    city: pd.DataFrame, x_name: str, y_name: str,
    x_values: np.ndarray, y_values: np.ndarray, requested_k: int,
) -> tuple[np.ndarray, list[dict]]:
    xx, yy = np.meshgrid(x_values, y_values)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    masks, diagnostics = [], []
    for event_type in EVENT_ORDER:
        observed = city.loc[
            city["event_type"].astype(str).eq(event_type), [x_name, y_name]
        ].to_numpy(float)
        observed = observed[np.isfinite(observed).all(axis=1)]
        if len(observed) < 5:
            masks.append(np.zeros(len(grid), dtype=bool))
            diagnostics.append({
                "event_type": event_type, "n": len(observed), "k": 0,
                "distance_limit": np.nan,
            })
            continue
        observed_z, grid_z = robust_xy(observed, grid)
        tree = cKDTree(observed_z)
        k = min(max(5, int(requested_k)), len(observed) - 1)
        observed_distance = tree.query(observed_z, k=k + 1)[0]
        observed_k = observed_distance[:, -1]
        limit = float(np.nanquantile(observed_k, SUPPORT_REFERENCE_QUANTILE))
        grid_distance = tree.query(grid_z, k=k)[0]
        grid_k = grid_distance[:, -1] if grid_distance.ndim == 2 else grid_distance
        masks.append(grid_k <= limit)
        diagnostics.append({
            "event_type": event_type, "n": len(observed), "k": k,
            "distance_limit": limit,
        })
    return np.logical_and.reduce(masks).reshape(len(y_values), len(x_values)), diagnostics


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = read_table(args.metrics)
    built_col = validate_source(source)
    city_data = {
        key: prepare_city_data(source, key, built_col) for key in OUTCOMES
    }
    pooled = reference_table(source, built_col)
    reference, ranges = reference_and_ranges(pooled)

    model_rows, surface_rows, effect_rows = [], [], []
    difference_rows, difference_summary, support_rows = [], [], []

    for outcome_index, key in enumerate(OUTCOMES, start=1):
        city = city_data[key]
        print(f"Fitting {key}: {len(city)} city/event-type rows, {int(city['n_events'].sum())} source events", flush=True)
        model, fixed_terms = fit_gamm(city, key)
        info = model.info
        model_rows.append({
            "outcome": key,
            "model": "Gaussian GAMM (mssm)",
            "event_type_structure": "joint factor-specific smooths",
            "random_effect": "city_id random intercept",
            "bivariate_smooth": " x ".join(OUTCOMES[key]["plane"]),
            "minimum_valid_events_per_city_type": MIN_EVENTS,
            "n_city_event_rows": len(city),
            "n_cities": city["city_id"].nunique(),
            "n_source_events": int(city["n_events"].sum()),
            "n_coefficients": len(model.coef),
            "effective_df": float(model.edf),
            "residual_variance": float(model.get_pars()[1]),
            "convergence_code": int(info.code),
            "outer_iterations": int(info.iter),
            "lambda_updates": int(info.lambda_updates),
            "response_scale": (
                "log1p days; conditional on observed threshold attainment"
                if key == "evi_recovery_time" else "identity"
            ),
        })
        posterior = model.sample_post(
            int(max(500, args.posterior_draws)), use_post=None,
            deviations=False, seed=args.seed + outcome_index,
        )
        city_token = str(city["city_id"].iloc[0])
        x_name, y_name = OUTCOMES[key]["plane"]
        x_values = np.linspace(*ranges[x_name], SURFACE_N)
        y_values = np.linspace(*ranges[y_name], SURFACE_N)
        xx, yy = np.meshgrid(x_values, y_values)
        support, diagnostics = common_knn_support(
            city, x_name, y_name, x_values, y_values, args.support_k
        )
        for diagnostic in diagnostics:
            support_rows.append({
                "outcome": key, **diagnostic,
                "common_supported_fraction": float(support.mean()),
                "rule": "common 95th-percentile kNN support in robust-scaled 2D space",
            })

        predictions, matrices = {}, {}
        for event_type in EVENT_ORDER:
            values = {
                "ai": xx if x_name == "ai" else np.full_like(xx, reference["ai"]),
                "green": yy if y_name == "green" else np.full_like(xx, reference["green"]),
                "vpd": xx if x_name == "vpd" else np.full_like(xx, reference["vpd"]),
                "built": yy if y_name == "built" else np.full_like(xx, reference["built"]),
            }
            new = prediction_data(
                values["ai"], values["green"], values["vpd"], values["built"],
                np.full_like(xx, event_type, dtype=object), city_token,
            )
            eta, matrix = predict_matrix(model, fixed_terms, new)
            prediction = backtransform(key, eta).reshape(SURFACE_N, SURFACE_N)
            predictions[event_type], matrices[event_type] = prediction, matrix
            for row, y_value in enumerate(y_values):
                for column, x_value in enumerate(x_values):
                    surface_rows.append({
                        "outcome": key, "event_type": event_type,
                        "x_variable": x_name, "y_variable": y_name,
                        "x": float(x_value), "y": float(y_value),
                        "prediction": float(prediction[row, column]),
                        "supported": bool(support[row, column]),
                    })

        draw_fd = backtransform(key, matrices["FD_only"] @ posterior)
        draw_transition = backtransform(key, matrices["FD_P"] @ posterior)
        contrast_draws = draw_transition - draw_fd
        contrast = predictions["FD_P"] - predictions["FD_only"]
        low = np.quantile(contrast_draws, 0.025, axis=1).reshape(SURFACE_N, SURFACE_N)
        high = np.quantile(contrast_draws, 0.975, axis=1).reshape(SURFACE_N, SURFACE_N)
        significant = ((low > 0) | (high < 0)) & support
        difference_summary.append({
            "outcome": key,
            "supported_cells": int(support.sum()),
            "significant_difference_cells": int(significant.sum()),
            "significant_fraction_of_supported": float(
                significant.sum() / max(support.sum(), 1)
            ),
            "contrast": "FD_P minus FD_only",
        })
        for row, y_value in enumerate(y_values):
            for column, x_value in enumerate(x_values):
                difference_rows.append({
                    "outcome": key, "x_variable": x_name, "y_variable": y_name,
                    "x": float(x_value), "y": float(y_value),
                    "difference_FD_P_minus_FD_only": float(contrast[row, column]),
                    "ci95_low": float(low[row, column]),
                    "ci95_high": float(high[row, column]),
                    "supported": bool(support[row, column]),
                    "ci_excludes_zero": bool(significant[row, column]),
                })

        climate_name = "vpd" if key == "peak_thermal_excess" else "ai"
        climate_values = np.linspace(*ranges[climate_name], CURVE_N)
        increment_half = 0.05
        for event_type in EVENT_ORDER:
            count = len(climate_values)
            plus = {name: np.full(count, reference[name]) for name in reference}
            minus = {name: values.copy() for name, values in plus.items()}
            plus[climate_name] = climate_values.copy()
            minus[climate_name] = climate_values.copy()
            urban_name = "built" if key == "peak_thermal_excess" else "green"
            # Exactly 10 percentage points, including reference values near
            # zero or one; record the actual evaluation location explicitly.
            urban_midpoint = float(np.clip(reference[urban_name], increment_half, 1.0 - increment_half))
            plus[urban_name][:] = urban_midpoint + increment_half
            minus[urban_name][:] = urban_midpoint - increment_half
            plus_data = prediction_data(
                plus["ai"], plus["green"], plus["vpd"], plus["built"],
                np.full(count, event_type, dtype=object), city_token,
            )
            minus_data = prediction_data(
                minus["ai"], minus["green"], minus["vpd"], minus["built"],
                np.full(count, event_type, dtype=object), city_token,
            )
            eta_plus, matrix_plus = predict_matrix(model, fixed_terms, plus_data)
            eta_minus, matrix_minus = predict_matrix(model, fixed_terms, minus_data)
            effect = backtransform(key, eta_plus) - backtransform(key, eta_minus)
            effect_draws = (
                backtransform(key, matrix_plus @ posterior)
                - backtransform(key, matrix_minus @ posterior)
            )
            effect_low = np.quantile(effect_draws, 0.025, axis=1)
            effect_high = np.quantile(effect_draws, 0.975, axis=1)
            for climate, estimate, ci_low, ci_high in zip(
                climate_values, effect, effect_low, effect_high
            ):
                effect_rows.append({
                    "outcome": key, "event_type": event_type,
                    "climate_variable": climate_name,
                    "urban_variable": urban_name,
                    "climate_value": float(climate),
                    "urban_increment": float(
                        plus[urban_name][0] - minus[urban_name][0]
                    ),
                    "urban_midpoint": urban_midpoint,
                    "marginal_effect": float(estimate),
                    "ci95_low": float(ci_low), "ci95_high": float(ci_high),
                })

        del model, posterior, matrices, draw_fd, draw_transition, contrast_draws
        gc.collect()

    if any(row["convergence_code"] != 0 for row in model_rows):
        raise RuntimeError("At least one GAMM did not converge; inspect the model summary.")
    write_csv(model_rows, args.output_dir / "gamm_model_summary.csv")
    write_csv(surface_rows, args.output_dir / "gamm_response_surfaces.csv")
    write_csv(effect_rows, args.output_dir / "gamm_marginal_effects.csv")
    write_csv(difference_rows, args.output_dir / "gamm_event_type_differences.csv")
    write_csv(
        difference_summary,
        args.output_dir / "gamm_event_type_difference_summary.csv",
    )
    write_csv(support_rows, args.output_dir / "gamm_support_diagnostics.csv")
    write_csv([reference], args.output_dir / "gamm_pooled_reference_values.csv")


if __name__ == "__main__":
    main()
