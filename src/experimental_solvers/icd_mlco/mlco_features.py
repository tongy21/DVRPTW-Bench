from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .common import travel_minutes


FEATURE_NAMES = (
    "feature_coords_x",
    "feature_coords_y",
    "feature_demands",
    "feature_serviceTime",
    "feature_timeWindowsStart",
    "feature_timeWindowsEnd",
    "timeDepotLocation",
    "relative_timeDepotLocation_windowEnd",
    "relative_time_window_start",
    "relative_time_window_end",
    "must_dispatch_feature",
    "q_0.01_time",
    "q_0.05_time",
    "q_0.1_time",
    "q_0.5_time",
    "q_0_timeWindows",
    "q_0.01_timeWindows",
    "q_0.05_timeWindows",
    "q_0.1_timeWindows",
    "q_0.5_timeWindows",
)

# The released feature builder appends the static-context features after the
# eleven online/dynamic features.  This prefix is therefore exactly the input
# expected by a model trained with ``--feature-set dynamic``.
DYNAMIC_FEATURE_NAMES = FEATURE_NAMES[:11]


def _duration_seconds(
    first: dict[str, Any],
    second: dict[str, Any],
    vehicle_speed: float,
    time_scale: float,
) -> float:
    return float(travel_minutes(first, second, vehicle_speed)) * time_scale


def duration_matrix(
    nodes: Sequence[dict[str, Any]],
    vehicle_speed: float,
    time_scale: float,
) -> np.ndarray:
    size = len(nodes)
    result = np.zeros((size, size), dtype=np.int64)
    for row in range(size):
        for col in range(size):
            if row != col:
                result[row, col] = int(
                    round(_duration_seconds(nodes[row], nodes[col], vehicle_speed, time_scale))
                )
    return result


def _quantiles(values: np.ndarray, probabilities: Sequence[float]) -> np.ndarray:
    if values.size == 0:
        return np.zeros(len(probabilities), dtype=float)
    return np.quantile(values.astype(float), q=np.asarray(probabilities))


def compute_mlco_features(
    depot: dict[str, Any],
    current_orders: Sequence[dict[str, Any]],
    reference_orders: Sequence[dict[str, Any]],
    must_dispatch_ids: set[int],
    current_time: float,
    vehicle_speed: float,
    horizon: float = 1440.0,
    time_scale: float = 60.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Computes the official NeuralNetwork ``dynamicstatic`` feature order.

    Absolute benchmark minutes are shifted to the current decision time and
    converted to seconds because the released ML-CO model was trained on
    second-valued time data. Static-distribution features use only the supplied
    reference set (normally all requests observed so far).
    """
    nodes = [dict(depot), *[dict(order) for order in current_orders]]
    references = [dict(depot), *[dict(order) for order in reference_orders]]
    if len(references) == 1:
        references.extend(dict(order) for order in current_orders)

    remaining_horizon = max((float(horizon) - float(current_time)) * time_scale, 1.0)
    reference_customers = references[1:]
    relevant_reference_customers = [
        order
        for order in reference_customers
        if float(order["tw_end"]) > float(current_time)
    ]
    average_reference_service = (
        float(np.mean([
            float(order.get("service_time", 0)) * time_scale
            for order in reference_customers
        ]))
        if reference_customers
        else 0.0
    )

    rows: list[list[float]] = []
    for node_idx, node in enumerate(nodes):
        is_depot = node_idx == 0
        if is_depot:
            tw_start = 0.0
            tw_end = remaining_horizon
            demand = 0.0
            service = 0.0
        else:
            tw_start = max(float(node["tw_start"]) - float(current_time), 0.0) * time_scale
            tw_end = max(float(node["tw_end"]) - float(current_time), 0.0) * time_scale
            demand = float(node["demand"])
            service = float(node.get("service_time", 0)) * time_scale

        depot_travel = 0.0 if is_depot else _duration_seconds(
            depot, node, vehicle_speed, time_scale
        )
        denominator = tw_end - service
        depot_ratio = depot_travel / (denominator if abs(denominator) > 1e-12 else 1.0)
        must_dispatch = (
            0.0 if is_depot else float(int(node["id"]) in must_dispatch_ids)
        )

        travel_to_reference = np.asarray(
            [
                _duration_seconds(node, reference, vehicle_speed, time_scale)
                for reference in references
            ],
            dtype=float,
        )
        travel_quantiles = _quantiles(travel_to_reference, (0.01, 0.05, 0.10, 0.50))

        compatibility_values: list[float] = []
        mean_from = float(np.mean(travel_to_reference)) if travel_to_reference.size else 0.0
        travel_from_reference = np.asarray(
            [
                _duration_seconds(reference, node, vehicle_speed, time_scale)
                for reference in references
            ],
            dtype=float,
        )
        mean_to = float(np.mean(travel_from_reference)) if travel_from_reference.size else 0.0
        for reference in relevant_reference_customers:
            ref_start = max(
                float(reference["tw_start"]) - float(current_time), 0.0
            ) * time_scale
            ref_end = max(
                float(reference["tw_end"]) - float(current_time), 0.0
            ) * time_scale
            after = max(ref_end - tw_start - service - mean_from, 0.0)
            before = max(tw_end - ref_start - average_reference_service - mean_to, 0.0)
            compatibility_values.append(max(after, before))
        compatibility_quantiles = _quantiles(
            np.asarray(compatibility_values, dtype=float),
            (0.0, 0.01, 0.05, 0.10, 0.50),
        )

        rows.append(
            [
                float(node["x"]),
                float(node["y"]),
                demand,
                service,
                tw_start,
                tw_end,
                depot_travel,
                depot_ratio,
                tw_start / remaining_horizon,
                tw_end / remaining_horizon,
                must_dispatch,
                *travel_quantiles.tolist(),
                *compatibility_quantiles.tolist(),
            ]
        )

    features = np.asarray(rows, dtype=np.float32)
    if features.shape != (len(nodes), len(FEATURE_NAMES)):
        raise RuntimeError(
            f"ML-CO feature shape mismatch: {features.shape}, expected "
            f"({len(nodes)}, {len(FEATURE_NAMES)})"
        )
    if not np.isfinite(features).all():
        raise RuntimeError("ML-CO features contain non-finite values")
    return features, duration_matrix(nodes, vehicle_speed, time_scale), nodes
