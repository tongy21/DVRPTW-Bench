from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any


def euclidean(a: dict[str, Any], b: dict[str, Any]) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def travel_minutes(a: dict[str, Any], b: dict[str, Any], speed: float) -> int:
    if speed <= 0:
        raise ValueError("vehicle speed must be positive")
    return int(math.ceil(euclidean(a, b) / speed))


def nearest_active_depot_distance(
    order: dict[str, Any],
    depots: Sequence[dict[str, Any]],
    available_vehicles: Sequence[dict[str, Any]],
) -> float:
    active = sorted({int(v["home_depot"]) for v in available_vehicles})
    if not active:
        return math.inf
    return min(euclidean(depots[idx], order) for idx in active)


def order_can_wait_until(
    order: dict[str, Any],
    wait_until: float,
    depots: Sequence[dict[str, Any]],
    available_vehicles: Sequence[dict[str, Any]],
    speed: float,
) -> bool:
    distance = nearest_active_depot_distance(order, depots, available_vehicles)
    if not math.isfinite(distance):
        return False
    earliest_arrival = wait_until + math.ceil(distance / speed)
    return earliest_arrival <= float(order["tw_end"])


def latest_feasible_departure(
    depot: dict[str, Any],
    orders: Sequence[dict[str, Any]],
    current_time: float,
    speed: float,
) -> float | None:
    """Returns the latest feasible departure for a fixed route order."""
    if not orders:
        return float(current_time)

    latest_service = float(orders[-1]["tw_end"])
    for idx in range(len(orders) - 2, -1, -1):
        current = orders[idx]
        successor = orders[idx + 1]
        latest_service = min(
            float(current["tw_end"]),
            latest_service
            - float(current.get("service_time", 0))
            - travel_minutes(current, successor, speed),
        )

    latest_departure = latest_service - travel_minutes(depot, orders[0], speed)
    if latest_departure + 1e-9 < float(current_time):
        return None
    return max(float(current_time), latest_departure)


def build_route_record(
    depot: dict[str, Any],
    orders: Sequence[dict[str, Any]],
    current_time: float,
    speed: float,
) -> dict[str, Any] | None:
    """Replays a route using the benchmark's conservative ceil travel time."""
    departure = latest_feasible_departure(depot, orders, current_time, speed)
    if departure is None:
        return None

    route_objs: list[dict[str, Any]] = [depot, *orders, depot]
    arrival_times: list[float] = [departure]
    clock = departure
    previous = depot

    for order in orders:
        clock += travel_minutes(previous, order, speed)
        clock = max(clock, float(order["tw_start"]), float(order.get("available_time", 0)))
        if clock > float(order["tw_end"]) + 1e-9:
            return None
        arrival_times.append(clock)
        clock += float(order.get("service_time", 0))
        previous = order

    clock += travel_minutes(previous, depot, speed)
    arrival_times.append(clock)
    return {
        "route_objs": route_objs,
        "arrival_times": arrival_times,
        "departure_time": departure,
        "return_time": clock,
        "customer_ids": [int(order["id"]) for order in orders],
    }


def build_route_record_at_departure(
    depot: dict[str, Any],
    orders: Sequence[dict[str, Any]],
    departure_time: float,
    speed: float,
) -> dict[str, Any] | None:
    """Replays a route at an exact dispatch-wave departure time."""
    route_objs: list[dict[str, Any]] = [depot, *orders, depot]
    arrival_times: list[float] = [float(departure_time)]
    clock = float(departure_time)
    previous = depot
    if any(
        float(departure_time) + 1e-9 < float(order.get("available_time", 0))
        for order in orders
    ):
        return None

    for order in orders:
        clock += travel_minutes(previous, order, speed)
        clock = max(clock, float(order["tw_start"]))
        if clock > float(order["tw_end"]) + 1e-9:
            return None
        arrival_times.append(clock)
        clock += float(order.get("service_time", 0))
        previous = order

    clock += travel_minutes(previous, depot, speed)
    arrival_times.append(clock)
    return {
        "route_objs": route_objs,
        "arrival_times": arrival_times,
        "departure_time": float(departure_time),
        "return_time": clock,
        "customer_ids": [int(order["id"]) for order in orders],
    }


def route_can_wait(
    depot: dict[str, Any],
    orders: Sequence[dict[str, Any]],
    next_decision_time: float,
    speed: float,
) -> bool:
    return build_route_record(depot, orders, next_decision_time, speed) is not None


def route_can_wait_release_literal(
    depot: dict[str, Any],
    orders: Sequence[dict[str, Any]],
    next_decision_time: float,
    speed: float,
    horizon: float = 10000.0,
) -> bool:
    """Reproduces the accepted public release's can_postpone_route code.

    This intentionally preserves its apparently reversed feasibility check.
    It is a code-reproduction ablation, not the intended route-slack test.
    """
    tour: list[dict[str, Any]] = [depot, *orders, depot]
    clock = float(next_decision_time)
    for predecessor, successor in zip(tour, tour[1:]):
        clock += travel_minutes(predecessor, successor, speed)
        if successor is depot:
            earliest, latest = 0.0, float(horizon)
        else:
            earliest = float(successor["tw_start"])
            latest = float(successor["tw_end"])
        clock = max(clock, earliest)
        if clock <= latest:
            return False
        clock += float(successor.get("service_time", 0))
    return True


def unique_orders(orders: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[int, dict[str, Any]] = {}
    for order in orders:
        order_id = int(order["id"])
        if order_id >= 0:
            by_id[order_id] = dict(order)
    return list(by_id.values())
