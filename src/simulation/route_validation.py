"""Environment-owned validation for solver-produced DVRP routes.

Solvers choose route order and departure time. The environment remains the
source of truth for travel-time, capacity, release-time, and time-window
feasibility, and recomputes the executable schedule before accepting a route.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Tuple


def _travel_time(first: Mapping, second: Mapping, speed: float) -> int:
    if speed <= 0:
        raise ValueError("vehicle_speed must be positive")
    distance = math.hypot(first["x"] - second["x"], first["y"] - second["y"])
    return int(math.ceil(distance / speed))


def _same_location(first: Mapping, second: Mapping) -> bool:
    return math.isclose(float(first["x"]), float(second["x"]), abs_tol=1e-9) and math.isclose(
        float(first["y"]), float(second["y"]), abs_tol=1e-9
    )


def _route_customer_ids(route_objs: Iterable[Mapping]) -> List[int]:
    return [node["id"] for node in route_objs if isinstance(node, Mapping) and "id" in node]


def validate_and_normalize_routes(
    assigned_routes: Mapping[int, Dict],
    current_batch: Iterable[Dict],
    available_vehicles: Iterable[Dict],
    depots: List[Dict],
    vehicle_capacity: float,
    current_time: float,
    vehicle_speed: float,
    allow_late_return: bool,
    simulation_end: float = 1440.0,
) -> Tuple[Dict[int, Dict], List[Dict]]:
    """Validates solver output and returns only executable routes.

    A rejected route leaves all of its customers in the environment's pending
    pool, so they may be considered again by a later replanning event.
    """

    batch_by_id = {customer["id"]: customer for customer in current_batch}
    vehicles_by_id = {vehicle["id"]: vehicle for vehicle in available_vehicles}
    accepted: Dict[int, Dict] = {}
    accepted_customer_ids = set()
    errors: List[Dict] = []

    for vehicle_id, raw_route in assigned_routes.items():
        route_objs = raw_route.get("route_objs", []) if isinstance(raw_route, Mapping) else []
        claimed_ids = _route_customer_ids(route_objs)

        def reject(reason: str, detail: str = "") -> None:
            errors.append(
                {
                    "vehicle_id": vehicle_id,
                    "reason": reason,
                    "detail": detail,
                    "customer_ids": claimed_ids,
                }
            )

        vehicle = vehicles_by_id.get(vehicle_id)
        if vehicle is None:
            reject("unknown_or_unavailable_vehicle")
            continue
        if len(route_objs) < 3:
            reject("malformed_route", "route must contain a depot, at least one customer, and a depot")
            continue

        home_depot_idx = vehicle.get("home_depot")
        if not isinstance(home_depot_idx, int) or not (0 <= home_depot_idx < len(depots)):
            reject("invalid_home_depot")
            continue
        home_depot = depots[home_depot_idx]
        if not _same_location(route_objs[0], home_depot) or not _same_location(route_objs[-1], home_depot):
            reject("wrong_route_depot")
            continue

        departure_time = float(raw_route.get("departure_time", current_time))
        if not math.isfinite(departure_time) or departure_time + 1e-9 < current_time:
            reject("departure_before_plan_time")
            continue

        route_ids: List[int] = []
        canonical_customers: List[Dict] = []
        malformed_customer = False
        for node in route_objs[1:-1]:
            if not isinstance(node, Mapping) or "id" not in node or node["id"] not in batch_by_id:
                malformed_customer = True
                break
            customer_id = node["id"]
            if customer_id in route_ids or customer_id in accepted_customer_ids:
                malformed_customer = True
                break
            route_ids.append(customer_id)
            canonical_customers.append(batch_by_id[customer_id])

        if malformed_customer:
            reject("unknown_or_duplicate_customer")
            continue
        if not route_ids:
            reject("empty_customer_route")
            continue

        if any(float(customer.get("available_time", 0.0)) > current_time + 1e-9 for customer in canonical_customers):
            reject("customer_not_released")
            continue

        total_demand = sum(float(customer.get("demand", 0.0)) for customer in canonical_customers)
        if total_demand > float(vehicle_capacity) + 1e-9:
            reject("capacity_exceeded", f"load={total_demand}, capacity={vehicle_capacity}")
            continue

        normalized_nodes = [home_depot, *canonical_customers, home_depot]
        arrival_times = [departure_time]
        clock = departure_time
        previous = normalized_nodes[0]
        invalid_reason = None
        invalid_detail = ""

        for customer in canonical_customers:
            arrival = max(
                clock + _travel_time(previous, customer, vehicle_speed),
                float(customer["tw_start"]),
            )
            if arrival > float(customer["tw_end"]) + 1e-9:
                invalid_reason = "time_window_violated"
                invalid_detail = (
                    f"customer={customer['id']}, arrival={arrival}, tw_end={customer['tw_end']}"
                )
                break
            arrival_times.append(arrival)
            clock = arrival + float(customer.get("service_time", 0.0))
            previous = customer

        if invalid_reason is not None:
            reject(invalid_reason, invalid_detail)
            continue

        return_time = clock + _travel_time(previous, home_depot, vehicle_speed)
        if not allow_late_return and return_time > simulation_end + 1e-9:
            reject("late_depot_return", f"return={return_time}, end={simulation_end}")
            continue
        arrival_times.append(return_time)

        normalized = dict(raw_route)
        normalized.update(
            {
                "route_objs": normalized_nodes,
                "arrival_times": arrival_times,
                "departure_time": departure_time,
                "return_time": return_time,
                "customer_ids": route_ids,
            }
        )
        accepted[vehicle_id] = normalized
        accepted_customer_ids.update(route_ids)

    return accepted, errors


def collect_ontime_served_orders(
    route_records: Iterable[Dict],
    customer_map: Mapping[int, Dict],
    simulation_end: float = 1440.0,
) -> set[int]:
    """Returns customers whose service actually started on time on a departed route."""

    served = set()
    for record in route_records:
        if not record.get("executed", False):
            continue
        route_path = record.get("route_path", [])
        arrival_times = record.get("times", [])
        if len(route_path) != len(arrival_times):
            continue
        departure_time = float(record.get("departure_time", math.inf))
        if not math.isfinite(departure_time) or departure_time > simulation_end + 1e-9:
            continue
        for node, arrival in zip(route_path, arrival_times):
            if not isinstance(node, Mapping) or node.get("id") not in customer_map:
                continue
            customer = customer_map[node["id"]]
            service_start = float(arrival)
            if (
                service_start + 1e-9 >= departure_time
                and service_start + 1e-9 >= float(customer.get("available_time", 0.0))
                and service_start + 1e-9 >= float(customer["tw_start"])
                and service_start <= float(customer["tw_end"]) + 1e-9
                and service_start <= simulation_end + 1e-9
            ):
                served.add(customer["id"])
    return served
