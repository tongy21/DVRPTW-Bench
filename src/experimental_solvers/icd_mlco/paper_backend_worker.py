#!/usr/bin/env python3
"""JSON-lines worker for the paper's dispatch-window PyVRP 0.6.3."""

from __future__ import annotations

import json
import math
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, List

import numpy as np
from pyvrp import (
    CostEvaluator,
    GeneticAlgorithm,
    Model,
    PenaltyManager,
    PenaltyParams,
    Population,
    PopulationParams,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.crossover import selective_route_exchange as srex
from pyvrp.diversity import broken_pairs_distance as bpd
from pyvrp.exceptions import EmptySolutionWarning
from pyvrp.search import Exchange10, LocalSearch, SwapRoutes, SwapStar, TwoOpt
from pyvrp.search import NODE_OPERATORS, ROUTE_OPERATORS, compute_neighbours as default_neighbours
from pyvrp.stop import MaxRuntime

warnings.filterwarnings("ignore", category=EmptySolutionWarning)
COORD_SCALE = 100
TIME_SCALE = 10


@dataclass
class NeighbourhoodParams:
    weight_wait_time: float = 0.2
    weight_time_warp: float = 1.0
    nb_granular: int = 40
    symmetric_proximity: bool = True
    symmetric_neighbours: bool = False


def compute_neighbours(data, params: NeighbourhoodParams = NeighbourhoodParams()):
    clients = [data.client(idx) for idx in range(data.num_clients + 1)]
    early = np.asarray([client.tw_early for client in clients])
    late = np.asarray([client.tw_late for client in clients])
    service = np.asarray([client.service_duration for client in clients])
    prize = np.asarray([client.prize for client in clients])
    release = np.asarray([client.release_time for client in clients])
    dispatch = np.asarray([client.dispatch_time for client in clients])
    duration = np.asarray(data.duration_matrix(), dtype=float)
    min_wait = early[None:] - duration - service[:, None] - late[:, None]
    earliest_release = np.maximum.outer(release, release) + duration[0, :]
    earliest_arrival = np.maximum(earliest_release, early[None, :])
    min_warp = np.maximum(
        earliest_arrival + service[None, :] + duration - late[None, :], 0
    )
    min_warp += np.maximum(np.subtract.outer(release, dispatch), 0)
    proximity = (
        np.asarray(data.distance_matrix(), dtype=float)
        + params.weight_wait_time * np.maximum(min_wait, 0)
        + params.weight_time_warp * np.maximum(min_warp, 0)
        - prize[None, :]
    )
    if params.symmetric_proximity:
        proximity = np.minimum(proximity, proximity.T)
    n = len(proximity)
    k = min(params.nb_granular, n - 2)
    np.fill_diagonal(proximity, np.inf)
    proximity[0, :] = np.inf
    proximity[:, 0] = np.inf
    top_k = np.argsort(proximity, axis=1, kind="stable")[1:, :k]
    if not params.symmetric_neighbours:
        return [[], *top_k.tolist()]
    adj = np.zeros_like(proximity, dtype=bool)
    rows = np.expand_dims(np.arange(1, n), axis=1)
    adj[rows, top_k] = True
    adj |= adj.transpose()
    return [np.flatnonzero(row).tolist() for row in adj]


def scenario_solve(
    model: Model, seed: int, time_limit: float, feasible_warm_start: bool
):
    data = model.data()
    pen_params = PenaltyParams(
        num_registrations_between_penalty_updates=10,
        penalty_increase=1.3,
        penalty_decrease=0.5,
    )
    pop_params = PopulationParams(
        min_pop_size=5, generation_size=3, nb_elite=2, nb_close=2
    )
    rng = RandomNumberGenerator(seed=seed)
    pen_manager = PenaltyManager(params=pen_params)
    pop = Population(bpd, params=pop_params)
    neighbours = compute_neighbours(data)
    ls = LocalSearch(data, rng, neighbours)
    for operator in (Exchange10, TwoOpt):
        ls.add_node_operator(operator(data))
    for operator in (SwapStar, SwapRoutes):
        ls.add_route_operator(operator(data))
    init = [Solution.make_random(data, rng) for _ in range(pop_params.min_pop_size)]
    if feasible_warm_start:
        singleton = Solution(
            data, [[index] for index in range(1, data.num_clients + 1)]
        )
        if singleton.is_feasible():
            init[0] = singleton
    algo = GeneticAlgorithm(data, pen_manager, rng, pop, ls, srex, init)
    return algo.run(MaxRuntime(time_limit))


def default_hgs_diverse_init(model: Model, seed: int, time_limit: float):
    """Runs default HGS with a route-count-diverse random population.

    PyVRP 0.6.3's make_random fixes the initial route count from the number
    of available vehicles. With one slot per request, every individual is a
    singleton solution and the route count does not recover. This keeps the
    normal default HGS search but randomises route counts across its initial
    population. No externally constructed incumbent or warm start is used.
    """
    data = model.data()
    hgs_rng = RandomNumberGenerator(seed=seed)
    local_search = LocalSearch(data, hgs_rng, default_neighbours(data))
    for operator in NODE_OPERATORS:
        local_search.add_node_operator(operator(data))
    for operator in ROUTE_OPERATORS:
        local_search.add_route_operator(operator(data))
    penalty_manager = PenaltyManager()
    population_params = PopulationParams()
    population = Population(bpd, population_params)
    num_clients = data.num_clients
    capacity = data.vehicle_type(0).capacity
    demand_bound = max(
        1,
        math.ceil(
            sum(data.client(idx).demand for idx in range(1, num_clients + 1))
            / capacity
        ),
    )
    max_routes = max(demand_bound, num_clients - 1)
    route_counts = np.rint(
        np.linspace(
            demand_bound,
            max_routes,
            population_params.min_pop_size,
        )
    ).astype(int)
    init_rng = np.random.default_rng(seed)
    initial = []
    for num_routes in route_counts:
        permutation = init_rng.permutation(
            np.arange(1, num_clients + 1, dtype=int)
        )
        routes = [
            values.tolist()
            for values in np.array_split(permutation, int(num_routes))
            if len(values)
        ]
        initial.append(Solution(data, routes))
    algorithm = GeneticAlgorithm(
        data,
        penalty_manager,
        hgs_rng,
        population,
        local_search,
        srex,
        initial,
    )
    return algorithm.run(MaxRuntime(time_limit))


def _route_is_feasible(
    route: list[int],
    orders: list[dict[str, Any]],
    depot: dict[str, Any],
    capacity: int,
    current: float,
    speed: float,
    allow_late_return: bool,
    horizon: float,
) -> bool:
    if not route:
        return False
    selected = [orders[index] for index in route]
    if sum(int(order["demand"]) for order in selected) > capacity:
        return False
    departure = max(
        current,
        max(
            float(
                order.get(
                    "_icd_dispatch_earliest",
                    order.get("available_time", 0),
                )
            )
            for order in selected
        ),
    )
    latest_departure = min(
        float(order.get("_icd_dispatch_latest", horizon))
        for order in selected
    )
    if departure > latest_departure + 1e-9:
        return False
    clock = departure
    previous = depot
    for order in selected:
        clock += math.ceil(euclidean(previous, order) / speed)
        clock = max(clock, float(order["tw_start"]))
        if clock > float(order["tw_end"]) + 1e-9:
            return False
        clock += float(order.get("service_time", 0))
        previous = order
    clock += math.ceil(euclidean(previous, depot) / speed)
    return allow_late_return or clock <= horizon + 1e-9


def nn_routes(
    orders: list[dict[str, Any]],
    depot: dict[str, Any],
    capacity: int,
    current: float,
    speed: float,
    allow_late_return: bool,
    horizon: float,
    seed: int,
    max_routes: int,
) -> list[list[int]]:
    """Builds a complete randomized NN solution using client indices.

    The route count is bounded by the vehicle type in the PyVRP model.  This
    mirrors the benchmark HGS solver, which sets ``num_available`` from the
    vehicles that are idle at the current decision epoch.
    """
    if max_routes <= 0:
        raise ValueError("max_routes must be positive")
    rng = np.random.default_rng(seed)
    unassigned = set(range(len(orders)))
    routes: list[list[int]] = []
    while unassigned:
        candidates = sorted(
            unassigned,
            key=lambda index: (
                float(orders[index]["tw_end"]),
                euclidean(depot, orders[index]),
                int(orders[index]["id"]),
            ),
        )
        seed_pool = candidates[: min(5, len(candidates))]
        first = int(seed_pool[int(rng.integers(0, len(seed_pool)))])
        route = [first]
        unassigned.remove(first)
        while unassigned:
            previous = orders[route[-1]]
            feasible = [
                index
                for index in unassigned
                if _route_is_feasible(
                    [*route, index],
                    orders,
                    depot,
                    capacity,
                    current,
                    speed,
                    allow_late_return,
                    horizon,
                )
            ]
            if not feasible:
                break
            feasible.sort(
                key=lambda index: (
                    euclidean(previous, orders[index]),
                    float(orders[index]["tw_end"]),
                    int(orders[index]["id"]),
                )
            )
            choice_pool = feasible[: min(3, len(feasible))]
            selected = int(choice_pool[int(rng.integers(0, len(choice_pool)))])
            route.append(selected)
            unassigned.remove(selected)
        routes.append([index + 1 for index in route])
        if unassigned and len(routes) >= max_routes:
            raise RuntimeError(
                "NN construction needs more routes than the available fleet"
            )
    return routes


def default_hgs_nn_init(
    model: Model,
    orders: list[dict[str, Any]],
    depot: dict[str, Any],
    capacity: int,
    current: float,
    speed: float,
    allow_late_return: bool,
    horizon: float,
    seed: int,
    time_limit: float,
):
    """Runs normal default HGS from randomized feasible NN individuals."""
    data = model.data()
    max_routes = sum(
        data.vehicle_type(index).num_available
        for index in range(data.num_vehicle_types)
    )
    hgs_rng = RandomNumberGenerator(seed=seed)
    local_search = LocalSearch(data, hgs_rng, default_neighbours(data))
    for operator in NODE_OPERATORS:
        local_search.add_node_operator(operator(data))
    for operator in ROUTE_OPERATORS:
        local_search.add_route_operator(operator(data))
    penalty_manager = PenaltyManager()
    population_params = PopulationParams()
    population = Population(bpd, population_params)
    initial = []
    construction_errors = []
    for index in range(population_params.min_pop_size):
        try:
            routes = nn_routes(
                orders,
                depot,
                capacity,
                current,
                speed,
                allow_late_return,
                horizon,
                seed + 1_000_003 * index,
                max_routes,
            )
            initial.append(Solution(data, routes))
        except RuntimeError as exc:
            construction_errors.append(str(exc))
    if len(initial) < population_params.min_pop_size:
        raise RuntimeError(
            "Could not construct a complete NN population within the "
            f"available fleet ({max_routes} vehicles): "
            f"{len(initial)}/{population_params.min_pop_size} initial solutions; "
            f"last error={construction_errors[-1] if construction_errors else 'unknown'}"
        )
    if not any(solution.is_feasible() for solution in initial):
        raise RuntimeError("NN initial population contains no feasible solution")
    algorithm = GeneticAlgorithm(
        data,
        penalty_manager,
        hgs_rng,
        population,
        local_search,
        srex,
        initial,
    )
    return algorithm.run(MaxRuntime(time_limit))


def euclidean(left: dict[str, Any], right: dict[str, Any]) -> float:
    return math.hypot(float(left["x"]) - float(right["x"]), float(left["y"]) - float(right["y"]))


def solve(payload: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    depots = payload["depots"]
    if len(depots) != 1:
        raise ValueError("Paper backend supports one depot only")
    orders = payload["orders"]
    current = float(payload["current_time"])
    speed = float(payload["vehicle_speed"])
    horizon = 10000 if payload["allow_late_return"] else 1440
    model = Model()
    depot = depots[0]
    depot_obj = model.add_depot(
        int(round(float(depot["x"]) * COORD_SCALE)),
        int(round(float(depot["y"]) * COORD_SCALE)),
        0,
        int(round(horizon * TIME_SCALE)),
    )
    clients = []
    for order in orders:
        route_release = max(
            current,
            float(order.get("_icd_dispatch_earliest", order.get("available_time", 0))),
        )
        latest = float(order.get("_icd_dispatch_latest", horizon))
        if not math.isfinite(latest):
            latest = horizon
        release = route_release
        clients.append(
            model.add_client(
                int(round(float(order["x"]) * COORD_SCALE)),
                int(round(float(order["y"]) * COORD_SCALE)),
                int(order["demand"]),
                int(round(float(order.get("service_time", 0)) * TIME_SCALE)),
                int(round(float(order["tw_start"]) * TIME_SCALE)),
                int(round(float(order["tw_end"]) * TIME_SCALE)),
                int(round(release * TIME_SCALE)),
                int(round(latest * TIME_SCALE)),
                0,
                True,
            )
        )
    model.add_vehicle_type(
        int(payload["capacity"]),
        len(payload["available_vehicles"]),
        tw_early=int(round(current * TIME_SCALE)),
        tw_late=int(round(horizon * TIME_SCALE)),
    )
    locations_obj = [depot_obj, *clients]
    locations_data = [depot, *orders]
    for frm, frm_obj in enumerate(locations_obj):
        for to, to_obj in enumerate(locations_obj):
            if frm == to:
                continue
            distance = int(math.ceil(euclidean(locations_data[frm], locations_data[to]) * COORD_SCALE))
            duration = int(math.ceil(euclidean(locations_data[frm], locations_data[to]) / speed) * TIME_SCALE)
            model.add_edge(frm_obj, to_obj, distance, duration)
    time_limit = max(float(payload["time_limit"]), 0.001)
    if payload["solver_kind"] == "scenario":
        scenario_kind = payload.get("scenario_solver_kind", "default_hgs")
        if scenario_kind == "default_hgs":
            result = model.solve(MaxRuntime(time_limit), seed=int(payload["seed"]))
        elif scenario_kind == "default_hgs_diverse_init":
            result = default_hgs_diverse_init(
                model, int(payload["seed"]), time_limit
            )
        elif scenario_kind == "default_hgs_nn_init":
            result = default_hgs_nn_init(
                model,
                orders,
                depot,
                int(payload["capacity"]),
                current,
                speed,
                bool(payload["allow_late_return"]),
                horizon,
                int(payload["seed"]),
                time_limit,
            )
        else:
            result = scenario_solve(
                model,
                int(payload["seed"]),
                time_limit,
                bool(payload.get("feasible_warm_start", False)),
            )
    elif payload.get("final_solver_kind", "default_hgs") == "default_hgs_nn_init":
        result = default_hgs_nn_init(
            model,
            orders,
            depot,
            int(payload["capacity"]),
            current,
            speed,
            bool(payload["allow_late_return"]),
            horizon,
            int(payload["seed"]),
            time_limit,
        )
    else:
        result = model.solve(MaxRuntime(time_limit), seed=int(payload["seed"]))
    if not result.best.is_feasible():
        raise RuntimeError("Paper PyVRP did not find a feasible solution")
    routes = []
    served = set()
    for route in result.best.get_routes():
        visits = [int(value) for value in route.visits()]
        order_ids = [int(orders[index - 1]["id"]) for index in visits]
        served.update(order_ids)
        routes.append(
            {
                "depot_idx": 0,
                "order_ids": order_ids,
                "start_time": float(
                    max(route.start_time(), route.release_time())
                )
                / TIME_SCALE,
            }
        )
    all_ids = {int(order["id"]) for order in orders}
    return {
        "status": "ok",
        "routes": routes,
        "unserved_ids": sorted(all_ids - served),
        "feasible": served == all_ids,
        "runtime_seconds": time.perf_counter() - started,
    }


for line in sys.stdin:
    try:
        print(json.dumps(solve(json.loads(line))), flush=True)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            ),
            flush=True,
        )
