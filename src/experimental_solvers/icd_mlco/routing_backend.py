from __future__ import annotations

import math
import random
import time
import warnings
from dataclasses import dataclass
from typing import Any, Sequence

from .common import (
    build_route_record,
    build_route_record_at_departure,
    euclidean,
    travel_minutes,
)


@dataclass(frozen=True)
class BackendRoute:
    depot_idx: int
    orders: tuple[dict[str, Any], ...]
    start_time: float | None = None

    @property
    def order_ids(self) -> tuple[int, ...]:
        return tuple(int(order["id"]) for order in self.orders)


@dataclass(frozen=True)
class BackendSolution:
    routes: tuple[BackendRoute, ...]
    unserved_ids: frozenset[int]
    feasible: bool
    runtime_seconds: float


class PyVRPRoutingBackend:
    """Small PyVRP adapter shared by ICD's scenario and dispatch solves.

    Imports are intentionally lazy.  This keeps the experimental package
    importable in lightweight analysis environments where PyVRP is absent.
    """

    coord_scale = 100
    time_scale = 10

    def __init__(
        self,
        depots: Sequence[dict[str, Any]],
        capacity: int,
        allow_late_return: bool = True,
    ) -> None:
        self.depots = [dict(depot) for depot in depots]
        self.capacity = int(capacity)
        self.allow_late_return = bool(allow_late_return)
        # Diagnostic only: counts calls for which default HGS did not return a
        # feasible incumbent and the Rolling-HGS-compatible greedy fallback
        # was used. This does not alter routing behaviour.
        self.greedy_fallback_count = 0

    @staticmethod
    def _load_pyvrp():
        try:
            from pyvrp import Model
            from pyvrp.stop import MaxRuntime
        except ImportError as exc:  # pragma: no cover - exercised remotely
            raise RuntimeError(
                "ICD requires PyVRP. Use the benchmark's svrp environment."
            ) from exc
        return Model, MaxRuntime

    def _reachable(
        self,
        order: dict[str, Any],
        active_depots: set[int],
        current_time: float,
        vehicle_speed: float,
    ) -> bool:
        if int(order.get("demand", 0)) > self.capacity:
            return False

        release = max(
            current_time,
            float(
                order.get(
                    "_icd_dispatch_earliest",
                    order.get("available_time", 0),
                )
            ),
        )
        for depot_idx in active_depots:
            depot = self.depots[depot_idx]
            arrival = max(
                release + travel_minutes(depot, order, vehicle_speed),
                float(order.get("tw_start", release)),
            )
            if arrival > float(order["tw_end"]):
                continue
            return_time = (
                arrival
                + float(order.get("service_time", 0))
                + travel_minutes(order, depot, vehicle_speed)
            )
            if self.allow_late_return or return_time <= 1440:
                return True
        return False

    def _randomised_nn_routes(
        self,
        orders: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
        max_routes: int,
        seed: int,
    ) -> list[list[int]]:
        """Builds a complete feasible NN solution using PyVRP client indices."""
        if len(self.depots) != 1:
            raise ValueError(
                "benchmark_hgs_nn_init currently supports one depot only"
            )
        if max_routes <= 0:
            raise ValueError("max_routes must be positive")
        rng = random.Random(int(seed))
        depot = self.depots[0]
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
            first = rng.choice(candidates[: min(5, len(candidates))])
            route = [first]
            unassigned.remove(first)
            while unassigned:
                previous = orders[route[-1]]
                feasible = []
                for index in unassigned:
                    proposed = [orders[value] for value in (*route, index)]
                    if sum(int(order["demand"]) for order in proposed) > self.capacity:
                        continue
                    record = build_route_record(
                        depot, proposed, current_time, vehicle_speed
                    )
                    if record is None:
                        continue
                    if not self.allow_late_return and record["return_time"] > 1440:
                        continue
                    feasible.append(index)
                if not feasible:
                    break
                feasible.sort(
                    key=lambda index: (
                        euclidean(previous, orders[index]),
                        float(orders[index]["tw_end"]),
                        int(orders[index]["id"]),
                    )
                )
                selected = rng.choice(feasible[: min(3, len(feasible))])
                route.append(selected)
                unassigned.remove(selected)
            # Single-depot models place the depot at index zero.
            routes.append([index + 1 for index in route])
            if unassigned and len(routes) >= max_routes:
                raise RuntimeError(
                    "NN construction needs more routes than the available fleet"
                )
        return routes

    def _solve_nn_initialised(
        self,
        model: Any,
        orders: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
        max_routes: int,
        time_limit: float,
        seed: int,
    ):
        """Runs PyVRP 0.11 default HGS from multi-customer NN seeds."""
        from pyvrp import (
            GeneticAlgorithm,
            PenaltyManager,
            Population,
            PopulationParams,
            RandomNumberGenerator,
            Solution,
        )
        from pyvrp.crossover import selective_route_exchange
        from pyvrp.diversity import broken_pairs_distance
        from pyvrp.search import (
            LocalSearch,
            NODE_OPERATORS,
            ROUTE_OPERATORS,
            compute_neighbours,
        )
        from pyvrp.stop import MaxRuntime

        data = model.data()
        hgs_rng = RandomNumberGenerator(seed=int(seed) & 0x7FFFFFFF)
        local_search = LocalSearch(data, hgs_rng, compute_neighbours(data))
        for operator in NODE_OPERATORS:
            local_search.add_node_operator(operator(data))
        for operator in ROUTE_OPERATORS:
            local_search.add_route_operator(operator(data))
        population_params = PopulationParams()
        initial = []
        attempt = 0
        max_attempts = 5 * population_params.min_pop_size
        while len(initial) < population_params.min_pop_size and attempt < max_attempts:
            try:
                routes = self._randomised_nn_routes(
                    orders,
                    current_time,
                    vehicle_speed,
                    max_routes,
                    seed + 1_000_003 * attempt,
                )
                solution = Solution(data, routes)
                if solution.is_feasible():
                    initial.append(solution)
            except RuntimeError:
                pass
            attempt += 1
        if len(initial) < population_params.min_pop_size:
            raise RuntimeError(
                "Could not construct a complete NN population within the "
                f"available fleet: {len(initial)}/{population_params.min_pop_size}"
            )
        algorithm = GeneticAlgorithm(
            data,
            PenaltyManager.init_from(data),
            hgs_rng,
            Population(broken_pairs_distance, population_params),
            local_search,
            selective_route_exchange,
            initial,
        )
        return algorithm.run(MaxRuntime(max(float(time_limit), 0.001)))

    def solve(
        self,
        orders: Sequence[dict[str, Any]],
        available_vehicles: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
        time_limit: float,
        seed: int,
        nn_initial_population: bool = False,
    ) -> BackendSolution:
        started = time.perf_counter()
        if not orders or not available_vehicles:
            return BackendSolution(
                routes=(),
                unserved_ids=frozenset(int(order["id"]) for order in orders),
                feasible=not orders,
                runtime_seconds=time.perf_counter() - started,
            )

        Model, MaxRuntime = self._load_pyvrp()
        active_depots = {int(vehicle["home_depot"]) for vehicle in available_vehicles}
        reachable = [
            dict(order)
            for order in orders
            if self._reachable(order, active_depots, current_time, vehicle_speed)
        ]
        unreachable_ids = {
            int(order["id"]) for order in orders
        } - {int(order["id"]) for order in reachable}
        if not reachable:
            return BackendSolution(
                routes=(),
                unserved_ids=frozenset(unreachable_ids),
                feasible=False,
                runtime_seconds=time.perf_counter() - started,
            )

        model = Model()
        horizon = 10000 if self.allow_late_return else 1440
        depot_objects = []
        for depot_idx, depot in enumerate(self.depots):
            name = f"depot_{depot_idx}"
            depot_objects.append(
                model.add_depot(
                    x=int(round(float(depot["x"]) * self.coord_scale)),
                    y=int(round(float(depot["y"]) * self.coord_scale)),
                    tw_early=int(round(current_time * self.time_scale)),
                    tw_late=int(horizon * self.time_scale),
                    name=name,
                )
            )

        client_objects = []
        order_by_name: dict[str, dict[str, Any]] = {}
        for pos, order in enumerate(reachable):
            name = f"request_{int(order['id'])}_{pos}"
            release = max(
                current_time,
                float(
                    order.get(
                        "_icd_dispatch_earliest",
                        order.get("available_time", 0),
                    )
                ),
            )
            client_objects.append(
                model.add_client(
                    x=int(round(float(order["x"]) * self.coord_scale)),
                    y=int(round(float(order["y"]) * self.coord_scale)),
                    delivery=int(order["demand"]),
                    service_duration=int(round(float(order.get("service_time", 0)) * self.time_scale)),
                    tw_early=int(round(float(order["tw_start"]) * self.time_scale)),
                    tw_late=int(round(float(order["tw_end"]) * self.time_scale)),
                    release_time=int(round(release * self.time_scale)),
                    required=True,
                    name=name,
                )
            )
            order_by_name[name] = order

        vehicles_by_depot: dict[int, list[dict[str, Any]]] = {}
        for vehicle in available_vehicles:
            vehicles_by_depot.setdefault(int(vehicle["home_depot"]), []).append(dict(vehicle))

        vehicle_type_to_depot: dict[int, int] = {}
        for depot_idx, vehicles in sorted(vehicles_by_depot.items()):
            vehicle_type_idx = len(vehicle_type_to_depot)
            model.add_vehicle_type(
                num_available=len(vehicles),
                capacity=self.capacity,
                start_depot=depot_objects[depot_idx],
                end_depot=depot_objects[depot_idx],
                tw_early=int(round(current_time * self.time_scale)),
                tw_late=int(horizon * self.time_scale),
            )
            vehicle_type_to_depot[vehicle_type_idx] = depot_idx

        locations_obj = depot_objects + client_objects
        locations_data = self.depots + reachable
        for from_idx, from_obj in enumerate(locations_obj):
            for to_idx, to_obj in enumerate(locations_obj):
                if from_idx == to_idx:
                    continue
                distance = int(math.ceil(euclidean(
                    locations_data[from_idx], locations_data[to_idx]
                ) * self.coord_scale))
                duration = travel_minutes(
                    locations_data[from_idx], locations_data[to_idx], vehicle_speed
                ) * self.time_scale
                model.add_edge(from_obj, to_obj, distance=distance, duration=int(duration))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if nn_initial_population:
                result = self._solve_nn_initialised(
                    model,
                    reachable,
                    current_time,
                    vehicle_speed,
                    len(available_vehicles),
                    time_limit,
                    seed,
                )
            else:
                result = model.solve(
                    stop=MaxRuntime(max(float(time_limit), 0.001)),
                    seed=int(seed) & 0x7FFFFFFF,
                    display=False,
                )

        if not result.best.is_feasible():
            return self._greedy_fallback(
                orders,
                available_vehicles,
                current_time,
                vehicle_speed,
                started,
            )

        routes: list[BackendRoute] = []
        served: set[int] = set()
        for route in result.best.routes():
            depot_idx = vehicle_type_to_depot[int(route.vehicle_type())]
            route_orders = []
            for location_idx in route.visits():
                location = model.locations[int(location_idx)]
                order = order_by_name[location.name]
                route_orders.append(order)
                served.add(int(order["id"]))
            if route_orders:
                routes.append(
                    BackendRoute(
                        depot_idx,
                        tuple(route_orders),
                        float(route.start_time()) / self.time_scale,
                    )
                )

        all_ids = {int(order["id"]) for order in orders}
        return BackendSolution(
            routes=tuple(routes),
            unserved_ids=frozenset((all_ids - served) | unreachable_ids),
            feasible=not (set(int(order["id"]) for order in reachable) - served),
            runtime_seconds=time.perf_counter() - started,
        )

    def _greedy_fallback(
        self,
        orders: Sequence[dict[str, Any]],
        available_vehicles: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
        started: float | None = None,
    ) -> BackendSolution:
        """Returns a deterministic feasible partial solution after an HGS miss.

        Short ICD scenario budgets may expire before PyVRP constructs a fully
        feasible mandatory solution. A partial route set is more informative
        for consensus than treating every request as postponed.
        """
        self.greedy_fallback_count += 1
        begin = time.perf_counter() if started is None else started
        remaining = sorted(
            (dict(order) for order in orders),
            key=lambda item: (float(item["tw_end"]), int(item["id"])),
        )
        routes: list[BackendRoute] = []
        vehicles = sorted(
            (dict(vehicle) for vehicle in available_vehicles),
            key=lambda item: (int(item["home_depot"]), int(item["id"])),
        )
        for vehicle in vehicles:
            if not remaining:
                break
            depot_idx = int(vehicle["home_depot"])
            depot = self.depots[depot_idx]
            route_orders: list[dict[str, Any]] = []
            load = 0
            while True:
                best: dict[str, Any] | None = None
                for candidate in remaining:
                    demand = int(candidate["demand"])
                    if load + demand > self.capacity:
                        continue
                    proposed = [*route_orders, candidate]
                    route_release = max(
                        float(current_time),
                        max(
                            float(
                                order.get(
                                    "_icd_dispatch_earliest",
                                    order.get("available_time", 0),
                                )
                            )
                            for order in proposed
                        ),
                    )
                    record = build_route_record(
                        depot, proposed, route_release, vehicle_speed
                    )
                    if record is None:
                        continue
                    if not self.allow_late_return and record["return_time"] > 1440:
                        continue
                    best = candidate
                    break
                if best is None:
                    break
                route_orders.append(best)
                load += int(best["demand"])
                remaining.remove(best)
            if route_orders:
                route_release = max(
                    float(current_time),
                    max(
                        float(
                            order.get(
                                "_icd_dispatch_earliest",
                                order.get("available_time", 0),
                            )
                        )
                        for order in route_orders
                    ),
                )
                routes.append(
                    BackendRoute(depot_idx, tuple(route_orders), route_release)
                )

        return BackendSolution(
            routes=tuple(routes),
            unserved_ids=frozenset(int(order["id"]) for order in remaining),
            feasible=not remaining,
            runtime_seconds=time.perf_counter() - begin,
        )

    def routes_to_assignments(
        self,
        routes: Sequence[BackendRoute],
        available_vehicles: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
        fixed_departure: float | None = None,
    ) -> tuple[dict[int, dict[str, Any]], set[int]]:
        """Maps backend routes onto concrete benchmark vehicles and replays time."""
        vehicles_by_depot: dict[int, list[dict[str, Any]]] = {}
        for vehicle in available_vehicles:
            vehicles_by_depot.setdefault(int(vehicle["home_depot"]), []).append(dict(vehicle))
        for vehicles in vehicles_by_depot.values():
            vehicles.sort(key=lambda item: int(item["id"]))

        assignments: dict[int, dict[str, Any]] = {}
        rejected: set[int] = set()
        counters = {depot_idx: 0 for depot_idx in vehicles_by_depot}
        for route in routes:
            depot_idx = route.depot_idx
            cursor = counters.get(depot_idx, 0)
            vehicles = vehicles_by_depot.get(depot_idx, [])
            if cursor >= len(vehicles):
                rejected.update(route.order_ids)
                continue
            if fixed_departure is None:
                record = build_route_record(
                    self.depots[depot_idx], route.orders, current_time, vehicle_speed
                )
            else:
                record = build_route_record_at_departure(
                    self.depots[depot_idx],
                    route.orders,
                    fixed_departure,
                    vehicle_speed,
                )
            if record is None:
                rejected.update(route.order_ids)
                continue
            if not self.allow_late_return and record["return_time"] > 1440:
                rejected.update(route.order_ids)
                continue
            vehicle_id = int(vehicles[cursor]["id"])
            counters[depot_idx] = cursor + 1
            assignments[vehicle_id] = record
        return assignments, rejected
