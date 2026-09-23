from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from solvers.base_solver import BaseSolver
from solvers.hgs_solver import HGSSolver

from .common import (
    build_route_record_at_departure,
    order_can_wait_until,
    route_can_wait,
    route_can_wait_release_literal,
    unique_orders,
)
from .config import ICDConfig
from .paper_routing_backend import PaperDispatchWindowRoutingBackend
from .routing_backend import BackendRoute, PyVRPRoutingBackend
from .scenario_sampler import OnlineScenarioSampler


class ICDSolver(BaseSolver):
    """Iterative Conditional Dispatch with fixed double-threshold consensus.

    The policy structure and default hyperparameters follow the public ICD
    implementation.  Scenario generation is adapted to this benchmark's
    temporal/spatial generator and never reads unrevealed customer rows.
    """

    runtime_config = ICDConfig()

    def __init__(
        self,
        depots: list[dict[str, Any]],
        capacity: int,
        time_limit: int = 2,
        allow_late_return: bool = True,
    ) -> None:
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = int(capacity)
        self.config = self.__class__.runtime_config
        if self.config.routing_backend == "paper_dispatch_window":
            self.backend = PaperDispatchWindowRoutingBackend(
                depots,
                capacity,
                allow_late_return,
                feasible_warm_start=self.config.paper_feasible_warm_start,
                scenario_solver_kind=self.config.scenario_solver_kind,
                final_solver_kind=self.config.final_solver_kind,
            )
        else:
            self.backend = PyVRPRoutingBackend(depots, capacity, allow_late_return)
        self.benchmark_final_solver = (
            HGSSolver(
                depots,
                capacity,
                time_limit=time_limit,
                allow_late_return=allow_late_return,
            )
            if self.config.final_solver_kind == "benchmark_hgs"
            else None
        )
        self.benchmark_nn_final_backend = (
            PyVRPRoutingBackend(depots, capacity, allow_late_return)
            if self.config.final_solver_kind == "benchmark_hgs_nn_init"
            else None
        )
        self.sampler = OnlineScenarioSampler(
            self.config.max_sampled_requests,
            knowledge_mode=self.config.sampler_knowledge,
        )
        self.diagnostics_history: list[dict[str, Any]] = []
        self._call_index = 0
        self._postponement_streaks: dict[int, int] = {}

    def configure_instance(
        self,
        raw_data: dict[str, Any],
        data_file: str | None = None,
    ) -> None:
        self.sampler.configure_instance(raw_data)
        self.data_file = data_file

    def _is_forced(
        self,
        order: dict[str, Any],
        available_vehicles: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
    ) -> bool:
        next_time = current_time + self.config.decision_interval
        return not order_can_wait_until(
            order,
            next_time,
            self.depots,
            available_vehicles,
            vehicle_speed,
        )

    @staticmethod
    def _route_distance(
        depot: dict[str, Any], orders: Sequence[dict[str, Any]]
    ) -> float:
        if not orders:
            return 0.0
        tour = [depot, *orders, depot]
        return sum(
            float(
                np.hypot(
                    float(left["x"]) - float(right["x"]),
                    float(left["y"]) - float(right["y"]),
                )
            )
            for left, right in zip(tour, tour[1:])
        )

    @classmethod
    def _forecast_merge_savings_ratio(
        cls, route: BackendRoute, depot: dict[str, Any]
    ) -> float | None:
        current = [order for order in route.orders if int(order["id"]) >= 0]
        future = [order for order in route.orders if int(order["id"]) < 0]
        if not current or not future:
            return None
        separate = cls._route_distance(depot, current) + cls._route_distance(
            depot, future
        )
        if separate <= 1e-12:
            return 0.0
        merged = cls._route_distance(depot, route.orders)
        return max(0.0, min(1.0, (separate - merged) / separate))

    @staticmethod
    def _route_is_dispatched(
        route: BackendRoute,
        to_dispatch: set[int],
        to_postpone: set[int],
        next_decision_time: float,
        depots: Sequence[dict[str, Any]],
        vehicle_speed: float,
        route_wait_policy: str = "legacy_latest_feasible",
        forecast_merge_min_savings_ratio: float = 0.10,
    ) -> bool:
        current_ids = {order_id for order_id in route.order_ids if order_id >= 0}
        if current_ids & to_dispatch:
            return True
        has_sampled = any(order_id < 0 for order_id in route.order_ids)
        if current_ids & to_postpone:
            return False
        current_orders = [
            order for order in route.orders if int(order["id"]) >= 0
        ]
        if route_wait_policy in {
            "forecast_merge_value",
            "forecast_merge_next_wave_value",
        }:
            wait_orders = (
                list(route.orders)
                if route_wait_policy == "forecast_merge_next_wave_value"
                else current_orders
            )
            route_survives_next_wave = build_route_record_at_departure(
                depots[route.depot_idx],
                wait_orders,
                next_decision_time,
                vehicle_speed,
            ) is not None
            if not route_survives_next_wave or not has_sampled:
                return True
            savings_ratio = ICDSolver._forecast_merge_savings_ratio(
                route, depots[route.depot_idx]
            )
            return (
                savings_ratio is None
                or savings_ratio + 1e-12 < forecast_merge_min_savings_ratio
            )
        if has_sampled:
            return False
        if route_wait_policy == "release_literal":
            return not route_can_wait_release_literal(
                depots[route.depot_idx],
                current_orders,
                next_decision_time,
                vehicle_speed,
            )
        if route_wait_policy == "paper_intended":
            return build_route_record_at_departure(
                depots[route.depot_idx],
                current_orders,
                next_decision_time,
                vehicle_speed,
            ) is None
        return not route_can_wait(
            depots[route.depot_idx],
            current_orders,
            next_decision_time,
            vehicle_speed,
        )

    def _scenario_orders(
        self,
        current_orders: Sequence[dict[str, Any]],
        sampled_orders: Sequence[dict[str, Any]],
        to_dispatch: set[int],
        to_postpone: set[int],
        current_time: float,
        next_decision_time: float,
    ) -> list[dict[str, Any]]:
        scenario: list[dict[str, Any]] = []
        for original in current_orders:
            order = dict(original)
            order_id = int(order["id"])
            release = float(order.get("available_time", 0))
            if order_id in to_postpone:
                dispatch_earliest = max(release, next_decision_time)
                dispatch_latest = float("inf")
            elif order_id in to_dispatch:
                dispatch_earliest = release
                dispatch_latest = current_time
            else:
                dispatch_earliest = release
                dispatch_latest = float("inf")
            order["_icd_dispatch_earliest"] = dispatch_earliest
            order["_icd_dispatch_latest"] = dispatch_latest
            scenario.append(order)
        for sampled in sampled_orders:
            order = dict(sampled)
            release = float(order.get("available_time", 0))
            order["_icd_dispatch_earliest"] = release
            order["_icd_dispatch_latest"] = float("inf")
            scenario.append(order)
        if any(
            float(candidate.get("available_time", 0))
            != float(original.get("available_time", 0))
            for original, candidate in zip(current_orders, scenario)
        ):
            raise AssertionError("ICD scenario construction mutated release times")
        return scenario

    def _routing_seed(
        self,
        call_index: int,
        phase: int,
        iteration: int = 0,
        scenario_index: int = 0,
    ) -> int:
        """Return a backend seed independent of sampler RNG consumption.

        Knowledge modes draw different numbers of random variates while
        sampling future requests.  Routing seeds must not therefore come from
        that same stream, otherwise a knowledge ablation also changes PyVRP's
        stochastic search trajectory.
        """
        sequence = np.random.SeedSequence(
            [
                int(self.config.seed),
                int(call_index),
                int(phase),
                int(iteration),
                int(scenario_index),
            ]
        )
        return int(sequence.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)

    def solve_batch(
        self,
        batch_orders: list[dict[str, Any]],
        available_vehicles: list[dict[str, Any]],
        current_time: int,
        vehicle_speed: float,
    ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        started = time.perf_counter()
        current_orders = unique_orders(batch_orders)
        if not current_orders or not available_vehicles:
            return {}, current_orders

        self.sampler.observe(current_orders)
        call_index = self._call_index
        call_seed = self.config.seed + 1_000_003 * call_index
        self._call_index += 1
        sampler_rng = np.random.default_rng(
            np.random.SeedSequence([int(call_seed), 0x53414D50])
        )
        current_by_id = {int(order["id"]): order for order in current_orders}
        all_current_ids = set(current_by_id)
        next_decision_time = float(current_time + self.config.decision_interval)
        max_streak_before_decision = max(
            (
                self._postponement_streaks.get(order_id, 0)
                for order_id in all_current_ids
            ),
            default=0,
        )

        urgency_forced = {
            order_id
            for order_id, order in current_by_id.items()
            if self._is_forced(
                order, available_vehicles, current_time, vehicle_speed
            )
        }
        age_forced = (
            {
                order_id
                for order_id in all_current_ids
                if self._postponement_streaks.get(order_id, 0)
                >= self.config.max_consecutive_postponements
            }
            if self.config.max_consecutive_postponements is not None
            else set()
        )
        to_dispatch = urgency_forced | age_forced
        to_postpone: set[int] = set()

        if self.config.scenario_time_limit is not None:
            scenario_time = float(self.config.scenario_time_limit)
            budget_policy = "explicit_scenario_and_dispatch_limits"
        else:
            strategy_budget = max(
                float(self.time_limit) * self.config.strategy_time_fraction, 0.001
            )
            scenario_time = max(
                strategy_budget
                / (self.config.num_iterations * self.config.num_scenarios),
                self.config.min_scenario_time,
            )
            budget_policy = "fraction_of_total_with_minimums"
        sample_sizes: list[int] = []
        scenario_runtimes: list[float] = []
        scenario_greedy_fallbacks_before = int(
            getattr(self.backend, "greedy_fallback_count", 0)
        )
        scenario_vehicle_counts: list[int] = []
        forecast_merge_savings_ratios: list[float] = []
        forecast_mixed_next_wave_feasible = 0
        forecast_mixed_next_wave_infeasible = 0
        iteration_summaries: list[dict[str, Any]] = []

        for iteration in range(self.config.num_iterations):
            dispatch_counts = {order_id: 0 for order_id in all_current_ids}
            for scenario_index in range(self.config.num_scenarios):
                sampled = self.sampler.sample(
                    sampler_rng,
                    current_time,
                    self.config.decision_interval * self.config.num_lookahead,
                )
                sample_sizes.append(len(sampled))
                scenario_orders = self._scenario_orders(
                    current_orders,
                    sampled,
                    to_dispatch,
                    to_postpone,
                    current_time,
                    next_decision_time,
                )
                scenario_seed = self._routing_seed(
                    call_index,
                    phase=0,
                    iteration=iteration,
                    scenario_index=scenario_index,
                )
                scenario_vehicles = available_vehicles
                if self.config.scenario_fleet_policy == "unlimited":
                    depot_idx = int(available_vehicles[0]["home_depot"])
                    scenario_vehicles = [
                        {"id": idx, "home_depot": depot_idx}
                        for idx in range(len(scenario_orders))
                    ]
                elif self.config.scenario_fleet_policy == "scaled":
                    depot_idx = int(available_vehicles[0]["home_depot"])
                    demand_bound = int(
                        np.ceil(
                            sum(int(order["demand"]) for order in scenario_orders)
                            / self.capacity
                        )
                    )
                    size_bound = int(
                        np.ceil(
                            len(scenario_orders)
                            / self.config.scenario_initial_customers_per_route
                        )
                    )
                    num_scenario_vehicles = min(
                        len(scenario_orders), max(demand_bound, size_bound, 1)
                    )
                    scenario_vehicles = [
                        {"id": idx, "home_depot": depot_idx}
                        for idx in range(num_scenario_vehicles)
                    ]
                scenario_vehicle_counts.append(len(scenario_vehicles))
                solution = self.backend.solve(
                    scenario_orders,
                    scenario_vehicles,
                    current_time,
                    vehicle_speed,
                    scenario_time,
                    scenario_seed,
                )
                scenario_runtimes.append(solution.runtime_seconds)
                for route in solution.routes:
                    savings_ratio = self._forecast_merge_savings_ratio(
                        route, self.depots[route.depot_idx]
                    )
                    if savings_ratio is not None:
                        forecast_merge_savings_ratios.append(savings_ratio)
                        full_route_waitable = build_route_record_at_departure(
                            self.depots[route.depot_idx],
                            route.orders,
                            next_decision_time,
                            vehicle_speed,
                        ) is not None
                        if full_route_waitable:
                            forecast_mixed_next_wave_feasible += 1
                        else:
                            forecast_mixed_next_wave_infeasible += 1
                    if self._route_is_dispatched(
                        route,
                        to_dispatch,
                        to_postpone,
                        next_decision_time,
                        self.depots,
                        vehicle_speed,
                        self.config.route_wait_policy,
                        self.config.forecast_merge_min_savings_ratio,
                    ):
                        for order_id in route.order_ids:
                            if order_id in dispatch_counts:
                                dispatch_counts[order_id] += 1

            old_dispatch = set(to_dispatch)
            old_postpone = set(to_postpone)
            undecided_before = sorted(all_current_ids - old_dispatch - old_postpone)
            frequency_counts = {str(count): 0 for count in range(self.config.num_scenarios + 1)}
            for order_id in undecided_before:
                frequency_counts[str(dispatch_counts[order_id])] += 1
            for order_id in all_current_ids - old_dispatch - old_postpone:
                frequency = dispatch_counts[order_id] / self.config.num_scenarios
                if frequency >= self.config.dispatch_threshold:
                    to_dispatch.add(order_id)
                elif (
                    (1.0 - frequency) > self.config.postpone_threshold
                    if self.config.postpone_boundary == "paper_strict"
                    else (1.0 - frequency) >= self.config.postpone_threshold
                ):
                    to_postpone.add(order_id)
            iteration_summaries.append(
                {
                    "iteration": iteration + 1,
                    "dispatch_count": len(to_dispatch),
                    "postpone_count": len(to_postpone),
                    "undecided_count": len(all_current_ids - to_dispatch - to_postpone),
                    "dispatch_frequency_denominator": self.config.num_scenarios,
                    "dispatch_frequency_count_histogram": frequency_counts,
                }
            )
            if to_dispatch | to_postpone == all_current_ids:
                break

        if self.config.dispatch_choice == "not_postpone":
            selected_ids = all_current_ids - to_postpone
        else:
            selected_ids = set(to_dispatch)
        selected_orders = [
            current_by_id[order_id] for order_id in sorted(selected_ids)
        ]
        routing_orders = selected_orders
        if self.config.routing_backend == "paper_dispatch_window":
            routing_orders = []
            for original in selected_orders:
                order = dict(original)
                order["_icd_dispatch_earliest"] = float(
                    order.get("available_time", 0)
                )
                order["_icd_dispatch_latest"] = float(current_time)
                order["_icd_final_dispatch"] = True
                routing_orders.append(order)

        assignments: dict[int, dict[str, Any]] = {}
        backend_unserved: set[int] = set()
        final_runtime = 0.0
        final_greedy_fallbacks_before = (
            int(getattr(self.benchmark_final_solver, "greedy_fallback_count", 0))
            if self.benchmark_final_solver is not None
            else (
                int(getattr(self.benchmark_nn_final_backend, "greedy_fallback_count", 0))
                if self.benchmark_nn_final_backend is not None
                else int(getattr(self.backend, "greedy_fallback_count", 0))
            )
        )
        if routing_orders:
            if self.config.dispatch_time_limit is not None:
                final_time_limit = float(self.config.dispatch_time_limit)
            else:
                elapsed = time.perf_counter() - started
                final_time_limit = max(
                    float(self.time_limit) - elapsed, self.config.min_final_time
                )
            if self.benchmark_nn_final_backend is not None:
                final_solution = self.benchmark_nn_final_backend.solve(
                    selected_orders,
                    available_vehicles,
                    current_time,
                    vehicle_speed,
                    final_time_limit,
                    self._routing_seed(call_index, phase=1),
                    nn_initial_population=True,
                )
                final_runtime = final_solution.runtime_seconds
                assignments, rejected = (
                    self.benchmark_nn_final_backend.routes_to_assignments(
                        final_solution.routes,
                        available_vehicles,
                        current_time,
                        vehicle_speed,
                    )
                )
                backend_unserved = set(final_solution.unserved_ids) | rejected
            elif self.benchmark_final_solver is not None:
                self.benchmark_final_solver.time_limit = final_time_limit
                final_started = time.perf_counter()
                assignments, dropped_final = self.benchmark_final_solver.solve_batch(
                    selected_orders,
                    available_vehicles,
                    current_time,
                    vehicle_speed,
                )
                final_runtime = time.perf_counter() - final_started
                backend_unserved = {
                    int(order["id"]) for order in dropped_final
                }
            else:
                final_solution = self.backend.solve(
                    routing_orders,
                    available_vehicles,
                    current_time,
                    vehicle_speed,
                    final_time_limit,
                    self._routing_seed(call_index, phase=1),
                )
                final_runtime = final_solution.runtime_seconds
                assignments, rejected = self.backend.routes_to_assignments(
                    final_solution.routes,
                    available_vehicles,
                    current_time,
                    vehicle_speed,
                    fixed_departure=(
                        float(current_time)
                        if self.config.dispatch_at_wave_start
                        else None
                    ),
                )
                backend_unserved = set(final_solution.unserved_ids) | rejected

        assigned_ids = {
            int(order_id)
            for route in assignments.values()
            for order_id in route.get("customer_ids", [])
        }
        for order_id in assigned_ids:
            self._postponement_streaks.pop(order_id, None)
        for order_id in all_current_ids - assigned_ids:
            self._postponement_streaks[order_id] = (
                self._postponement_streaks.get(order_id, 0) + 1
            )
        dropped = [
            order for order in current_orders if int(order["id"]) not in assigned_ids
        ]
        self.diagnostics_history.append(
            {
                "time": int(current_time),
                "batch_size": len(current_orders),
                "forced_count": len(
                    urgency_forced | age_forced
                ),
                "urgency_forced_count": len(urgency_forced),
                "postponement_cap_forced_count": len(age_forced),
                "max_postponement_streak_before_decision": max(
                    max_streak_before_decision, 0
                ),
                "selected_count": len(selected_ids),
                "assigned_count": len(assigned_ids),
                "backend_unserved_ids": sorted(backend_unserved),
                "iterations": iteration_summaries,
                "scenario_count": len(scenario_runtimes),
                "scenario_greedy_fallback_count": (
                    int(getattr(self.backend, "greedy_fallback_count", 0))
                    - scenario_greedy_fallbacks_before
                ),
                "mean_sampled_requests": (
                    float(np.mean(sample_sizes)) if sample_sizes else 0.0
                ),
                "mean_scenario_runtime_seconds": (
                    float(np.mean(scenario_runtimes)) if scenario_runtimes else 0.0
                ),
                "forecast_mixed_route_count": len(
                    forecast_merge_savings_ratios
                ),
                "forecast_mixed_next_wave_feasible_count": (
                    forecast_mixed_next_wave_feasible
                ),
                "forecast_mixed_next_wave_infeasible_count": (
                    forecast_mixed_next_wave_infeasible
                ),
                "forecast_merge_savings_ratio_mean": (
                    float(np.mean(forecast_merge_savings_ratios))
                    if forecast_merge_savings_ratios
                    else None
                ),
                "forecast_merge_savings_ratio_median": (
                    float(np.median(forecast_merge_savings_ratios))
                    if forecast_merge_savings_ratios
                    else None
                ),
                "available_vehicle_count": len(available_vehicles),
                "scenario_vehicle_count_min": (
                    min(scenario_vehicle_counts) if scenario_vehicle_counts else 0
                ),
                "scenario_vehicle_count_mean": (
                    float(np.mean(scenario_vehicle_counts))
                    if scenario_vehicle_counts
                    else 0.0
                ),
                "scenario_vehicle_count_max": (
                    max(scenario_vehicle_counts) if scenario_vehicle_counts else 0
                ),
                "final_runtime_seconds": final_runtime,
                "final_greedy_fallback_count": (
                    (
                        int(getattr(self.benchmark_final_solver, "greedy_fallback_count", 0))
                        if self.benchmark_final_solver is not None
                        else (
                            int(
                                getattr(
                                    self.benchmark_nn_final_backend,
                                    "greedy_fallback_count",
                                    0,
                                )
                            )
                            if self.benchmark_nn_final_backend is not None
                            else int(getattr(self.backend, "greedy_fallback_count", 0))
                        )
                    )
                    - final_greedy_fallbacks_before
                ),
                "scenario_time_limit_seconds": scenario_time,
                "dispatch_time_limit_seconds": (
                    float(self.config.dispatch_time_limit)
                    if self.config.dispatch_time_limit is not None
                    else None
                ),
                "budget_policy": budget_policy,
                "wall_runtime_seconds": time.perf_counter() - started,
                "routing_seed_policy": "separate_from_sampler_rng",
                "release_time_policy": "immutable_original_release",
                "routing_backend": self.config.routing_backend,
                "route_wait_policy": self.config.route_wait_policy,
                "forecast_merge_min_savings_ratio": (
                    self.config.forecast_merge_min_savings_ratio
                ),
                "max_consecutive_postponements": (
                    self.config.max_consecutive_postponements
                ),
                "scenario_fleet_policy": self.config.scenario_fleet_policy,
                "scenario_initial_customers_per_route": (
                    self.config.scenario_initial_customers_per_route
                ),
                "scenario_solver_kind": self.config.scenario_solver_kind,
                "final_solver_kind": self.config.final_solver_kind,
                "final_departure_policy": (
                    "benchmark_latest_feasible"
                    if (
                        self.benchmark_final_solver is not None
                        or self.benchmark_nn_final_backend is not None
                    )
                    else (
                        "fixed_wave_start"
                        if self.config.dispatch_at_wave_start
                        else "backend_latest_feasible"
                    )
                ),
                "paper_feasible_warm_start": (
                    self.config.paper_feasible_warm_start
                ),
                "postpone_boundary": self.config.postpone_boundary,
                "sampler": self.sampler.public_diagnostics(),
            }
        )
        return assignments, dropped

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "solver": "ICD-double-threshold",
            "config": asdict(self.config),
            "scenario_knowledge": self.config.sampler_knowledge,
            "decision_rounds": self.diagnostics_history,
        }
