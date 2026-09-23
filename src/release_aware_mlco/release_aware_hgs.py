#!/usr/bin/env python3
"""Release-aware HGS oracle and ML-CO training-data exporter.

The HGS variant released with the EURO/NeurIPS dynamic VRPTW benchmark treats
the maximum customer release time on a route as that route's earliest depot
departure.  This module adapts that exact semantic to SDVRP_Bench JSON files.

For a fixed-period policy, customer release times are rounded up to the next
decision epoch before solving.  Consequently every exported route can be
dispatched at a real decision epoch, and every customer on that route is known
before the vehicle leaves the depot.  No in-route insertion is assumed.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


TIME_SCALE = 60  # benchmark minutes -> integer seconds used by official ML-CO


def ceil_to_interval(value: float, interval: int) -> int:
    if interval <= 0:
        raise ValueError("interval must be positive")
    return int(math.ceil(max(float(value), 0.0) / interval) * interval)


def euclidean(first: dict[str, Any], second: dict[str, Any]) -> float:
    return math.hypot(
        float(first["x"]) - float(second["x"]),
        float(first["y"]) - float(second["y"]),
    )


def travel_minutes(first: dict[str, Any], second: dict[str, Any], speed: float) -> int:
    if speed <= 0:
        raise ValueError("speed must be positive")
    return int(math.ceil(euclidean(first, second) / float(speed)))


@dataclass(frozen=True)
class ReleaseAwareProblem:
    name: str
    capacity: int
    interval_minutes: int
    speed: float
    horizon_seconds: int
    customer_ids: tuple[int, ...]
    coords: tuple[tuple[int, int], ...]
    demands: tuple[int, ...]
    service_seconds: tuple[int, ...]
    time_windows_seconds: tuple[tuple[int, int], ...]
    release_seconds: tuple[int, ...]
    duration_seconds: tuple[tuple[int, ...], ...]

    @property
    def size(self) -> int:
        return len(self.customer_ids)

    @property
    def id_to_local(self) -> dict[int, int]:
        return {customer_id: idx + 1 for idx, customer_id in enumerate(self.customer_ids)}


@dataclass(frozen=True)
class RouteAudit:
    customer_ids: tuple[int, ...]
    local_indices: tuple[int, ...]
    dispatch_seconds: int
    return_seconds: int
    load: int
    driving_seconds: int


@dataclass(frozen=True)
class OracleSolution:
    routes: tuple[RouteAudit, ...]
    objective_driving_seconds: int
    solver_runtime_seconds: float
    solver_seed: int
    solver_time_limit_seconds: int
    unserviceable_customer_ids: tuple[int, ...] = ()


def individually_unserviceable_customers(
    problem: ReleaseAwareProblem,
) -> tuple[int, ...]:
    """Customers impossible even on a dedicated route at their release epoch."""
    rejected: list[int] = []
    depot_close = problem.time_windows_seconds[0][1]
    for local in range(1, problem.size + 1):
        arrival = max(
            problem.release_seconds[local] + problem.duration_seconds[0][local],
            problem.time_windows_seconds[local][0],
        )
        return_time = (
            arrival
            + problem.service_seconds[local]
            + problem.duration_seconds[local][0]
        )
        if (
            problem.demands[local] > problem.capacity
            or arrival > problem.time_windows_seconds[local][1]
            or return_time > depot_close
        ):
            rejected.append(problem.customer_ids[local - 1])
    return tuple(rejected)


def _restrict_problem(
    problem: ReleaseAwareProblem,
    retained_customer_ids: set[int],
) -> ReleaseAwareProblem:
    retained_locals = [
        local
        for local in range(1, problem.size + 1)
        if problem.customer_ids[local - 1] in retained_customer_ids
    ]
    indices = [0, *retained_locals]
    return ReleaseAwareProblem(
        name=f"{problem.name}-serviceable-subset",
        capacity=problem.capacity,
        interval_minutes=problem.interval_minutes,
        speed=problem.speed,
        horizon_seconds=problem.horizon_seconds,
        customer_ids=tuple(problem.customer_ids[local - 1] for local in retained_locals),
        coords=tuple(problem.coords[local] for local in indices),
        demands=tuple(problem.demands[local] for local in indices),
        service_seconds=tuple(problem.service_seconds[local] for local in indices),
        time_windows_seconds=tuple(
            problem.time_windows_seconds[local] for local in indices
        ),
        release_seconds=tuple(problem.release_seconds[local] for local in indices),
        duration_seconds=tuple(
            tuple(problem.duration_seconds[row][column] for column in indices)
            for row in indices
        ),
    )


def problem_from_benchmark_json(
    raw_data: dict[str, Any],
    interval_minutes: int = 30,
    speed: float = 5.0,
    allow_late_return: bool = True,
    round_releases_to_interval: bool = True,
) -> ReleaseAwareProblem:
    depots = list(raw_data.get("depots") or [])
    if len(depots) != 1:
        raise ValueError("The released release-aware HGS supports exactly one depot")
    customers = sorted(raw_data.get("customers") or [], key=lambda row: int(row["id"]))
    if not customers:
        raise ValueError("Instance has no customers")
    customer_ids = tuple(int(row["id"]) for row in customers)
    if len(set(customer_ids)) != len(customer_ids) or 0 in customer_ids:
        raise ValueError("Customer IDs must be unique and non-zero")

    config = raw_data.get("generation_config") or {}
    horizon_minutes = int(config.get("horizon", 1440))
    depot_horizon_minutes = 10000 if allow_late_return else horizon_minutes
    nodes = [depots[0], *customers]
    duration = tuple(
        tuple(travel_minutes(first, second, speed) * TIME_SCALE for second in nodes)
        for first in nodes
    )
    coords = tuple(
        (int(round(float(node["x"]))), int(round(float(node["y"]))))
        for node in nodes
    )
    demands = (0, *(int(row["demand"]) for row in customers))
    services = (0, *(int(round(float(row.get("service_time", 0)) * TIME_SCALE)) for row in customers))
    time_windows = (
        (0, depot_horizon_minutes * TIME_SCALE),
        *(
            (
                int(round(float(row["tw_start"]) * TIME_SCALE)),
                int(round(float(row["tw_end"]) * TIME_SCALE)),
            )
            for row in customers
        ),
    )
    releases = (
        0,
        *(
            (
                ceil_to_interval(float(row.get("available_time", 0)), interval_minutes)
                if round_releases_to_interval
                else int(round(max(float(row.get("available_time", 0)), 0.0)))
            )
            * TIME_SCALE
            for row in customers
        ),
    )
    return ReleaseAwareProblem(
        name=str(raw_data.get("name", "sdvrp_instance")),
        capacity=int(raw_data["capacity"]),
        interval_minutes=int(interval_minutes),
        speed=float(speed),
        horizon_seconds=horizon_minutes * TIME_SCALE,
        customer_ids=customer_ids,
        coords=coords,
        demands=tuple(demands),
        service_seconds=tuple(services),
        time_windows_seconds=tuple(time_windows),
        release_seconds=tuple(releases),
        duration_seconds=duration,
    )


def write_vrplib(
    path: Path,
    problem: ReleaseAwareProblem,
    *,
    include_release_times: bool = True,
) -> None:
    """Write either an HGS input or an ML-CO static-instance file.

    The official release-aware HGS parser consumes ``RELEASE_TIME_SECTION``.
    The separately released ML-CO Python loader does not recognize that
    section: release information reaches it through the epoch observations and
    oracle route epochs instead.  Keeping the switch here prevents one file
    format from being incorrectly reused for both consumers.
    """
    lines = [
        f"NAME : {problem.name}",
        "COMMENT : SDVRP_Bench release-aware oracle",
        "TYPE : CVRP",
        f"DIMENSION : {problem.size + 1}",
        "EDGE_WEIGHT_TYPE : EXPLICIT",
        "EDGE_WEIGHT_FORMAT : FULL_MATRIX",
        f"CAPACITY : {problem.capacity}",
        "EDGE_WEIGHT_SECTION",
    ]
    lines.extend("\t".join(map(str, row)) for row in problem.duration_seconds)
    lines.append("NODE_COORD_SECTION")
    lines.extend(
        f"{idx + 1}\t{x}\t{y}" for idx, (x, y) in enumerate(problem.coords)
    )
    lines.append("DEMAND_SECTION")
    lines.extend(f"{idx + 1}\t{value}" for idx, value in enumerate(problem.demands))
    lines.extend(("DEPOT_SECTION", "1", "-1", "SERVICE_TIME_SECTION"))
    lines.extend(
        f"{idx + 1}\t{value}" for idx, value in enumerate(problem.service_seconds)
    )
    lines.append("TIME_WINDOW_SECTION")
    lines.extend(
        f"{idx + 1}\t{lower}\t{upper}"
        for idx, (lower, upper) in enumerate(problem.time_windows_seconds)
    )
    if include_release_times:
        lines.append("RELEASE_TIME_SECTION")
        lines.extend(
            f"{idx + 1}\t{value}"
            for idx, value in enumerate(problem.release_seconds)
        )
    lines.extend(("EOF", ""))
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_hgs_stdout(stdout: str) -> tuple[list[list[int]], int]:
    """Return the last complete incumbent printed by HGS."""
    current: list[list[int]] = []
    incumbents: list[tuple[list[list[int]], int]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Route"):
            _, route_text = line.split(":", 1)
            route = [int(value) for value in route_text.split()]
            current.append(route)
        elif line.startswith("Cost"):
            cost = int(round(float(line.split()[-1])))
            incumbents.append(([list(route) for route in current], cost))
            current = []
        elif "EXCEPTION" in line:
            raise RuntimeError(f"HGS exception: {line}")
    if not incumbents:
        raise RuntimeError(f"HGS returned no complete incumbent. Output tail:\n{stdout[-2000:]}")
    return incumbents[-1]


def audit_routes(
    problem: ReleaseAwareProblem,
    routes: Sequence[Sequence[int]],
) -> tuple[RouteAudit, ...]:
    """Validate capacity, route-level release, time windows, and coverage."""
    expected = set(range(1, problem.size + 1))
    flattened = [int(local) for route in routes for local in route]
    if len(flattened) != len(set(flattened)):
        raise ValueError("HGS solution contains duplicate customers")
    if set(flattened) != expected:
        missing = sorted(expected - set(flattened))
        extra = sorted(set(flattened) - expected)
        raise ValueError(f"HGS coverage mismatch: missing={missing}, extra={extra}")

    audited: list[RouteAudit] = []
    for raw_route in routes:
        route = tuple(int(local) for local in raw_route)
        if not route:
            continue
        load = sum(problem.demands[local] for local in route)
        if load > problem.capacity:
            raise ValueError(f"Route capacity violation: {load} > {problem.capacity}")
        dispatch = max(problem.release_seconds[local] for local in route)
        clock = dispatch
        previous = 0
        driving = 0
        for local in route:
            if dispatch < problem.release_seconds[local]:
                raise ValueError("Mid-route insertion detected by route-level release audit")
            leg = problem.duration_seconds[previous][local]
            driving += leg
            clock += leg
            tw_start, tw_end = problem.time_windows_seconds[local]
            clock = max(clock, tw_start)
            if clock > tw_end:
                raise ValueError(
                    f"Time-window violation for customer local={local}: {clock} > {tw_end}"
                )
            clock += problem.service_seconds[local]
            previous = local
        leg = problem.duration_seconds[previous][0]
        driving += leg
        clock += leg
        if clock > problem.time_windows_seconds[0][1]:
            raise ValueError("Depot return time-window violation")
        audited.append(
            RouteAudit(
                customer_ids=tuple(problem.customer_ids[local - 1] for local in route),
                local_indices=route,
                dispatch_seconds=dispatch,
                return_seconds=clock,
                load=load,
                driving_seconds=driving,
            )
        )
    return tuple(audited)


def solve_release_aware_hgs(
    problem: ReleaseAwareProblem,
    hgs_binary: Path,
    time_limit_seconds: int = 5,
    seed: int = 1,
    drop_individually_infeasible: bool = False,
) -> OracleSolution:
    hgs_binary = Path(hgs_binary).expanduser().resolve()
    if not hgs_binary.is_file():
        raise FileNotFoundError(hgs_binary)
    time_limit_seconds = max(int(time_limit_seconds), 1)
    unserviceable = individually_unserviceable_customers(problem)
    if unserviceable and not drop_individually_infeasible:
        raise ValueError(
            "Release-rounded instance has individually unserviceable customers: "
            f"{list(unserviceable)}. Use --drop-individually-infeasible only when "
            "training a maximal-serviceable-subset oracle."
        )
    retained = set(problem.customer_ids) - set(unserviceable)
    hgs_problem = _restrict_problem(problem, retained)
    if hgs_problem.size == 0:
        return OracleSolution(
            routes=(),
            objective_driving_seconds=0,
            solver_runtime_seconds=0.0,
            solver_seed=int(seed),
            solver_time_limit_seconds=time_limit_seconds,
            unserviceable_customer_ids=unserviceable,
        )
    with tempfile.TemporaryDirectory(prefix="release_aware_hgs_") as directory:
        instance_path = Path(directory) / "instance.vrp"
        write_vrplib(instance_path, hgs_problem, include_release_times=True)
        command = [
            str(hgs_binary),
            str(instance_path),
            str(time_limit_seconds),
            "-seed",
            str(int(seed)),
            "-veh",
            "-1",
            "-useWallClockTime",
            "1",
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=time_limit_seconds + 30,
            check=False,
        )
        runtime = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(
            f"HGS exited with {completed.returncode}: {completed.stderr[-2000:]}"
        )
    routes, printed_cost = parse_hgs_stdout(completed.stdout)
    audited_subset = audit_routes(hgs_problem, routes)
    original_local = problem.id_to_local
    audited = tuple(
        RouteAudit(
            customer_ids=route.customer_ids,
            local_indices=tuple(original_local[value] for value in route.customer_ids),
            dispatch_seconds=route.dispatch_seconds,
            return_seconds=route.return_seconds,
            load=route.load,
            driving_seconds=route.driving_seconds,
        )
        for route in audited_subset
    )
    audited_cost = sum(route.driving_seconds for route in audited)
    if audited_cost != printed_cost:
        raise ValueError(f"HGS cost mismatch: printed={printed_cost}, audited={audited_cost}")
    return OracleSolution(
        routes=audited,
        objective_driving_seconds=audited_cost,
        solver_runtime_seconds=runtime,
        solver_seed=int(seed),
        solver_time_limit_seconds=time_limit_seconds,
        unserviceable_customer_ids=unserviceable,
    )


def _dense_submatrix(matrix: Sequence[Sequence[int]], indices: Sequence[int]) -> list[list[int]]:
    return [[int(matrix[row][col]) for col in indices] for row in indices]


def build_epoch_observations(
    problem: ReleaseAwareProblem,
    solution: OracleSolution,
) -> list[tuple[int, dict[str, Any], list[list[int]]]]:
    """Replay the oracle into ML-CO-style observations and dispatch labels."""
    routes_by_dispatch: dict[int, list[list[int]]] = {}
    for route in solution.routes:
        routes_by_dispatch.setdefault(route.dispatch_seconds, []).append(list(route.customer_ids))
    last_dispatch = max(routes_by_dispatch, default=0)
    interval_seconds = problem.interval_minutes * TIME_SCALE
    last_release = max(problem.release_seconds[1:], default=0)
    replay_end = max(last_dispatch, last_release) + interval_seconds
    epochs: list[tuple[int, dict[str, Any], list[list[int]]]] = []
    already_dispatched: set[int] = set()
    epoch_key = 1
    for current in range(0, replay_end + 1, interval_seconds):
        visible_locals = [
            local
            for local in range(1, problem.size + 1)
            if problem.release_seconds[local] <= current
            and problem.customer_ids[local - 1] not in already_dispatched
        ]
        target_routes = routes_by_dispatch.get(current, [])
        if visible_locals:
            indices = [0, *visible_locals]
            request_ids = [0, *(problem.customer_ids[local - 1] for local in visible_locals)]
            relative_windows = [
                [max(problem.time_windows_seconds[local][0] - current, 0),
                 max(problem.time_windows_seconds[local][1] - current, 0)]
                for local in indices
            ]
            must_dispatch = [False]
            for local in visible_locals:
                earliest_next_arrival = (
                    current + interval_seconds + problem.duration_seconds[0][local]
                )
                must_dispatch.append(
                    earliest_next_arrival > problem.time_windows_seconds[local][1]
                )
            observation = {
                "current_epoch": epoch_key,
                "current_time": current,
                "planning_starttime": current,
                "benchmark_horizon_seconds": problem.horizon_seconds,
                "epoch_instance": {
                    "is_depot": [True, *([False] * len(visible_locals))],
                    "customer_idx": indices,
                    "request_idx": request_ids,
                    "coords": [list(problem.coords[local]) for local in indices],
                    "demands": [problem.demands[local] for local in indices],
                    "capacity": problem.capacity,
                    "time_windows": relative_windows,
                    "service_times": [problem.service_seconds[local] for local in indices],
                    "duration_matrix": _dense_submatrix(problem.duration_seconds, indices),
                    "must_dispatch": must_dispatch,
                },
            }
            target_ids = {customer_id for route in target_routes for customer_id in route}
            if not target_ids.issubset(set(request_ids)):
                raise ValueError(
                    f"Oracle dispatches unrevealed customer(s) at t={current}: "
                    f"{sorted(target_ids - set(request_ids))}"
                )
            epochs.append((epoch_key, observation, target_routes))
            epoch_key += 1
        for route in target_routes:
            already_dispatched.update(route)
    expected_dispatched = set(problem.customer_ids) - set(
        solution.unserviceable_customer_ids
    )
    if already_dispatched != expected_dispatched:
        raise ValueError(
            "Epoch replay coverage mismatch: expected serviceable customers "
            f"{sorted(expected_dispatched)}, dispatched {sorted(already_dispatched)}"
        )
    return epochs


def _safe_base_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9-]+", "-", name).strip("-")
    return cleaned or fallback


def export_official_mlco_layout(
    raw_data: dict[str, Any],
    problem: ReleaseAwareProblem,
    solution: OracleSolution,
    output_root: Path,
    instance_key: str,
) -> dict[str, Any]:
    """Export files consumed by the released ML-CO training loader."""
    output_root = Path(output_root).expanduser().resolve()
    instances_dir = output_root / "instances"
    oracle_dir = output_root / "oracle_solutions"
    instances_dir.mkdir(parents=True, exist_ok=True)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    base = _safe_base_name(instance_key, "sdvrp-instance")
    run_dir = oracle_dir / f"{base}_releaseaware"
    run_dir.mkdir(parents=True, exist_ok=True)

    # This is the static/mother instance read by ML-CO's Python feature loader.
    # Release times live in observation_*.json and route epochs, so this parser-
    # compatible copy deliberately omits RELEASE_TIME_SECTION.
    write_vrplib(
        instances_dir / f"{base}.txt",
        problem,
        include_release_times=False,
    )
    epochs = build_epoch_observations(problem, solution)
    for epoch_key, observation, _target in epochs:
        (run_dir / f"observation_{epoch_key}.json").write_text(
            json.dumps(observation, separators=(",", ":")), encoding="utf-8"
        )
    epoch_by_dispatch = {
        observation["planning_starttime"]: epoch_key
        for epoch_key, observation, _target in epochs
    }
    serialized_routes = [
        {
            "customers": list(route.customer_ids),
            "epoch": epoch_by_dispatch[route.dispatch_seconds],
        }
        for route in solution.routes
    ]
    best_solution = [
        {
            "routes": serialized_routes,
            "cost": solution.objective_driving_seconds,
            "found_after_ms": int(round(solution.solver_runtime_seconds * 1000)),
        }
    ]
    (run_dir / "best-sol.json").write_text(
        json.dumps(best_solution, indent=2), encoding="utf-8"
    )
    (run_dir / "seeds.json").write_text(
        json.dumps({"instance_seed": 1, "solver_seed": solution.solver_seed}, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "source_name": raw_data.get("name"),
        "oracle_semantics": "route_departure_at_max_rounded_customer_release",
        "no_mid_route_insertion": True,
        "release_rounding": "ceil_to_fixed_decision_interval",
        "interval_minutes": problem.interval_minutes,
        "speed": problem.speed,
        "time_scale": TIME_SCALE,
        "num_customers": problem.size,
        "num_routes": len(solution.routes),
        "num_serviceable_customers": problem.size
        - len(solution.unserviceable_customer_ids),
        "num_individually_unserviceable_customers": len(
            solution.unserviceable_customer_ids
        ),
        "individually_unserviceable_customer_ids": list(
            solution.unserviceable_customer_ids
        ),
        "oracle_coverage": (
            "all_customers"
            if not solution.unserviceable_customer_ids
            else "maximal_individually_serviceable_subset"
        ),
        "num_observations": len(epochs),
        "objective_driving_seconds": solution.objective_driving_seconds,
        "solver_runtime_seconds": solution.solver_runtime_seconds,
        "solver_time_limit_seconds": solution.solver_time_limit_seconds,
        "solver_seed": solution.solver_seed,
        "training_compatibility": {
            "official_loader": True,
            "static_instance_release_section": "omitted_for_official_python_parser",
            "static_feature_context": (
                "full realized instance; useful for pipeline reproduction but future-location "
                "leakage relative to the current benchmark information policy"
            ),
            "fleet_model": "unlimited routes as in released ML-CO oracle",
        },
        "route_audits": [asdict(route) for route in solution.routes],
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "instance_file": str(instances_dir / f"{base}.txt"),
        "oracle_run_directory": str(run_dir),
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hgs-binary", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--instance-key", default=None)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--time-limit", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--forbid-late-return", action="store_true")
    parser.add_argument(
        "--drop-individually-infeasible",
        action="store_true",
        help=(
            "Solve/export only the customers feasible on a dedicated route after "
            "release rounding; rejected IDs remain in observations and metadata."
        ),
    )
    args = parser.parse_args()

    raw_data = json.loads(args.input.expanduser().resolve().read_text(encoding="utf-8"))
    problem = problem_from_benchmark_json(
        raw_data,
        interval_minutes=args.interval,
        speed=args.speed,
        allow_late_return=not args.forbid_late_return,
    )
    solution = solve_release_aware_hgs(
        problem,
        args.hgs_binary,
        time_limit_seconds=args.time_limit,
        seed=args.seed,
        drop_individually_infeasible=args.drop_individually_infeasible,
    )
    key = args.instance_key or args.input.stem
    result = export_official_mlco_layout(
        raw_data, problem, solution, args.output_root, instance_key=key
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
