from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .routing_backend import (
    BackendRoute,
    BackendSolution,
    PyVRPRoutingBackend,
)
from .common import build_route_record_at_departure


class PaperDispatchWindowRoutingBackend(PyVRPRoutingBackend):
    """Routes through the paper's isolated dispatch-window PyVRP build."""

    def __init__(
        self,
        depots: Sequence[dict[str, Any]],
        capacity: int,
        allow_late_return: bool = True,
        feasible_warm_start: bool = False,
        scenario_solver_kind: str = "default_hgs",
        final_solver_kind: str = "default_hgs",
    ) -> None:
        super().__init__(depots, capacity, allow_late_return)
        if len(self.depots) != 1:
            raise ValueError(
                "The paper dispatch-window PyVRP backend supports one depot only"
            )
        self.python = Path(
            os.environ.get(
                "ICD_PAPER_PYTHON",
                sys.executable,
            )
        )
        self.worker_script = Path(__file__).with_name("paper_backend_worker.py")
        self.feasible_warm_start = bool(feasible_warm_start)
        self.scenario_solver_kind = str(scenario_solver_kind)
        self.final_solver_kind = str(final_solver_kind)
        if self.scenario_solver_kind not in {
            "paper_short_hgs",
            "default_hgs",
            "default_hgs_diverse_init",
            "default_hgs_nn_init",
        }:
            raise ValueError(
                "scenario_solver_kind must be paper_short_hgs, default_hgs, "
                "default_hgs_diverse_init, or default_hgs_nn_init"
            )
        if self.final_solver_kind not in {
            "default_hgs",
            "default_hgs_nn_init",
            "benchmark_hgs",
            "benchmark_hgs_nn_init",
        }:
            raise ValueError(
                "final_solver_kind must be default_hgs, default_hgs_nn_init, "
                "benchmark_hgs, or benchmark_hgs_nn_init"
            )
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        atexit.register(self.close)

    def _start(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if not self.python.is_file():
            raise FileNotFoundError(
                f"Paper PyVRP interpreter does not exist: {self.python}"
            )
        self._process = subprocess.Popen(
            [str(self.python), "-u", str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        return self._process

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        try:
            if process.stdin:
                process.stdin.close()
            process.terminate()
            process.wait(timeout=2)
        except Exception:
            process.kill()

    def solve(
        self,
        orders: Sequence[dict[str, Any]],
        available_vehicles: Sequence[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
        time_limit: float,
        seed: int,
    ) -> BackendSolution:
        if not orders or not available_vehicles:
            return BackendSolution(
                routes=(),
                unserved_ids=frozenset(int(order["id"]) for order in orders),
                feasible=not orders,
                runtime_seconds=0.0,
            )
        active_depots = {int(vehicle["home_depot"]) for vehicle in available_vehicles}
        reachable = [
            dict(order)
            for order in orders
            if self._reachable(order, active_depots, current_time, vehicle_speed)
        ]
        unreachable_ids = {int(order["id"]) for order in orders} - {
            int(order["id"]) for order in reachable
        }
        if not reachable:
            return BackendSolution(
                routes=(),
                unserved_ids=frozenset(unreachable_ids),
                feasible=False,
                runtime_seconds=0.0,
            )
        payload = {
            "depots": self.depots,
            "capacity": self.capacity,
            "allow_late_return": self.allow_late_return,
            "orders": reachable,
            "available_vehicles": list(available_vehicles),
            "current_time": current_time,
            "vehicle_speed": vehicle_speed,
            "time_limit": time_limit,
            "seed": int(seed),
            "solver_kind": (
                "default"
                if all(bool(order.get("_icd_final_dispatch")) for order in reachable)
                else "scenario"
            ),
            "feasible_warm_start": self.feasible_warm_start,
            "scenario_solver_kind": self.scenario_solver_kind,
            "final_solver_kind": self.final_solver_kind,
        }
        with self._lock:
            process = self._start()
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            if not line:
                stderr = process.stderr.read() if process.stderr else ""
                raise RuntimeError(
                    f"Paper PyVRP worker exited unexpectedly: {stderr[-4000:]}"
                )
        response = json.loads(line)
        if response.get("status") != "ok":
            finite_latest = sum(
                float(order.get("_icd_dispatch_latest", float("inf")))
                < float("inf")
                for order in reachable
            )
            postponed = sum(
                float(order.get("_icd_dispatch_earliest", 0)) > current_time
                for order in reachable
            )
            sampled = sum(int(order["id"]) < 0 for order in reachable)
            if payload["solver_kind"] == "default":
                return self._fixed_departure_partial_fallback(
                    reachable,
                    available_vehicles,
                    current_time,
                    vehicle_speed,
                    unreachable_ids,
                )
            raise RuntimeError(
                "Paper PyVRP worker failed "
                f"(t={current_time}, orders={len(reachable)}, "
                f"filtered={len(unreachable_ids)}, "
                f"finite_latest={finite_latest}, postponed={postponed}, "
                f"sampled={sampled}, solver={payload['solver_kind']}): "
                f"{response.get('error', response)}; "
                f"{response.get('traceback', '')[-2000:]}"
            )
        by_id = {int(order["id"]): order for order in reachable}
        routes = tuple(
            BackendRoute(
                int(route["depot_idx"]),
                tuple(by_id[int(order_id)] for order_id in route["order_ids"]),
                float(route["start_time"]),
            )
            for route in response["routes"]
        )
        return BackendSolution(
            routes=routes,
            unserved_ids=frozenset(
                {int(value) for value in response["unserved_ids"]}
                | unreachable_ids
            ),
            feasible=bool(response["feasible"]),
            runtime_seconds=float(response["runtime_seconds"]),
        )

    def _fixed_departure_partial_fallback(
        self,
        orders: Sequence[dict[str, Any]],
        available_vehicles: Sequence[dict[str, Any]],
        departure_time: float,
        vehicle_speed: float,
        already_unserved: set[int],
    ) -> BackendSolution:
        """Builds a maximal feasible current-wave plan for finite fleets."""
        started = time.perf_counter()
        remaining = sorted(
            (dict(order) for order in orders),
            key=lambda order: (float(order["tw_end"]), int(order["id"])),
        )
        routes: list[BackendRoute] = []
        for _vehicle in available_vehicles:
            if not remaining:
                break
            route_orders: list[dict[str, Any]] = []
            load = 0
            while True:
                accepted = None
                for candidate in remaining:
                    demand = int(candidate["demand"])
                    if load + demand > self.capacity:
                        continue
                    proposed = [*route_orders, candidate]
                    record = build_route_record_at_departure(
                        self.depots[0],
                        proposed,
                        departure_time,
                        vehicle_speed,
                    )
                    if record is None:
                        continue
                    if not self.allow_late_return and record["return_time"] > 1440:
                        continue
                    accepted = candidate
                    break
                if accepted is None:
                    break
                route_orders.append(accepted)
                load += int(accepted["demand"])
                remaining.remove(accepted)
            if route_orders:
                routes.append(BackendRoute(0, tuple(route_orders), departure_time))
        unserved = already_unserved | {int(order["id"]) for order in remaining}
        return BackendSolution(
            routes=tuple(routes),
            unserved_ids=frozenset(unserved),
            feasible=not unserved,
            runtime_seconds=time.perf_counter() - started,
        )
