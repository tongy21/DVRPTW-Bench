from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class PCHGSSolution:
    routes: tuple[tuple[int, ...], ...]
    selected_ids: frozenset[int]
    cost: float
    runtime_seconds: float
    raw_summary: dict[str, Any]


class PCHGSBackend:
    """JSON/stdin adapter for the official profit-collecting HGS release."""

    def __init__(self, binary: Path) -> None:
        self.binary = Path(binary)

    def validate(self) -> None:
        if not self.binary.is_file():
            raise FileNotFoundError(
                f"PC-HGS executable not found: {self.binary}. Build the vendored "
                "source with CMake before running ML-CO."
            )
        if not os.access(self.binary, os.X_OK):
            raise PermissionError(f"PC-HGS executable is not executable: {self.binary}")

    @staticmethod
    def format_instance(
        nodes: Sequence[dict[str, Any]],
        capacity: int,
        durations: np.ndarray,
        profits: np.ndarray,
        current_time: float,
        horizon: float,
        time_scale: float,
    ) -> dict[str, Any]:
        if len(nodes) != len(profits) or durations.shape != (len(nodes), len(nodes)):
            raise ValueError("PC-HGS node, duration, and profit dimensions disagree")

        json_nodes = []
        for idx, node in enumerate(nodes):
            if idx == 0:
                request_idx = 0
                demand = 0
                service = 0
                tw_start = 0
                tw_end = int(round(max(horizon - current_time, 1.0) * time_scale))
            else:
                request_idx = int(node["id"])
                demand = int(node["demand"])
                service = int(round(float(node.get("service_time", 0)) * time_scale))
                tw_start = int(round(max(float(node["tw_start"]) - current_time, 0.0) * time_scale))
                tw_end = int(round(max(float(node["tw_end"]) - current_time, 0.0) * time_scale))
            json_nodes.append(
                {
                    "coords": [int(round(float(node["x"]))), int(round(float(node["y"])))],
                    "service_time": service,
                    "demand": demand,
                    "time_window": [tw_start, tw_end],
                    "release_time": 0,
                    "profit": int(profits[idx]),
                    "request_idx": request_idx,
                }
            )
        matrix = durations.astype(np.int64).tolist()
        return {
            "capacity": int(capacity),
            "nodes": json_nodes,
            "durations": matrix,
            "cost": matrix,
        }

    def solve(
        self,
        instance: dict[str, Any],
        num_vehicles: int,
        time_limit: float,
        seed: int,
        must_dispatch_ids: Sequence[int] = (),
        allow_fractional_seconds: bool = False,
    ) -> PCHGSSolution:
        self.validate()
        if num_vehicles <= 0:
            return PCHGSSolution((), frozenset(), 0.0, 0.0, {})

        if allow_fractional_seconds:
            nominal_seconds: float | int = float(time_limit)
            if nominal_seconds <= 0:
                raise ValueError("Fractional PC-HGS time limit must be positive")
        else:
            # The unmodified released PC-HGS command line accepts integral seconds only.
            nominal_seconds = max(int(math.ceil(float(time_limit))), 1)
        # The official executable accepts JSON files but does not read an
        # instance from stdin. Temporary files keep that interface isolated
        # from the benchmark dataset and result directories.
        attempts: list[dict[str, Any]] = []
        total_runtime = 0.0
        completed = None
        stdout = stderr = solution_text = ""
        # PC-HGS very rarely exits successfully without writing output under a
        # one-second limit. Retry only this empty-output case with deterministic
        # alternate seeds; the per-attempt time budget stays unchanged.
        for attempt in range(3):
            attempt_seed = (int(seed) + attempt * 1_000_003) & 0x7FFFFFFF
            with tempfile.TemporaryDirectory(prefix="sdvrp_pchgs_") as directory:
                input_path = Path(directory) / "instance.json"
                output_path = Path(directory) / "solution.json"
                seed_path = Path(directory) / "feasible_empty_seed.json"
                input_path.write_text(
                    json.dumps(instance, separators=(",", ":")), encoding="utf-8"
                )
                # The upstream parser augments this empty feasible seed with every
                # certainly-profitable (must-dispatch) customer as singleton routes.
                seed_path.write_text('{"routes":[]}', encoding="utf-8")
                command = [
                    str(self.binary), str(input_path), "-t", str(nominal_seconds),
                    "-assumeJSONInput", "1", "-seed", str(attempt_seed),
                    "-veh", str(int(num_vehicles)), "-quiet", "1",
                    "-useWallClockTime", "1", "-outputJSONPath", str(output_path),
                    "-seedSolutions", str(seed_path),
                ]
                started = time.perf_counter()
                completed = subprocess.run(
                    command, text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=float(nominal_seconds) + 30, check=False,
                )
                runtime = time.perf_counter() - started
                total_runtime += runtime
                stdout = completed.stdout.strip(); stderr = completed.stderr.strip()
                solution_text = (
                    output_path.read_text(encoding="utf-8").strip()
                    if output_path.is_file() else stdout
                )
            attempts.append({
                "attempt": attempt + 1, "seed": attempt_seed,
                "runtime_seconds": runtime, "returncode": completed.returncode,
                "empty_output": not bool(solution_text),
            })
            if completed.returncode != 0 or stdout.startswith("EXCEPTION") or solution_text:
                break
        assert completed is not None
        if completed.returncode != 0:
            raise RuntimeError(
                f"PC-HGS exited with code {completed.returncode}: "
                f"{stderr[-1000:]}"
            )
        if stdout.startswith("EXCEPTION"):
            raise RuntimeError(f"PC-HGS failed: {stdout}")
        fallback_ids = tuple(int(value) for value in must_dispatch_ids)
        if not solution_text:
            if len(fallback_ids) > num_vehicles:
                raise RuntimeError(
                    "PC-HGS returned no solution and the mandatory singleton "
                    f"fallback needs {len(fallback_ids)} vehicles, but only "
                    f"{num_vehicles} are available"
                )
            # The same singleton incumbent is supplied to PC-HGS. If all
            # deterministic attempts terminate without output, return that
            # known-feasible incumbent rather than aborting the dynamic run.
            # Optional requests stay pending and may be reconsidered next wave.
            return PCHGSSolution(
                routes=tuple((order_id,) for order_id in fallback_ids),
                selected_ids=frozenset(fallback_ids),
                cost=0.0,
                runtime_seconds=total_runtime,
                raw_summary={
                    "cost": None, "prize": None,
                    "num_routes": len(fallback_ids),
                    "nominal_seconds": nominal_seconds,
                    "feasible_empty_seed_warm_start": True,
                    "empty_output_fallback": "mandatory_singletons",
                    "attempts": attempts,
                },
            )
        try:
            payload = json.loads(solution_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PC-HGS did not return JSON. "
                f"stdout tail: {stdout[-1000:]}; stderr tail: {stderr[-1000:]}"
            ) from exc

        known_ids = {
            int(node["request_idx"])
            for node in instance["nodes"]
            if int(node["request_idx"]) != 0
        }
        routes: list[tuple[int, ...]] = []
        selected: set[int] = set()
        for route_payload in payload.get("routes", []):
            route = tuple(int(value) for value in route_payload.get("requests", []))
            if not route:
                continue
            unknown = set(route) - known_ids
            if unknown:
                raise RuntimeError(f"PC-HGS returned unknown request IDs: {sorted(unknown)}")
            duplicate = selected & set(route)
            if duplicate:
                raise RuntimeError(f"PC-HGS returned duplicate request IDs: {sorted(duplicate)}")
            selected.update(route)
            routes.append(route)

        return PCHGSSolution(
            routes=tuple(routes),
            selected_ids=frozenset(selected),
            cost=float(payload.get("cost", 0.0)),
            runtime_seconds=total_runtime,
            raw_summary={
                "cost": payload.get("cost"),
                "prize": payload.get("prize"),
                "num_routes": len(routes),
                "nominal_seconds": nominal_seconds,
                "feasible_empty_seed_warm_start": True,
                "attempts": attempts,
            },
        )
