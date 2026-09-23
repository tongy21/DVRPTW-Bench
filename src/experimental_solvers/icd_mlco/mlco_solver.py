from __future__ import annotations

import time
import hashlib
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from solvers.base_solver import BaseSolver

from .common import order_can_wait_until, unique_orders
from .config import MLCOConfig
from .mlco_features import DYNAMIC_FEATURE_NAMES, FEATURE_NAMES, compute_mlco_features
from .mlco_predictor import NumpyMLCOPredictor
from .pc_hgs_backend import PCHGSBackend
from .routing_backend import BackendRoute, PyVRPRoutingBackend


class MLCOSolver(BaseSolver):
    """Released ML-CO neural profit predictor coupled to official PC-HGS."""

    runtime_config = MLCOConfig()

    def __init__(
        self,
        depots: list[dict[str, Any]],
        capacity: int,
        time_limit: int = 2,
        allow_late_return: bool = True,
    ) -> None:
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        if len(depots) != 1:
            raise ValueError(
                "The released PC-HGS model supports one depot; this ML-CO adapter "
                "therefore rejects multi-depot instances explicitly."
            )
        self.capacity = int(capacity)
        self.config = self.__class__.runtime_config
        self.package_dir = Path(__file__).resolve().parent
        self.model_path = self.config.resolved_model_path(self.package_dir)
        self.weights_path = self.config.resolved_weights_path(self.package_dir)
        self.pchgs = PCHGSBackend(
            self.config.resolved_pchgs_binary(self.package_dir)
        )
        self.route_adapter = PyVRPRoutingBackend(depots, capacity, allow_late_return)
        self._predictor: NumpyMLCOPredictor | None = None
        self._inference_backend: str | None = None
        self._model_source_sha256: str | None = None
        self._observed: dict[int, dict[str, Any]] = {}
        self._horizon = 1440.0
        self._call_index = 0
        self.data_file: str | None = None
        self.instance_name: str | None = None
        self.diagnostics_history: list[dict[str, Any]] = []

    def configure_instance(
        self,
        raw_data: dict[str, Any],
        data_file: str | None = None,
    ) -> None:
        config = raw_data.get("generation_config") or {}
        self._horizon = float(config.get("horizon", 1440))
        self.instance_name = str(raw_data.get("name", "unknown"))
        self.data_file = data_file
        self._observed.clear()

    def _load_predictor(self) -> NumpyMLCOPredictor:
        if self._predictor is not None:
            return self._predictor
        if not self.weights_path.is_file():
            raise FileNotFoundError(
                f"Exported official ML-CO weights not found: {self.weights_path}. "
                "Run .venv_mlco/bin/python scripts/export_mlco_weights.py once."
            )
        self._predictor = NumpyMLCOPredictor(self.weights_path)
        saved_model_file = self.model_path / "saved_model.pb"
        if saved_model_file.is_file():
            digest = hashlib.sha256()
            with saved_model_file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            self._model_source_sha256 = digest.hexdigest()
            if (
                self._predictor.source_saved_model_sha256 != "unknown"
                and self._predictor.source_saved_model_sha256
                != self._model_source_sha256
            ):
                raise RuntimeError(
                    "ML-CO exported weights do not match the configured SavedModel: "
                    f"weights={self._predictor.source_saved_model_sha256}, "
                    f"SavedModel={self._model_source_sha256}"
                )
        self._inference_backend = "numpy-cpu"
        return self._predictor

    def _predict_profits(self, features: np.ndarray) -> np.ndarray:
        predictor = self._load_predictor()
        if predictor.input_dimension == len(FEATURE_NAMES):
            model_features = features
        elif predictor.input_dimension == len(DYNAMIC_FEATURE_NAMES):
            model_features = features[:, : len(DYNAMIC_FEATURE_NAMES)]
        else:  # guarded by NumpyMLCOPredictor, kept defensive for subclasses
            raise RuntimeError(
                f"Unsupported ML-CO model input dimension {predictor.input_dimension}"
            )
        prediction = predictor.predict(model_features)
        profits = np.asarray(prediction, dtype=float).reshape(-1)
        if profits.shape != (len(features),):
            raise RuntimeError(
                f"ML-CO model returned shape {profits.shape} for {len(features)} nodes"
            )
        if not np.isfinite(profits).all():
            raise RuntimeError("ML-CO model returned non-finite profits")
        profits[0] = 0.0
        return profits

    def _must_dispatch_ids(
        self,
        orders: list[dict[str, Any]],
        available_vehicles: list[dict[str, Any]],
        current_time: float,
        vehicle_speed: float,
    ) -> set[int]:
        next_time = current_time + self.config.decision_interval
        if next_time >= self._horizon:
            return {int(order["id"]) for order in orders}
        return {
            int(order["id"])
            for order in orders
            if not order_can_wait_until(
                order,
                next_time,
                self.depots,
                available_vehicles,
                vehicle_speed,
            )
        }

    def solve_batch(
        self,
        batch_orders: list[dict[str, Any]],
        available_vehicles: list[dict[str, Any]],
        current_time: int,
        vehicle_speed: float,
    ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
        started = time.perf_counter()
        orders = unique_orders(batch_orders)
        if not orders or not available_vehicles:
            return {}, orders
        if any(int(vehicle["home_depot"]) != 0 for vehicle in available_vehicles):
            raise ValueError("ML-CO/PC-HGS adapter only supports vehicles based at depot 0")

        for order in orders:
            self._observed[int(order["id"])] = dict(order)
        # A strict fixed-period trigger may first reconsider an order after its
        # latest feasible depot departure, even though tw_end itself has not
        # passed yet. Such an order cannot occur in the official hourly
        # environment, which filters requests for reachability at each epoch.
        # Do not turn it into an impossible must-dispatch customer: one forced
        # infeasible customer makes the complete PC-HGS instance infeasible and
        # prevents otherwise feasible orders from being selected.
        dispatchable_orders = [
            order
            for order in orders
            if order_can_wait_until(
                order,
                current_time,
                self.depots,
                available_vehicles,
                vehicle_speed,
            )
        ]
        currently_unreachable_ids = sorted(
            {int(order["id"]) for order in orders}
            - {int(order["id"]) for order in dispatchable_orders}
        )
        if not dispatchable_orders:
            self.diagnostics_history.append(
                {
                    "time": int(current_time),
                    "batch_size": len(orders),
                    "dispatchable_count": 0,
                    "currently_unreachable_ids": currently_unreachable_ids,
                    "must_dispatch_count": 0,
                    "selected_count": 0,
                    "assigned_count": 0,
                    "rejected_after_environment_replay": [],
                    "pchgs": {"skipped_no_dispatchable_orders": True},
                    "pchgs_runtime_seconds": 0.0,
                    "wall_runtime_seconds": time.perf_counter() - started,
                }
            )
            return {}, orders
        must_dispatch_ids = self._must_dispatch_ids(
            dispatchable_orders, available_vehicles, current_time, vehicle_speed
        )
        if self.config.feature_reference == "current":
            references = dispatchable_orders
        else:
            references = list(self._observed.values())

        features, durations, nodes = compute_mlco_features(
            self.depots[0],
            dispatchable_orders,
            references,
            must_dispatch_ids,
            current_time,
            vehicle_speed,
            horizon=self._horizon,
            time_scale=self.config.feature_time_scale,
        )
        try:
            profits = self._predict_profits(features)
        except Exception:
            if self.config.strict_model:
                raise
            profits = np.zeros(len(nodes), dtype=float)

        must_mask = np.asarray(
            [
                False,
                *[
                    int(order["id"]) in must_dispatch_ids
                    for order in dispatchable_orders
                ],
            ],
            dtype=bool,
        )
        max_from = np.max(durations, axis=1)
        max_to = np.max(durations, axis=0)
        forced_profits = 2 * (max_from + max_to)
        profits[must_mask] = forced_profits[must_mask]
        integer_profits = profits.astype(np.int64)
        integer_profits[0] = 0

        instance = self.pchgs.format_instance(
            nodes,
            self.capacity,
            durations,
            integer_profits,
            current_time,
            10000.0 if self.allow_late_return else self._horizon,
            self.config.feature_time_scale,
        )
        seed = self.config.seed + 1_000_003 * self._call_index
        self._call_index += 1
        solution = self.pchgs.solve(
            instance,
            num_vehicles=len(available_vehicles),
            time_limit=max(float(self.time_limit), self.config.min_solver_seconds),
            seed=seed,
            must_dispatch_ids=sorted(must_dispatch_ids),
            allow_fractional_seconds=self.config.allow_fractional_pchgs_budget,
        )

        order_by_id = {int(order["id"]): order for order in dispatchable_orders}
        backend_routes = [
            BackendRoute(0, tuple(order_by_id[order_id] for order_id in route))
            for route in solution.routes
        ]
        assignments, rejected = self.route_adapter.routes_to_assignments(
            backend_routes,
            available_vehicles,
            current_time,
            vehicle_speed,
        )
        assigned_ids = {
            int(order_id)
            for route in assignments.values()
            for order_id in route.get("customer_ids", [])
        }
        dropped = [order for order in orders if int(order["id"]) not in assigned_ids]
        self.diagnostics_history.append(
            {
                "time": int(current_time),
                "batch_size": len(orders),
                "dispatchable_count": len(dispatchable_orders),
                "currently_unreachable_ids": currently_unreachable_ids,
                "observed_reference_size": len(references),
                "must_dispatch_count": len(must_dispatch_ids),
                "selected_count": len(solution.selected_ids),
                "assigned_count": len(assigned_ids),
                "rejected_after_environment_replay": sorted(rejected),
                "predicted_profit_min": float(np.min(profits[1:])),
                "predicted_profit_mean": float(np.mean(profits[1:])),
                "predicted_profit_max": float(np.max(profits[1:])),
                "pchgs": solution.raw_summary,
                "pchgs_runtime_seconds": solution.runtime_seconds,
                "wall_runtime_seconds": time.perf_counter() - started,
            }
        )
        return assignments, dropped

    def get_diagnostics(self) -> dict[str, Any]:
        return {
            "solver": "ML-CO/PC-HGS",
            "config": asdict(self.config),
            "official_feature_count": len(FEATURE_NAMES),
            "dynamic_feature_count": len(DYNAMIC_FEATURE_NAMES),
            "model_input_feature_count": (
                self._predictor.input_dimension if self._predictor is not None else None
            ),
            "official_saved_model": str(self.model_path),
            "exported_weights": str(self.weights_path),
            "inference_backend": self._inference_backend or "not_loaded",
            "saved_model_sha256": self._model_source_sha256,
            "pchgs_binary": str(self.pchgs.binary),
            "feature_reference_policy": self.config.feature_reference,
            "future_customer_rows_retained": False,
            "transfer_note": (
                "The released predictor is evaluated zero-shot; its training distribution "
                "differs from this benchmark's synthetic instances."
            ),
            "decision_rounds": self.diagnostics_history,
        }
