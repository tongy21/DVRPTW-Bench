from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ICDConfig:
    """Parameters of the double-threshold ICD policy.

    The reference paper uses three iterations, thirty scenarios, one lookahead
    epoch, a 0.50 dispatch threshold, and a 0.80 postpone threshold. The
    repository defaults use instance-configuration sampling and unmodified HGS
    for both scenario lookahead and final routing, with one second per solve.
    Experiments may override these values through the runner.
    """

    seed: int = 42
    num_iterations: int = 3
    num_scenarios: int = 30
    decision_interval: int = 30
    num_lookahead: int = 1
    dispatch_threshold: float = 0.50
    postpone_threshold: float = 0.80
    dispatch_choice: str = "not_postpone"
    strategy_time_fraction: float = 0.70
    min_scenario_time: float = 0.01
    min_final_time: float = 0.05
    scenario_time_limit: float | None = 1.0
    dispatch_time_limit: float | None = 1.0
    max_sampled_requests: int = 200
    sampler_knowledge: str = "instance_config"
    routing_backend: str = "paper_dispatch_window"
    route_wait_policy: str = "forecast_merge_next_wave_value"
    forecast_merge_min_savings_ratio: float = 0.0
    max_consecutive_postponements: int | None = None
    dispatch_at_wave_start: bool = False
    scenario_fleet_policy: str = "available"
    scenario_initial_customers_per_route: float = 3.0
    scenario_solver_kind: str = "default_hgs"
    final_solver_kind: str = "default_hgs"
    paper_feasible_warm_start: bool = False
    postpone_boundary: str = "paper_strict"

    def __post_init__(self) -> None:
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.num_scenarios <= 0:
            raise ValueError("num_scenarios must be positive")
        if self.decision_interval <= 0 or self.num_lookahead <= 0:
            raise ValueError("decision_interval and num_lookahead must be positive")
        if not 0 <= self.dispatch_threshold <= 1:
            raise ValueError("dispatch_threshold must be in [0, 1]")
        if not 0 <= self.postpone_threshold <= 1:
            raise ValueError("postpone_threshold must be in [0, 1]")
        if self.dispatch_choice not in {"dispatch", "not_postpone"}:
            raise ValueError("dispatch_choice must be 'dispatch' or 'not_postpone'")
        if not 0 < self.strategy_time_fraction < 1:
            raise ValueError("strategy_time_fraction must be in (0, 1)")
        if self.scenario_time_limit is not None and self.scenario_time_limit <= 0:
            raise ValueError("scenario_time_limit must be positive when provided")
        if self.dispatch_time_limit is not None and self.dispatch_time_limit <= 0:
            raise ValueError("dispatch_time_limit must be positive when provided")
        if self.routing_backend not in {"standard", "paper_dispatch_window"}:
            raise ValueError(
                "routing_backend must be 'standard' or 'paper_dispatch_window'"
            )
        if self.route_wait_policy not in {
            "legacy_latest_feasible",
            "paper_intended",
            "release_literal",
            "forecast_merge_value",
            "forecast_merge_next_wave_value",
        }:
            raise ValueError(
                "route_wait_policy must be legacy_latest_feasible, "
                "paper_intended, release_literal, forecast_merge_value, or "
                "forecast_merge_next_wave_value"
            )
        if not 0 <= self.forecast_merge_min_savings_ratio <= 1:
            raise ValueError(
                "forecast_merge_min_savings_ratio must be in [0, 1]"
            )
        if (
            self.max_consecutive_postponements is not None
            and self.max_consecutive_postponements < 0
        ):
            raise ValueError(
                "max_consecutive_postponements must be non-negative or None"
            )
        if self.scenario_fleet_policy not in {"available", "unlimited", "scaled"}:
            raise ValueError(
                "scenario_fleet_policy must be available, unlimited, or scaled"
            )
        if self.scenario_initial_customers_per_route <= 0:
            raise ValueError("scenario_initial_customers_per_route must be positive")
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
        if self.postpone_boundary not in {"release_inclusive", "paper_strict"}:
            raise ValueError(
                "postpone_boundary must be release_inclusive or paper_strict"
            )
        if self.sampler_knowledge not in {
            "none",
            "observed_history",
            "scenario_family",
            "instance_peak_centers",
            "instance_temporal",
            "instance_hotspot_schedule",
            "instance_spatial",
            "instance_peak_centers_hotspot_schedule",
            "instance_peak_centers_spatial",
            "instance_temporal_hotspot_schedule",
            "instance_config",
            "realized_future",
        }:
            raise ValueError(
                "sampler_knowledge must be one of none, observed_history, "
                "scenario_family, instance_peak_centers, instance_temporal, "
                "instance_hotspot_schedule, instance_spatial, "
                "instance_peak_centers_hotspot_schedule, "
                "instance_peak_centers_spatial, "
                "instance_temporal_hotspot_schedule, "
                "instance_config, or realized_future"
            )


@dataclass(frozen=True)
class MLCOConfig:
    """Runtime configuration for the official ML-CO model and PC-HGS."""

    seed: int = 42
    decision_interval: int = 30
    model_path: str | None = None
    weights_path: str | None = None
    pchgs_binary: str | None = None
    feature_time_scale: float = 60.0
    feature_reference: str = "observed"
    strict_model: bool = True
    min_solver_seconds: float = 1.0
    allow_fractional_pchgs_budget: bool = False

    def resolved_model_path(self, package_dir: Path) -> Path:
        if self.model_path:
            return Path(self.model_path).expanduser().resolve()
        return package_dir / "assets" / "mlco_official_saved_model"

    def resolved_weights_path(self, package_dir: Path) -> Path:
        if self.weights_path:
            return Path(self.weights_path).expanduser().resolve()
        return package_dir / "assets" / "mlco_official_weights.npz"

    def resolved_pchgs_binary(self, package_dir: Path) -> Path:
        if self.pchgs_binary:
            return Path(self.pchgs_binary).expanduser().resolve()
        return package_dir / "vendor" / "pchgs" / "build" / "PCHGS"

    def __post_init__(self) -> None:
        if self.decision_interval <= 0:
            raise ValueError("decision_interval must be positive")
        if self.feature_time_scale <= 0:
            raise ValueError("feature_time_scale must be positive")
        if self.feature_reference not in {"observed", "current"}:
            raise ValueError("feature_reference must be 'observed' or 'current'")
        if self.min_solver_seconds <= 0:
            raise ValueError("min_solver_seconds must be positive")
