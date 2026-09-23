from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = WORKSPACE_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import main_test  # noqa: E402

from .config import ICDConfig, MLCOConfig  # noqa: E402
from .icd_solver import ICDSolver  # noqa: E402
from .mlco_solver import MLCOSolver  # noqa: E402


class ExperimentalDynamicSimulation(main_test.DynamicSimulation):
    """Uses the unchanged benchmark environment and configures its new solver."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        configure = getattr(self.solver, "configure_instance", None)
        if configure is not None:
            configure(self.raw_data, self.data_file_name)

    def _should_replan(self, current_time, pending_order_ids):
        """Preserve strict periodic semantics for the fixed trigger.

        The copied environment suppresses a fixed-period call when the order
        and idle-vehicle sets have not changed since the previous call.  Time
        itself changes feasibility and ``must_dispatch`` status, so that
        suppression turns fixed-30 into an event-driven trigger and can let a
        deferred order expire.  This isolated runner override retains all
        other trigger modes from the copied environment.
        """
        if self.replan_strategy != "fixed":
            return super()._should_replan(current_time, pending_order_ids)
        if current_time % self.interval != 0:
            return False, None
        if current_time == self.last_replan_state["time"]:
            return False, None
        has_available_backlog = any(
            self.customer_map[customer_id]["available_time"] <= current_time
            <= self.customer_map[customer_id]["tw_end"]
            for customer_id in pending_order_ids
        )
        return (
            (True, "fixed_interval")
            if has_available_backlog
            else (False, None)
        )


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-encode {type(value).__name__}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated ICD or ML-CO experiments with the unchanged benchmark environment."
    )
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data-file", action="append", help="Exact benchmark JSON; repeatable.")
    inputs.add_argument("--data-dir", help="Directory containing benchmark JSON files.")
    parser.add_argument("--pattern", default="*.json", help="Glob used with --data-dir.")
    parser.add_argument("--max-instances", type=int, default=None)
    parser.add_argument("--solver", choices=("icd", "mlco"), required=True)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--time-limit", type=float, default=1.0)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument(
        "--replan-strategy",
        choices=("fixed", "new_order", "demand", "urgent", "urgency"),
        default="fixed",
    )
    parser.add_argument("--urgency-thresh", type=float, default=10.0)
    parser.add_argument("--capacity-thresh", type=float, default=1.0)
    parser.add_argument("--commit-lead-time", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--forbid-late-return", action="store_true")

    icd = parser.add_argument_group("ICD")
    icd.add_argument("--icd-iterations", type=int, default=3)
    icd.add_argument("--icd-scenarios", type=int, default=30)
    icd.add_argument("--icd-lookahead", type=int, default=1)
    icd.add_argument("--icd-dispatch-threshold", type=float, default=0.50)
    icd.add_argument("--icd-postpone-threshold", type=float, default=0.80)
    icd.add_argument(
        "--icd-dispatch-choice",
        choices=("dispatch", "not_postpone"),
        default="not_postpone",
    )
    icd.add_argument("--icd-paper-feasible-warm-start", action="store_true")
    icd.add_argument(
        "--icd-postpone-boundary",
        choices=("release_inclusive", "paper_strict"),
        default="paper_strict",
    )
    icd.add_argument("--icd-max-sampled", type=int, default=200)
    icd.add_argument("--icd-scenario-time-limit", type=float, default=1.0)
    icd.add_argument("--icd-dispatch-time-limit", type=float, default=1.0)
    icd.add_argument(
        "--icd-budget-preset",
        choices=("custom", "repo_demo_5s", "paper_formal_120s"),
        default="custom",
    )
    icd.add_argument(
        "--icd-routing-backend",
        choices=("standard", "paper_dispatch_window"),
        default="paper_dispatch_window",
    )
    icd.add_argument(
        "--icd-route-wait-policy",
        choices=(
            "legacy_latest_feasible",
            "paper_intended",
            "release_literal",
            "forecast_merge_value",
            "forecast_merge_next_wave_value",
        ),
        default="forecast_merge_next_wave_value",
    )
    icd.add_argument("--icd-dispatch-at-wave-start", action="store_true")
    icd.add_argument(
        "--icd-forecast-merge-min-savings-ratio", type=float, default=0.0
    )
    icd.add_argument("--icd-max-consecutive-postponements", type=int)
    icd.add_argument(
        "--icd-scenario-fleet-policy",
        choices=("available", "unlimited", "scaled"),
        default="available",
    )
    icd.add_argument(
        "--icd-scenario-initial-customers-per-route",
        type=float,
        default=3.0,
    )
    icd.add_argument(
        "--icd-scenario-solver-kind",
        choices=(
            "paper_short_hgs",
            "default_hgs",
            "default_hgs_diverse_init",
            "default_hgs_nn_init",
        ),
        default="default_hgs",
    )
    icd.add_argument(
        "--icd-final-solver-kind",
        choices=(
            "default_hgs",
            "default_hgs_nn_init",
            "benchmark_hgs",
            "benchmark_hgs_nn_init",
        ),
        default="default_hgs",
    )
    icd.add_argument(
        "--icd-sampler-knowledge",
        choices=(
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
        ),
        default="instance_config",
        help=(
            "Future-information ablation. realized_future reads future rows and is "
            "a perfect-information reference, not a fair online policy and not a "
            "guaranteed performance upper bound for the fixed ICD heuristic."
        ),
    )

    mlco = parser.add_argument_group("ML-CO")
    mlco.add_argument("--mlco-model", default=None)
    mlco.add_argument("--mlco-weights", default=None)
    mlco.add_argument("--pchgs-binary", default=None)
    mlco.add_argument(
        "--mlco-feature-reference", choices=("observed", "current"), default="observed"
    )
    mlco.add_argument("--mlco-time-scale", type=float, default=60.0)
    mlco.add_argument("--mlco-nonstrict-model", action="store_true")
    return parser


def _resolve_files(args: argparse.Namespace) -> list[Path]:
    if args.data_file:
        files = [Path(value).expanduser().resolve() for value in args.data_file]
    else:
        files = sorted(Path(args.data_dir).expanduser().resolve().glob(args.pattern))
    if args.max_instances is not None:
        files = files[: args.max_instances]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Input files do not exist: {missing}")
    if not files:
        raise FileNotFoundError("No input JSON files matched")
    return files


def _configure_solver(args: argparse.Namespace):
    if args.solver == "icd":
        scenario_time_limit = args.icd_scenario_time_limit
        dispatch_time_limit = args.icd_dispatch_time_limit
        if args.icd_budget_preset == "repo_demo_5s":
            scenario_time_limit = 3.0 / (3 * 30)
            dispatch_time_limit = 2.0
        elif args.icd_budget_preset == "paper_formal_120s":
            scenario_time_limit = 1.0
            dispatch_time_limit = 30.0
        config = ICDConfig(
            seed=args.seed,
            num_iterations=args.icd_iterations,
            num_scenarios=args.icd_scenarios,
            decision_interval=args.interval,
            num_lookahead=args.icd_lookahead,
            dispatch_threshold=args.icd_dispatch_threshold,
            postpone_threshold=args.icd_postpone_threshold,
            dispatch_choice=args.icd_dispatch_choice,
            max_sampled_requests=args.icd_max_sampled,
            sampler_knowledge=args.icd_sampler_knowledge,
            scenario_time_limit=scenario_time_limit,
            dispatch_time_limit=dispatch_time_limit,
            routing_backend=args.icd_routing_backend,
            route_wait_policy=args.icd_route_wait_policy,
            forecast_merge_min_savings_ratio=(
                args.icd_forecast_merge_min_savings_ratio
            ),
            max_consecutive_postponements=(
                args.icd_max_consecutive_postponements
            ),
            dispatch_at_wave_start=args.icd_dispatch_at_wave_start,
            scenario_fleet_policy=args.icd_scenario_fleet_policy,
            scenario_initial_customers_per_route=(
                args.icd_scenario_initial_customers_per_route
            ),
            scenario_solver_kind=args.icd_scenario_solver_kind,
            final_solver_kind=args.icd_final_solver_kind,
            paper_feasible_warm_start=args.icd_paper_feasible_warm_start,
            postpone_boundary=args.icd_postpone_boundary,
        )
        ICDSolver.runtime_config = config
        return ICDSolver, config

    config = MLCOConfig(
        seed=args.seed,
        decision_interval=args.interval,
        model_path=args.mlco_model,
        weights_path=args.mlco_weights,
        pchgs_binary=args.pchgs_binary,
        feature_time_scale=args.mlco_time_scale,
        feature_reference=args.mlco_feature_reference,
        strict_model=not args.mlco_nonstrict_model,
    )
    MLCOSolver.runtime_config = config
    return MLCOSolver, config


def _result_directory(output_root: Path, data_file: Path) -> Path:
    """Mirrors dataset subdirectories so equal stems never overwrite."""
    dataset_root = (PROJECT_ROOT / "data").resolve()
    try:
        relative_parent = data_file.resolve().relative_to(dataset_root).parent
        return output_root / relative_parent
    except ValueError:
        digest = hashlib.sha1(str(data_file.resolve()).encode("utf-8")).hexdigest()[:10]
        return output_root / "_external" / f"{data_file.parent.name}_{digest}"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if hasattr(main_test, "set_seed"):
        main_test.set_seed(seed)


def run(args: argparse.Namespace) -> int:
    files = _resolve_files(args)
    solver_class, config = _configure_solver(args)
    main_test.SOLVER_TIME_LIMIT = float(args.time_limit)
    main_test.ALLOW_LATE_RETURN = not args.forbid_late_return

    output_root = Path(args.output_dir).expanduser().resolve() / args.solver
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "solver": args.solver,
        "solver_class": solver_class.__name__,
        "solver_config": dataclasses.asdict(config),
        "benchmark_environment_file": str(Path(main_test.__file__).resolve()),
        "original_environment_modified": False,
        "started_at_unix": time.time(),
        "runs": [],
    }

    failed = False
    for data_file in files:
        stem = data_file.stem
        result_directory = _result_directory(output_root, data_file)
        result_directory.mkdir(parents=True, exist_ok=True)
        result_path = result_directory / f"res_{stem}_{args.solver}.json"
        diagnostics_path = result_directory / f"diag_{stem}_{args.solver}.json"
        if result_path.exists() and not args.overwrite:
            print(f"[skip] {result_path}")
            summary["runs"].append(
                {"data_file": str(data_file), "result": str(result_path), "status": "skipped"}
            )
            continue

        print(f"[run] solver={args.solver} instance={data_file.name}")
        _set_seed(args.seed)
        try:
            simulation = ExperimentalDynamicSimulation(
                data_file=str(data_file),
                solver_class=solver_class,
                interval=args.interval,
                speed=args.speed,
                replan_strategy=(
                    "urgent" if args.replan_strategy == "urgency" else args.replan_strategy
                ),
                strategy_kwargs={
                    "urgency_thresh": args.urgency_thresh,
                    "capacity_thresh": args.capacity_thresh,
                },
                commit_lead_time=args.commit_lead_time,
            )
            service_rate = simulation.run()
            simulation.save_and_visualize(
                str(result_path), solver_name=solver_class.__name__, render_formats=[]
            )
            diagnostics = simulation.solver.get_diagnostics()
            diagnostics.update(
                {
                    "data_file": str(data_file),
                    "result_file": str(result_path),
                    "service_rate": service_rate,
                }
            )
            diagnostics_path.write_text(
                json.dumps(diagnostics, indent=2, default=_json_default), encoding="utf-8"
            )
            summary["runs"].append(
                {
                    "data_file": str(data_file),
                    "result": str(result_path),
                    "diagnostics": str(diagnostics_path),
                    "service_rate": service_rate,
                    "status": "completed",
                }
            )
            print(f"[done] {result_path}")
        except Exception as exc:
            failed = True
            traceback.print_exc()
            summary["runs"].append(
                {
                    "data_file": str(data_file),
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    summary["finished_at_unix"] = time.time()
    summary_path = output_root / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    print(f"[summary] {summary_path}")
    return 1 if failed else 0


def main() -> None:
    raise SystemExit(run(_parser().parse_args()))


if __name__ == "__main__":
    main()
