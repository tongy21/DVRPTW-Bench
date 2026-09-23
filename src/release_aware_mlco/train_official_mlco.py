#!/usr/bin/env python3
"""Train the released ML-CO model on exported SDVRP release-aware labels.

This wrapper deliberately imports the released ML-CO implementation instead
of copying/reimplementing its learning objective.  It changes only two pieces
of execution plumbing:

* bound PC-HGS perturbation solves to either a serial pool or a reusable
  thread pool, so it never spawns ``cpu_count() - 2`` workers beside unrelated
  benchmark training;
* adapt the released feature computer's hard-coded one-hour epoch constants to
  the decision interval used by the exported SDVRP observations.

The neural network, predict-and-optimize gradient, perturbations, and PC-HGS
subproblem solver remain the official implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np


def _with_trailing_separator(path: Path) -> str:
    return str(path.expanduser().resolve()) + os.sep


class SerialPool:
    """Small context-manager substitute for multiprocessing.Pool."""

    def __enter__(self) -> "SerialPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False

    def map(self, function: Any, values: Iterable[Any]) -> list[Any]:
        return [function(value) for value in values]


class ReusableThreadPool:
    """Pool-compatible facade over one bounded executor shared by all states."""

    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self.executor = executor
        self.map_index = 0

    def __enter__(self) -> "ReusableThreadPool":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False

    def map(self, function: Any, values: Iterable[Any]) -> list[Any]:
        current_map = self.map_index
        self.map_index += 1
        return list(
            self.executor.map(
                lambda value: function((current_map, int(value))),
                values,
            )
        )


def _install_deterministic_perturbations(
    optimization: Any,
    util: Any,
    evaluation_tools: Any,
    seed: int,
) -> None:
    """Use a distinct deterministic RNG and PC-HGS seed per perturbation."""

    lock = threading.Lock()
    audit: list[dict[str, Any]] = []
    retry_audit: list[dict[str, Any]] = []

    def seeded_loss_for_perturbation(kwargs: Any, identity: Any):
        args, training_instance, costs, profits = kwargs
        if isinstance(identity, tuple):
            observation_update, perturbation_num = identity
        else:
            observation_update, perturbation_num = 0, int(identity)
        digest = hashlib.blake2b(digest_size=8)
        digest.update(np.asarray(costs).shape.__repr__().encode("ascii"))
        digest.update(np.asarray(training_instance["nodes"]).tobytes())
        digest.update(str(int(seed)).encode("ascii"))
        digest.update(str(int(observation_update)).encode("ascii"))
        digest.update(str(int(perturbation_num)).encode("ascii"))
        base_seed = int.from_bytes(digest.digest(), "little") % 2_147_483_646 + 1
        profit_values = np.asarray(profits)
        solution = None
        noise = None
        perturbed = None
        perturbation_seed = None
        for retry in range(int(args.pchgs_max_attempts)):
            perturbation_seed = (
                (base_seed + retry * 1_000_003 - 1) % 2_147_483_646
            ) + 1
            rng = np.random.default_rng(perturbation_seed)
            noise = rng.normal(
                loc=0, scale=args.sd_perturbation, size=profit_values.shape
            )
            perturbed = profit_values + noise
            perturbed[0] = 0
            perturbed = util.make_to_ints(perturbed)
            instance = evaluation_tools.format_pchgs_instance_as_json(
                args,
                instance=training_instance["epoch_instance_numpy"],
                profits=perturbed,
            )
            try:
                retry_time_limit = (
                    args.time_limit
                    if retry == 0
                    else max(args.time_limit, args.pchgs_retry_time_limit)
                )
                solution = evaluation_tools.solve_pchgs(
                    args=args,
                    instance=instance,
                    executable=args.pchgs_executable,
                    time_limit=retry_time_limit,
                    seed=perturbation_seed,
                )
                break
            except (json.JSONDecodeError, RuntimeError, ValueError) as exc:
                with lock:
                    retry_audit.append(
                        {
                            "observation_update": int(observation_update),
                            "perturbation": int(perturbation_num),
                            "retry": retry + 1,
                            "perturbation_seed": perturbation_seed,
                            "time_limit_seconds": retry_time_limit,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                if retry == int(args.pchgs_max_attempts) - 1:
                    raise
                time.sleep(0.1 * (retry + 1))
        assert solution is not None and noise is not None and perturbed is not None
        assert perturbation_seed is not None
        cost_hat, y_hat, n_hat = util.decode_solution(
            solution, edges=training_instance["edges"], nodes=training_instance["nodes"]
        )
        loss_hat = np.sum(perturbed[n_hat]) - np.sum(costs.flatten(order="C")[y_hat])
        if getattr(args, "audit_perturbations", False):
            with lock:
                if len(audit) < int(args.num_perturbations):
                    audit.append(
                        {
                            "perturbation": int(perturbation_num),
                            "observation_update": int(observation_update),
                            "seed": perturbation_seed,
                            "noise_sha256": hashlib.sha256(
                                np.asarray(noise, dtype=np.float64).tobytes()
                            ).hexdigest(),
                            "profit_sha256": hashlib.sha256(
                                np.asarray(perturbed).tobytes()
                            ).hexdigest(),
                            "selected_nodes": sorted(int(value) for value in n_hat),
                        }
                    )
        return loss_hat, y_hat, n_hat

    optimization.loss_for_perturbation = seeded_loss_for_perturbation
    optimization._sdvrp_perturbation_audit = audit
    optimization._sdvrp_retry_audit = retry_audit


def _install_interval_adapter(
    feature_module: Any,
    epoch_duration_seconds: int,
    dispatch_margin_seconds: int,
) -> None:
    official_class = feature_module.FeatureComputer

    class IntervalAwareFeatureComputer(official_class):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.EPOCH_DURATION = int(epoch_duration_seconds)
            self.MARGIN_DISPATCH = int(dispatch_margin_seconds)

    IntervalAwareFeatureComputer.__name__ = "IntervalAwareFeatureComputer"
    feature_module.FeatureComputer = IntervalAwareFeatureComputer


def _repair_static_info_for_interval(
    training_instances: list[dict[str, Any]],
    epoch_duration_seconds: int,
    dispatch_margin_seconds: int,
) -> None:
    """Replace metadata computed by the official hourly simulator.

    The observations themselves are exported by SDVRP_Bench.  The official
    loader nevertheless calls its own hourly environment solely to obtain
    ``static_info``.  Only normalized-time features use this information.
    """
    for instance in training_instances:
        observations = instance["epoch_observations"]
        if not observations:
            raise ValueError("Training instance contains no retained observations")
        recorded_horizons = {
            int(observation["benchmark_horizon_seconds"])
            for observation in observations
            if "benchmark_horizon_seconds" in observation
        }
        if len(recorded_horizons) > 1:
            raise ValueError(
                f"Inconsistent benchmark horizons in one training instance: {recorded_horizons}"
            )
        if recorded_horizons:
            final_training_time = recorded_horizons.pop()
        else:
            # Compatibility fallback for the pre-fix smoke corpus.
            final_training_time = max(
                int(observation["planning_starttime"])
                for observation in observations
            ) + int(epoch_duration_seconds)
        end_epoch = max(
            int(
                math.ceil(
                    (final_training_time - dispatch_margin_seconds)
                    / epoch_duration_seconds
                )
            ),
            1,
        )
        instance["static_info"].update(
            {
                "start_epoch": 0,
                "end_epoch": end_epoch,
                "num_epochs": end_epoch + 1,
            }
        )


def _validate_feature_labels(training_set: list[dict[str, Any]]) -> dict[str, Any]:
    if not training_set:
        raise ValueError("Official feature builder returned an empty training set")
    positive_targets = 0
    label_rows = 0
    for index, sample in enumerate(training_set):
        frame = sample["epoch_instance"]
        if "target" not in frame:
            raise ValueError(f"Training sample {index} has no target column")
        if bool(frame.iloc[0]["target"]):
            raise ValueError(f"Depot has a positive target in sample {index}")
        positive_targets += int(frame["target"].sum())
        label_rows += int(len(frame))
    return {
        "num_training_observations": len(training_set),
        "num_features": int(training_set[0]["features"].shape[1]),
        "num_edge_features": int(training_set[0]["edge_features"].shape[1]),
        "feature_names": list(training_set[0]["features"].columns),
        "edge_feature_names": list(training_set[0]["edge_features"].columns),
        "num_label_rows": label_rows,
        "num_positive_targets": positive_targets,
    }


def _subsample_training_set(
    training_set: list[dict[str, Any]],
    maximum: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], int]:
    """Deterministically cap expensive predict-and-optimize observations."""
    original_size = len(training_set)
    if maximum is None or original_size <= maximum:
        return training_set, original_size
    import numpy as np

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(original_size, size=maximum, replace=False))
    return [training_set[int(index)] for index in selected], original_size


def _parse_scale_observation_quotas(raw: str | None) -> dict[int, int]:
    if not raw:
        return {}
    quotas: dict[int, int] = {}
    for item in raw.split(","):
        scale, quota = item.split(":", 1)
        scale_value, quota_value = int(scale), int(quota)
        if scale_value <= 0 or quota_value <= 0:
            raise ValueError("scale observation quotas must be positive")
        quotas[scale_value] = quota_value
    return quotas


def _stratify_training_set_by_scale(
    training_set: list[dict[str, Any]], quotas: dict[int, int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Retain evenly spaced observations per trajectory by customer scale."""
    if not quotas:
        return training_set, {}
    grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, sample in enumerate(training_set):
        source = str(sample["_source_metadata"]["source_directory"])
        grouped.setdefault(source, []).append((index, sample))

    selected_indices: set[int] = set()
    counts: dict[str, int] = {}
    for source, indexed_samples in grouped.items():
        match = re.search(r"(?:^|-)n(\d+)(?:-|_)", source)
        if not match:
            raise ValueError(f"Cannot parse customer scale from {source}")
        scale = int(match.group(1))
        if scale not in quotas:
            raise ValueError(f"No observation quota configured for scale {scale}")
        quota = quotas[scale]
        ordered = sorted(
            indexed_samples,
            key=lambda pair: int(pair[1]["_source_metadata"]["current_epoch"]),
        )
        if len(ordered) < quota:
            raise ValueError(
                f"{source} needs {quota} observations but only {len(ordered)} exist"
            )
        positions = (
            [len(ordered) // 2]
            if quota == 1
            else [round(i * (len(ordered) - 1) / (quota - 1)) for i in range(quota)]
        )
        chosen = [ordered[position][0] for position in positions]
        if len(set(chosen)) != quota:
            raise RuntimeError(f"Observation stratification duplicated {source}")
        selected_indices.update(chosen)
        counts[source] = quota
    return (
        [sample for index, sample in enumerate(training_set) if index in selected_indices],
        counts,
    )


def _tag_training_sources(util: Any) -> None:
    original = util.get_observation_and_solution

    def tagged(args: Any, training_instance_directory: str):
        result = original(args, training_instance_directory)
        result["_source_directory"] = training_instance_directory
        return result

    util.get_observation_and_solution = tagged


def _attach_sample_sources(
    loaded: list[dict[str, Any]], training_set: list[dict[str, Any]]
) -> None:
    metadata = [
        {
            "source_directory": instance["_source_directory"],
            "current_epoch": int(observation["current_epoch"]),
            "planning_starttime": int(observation["planning_starttime"]),
        }
        for instance in loaded
        for observation in instance["epoch_observations"]
    ]
    if len(metadata) != len(training_set):
        raise RuntimeError(
            f"Training sample/source mismatch: {len(training_set)} vs {len(metadata)}"
        )
    for sample, source in zip(training_set, metadata):
        sample["_source_metadata"] = source


def _robust_train(
    optimizer: Any,
    training_instances: list[dict[str, Any]],
    util: Any,
    run_directory: Path,
    maximum_skipped_updates: int,
) -> list[dict[str, Any]]:
    skipped: list[dict[str, Any]] = []
    skip_log = run_directory / "skipped_observation_updates.jsonl"
    optimizer.default_accuracy_n = util.calculate_default_accuracy(
        training_instances=training_instances
    )
    print("Start robust training ...")
    for epoch in range(optimizer.args.num_training_epochs):
        loss_values = []
        accuracies = []
        for observation_index, training_instance in enumerate(training_instances):
            try:
                loss_value, n, n_saver = optimizer.loss(training_instance)
            except json.JSONDecodeError as exc:
                record = {
                    "epoch": epoch,
                    "observation_index": observation_index,
                    **training_instance.get("_source_metadata", {}),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                skipped.append(record)
                with skip_log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
                print(f"Skipping failed observation update: {record}", flush=True)
                if len(skipped) > maximum_skipped_updates:
                    raise RuntimeError(
                        f"Skipped update limit exceeded: {len(skipped)} > "
                        f"{maximum_skipped_updates}"
                    ) from exc
                continue
            accuracy = util.calculate_accuracy_mean(
                n=np.asarray(n), n_hat=np.asarray(n_saver)
            )
            loss_values.append(loss_value)
            accuracies.append(accuracy)
        if not loss_values:
            raise RuntimeError(f"Every observation failed in epoch {epoch}")
        optimizer.overall_loss.append(float(np.mean(loss_values)))
        optimizer.overall_accuracy_n.append(float(np.mean(accuracies)))
        optimizer.model.save_model(
            directory=optimizer.args.dir_models + optimizer.args.model_name,
            count=epoch,
        )
        optimizer.save_learning_evaluation()
        print(
            f"Epoch: {epoch} ----> Loss: {np.mean(loss_values)} ---- "
            f"Accuracy_n: {np.mean(accuracies)} ---- Skipped: {len(skipped)}",
            flush=True,
        )
    return skipped


def main() -> None:
    total_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repository", required=True, type=Path)
    parser.add_argument("--oracle-solutions-directory", required=True, type=Path)
    parser.add_argument("--instances-directory", required=True, type=Path)
    parser.add_argument("--pchgs-executable", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument("--model-name", default="NeuralNetwork_sdvrp_releaseaware_dynamic")
    parser.add_argument("--predictor", default="NeuralNetwork")
    parser.add_argument("--feature-set", choices=("dynamic", "static", "dynamicstatic"), default="dynamic")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--time-limit", type=int, default=1)
    parser.add_argument("--num-perturbations", type=int, default=1)
    parser.add_argument("--sd-perturbation", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--epoch-duration-seconds", type=int, default=1800)
    parser.add_argument("--dispatch-margin-seconds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--parallel-perturbation-workers", type=int, default=1)
    parser.add_argument("--deterministic-perturbations", action="store_true")
    parser.add_argument("--audit-perturbations", action="store_true")
    parser.add_argument("--pchgs-max-attempts", type=int, default=5)
    parser.add_argument("--pchgs-retry-time-limit", type=int, default=5)
    parser.add_argument("--skip-failed-observation-updates", action="store_true")
    parser.add_argument("--max-skipped-observation-updates", type=int, default=10)
    parser.add_argument(
        "--max-training-observations",
        type=int,
        default=None,
        help=(
            "Optional deterministic cap on epoch observations. This bounds the "
            "number of one-second PC-HGS perturbation calls in pilot studies."
        ),
    )
    parser.add_argument(
        "--scale-observation-quotas",
        help="Per-trajectory quotas such as 20:10,50:6,100:4,200:3,500:2",
    )
    args_cli = parser.parse_args()

    for value, label in (
        (args_cli.epochs, "epochs"),
        (args_cli.time_limit, "time limit"),
        (args_cli.num_perturbations, "number of perturbations"),
        (args_cli.epoch_duration_seconds, "epoch duration"),
        (args_cli.parallel_perturbation_workers, "parallel perturbation workers"),
        (args_cli.pchgs_max_attempts, "PC-HGS max attempts"),
        (args_cli.pchgs_retry_time_limit, "PC-HGS retry time limit"),
    ):
        if int(value) <= 0:
            raise ValueError(f"{label} must be positive")
    if args_cli.max_skipped_observation_updates < 0:
        raise ValueError("max skipped observation updates cannot be negative")
    if (
        args_cli.max_training_observations is not None
        and args_cli.max_training_observations <= 0
    ):
        raise ValueError("max training observations must be positive")

    official_repository = args_cli.official_repository.expanduser().resolve()
    official_training = official_repository / "training"
    if not (official_training / "optimization.py").is_file():
        raise FileNotFoundError(f"Not an official ML-CO checkout: {official_repository}")
    pchgs = args_cli.pchgs_executable.expanduser().resolve()
    if not pchgs.is_file():
        raise FileNotFoundError(pchgs)
    oracle_solutions_directory = (
        args_cli.oracle_solutions_directory.expanduser().resolve()
    )
    instances_directory = args_cli.instances_directory.expanduser().resolve()
    if not oracle_solutions_directory.is_dir():
        raise FileNotFoundError(oracle_solutions_directory)
    if not instances_directory.is_dir():
        raise FileNotFoundError(instances_directory)

    run_directory = args_cli.run_directory.expanduser().resolve()
    model_directory = run_directory / "models"
    executable_directory = run_directory / "bin"
    run_directory.mkdir(parents=True, exist_ok=True)
    model_directory.mkdir(parents=True, exist_ok=True)
    executable_directory.mkdir(parents=True, exist_ok=True)
    local_pchgs = executable_directory / "PCHGS"
    if local_pchgs.exists() or local_pchgs.is_symlink():
        if local_pchgs.resolve() != pchgs:
            raise ValueError(f"Existing PCHGS link points elsewhere: {local_pchgs}")
    else:
        local_pchgs.symlink_to(pchgs)

    # TensorFlow reads these before import.  Bound its CPU footprint so this
    # experiment can safely coexist with the user's ongoing RL training.
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    sys.path.insert(0, str(official_repository))
    sys.path.insert(0, str(official_training))
    previous_cwd = Path.cwd()
    os.chdir(run_directory)
    try:
        util = importlib.import_module("src.util")
        _tag_training_sources(util)
        feature_module = importlib.import_module("features.FeatureComputer")
        optimization = importlib.import_module("optimization")
        _install_interval_adapter(
            feature_module,
            args_cli.epoch_duration_seconds,
            args_cli.dispatch_margin_seconds,
        )
        # optimization.py imported Pool directly, so patch that module-level
        # symbol without changing the released source tree.
        optimization.Pool = lambda _workers: SerialPool()

        training_args = SimpleNamespace(
            oracle_solutions_directory=_with_trailing_separator(
                oracle_solutions_directory
            ),
            instances_directory=_with_trailing_separator(instances_directory),
            predictor=args_cli.predictor,
            num_training_epochs=int(args_cli.epochs),
            time_limit=int(args_cli.time_limit),
            learning_rate=float(args_cli.learning_rate),
            dir_models=_with_trailing_separator(model_directory),
            num_perturbations=int(args_cli.num_perturbations),
            audit_perturbations=bool(args_cli.audit_perturbations),
            pchgs_max_attempts=int(args_cli.pchgs_max_attempts),
            pchgs_retry_time_limit=int(args_cli.pchgs_retry_time_limit),
            sd_perturbation=float(args_cli.sd_perturbation),
            feature_set=args_cli.feature_set,
            # evaluation.tools prefixes "./", hence use a path relative to the
            # isolated run directory rather than an absolute executable path.
            pchgs_executable=str(Path("bin") / "PCHGS"),
            model_name=args_cli.model_name,
        )

        random.seed(args_cli.seed)
        np.random.seed(args_cli.seed)
        import tensorflow as tf

        tf.keras.utils.set_random_seed(args_cli.seed)
        executor = None
        if (
            args_cli.parallel_perturbation_workers > 1
            and not args_cli.deterministic_perturbations
        ):
            raise ValueError(
                "Parallel perturbations require --deterministic-perturbations "
                "to avoid inherited NumPy RNG state"
            )
        if args_cli.parallel_perturbation_workers > 1:
            executor = ThreadPoolExecutor(
                max_workers=min(
                    args_cli.parallel_perturbation_workers,
                    args_cli.num_perturbations,
                ),
                thread_name_prefix="mlco-pchgs",
            )
            reusable_pool = ReusableThreadPool(executor)
            optimization.Pool = lambda _workers: reusable_pool
        elif args_cli.deterministic_perturbations:
            executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mlco-pchgs-serial"
            )
            reusable_pool = ReusableThreadPool(executor)
            optimization.Pool = lambda _workers: reusable_pool
        if args_cli.deterministic_perturbations:
            _install_deterministic_perturbations(
                optimization, util, importlib.import_module("evaluation.tools"), args_cli.seed
            )
        load_started = time.perf_counter()
        loaded = util.load_training_instances(training_args)
        load_seconds = time.perf_counter() - load_started
        if not loaded:
            raise ValueError("Official loader found no exported training instances")
        _repair_static_info_for_interval(
            loaded,
            args_cli.epoch_duration_seconds,
            args_cli.dispatch_margin_seconds,
        )
        feature_started = time.perf_counter()
        training_set = feature_module.create_features(training_args, loaded)
        _attach_sample_sources(loaded, training_set)
        num_available_observations = len(training_set)
        scale_quotas = _parse_scale_observation_quotas(
            args_cli.scale_observation_quotas
        )
        training_set, scale_quota_counts = _stratify_training_set_by_scale(
            training_set, scale_quotas
        )
        training_set, _ = _subsample_training_set(
            training_set,
            args_cli.max_training_observations,
            args_cli.seed,
        )
        feature_seconds = time.perf_counter() - feature_started
        summary = _validate_feature_labels(training_set)

        optimizer = optimization.Optimizer(
            args=training_args,
            num_features=summary["num_features"],
            num_edge_features=summary["num_edge_features"],
        )
        training_started = time.perf_counter()
        skipped_observation_updates = (
            _robust_train(
                optimizer,
                training_set,
                util,
                run_directory,
                args_cli.max_skipped_observation_updates,
            )
            if args_cli.skip_failed_observation_updates
            else []
        )
        if not args_cli.skip_failed_observation_updates:
            optimizer.train(training_set)
        training_seconds = time.perf_counter() - training_started

        manifest = {
            "official_repository": str(official_repository),
            "oracle_solutions_directory": training_args.oracle_solutions_directory,
            "instances_directory": training_args.instances_directory,
            "pchgs_executable": str(pchgs),
            "model_name": training_args.model_name,
            "predictor": training_args.predictor,
            "feature_set": training_args.feature_set,
            "epochs": training_args.num_training_epochs,
            "time_limit_seconds_per_pchgs_call": training_args.time_limit,
            "num_perturbations": training_args.num_perturbations,
            "epoch_duration_seconds": args_cli.epoch_duration_seconds,
            "dispatch_margin_seconds": args_cli.dispatch_margin_seconds,
            "seed": args_cli.seed,
            "num_available_training_observations": num_available_observations,
            "max_training_observations": args_cli.max_training_observations,
            "scale_observation_quotas": scale_quotas,
            "scale_observation_quota_counts": scale_quota_counts,
            "execution_pool": (
                "bounded_reusable_thread_pool"
                if args_cli.parallel_perturbation_workers > 1
                else "serial_safety_wrapper"
            ),
            "parallel_perturbation_workers": args_cli.parallel_perturbation_workers,
            "pchgs_max_attempts": args_cli.pchgs_max_attempts,
            "pchgs_retry_time_limit_seconds": args_cli.pchgs_retry_time_limit,
            "skip_failed_observation_updates": args_cli.skip_failed_observation_updates,
            "skipped_observation_updates": skipped_observation_updates,
            "perturbation_rng": (
                "deterministic_per_input_and_perturbation"
                if args_cli.deterministic_perturbations
                else "official_global_numpy_rng"
            ),
            "official_learning_objective_modified": False,
            "official_execution_plumbing_modified": True,
            "interval_feature_adapter": True,
            **summary,
            "training_accuracy": optimizer.overall_accuracy_n,
            "training_loss": optimizer.overall_loss,
            "default_accuracy": optimizer.default_accuracy_n,
            "timing_seconds": {
                "load_training_instances": load_seconds,
                "feature_construction_and_subsampling": feature_seconds,
                "predict_and_optimize_training": training_seconds,
                "mean_per_epoch": training_seconds / training_args.num_training_epochs,
                "mean_per_observation_epoch": training_seconds
                / (
                    training_args.num_training_epochs
                    * summary["num_training_observations"]
                ),
                "total_inside_wrapper": time.perf_counter() - total_started,
            },
        }
        if args_cli.audit_perturbations:
            perturbation_audit = sorted(
                optimization._sdvrp_perturbation_audit,
                key=lambda row: row["perturbation"],
            )
            unique_vectors = len({row["profit_sha256"] for row in perturbation_audit})
            unique_seeds = len({row["seed"] for row in perturbation_audit})
            unique_noise = len({row["noise_sha256"] for row in perturbation_audit})
            manifest["perturbation_audit"] = {
                "records": perturbation_audit,
                "unique_profit_vectors": unique_vectors,
                "unique_seeds": unique_seeds,
                "unique_noise_vectors": unique_noise,
                "expected_unique_seeds": args_cli.num_perturbations,
                "pchgs_retry_events": optimization._sdvrp_retry_audit,
            }
            if (
                unique_seeds != args_cli.num_perturbations
                or unique_noise != args_cli.num_perturbations
            ):
                raise RuntimeError(
                    f"Perturbation audit failed: seeds={unique_seeds}, "
                    f"noise_vectors={unique_noise}, profit_vectors={unique_vectors}, "
                    f"expected unique count="
                    f"{args_cli.num_perturbations}"
                )
        manifest_path = run_directory / "training_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
    finally:
        if "executor" in locals() and executor is not None:
            executor.shutdown(wait=True)
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
