#!/usr/bin/env python3
"""Build strict ML-CO corpora, training views, and a large high-DoD pilot set."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[1]
DATA_GENERATION = WORKSPACE / "data_generation"
sys.path[:0] = [str(WORKSPACE), str(DATA_GENERATION), str(Path(__file__).parent)]

from svrp_generator_spatiotemporal import (  # noqa: E402
    ImmediateReachabilityError,
    SpatioTemporalSVRPGenerator,
)
from release_aware_hgs import (  # noqa: E402
    export_official_mlco_layout,
    individually_unserviceable_customers,
    problem_from_benchmark_json,
    solve_release_aware_hgs,
)


SCENARIOS = {
    "US": ("uniform", "stationary"),
    "UV": ("uniform", "time_varying"),
    "BS": ("bursty", "stationary"),
    "BV": ("bursty", "time_varying"),
    "RS": ("realistic_bursty", "stationary"),
    "RV": ("realistic_bursty", "time_varying"),
}


@dataclass(frozen=True)
class InstanceSpec:
    key: str
    corpus: str
    scenario: str
    scale: int
    dod: float
    seed: int
    temporal_mode: str
    spatial_mode: str
    burst_count: int | None = None
    burst_width_range: tuple[float, float] | None = None
    burst_strength_range: tuple[float, float] | None = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _matched_specs() -> list[InstanceSpec]:
    specs = []
    position = 0
    for scenario, (temporal, spatial) in SCENARIOS.items():
        for scale in (20, 50):
            for dod in (0.5, 0.9):
                specs.append(
                    InstanceSpec(
                        key=f"{scenario.lower()}-n{scale}-d{int(dod * 100)}-r0",
                        corpus="matched6",
                        scenario=scenario,
                        scale=scale,
                        dod=dod,
                        seed=310_000 + position * 100,
                        temporal_mode=temporal,
                        spatial_mode=spatial,
                    )
                )
                position += 1
    return specs


def _domain_randomized_specs(count: int = 24) -> list[InstanceSpec]:
    rng = random.Random(20260818)
    specs = []
    modes = list(SCENARIOS.values())
    for position in range(count):
        temporal, spatial = rng.choice(modes)
        scale = rng.choice((20, 50))
        dod = rng.choice((0.3, 0.5, 0.7, 0.9))
        specs.append(
            InstanceSpec(
                key=f"dr-n{scale}-d{int(dod * 100)}-r{position}",
                corpus="domain-randomized",
                scenario="DR",
                scale=scale,
                dod=dod,
                seed=410_000 + position * 100,
                temporal_mode=temporal,
                spatial_mode=spatial,
                burst_count=(rng.randint(2, 6) if temporal != "uniform" else None),
                burst_width_range=(
                    (10.0, 90.0) if temporal != "uniform" else None
                ),
                burst_strength_range=(
                    (0.40, 0.98) if temporal != "uniform" else None
                ),
            )
        )
    return specs


def _test_specs() -> list[InstanceSpec]:
    return [
        InstanceSpec(
            key=f"test-{scenario.lower()}-n200-d90-r0",
            corpus="large-highdod-test",
            scenario=scenario,
            scale=200,
            dod=0.9,
            seed=510_000 + position * 100,
            temporal_mode=temporal,
            spatial_mode=spatial,
        )
        for position, (scenario, (temporal, spatial)) in enumerate(SCENARIOS.items())
    ]


def _generate_strict(
    spec: InstanceSpec,
    output: Path,
    interval: int,
    speed: float,
    max_attempts: int,
) -> tuple[dict[str, Any], int]:
    if output.is_file():
        raw = json.loads(output.read_text(encoding="utf-8"))
        config = raw.get("generation_config") or {}
        if config.get("immediate_reachability_constraint") != (
            "available_time + min_depot_travel <= tw_end"
        ):
            raise ValueError(f"Existing instance lacks strict metadata: {output}")
        problem = problem_from_benchmark_json(raw, interval, speed)
        if individually_unserviceable_customers(problem):
            raise ValueError(f"Existing instance is infeasible after epoch rounding: {output}")
        return raw, int(config.get("seed", spec.seed))

    output.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    for attempt in range(max_attempts):
        seed = spec.seed + attempt
        generator = SpatioTemporalSVRPGenerator(
            num_customers=spec.scale,
            map_size=1000,
            dod=spec.dod,
            fixed_vehicle_num=200,
            fixed_vehicle_capacity=500,
            depot_type="single",
            depot_placement="center",
            service_time_mode="linear",
            seed=seed,
            temporal_mode=spec.temporal_mode,
            spatial_mode=spec.spatial_mode,
            burst_count=spec.burst_count,
            burst_width_range=spec.burst_width_range,
            burst_strength_range=spec.burst_strength_range,
            feasibility_vehicle_speed=speed,
        )
        try:
            generator.generate_all()
        except ImmediateReachabilityError as exc:
            failures.append(f"seed={seed}: raw={exc}")
            continue
        generator.save_dataset(str(output))
        raw = json.loads(output.read_text(encoding="utf-8"))
        problem = problem_from_benchmark_json(raw, interval, speed)
        rounded_failures = individually_unserviceable_customers(problem)
        if rounded_failures:
            output.unlink()
            failures.append(f"seed={seed}: epoch-rounded={list(rounded_failures)}")
            continue
        return raw, seed
    raise RuntimeError(
        f"Unable to generate {spec.key} after {max_attempts} attempts; "
        + " | ".join(failures[-5:])
    )


def _export_corpus(
    specs: list[InstanceSpec],
    corpus_root: Path,
    hgs_binary: Path,
    interval: int,
    speed: float,
    oracle_seconds: int,
    small_oracle_seconds: int,
    max_attempts: int,
) -> list[dict[str, Any]]:
    records = []
    for position, spec in enumerate(specs, start=1):
        source = corpus_root / "source_instances" / f"{spec.key}.json"
        raw, accepted_seed = _generate_strict(
            spec, source, interval, speed, max_attempts
        )
        run_dir = corpus_root / "oracle_solutions" / f"{spec.key}_releaseaware"
        metadata_path = run_dir / "metadata.json"
        horizon_seconds = int(
            (raw.get("generation_config") or {}).get("horizon", 1440) * 60
        )
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        else:
            problem = problem_from_benchmark_json(raw, interval, speed)
            instance_oracle_seconds = (
                small_oracle_seconds if spec.scale <= 20 else oracle_seconds
            )
            solution = solve_release_aware_hgs(
                problem,
                hgs_binary,
                time_limit_seconds=instance_oracle_seconds,
                seed=accepted_seed,
            )
            exported = export_official_mlco_layout(
                raw, problem, solution, corpus_root, instance_key=spec.key
            )
            metadata = exported["metadata"]
        for observation_path in run_dir.glob("observation_*.json"):
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
            if observation.get("benchmark_horizon_seconds") != horizon_seconds:
                observation["benchmark_horizon_seconds"] = horizon_seconds
                observation_path.write_text(
                    json.dumps(observation, separators=(",", ":")),
                    encoding="utf-8",
                )
        records.append(
            {
                **asdict(spec),
                "accepted_seed": accepted_seed,
                "source": str(source),
                "source_sha256": _sha256(source),
                "instance_file": str(corpus_root / "instances" / f"{spec.key}.txt"),
                "oracle_directory": str(run_dir),
                "num_observations": metadata["num_observations"],
                "oracle_distance_seconds": metadata["objective_driving_seconds"],
                "oracle_solver_time_limit_seconds": metadata.get(
                    "solver_time_limit_seconds"
                ),
                "oracle_solver_runtime_seconds": metadata[
                    "solver_runtime_seconds"
                ],
                "raw_strict_reachability": True,
                "epoch_rounded_individual_reachability": True,
            }
        )
        print(f"[{position}/{len(specs)}] prepared {spec.key}")
    return records


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise ValueError(f"Training-view collision: {destination}")
        return
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


def _build_view(
    root: Path,
    name: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    view = root / name
    for record in records:
        key = record["key"]
        _link(
            Path(record["instance_file"]),
            view / "instances" / f"{key}.txt",
        )
        _link(
            Path(record["oracle_directory"]),
            view / "oracle_solutions" / f"{key}_releaseaware",
        )
    manifest = {
        "view": name,
        "num_instances": len(records),
        "scenario_counts": {
            scenario: sum(record["scenario"] == scenario for record in records)
            for scenario in sorted({record["scenario"] for record in records})
        },
        "keys": [record["key"] for record in records],
    }
    (view / "view_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--hgs-binary", required=True, type=Path)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--speed", type=float, default=5.0)
    parser.add_argument("--oracle-seconds", type=int, default=1)
    parser.add_argument("--small-oracle-seconds", type=int, default=5)
    parser.add_argument("--max-attempts", type=int, default=200)
    args = parser.parse_args()

    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    matched = _export_corpus(
        _matched_specs(),
        root / "corpora" / "matched6",
        args.hgs_binary,
        args.interval,
        args.speed,
        args.oracle_seconds,
        args.small_oracle_seconds,
        args.max_attempts,
    )
    randomized = _export_corpus(
        _domain_randomized_specs(),
        root / "corpora" / "domain-randomized",
        args.hgs_binary,
        args.interval,
        args.speed,
        args.oracle_seconds,
        args.small_oracle_seconds,
        args.max_attempts,
    )

    views = [_build_view(root / "training_views", "mixed6", matched)]
    views.append(
        _build_view(
            root / "training_views", "domain-randomized", randomized
        )
    )
    for held in SCENARIOS:
        views.append(
            _build_view(
                root / "training_views",
                f"single-{held.lower()}",
                [record for record in matched if record["scenario"] == held],
            )
        )
        views.append(
            _build_view(
                root / "training_views",
                f"loo-{held.lower()}",
                [record for record in matched if record["scenario"] != held],
            )
        )

    test_records = []
    for spec in _test_specs():
        source = root / "large_highdod_test" / f"{spec.key}.json"
        raw, accepted_seed = _generate_strict(
            spec, source, args.interval, args.speed, args.max_attempts
        )
        test_records.append(
            {
                **asdict(spec),
                "accepted_seed": accepted_seed,
                "source": str(source),
                "source_sha256": _sha256(source),
                "num_customers": len(raw["customers"]),
            }
        )

    manifest = {
        "protocol": "mlco-generalization-pilot-v1",
        "feature_set": "dynamic",
        "interval_minutes": args.interval,
        "vehicle_speed": args.speed,
        "oracle_time_limit_seconds": args.oracle_seconds,
        "small_instance_oracle_time_limit_seconds": args.small_oracle_seconds,
        "generator_file": str(DATA_GENERATION / "svrp_generator_spatiotemporal.py"),
        "generator_sha256": _sha256(
            DATA_GENERATION / "svrp_generator_spatiotemporal.py"
        ),
        "matched_records": matched,
        "domain_randomized_records": randomized,
        "training_views": views,
        "large_highdod_test": test_records,
        "notes": [
            "Training and test seeds are disjoint.",
            "All raw instances satisfy strict immediate reachability.",
            "Training instances additionally remain individually feasible after 30-minute release rounding.",
            "The release-aware oracle uses unlimited routes and is not a finite-fleet dynamic optimum.",
        ],
    }
    (root / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"manifest": str(root / "protocol_manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
