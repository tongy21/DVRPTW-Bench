"""Batch generation for a selected temporal-spatial SDVRP regime.

The output layout mirrors ``dataset/dataset_final_burst``:

    dataset_final_<dod>/
        <scale>_single_depot_dod<percent>_<replica>.json
        <scale>_single_depot_dod<percent>_<replica>_viz_data.json
        <scale>_single_depot_dod<percent>_<replica>_solution.png
        <scale>_single_depot_dod<percent>_<replica>_solution.svg

The script is resumable: a replica is skipped only when all four files exist.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

try:
    from .dynamic_validator_inf import DynamicValidatorInfCommit
    from .svrp_generator_spatiotemporal import (
        ImmediateReachabilityError,
        SpatioTemporalSVRPGenerator,
    )
except ImportError:  # Support direct execution from this directory.
    from dynamic_validator_inf import DynamicValidatorInfCommit
    from svrp_generator_spatiotemporal import (
        ImmediateReachabilityError,
        SpatioTemporalSVRPGenerator,
    )


DEFAULT_SCALES = [20, 50, 100, 200, 500]
DEFAULT_DODS = [0.0, 0.2, 0.5, 0.8, 0.9, 0.95]


def _parse_list(text: str, cast) -> List:
    return [cast(item.strip()) for item in text.split(",") if item.strip()]


def _expected_paths(output_dir: Path, stem: str) -> List[Path]:
    return [
        output_dir / f"{stem}.json",
        output_dir / f"{stem}_viz_data.json",
        output_dir / f"{stem}_solution.png",
        output_dir / f"{stem}_solution.svg",
    ]


def _instance_seed(
    base_seed: int,
    scale_position: int,
    dod_position: int,
    replica: int,
    attempt: int,
) -> int:
    return (
        int(base_seed)
        + scale_position * 1_000_000
        + dod_position * 10_000
        + replica * 1_000
        + attempt
    )


def generate_dataset(
    output_root: Path,
    scales: Iterable[int],
    dods: Iterable[float],
    replicas: int,
    target_service_rate: float,
    max_attempts: int,
    base_seed: int,
    temporal_mode: str,
    spatial_mode: str,
    burst_count: Optional[int],
    burst_width: Optional[float],
    burst_strength: Optional[float],
    burst_width_range: Optional[Sequence[float]] = None,
    burst_strength_range: Optional[Sequence[float]] = None,
    vehicle_speed: float = 5.0,
) -> None:
    scales = list(scales)
    dods = list(dods)
    output_root.mkdir(parents=True, exist_ok=True)

    total_instances = len(scales) * len(dods) * replicas
    completed = 0
    print(
        f"Generating {total_instances} instances under {output_root} "
        f"({temporal_mode} + {spatial_mode})"
    )

    for dod_position, dod in enumerate(dods):
        dod_dir = output_root / f"dataset_final_{dod}"
        dod_dir.mkdir(parents=True, exist_ok=True)

        for scale_position, scale in enumerate(scales):
            for replica in range(replicas):
                stem = (
                    f"{scale}_single_depot_dod{int(round(dod * 100))}_{replica}"
                )
                expected = _expected_paths(dod_dir, stem)
                if all(path.exists() for path in expected):
                    completed += 1
                    print(
                        f"[{completed}/{total_instances}] SKIP complete: "
                        f"scale={scale}, dod={dod}, replica={replica}"
                    )
                    continue

                accepted = False
                for attempt in range(1, max_attempts + 1):
                    seed = _instance_seed(
                        base_seed,
                        scale_position,
                        dod_position,
                        replica,
                        attempt,
                    )
                    temp_fd, temp_name = tempfile.mkstemp(
                        prefix="sdvrp_spatiotemporal_",
                        suffix=".json",
                    )
                    os.close(temp_fd)
                    temp_json = Path(temp_name)

                    try:
                        generator = SpatioTemporalSVRPGenerator(
                            num_customers=scale,
                            map_size=1000,
                            dod=dod,
                            fixed_vehicle_num=200,
                            fixed_vehicle_capacity=500,
                            depot_type="single",
                            depot_placement="center",
                            service_time_mode="linear",
                            seed=seed,
                            verbose=False,
                            temporal_mode=temporal_mode,
                            spatial_mode=spatial_mode,
                            burst_count=burst_count,
                            burst_width=burst_width,
                            burst_strength=burst_strength,
                            burst_width_range=burst_width_range,
                            burst_strength_range=burst_strength_range,
                            feasibility_vehicle_speed=vehicle_speed,
                        )
                        try:
                            generator.generate_all()
                        except ImmediateReachabilityError as error:
                            print(
                                f"  attempt={attempt}, seed={seed}, "
                                f"rejected={error}"
                            )
                            continue
                        generator.save_dataset(str(temp_json))

                        validator = DynamicValidatorInfCommit(
                            data_file=str(temp_json),
                            optimization_interval=30,
                            vehicle_speed=vehicle_speed,
                            verbose=False,
                        )
                        service_rate = validator.run_simulation()
                        print(
                            f"  attempt={attempt}, seed={seed}, "
                            f"service_rate={service_rate:.2f}%"
                        )
                        if service_rate < target_service_rate:
                            continue

                        final_json, final_viz, final_png, _ = expected
                        shutil.copy2(temp_json, final_json)
                        validator.save_results(
                            str(final_viz),
                            str(final_png),
                            time_display=False,
                            legend_display=False,
                        )

                        missing = [path for path in expected if not path.exists()]
                        if missing:
                            raise RuntimeError(
                                "Validator did not create expected files: "
                                + ", ".join(str(path) for path in missing)
                            )

                        accepted = True
                        completed += 1
                        print(
                            f"[{completed}/{total_instances}] ACCEPT: "
                            f"scale={scale}, dod={dod}, replica={replica}, "
                            f"seed={seed}, service_rate={service_rate:.2f}%"
                        )
                        break
                    finally:
                        if temp_json.exists():
                            temp_json.unlink()

                if not accepted:
                    raise RuntimeError(
                        f"Could not generate an accepted instance after "
                        f"{max_attempts} attempts: scale={scale}, dod={dod}, "
                        f"replica={replica}"
                    )

    print(f"Generation complete: {completed}/{total_instances} instances.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a full temporal-spatial dataset with the same "
            "directory and artifact structure as dataset_final_burst."
        )
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("dataset/dataset_final_bursty_spatialvarying"),
    )
    parser.add_argument(
        "--scales",
        default="20,50,100,200,500",
        help="Comma-separated customer counts.",
    )
    parser.add_argument(
        "--dods",
        default="0.0,0.2,0.5,0.8,0.9,0.95",
        help="Comma-separated degrees of dynamism.",
    )
    parser.add_argument("--replicas", type=int, default=10)
    parser.add_argument("--target-service-rate", type=float, default=95.0)
    parser.add_argument(
        "--vehicle-speed",
        type=float,
        default=5.0,
        help=(
            "Shared speed for generation feasibility and dynamic validation."
        ),
    )
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--base-seed", type=int, default=20260730)
    parser.add_argument(
        "--temporal-mode",
        choices=["uniform", "bursty", "realistic_bursty"],
        default="bursty",
    )
    parser.add_argument(
        "--spatial-mode",
        choices=["stationary", "time_varying"],
        default="time_varying",
    )
    parser.add_argument(
        "--burst-count",
        type=int,
        default=None,
        help=(
            "Optional burst-center override. Defaults: bursty=4; "
            "realistic_bursty=two empirical lunch/dinner peaks."
        ),
    )
    parser.add_argument(
        "--burst-width",
        type=float,
        default=None,
        help=(
            "Optional Gaussian-width override. Defaults: bursty=15 minutes; "
            "realistic_bursty samples separate lunch/dinner widths."
        ),
    )
    parser.add_argument(
        "--burst-width-range",
        default=None,
        help="Optional instance-level Gaussian-width range as min,max.",
    )
    parser.add_argument(
        "--burst-strength",
        type=float,
        default=None,
        help=(
            "Optional burst-mixture override. Defaults: bursty=0.95; "
            "realistic_bursty samples empirical lunch/dinner window shares."
        ),
    )
    parser.add_argument(
        "--burst-strength-range",
        default=None,
        help="Optional instance-level burst-mixture range as min,max.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    generate_dataset(
        output_root=args.output_root,
        scales=_parse_list(args.scales, int),
        dods=_parse_list(args.dods, float),
        replicas=args.replicas,
        target_service_rate=args.target_service_rate,
        max_attempts=args.max_attempts,
        base_seed=args.base_seed,
        vehicle_speed=args.vehicle_speed,
        temporal_mode=args.temporal_mode,
        spatial_mode=args.spatial_mode,
        burst_count=args.burst_count,
        burst_width=args.burst_width,
        burst_strength=args.burst_strength,
        burst_width_range=(
            None
            if args.burst_width_range is None
            else _parse_list(args.burst_width_range, float)
        ),
        burst_strength_range=(
            None
            if args.burst_strength_range is None
            else _parse_list(args.burst_strength_range, float)
        ),
    )


if __name__ == "__main__":
    main()
