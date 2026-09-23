#!/usr/bin/env python3
"""Generate the complete DVRPTW dataset matrix."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

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


SCENARIOS = {
    "US": ("uniform", "stationary"),
    "RS": ("realistic_bursty", "stationary"),
    "BS": ("bursty", "stationary"),
    "UV": ("uniform", "time_varying"),
    "RV": ("realistic_bursty", "time_varying"),
    "BV": ("bursty", "time_varying"),
}
DEFAULT_SCALES = (20, 50, 100, 200, 500)
DEFAULT_DODS = (0.0, 0.2, 0.5, 0.8, 0.9, 0.95)


def parse_values(value: str, cast):
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def candidate_seed(
    base_seed: int, scale_position: int, dod_position: int, replica: int, attempt: int
) -> int:
    """Return a collision-free seed for up to 999 attempts per cell."""
    return (
        int(base_seed)
        + scale_position * 1_000_000
        + dod_position * 10_000
        + replica * 1_000
        + attempt
    )


def generate(args: argparse.Namespace) -> None:
    scales = parse_values(args.scales, int)
    dods = parse_values(args.dods, float)
    scenarios = parse_values(args.scenarios, str)
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")
    if not 1 <= args.replicas <= 10:
        raise ValueError("replicas must be between 1 and 10")
    if not 1 <= args.max_attempts < 1_000:
        raise ValueError("max-attempts must be between 1 and 999")

    output_root = args.output_root.expanduser().resolve()
    total = len(scenarios) * len(scales) * len(dods) * args.replicas
    completed = 0
    for scenario in scenarios:
        temporal_mode, spatial_mode = SCENARIOS[scenario]
        for scale_position, scale in enumerate(scales):
            for dod_position, dod in enumerate(dods):
                dod_pct = int(round(100 * dod))
                for replica in range(args.replicas):
                    output = (
                        output_root / scenario / f"n{scale}" / f"dod{dod_pct}"
                        / f"replica{replica}.json"
                    )
                    if output.is_file() and not args.overwrite:
                        completed += 1
                        print(f"[{completed}/{total}] skip {output}", flush=True)
                        continue
                    accepted = False
                    for attempt in range(1, args.max_attempts + 1):
                        seed = candidate_seed(
                            args.base_seed, scale_position, dod_position, replica, attempt
                        )
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
                            feasibility_vehicle_speed=args.vehicle_speed,
                        )
                        try:
                            generator.generate_all()
                        except ImmediateReachabilityError:
                            continue
                        descriptor, temporary_name = tempfile.mkstemp(
                            prefix="dvrptw_candidate_", suffix=".json"
                        )
                        os.close(descriptor)
                        temporary = Path(temporary_name)
                        try:
                            generator.save_dataset(str(temporary))
                            validator = DynamicValidatorInfCommit(
                                data_file=str(temporary),
                                optimization_interval=30,
                                vehicle_speed=args.vehicle_speed,
                                verbose=False,
                            )
                            if validator.run_simulation() < args.min_service_rate:
                                continue
                            output.parent.mkdir(parents=True, exist_ok=True)
                            os.replace(temporary, output)
                            accepted = True
                            completed += 1
                            print(
                                f"[{completed}/{total}] generated {output} seed={seed}",
                                flush=True,
                            )
                            break
                        finally:
                            temporary.unlink(missing_ok=True)
                    if not accepted:
                        raise RuntimeError(
                            f"No accepted candidate for {scenario}, n={scale}, "
                            f"DoD={dod}, replica={replica}"
                        )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-root", type=Path, default=Path("data"))
    result.add_argument("--scenarios", default=",".join(SCENARIOS))
    result.add_argument("--scales", default=",".join(map(str, DEFAULT_SCALES)))
    result.add_argument("--dods", default=",".join(map(str, DEFAULT_DODS)))
    result.add_argument("--replicas", type=int, default=10)
    result.add_argument("--min-service-rate", type=float, default=95.0)
    result.add_argument("--vehicle-speed", type=float, default=5.0)
    result.add_argument("--max-attempts", type=int, default=300)
    result.add_argument("--base-seed", type=int, default=20260730)
    result.add_argument("--overwrite", action="store_true")
    return result


if __name__ == "__main__":
    generate(parser().parse_args())
