#!/usr/bin/env python3
"""Build the canonical release tree from formal and extra-5 source roots."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "US": ("dataset_final_uniform_stationary_strict_v2", "dataset_final_uniform_stationary_strict_v2_extra5"),
    "UV": ("dataset_final_uniform_spatialvarying_balanced_strict_v2", "dataset_final_uniform_spatialvarying_balanced_strict_v2_extra5"),
    "BS": ("dataset_final_bursty_stationary_strong_strict_v2", "dataset_final_bursty_stationary_strong_strict_v2_extra5"),
    "BV": ("dataset_final_bursty_spatialvarying_strong_strict_v2", "dataset_final_bursty_spatialvarying_strong_strict_v2_extra5"),
    "RS": ("dataset_final_realistic_bursty_stationary_v2_strict_v2", "dataset_final_realistic_bursty_stationary_v2_strict_v2_extra5"),
    "RV": ("dataset_final_realistic_bursty_spatialvarying_v2_strict_v2", "dataset_final_realistic_bursty_spatialvarying_v2_strict_v2_extra5"),
}
PATTERN = re.compile(r"^(20|50|100|200|500)_single_depot_dod(0|20|50|80|90|95)_([0-9])$")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--extra-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for scenario, (formal_name, extra_name) in SCENARIOS.items():
        for source_root, directory, expected_replicas, block in (
            (args.formal_root, formal_name, set(range(5)), "formal"),
            (args.extra_root, extra_name, set(range(5, 10)), "extra5"),
        ):
            for source in sorted((source_root / directory).glob("dataset_final_*/*.json")):
                if source.name.endswith("_viz_data.json"):
                    continue
                match = PATTERN.match(source.stem)
                if not match:
                    continue
                scale, dod, replica = map(int, match.groups())
                if replica not in expected_replicas:
                    continue
                target = ROOT / "data" / scenario / f"n{scale}" / f"dod{dod}" / f"replica{replica}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                relative = str(target.relative_to(ROOT))
                rows.append({
                    "scenario": scenario, "scale": scale, "dod": dod / 100,
                    "replica": replica, "replica_block": block,
                    "path": relative, "bytes": target.stat().st_size,
                })
    if len(rows) != 1800:
        raise RuntimeError(f"Expected 1800 instances, copied {len(rows)}")
    metadata = ROOT / "metadata"
    metadata.mkdir(exist_ok=True)
    with (metadata / "instances.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["scenario"], row["scale"], row["dod"], row["replica"])))


if __name__ == "__main__":
    main()
