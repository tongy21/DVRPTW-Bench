# DVRPTW Strict-v2 Dataset

This repository contains the strict-v2 dynamic vehicle-routing benchmark used
for the DVRPTW experiments. The release covers six spatiotemporal demand
regimes, five nominal instance sizes, six degrees of dynamism, and ten
replicas.

## Release matrix

| Dimension | Values |
|---|---|
| Scenario | `US`, `UV`, `BS`, `BV`, `RS`, `RV` |
| Customers | `20`, `50`, `100`, `200`, `500` |
| DoD | `0`, `0.2`, `0.5`, `0.8`, `0.9`, `0.95` |
| Replica | `0`--`9` |

The complete release contains 1,800 problem JSON files. Replicas `0`--`4`
come from the original formal block and replicas `5`--`9` from the independently
generated extra-5 block.

## Repository layout

```text
data/
  US/
    n20/dod0/replica0.json
    ...
  UV/
  BS/
  BV/
  RS/
  RV/
docs/
  DATASET_CARD.md
  GENERATION.md
  SCHEMA.md
metadata/
  instances.csv
scripts/
  build_release.py
```

Only canonical problem JSON files belong under `data/`. Screening
visualizations, solver outputs, paper figures, checkpoints, and machine-local
paths are intentionally excluded.

## Scenario codes

| Code | Temporal demand | Spatial demand |
|---|---|---|
| `US` | Uniform | Stationary |
| `UV` | Uniform | Time-varying hotspots |
| `BS` | Strong burst | Stationary |
| `BV` | Strong burst | Time-varying hotspots |
| `RS` | Realistic double peak | Stationary |
| `RV` | Realistic double peak | Time-varying hotspots |

## Citation and license

Citation metadata and the release license will be finalized before the
repository is made public.
