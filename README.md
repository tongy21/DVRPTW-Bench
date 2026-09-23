# DVRPTW dataset

This repository provides the dataset and implementation used to study dynamic
vehicle routing with time windows under different temporal arrival profiles
and spatial demand patterns. It contains 1,800 single-depot problem instances
and the code framework for data generation, dynamic simulation, and the
evaluated routing methods.

## Dataset design

| Dimension | Values |
|---|---|
| Scenario | `US`, `RS`, `BS`, `UV`, `RV`, `BV` |
| Customers | `20`, `50`, `100`, `200`, `500` |
| Degree of dynamism | `0`, `0.2`, `0.5`, `0.8`, `0.9`, `0.95` |
| Replica | `0`--`9` |

Each of the 180 scenario--scale--DoD cells contains ten replicas. The six
scenario codes combine three temporal profiles with two spatial profiles:

| Code | Temporal profile | Spatial profile |
|---|---|---|
| `US` | Uniform | Stationary |
| `RS` | Double peak | Stationary |
| `BS` | Strong burst | Stationary |
| `UV` | Uniform | Time-varying hotspots |
| `RV` | Double peak | Time-varying hotspots |
| `BV` | Strong burst | Time-varying hotspots |

Customer locations are sampled from a scale-dependent Gaussian mixture. Each
request records its coordinates, demand, service duration, time window, release
time, dynamic/static status, temporal profile, and spatial cluster. Instances
use one depot, 200 vehicles of capacity 500, a 1,440-minute operating horizon,
and a vehicle speed of five distance units per minute.

## Repository structure

```text
data/
  US/
    n20/
      dod0/
        replica0.json
        ...
  RS/
  BS/
  UV/
  RV/
  BV/
metadata/
  instances.csv
src/
  data_generation/
  simulation/
  solvers/
  experimental_solvers/icd_mlco/
  release_aware_mlco/
requirements.txt
```

`metadata/instances.csv` provides one row per instance with its scenario,
temporal and spatial profiles, scale, DoD, replica, customer counts, minimum
reachability slack, and repository-relative path.

## Code framework

The source tree includes:

- the spatiotemporal instance generator and batch-generation entry points;
- the rolling-horizon simulator and common route-validation logic;
- HGS, OR-Tools, ALNS, ACO, tabu search, Lin--Kernighan, and nearest-neighbor
  with 2-opt solvers;
- RL4CO training and inference support for Attention, POMO, SymNCO, and
  PolyNet;
- ML-CO feature extraction, training, NumPy inference, and PC-HGS integration;
- ICD future-scenario sampling and iterative dispatch; and
- the release-aware HGS offline reference.

The default ICD configuration uses `instance_config` future sampling. Both the
sampled-scenario lookahead stage and the final routing stage use unmodified HGS
with a one-second time budget.

## Running the code

Install the Python dependencies and expose `src` on `PYTHONPATH`:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=src
```

Python 3.10 or newer is recommended. Build the vendored PC-HGS source with:

```bash
cmake -S src/experimental_solvers/icd_mlco/vendor/pchgs \
  -B src/experimental_solvers/icd_mlco/vendor/pchgs/build \
  -DCMAKE_BUILD_TYPE=Release
cmake --build src/experimental_solvers/icd_mlco/vendor/pchgs/build --parallel
```

The full dataset matrix can be generated with:

```bash
python -m data_generation.generate_dataset --output-root data
```

For example, run ICD on one instance with the repository defaults:

```bash
python -m experimental_solvers.icd_mlco.runner \
  --solver icd \
  --data-file data/US/n100/dod80/replica0.json \
  --output-dir results
```

The ML-CO adapter uses the vendored PC-HGS source under
`src/experimental_solvers/icd_mlco/vendor/pchgs`. Its Python configuration
accepts paths to trained model weights when ML-CO inference is requested. The
RL training entry points are `src/solvers/rl/train.py` and
`src/solvers/rl/train_mixed_scale.py`.
