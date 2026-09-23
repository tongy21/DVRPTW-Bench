# Dataset card

## Scope

The strict-v2 benchmark is a factorial DVRPTW dataset that varies temporal
demand concentration, spatial hotspot dynamics, nominal customer count, and
degree of dynamism. It is intended for controlled evaluation of dynamic
routing, dispatch, and anticipatory decision methods.

## Matrix

- Scenarios: US, UV, BS, BV, RS, RV.
- Customer counts: 20, 50, 100, 200, 500.
- Degrees of dynamism: 0, 0.2, 0.5, 0.8, 0.9, 0.95.
- Replicas: 0 through 9.
- Total problem instances: 1,800.

## Common parameters

- Map: `[0, 1000] x [0, 1000]`.
- Day horizon: 1,440 minutes.
- Latent release horizon: 1,200 minutes.
- Time-window lead: 30--120 minutes.
- Time-window width: 60--120 minutes.
- Dynamic reaction buffer: 30 minutes.
- Vehicle count: 200.
- Vehicle capacity: 500.
- Vehicle speed: 5 distance units per minute.
- Depot: one depot near the map center.
- Demand: discrete uniform on 1 through 100.
- Service time: `round(1 + 0.1 * demand)`.

## Integrity policy

Every customer satisfies strict immediate reachability:

```text
available_time + ceil(min_depot_distance / vehicle_speed) <= tw_end
```

Candidate instances that fail reachability are rejected rather than repaired.
The fixed-30 screening run must serve at least 95 percent of customers.

### Historical duplicate exceptions

The original formal replica block retains two documented same-scenario
duplicates caused by the early `replica * 100` seed stride overlapping after
repeated candidate rejection:

- `US/n500/dod90/replica0.json` and `replica1.json`;
- `RS/n500/dod95/replica3.json` and `replica4.json`.

They are retained unchanged because this repository publishes the exact
instances used by the reported experiments. The extra-5 block uses a
non-overlapping seed stride and has no within-scenario duplicates.

## Exclusions

The repository does not include trained model checkpoints, result logs,
machine-local source paths, credentials, or screening images. Those artifacts
are not part of the canonical problem definition.
