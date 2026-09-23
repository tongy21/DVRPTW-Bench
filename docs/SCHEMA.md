# Problem JSON schema

Each JSON document describes one single-depot DVRPTW instance. The canonical
solver-facing fields include the fleet, depot, customers, and generator
metadata.

Important customer fields include:

- `x`, `y`: integer coordinates;
- `demand`: customer demand;
- `service_time`: service duration;
- `tw_start`, `tw_end`: service-start time window;
- `available_time`: request release time;
- `is_dynamic`: whether the request is released after time zero.

Generator metadata records the temporal and spatial mode, random seed, and
strict reachability settings used to construct the instance.

The release layout renames only directory and file paths. JSON payloads remain
byte-for-byte identical to the audited source files.
