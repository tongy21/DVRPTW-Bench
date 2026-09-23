# Generation overview

Customer geometry is sampled from a scale-dependent Gaussian mixture. The
number of clusters is `max(1, floor(n / 50))`. Temporal profiles determine the
latent demand times and time windows, while the spatial mode controls how
dynamic profiles are assigned to the fixed customer geometry.

The six scenarios combine three temporal profiles with stationary or
time-varying spatial demand:

- Uniform: releases spread across the release horizon.
- Strong burst: most releases concentrate around four narrow burst centers.
- Realistic double peak: releases concentrate around two broader peaks.
- Stationary: hotspot assignment does not change over the day.
- Time-varying: active hotspot assignments change across four periods.

The original formal block contains replicas 0--4. A second independently
generated block contains replicas 5--9. Both blocks use the same factorial
design and strict reachability checks.
