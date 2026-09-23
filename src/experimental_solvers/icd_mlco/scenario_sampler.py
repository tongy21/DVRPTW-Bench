from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

import numpy as np


class OnlineScenarioSampler:
    """Samples future requests under an explicit information boundary.

    The fair modes never retain unrevealed customer rows. ``realized_future``
    is intentionally different: it is an oracle ablation used only to measure
    the value of perfect future information.
    """

    KNOWLEDGE_MODES = {
        "none",
        "observed_history",
        "scenario_family",
        "instance_peak_centers",
        "instance_temporal",
        "instance_hotspot_schedule",
        "instance_spatial",
        "instance_peak_centers_hotspot_schedule",
        "instance_peak_centers_spatial",
        "instance_temporal_hotspot_schedule",
        "instance_config",
        "realized_future",
    }

    def __init__(
        self,
        max_sampled_requests: int = 200,
        knowledge_mode: str = "scenario_family",
    ) -> None:
        self.max_sampled_requests = int(max_sampled_requests)
        if knowledge_mode not in self.KNOWLEDGE_MODES:
            raise ValueError(f"Unknown knowledge mode: {knowledge_mode}")
        self.knowledge_mode = knowledge_mode
        self.config: dict[str, Any] = {}
        self.dynamic_total = 0
        self.instance_name: str | None = None
        self._observed: dict[int, dict[str, Any]] = {}
        self._realized_future: tuple[dict[str, Any], ...] = ()
        self._release_minutes = np.asarray([1], dtype=int)
        self._release_pmf = np.asarray([1.0], dtype=float)

    def configure_instance(self, raw_data: dict[str, Any]) -> None:
        full_config = dict(raw_data.get("generation_config") or {})
        if self.knowledge_mode in {"instance_config", "realized_future"}:
            self.config = full_config
        elif self.knowledge_mode == "scenario_family":
            self.config = self._scenario_family_config(full_config)
        elif self.knowledge_mode == "instance_peak_centers":
            self.config = self._instance_peak_centers_config(full_config)
        elif self.knowledge_mode == "instance_temporal":
            self.config = self._instance_temporal_config(full_config)
        elif self.knowledge_mode == "instance_hotspot_schedule":
            self.config = self._instance_hotspot_schedule_config(full_config)
        elif self.knowledge_mode == "instance_spatial":
            self.config = self._instance_spatial_config(full_config)
        elif self.knowledge_mode == "instance_peak_centers_hotspot_schedule":
            self.config = self._combined_instance_config(
                full_config, temporal="centers", spatial="schedule"
            )
        elif self.knowledge_mode == "instance_peak_centers_spatial":
            self.config = self._combined_instance_config(
                full_config, temporal="centers", spatial="full"
            )
        elif self.knowledge_mode == "instance_temporal_hotspot_schedule":
            self.config = self._combined_instance_config(
                full_config, temporal="full", spatial="schedule"
            )
        else:
            self.config = {
                "horizon": int(full_config.get("horizon", 1440)),
                "release_horizon": int(full_config.get("release_horizon", 1200)),
                "temporal_mode": "observed_history",
                "spatial_mode": "observed_history",
            }
        self.instance_name = str(raw_data.get("name", "unknown"))
        stats = raw_data.get("generation_stats") or {}
        if "dynamic_count" in stats:
            self.dynamic_total = int(stats["dynamic_count"])
        else:
            total = len(raw_data.get("customers", []))
            dod = float(self.config.get("dod", 0.0))
            self.dynamic_total = int(round(total * dod))
        self._release_minutes, self._release_pmf = self._build_release_pmf()
        self._observed.clear()
        if self.knowledge_mode == "realized_future":
            self._realized_future = tuple(
                dict(customer)
                for customer in raw_data.get("customers", [])
                if bool(customer.get("is_dynamic", False))
            )
        else:
            self._realized_future = ()

    @staticmethod
    def _scenario_family_config(full_config: dict[str, Any]) -> dict[str, Any]:
        """Keep family-level assumptions but remove instance realizations."""
        temporal_mode = str(full_config.get("temporal_mode", "uniform")).replace(
            "-", "_"
        )
        release_horizon = int(full_config.get("release_horizon", 1200))
        result: dict[str, Any] = {
            "horizon": int(full_config.get("horizon", 1440)),
            "release_horizon": release_horizon,
            "dod": float(full_config.get("dod", 0.0)),
            "temporal_mode": temporal_mode,
            "spatial_mode": str(
                full_config.get("spatial_mode", "stationary")
            ).replace("-", "_"),
            "lead_time_range": list(full_config.get("lead_time_range", [30, 120])),
            "time_window_width_range": list(
                full_config.get("time_window_width_range", [60, 120])
            ),
        }
        if temporal_mode == "bursty":
            result.update(
                burst_count=4,
                burst_width=15.0,
                burst_strength=0.95,
            )
        elif temporal_mode == "realistic_bursty":
            scale = release_horizon / 1200.0
            result.update(
                burst_centers=[705.0 * scale, 1080.0 * scale],
                burst_widths=[55.0 * scale, 67.5 * scale],
                burst_component_shares=[0.30, 0.21],
                burst_strength=0.51,
                burst_peak_windows=[
                    [645.0 * scale, 780.0 * scale],
                    [1020.0 * scale, 1140.0 * scale],
                ],
                background_sampling="uniform_outside_peak_windows",
            )
        return result

    @staticmethod
    def _copy_matching_config(
        target: dict[str, Any],
        source: dict[str, Any],
        *,
        exact_keys: set[str],
        prefixes: tuple[str, ...],
    ) -> dict[str, Any]:
        result = dict(target)
        for key, value in source.items():
            if key in exact_keys or key.startswith(prefixes):
                result[key] = value
        return result

    @classmethod
    def _instance_peak_centers_config(
        cls, full_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Use exact peak locations but retain the family-level peak shape."""
        family = cls._scenario_family_config(full_config)
        return cls._copy_matching_config(
            family,
            full_config,
            exact_keys={"burst_centers"},
            prefixes=(),
        )

    @classmethod
    def _instance_temporal_config(
        cls, full_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Use exact release-process metadata but family-level spatial data."""
        family = cls._scenario_family_config(full_config)
        return cls._copy_matching_config(
            family,
            full_config,
            exact_keys={
                "temporal_mode",
                "background_sampling",
                "uniform_background_share",
                "release_horizon",
            },
            prefixes=("burst_", "realistic_calibration_"),
        )

    @classmethod
    def _instance_hotspot_schedule_config(
        cls, full_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Use only the exact hotspot identities for each spatial period."""
        family = cls._scenario_family_config(full_config)
        return cls._copy_matching_config(
            family,
            full_config,
            exact_keys={"hotspot_schedule"},
            prefixes=(),
        )

    @classmethod
    def _instance_spatial_config(
        cls, full_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Use exact hotspot metadata but family-level temporal data."""
        family = cls._scenario_family_config(full_config)
        return cls._copy_matching_config(
            family,
            full_config,
            exact_keys={"stationary_cluster_weights"},
            prefixes=("hotspot_", "spatial_"),
        )

    @classmethod
    def _combined_instance_config(
        cls,
        full_config: dict[str, Any],
        *,
        temporal: str,
        spatial: str,
    ) -> dict[str, Any]:
        """Compose temporal and spatial metadata levels independently."""
        if temporal == "centers":
            result = cls._instance_peak_centers_config(full_config)
        elif temporal == "full":
            result = cls._instance_temporal_config(full_config)
        else:
            raise ValueError(f"Unknown temporal metadata level: {temporal}")
        if spatial == "schedule":
            return cls._copy_matching_config(
                result,
                full_config,
                exact_keys={"hotspot_schedule"},
                prefixes=(),
            )
        if spatial == "full":
            return cls._copy_matching_config(
                result,
                full_config,
                exact_keys={"stationary_cluster_weights"},
                prefixes=("hotspot_", "spatial_"),
            )
        raise ValueError(f"Unknown spatial metadata level: {spatial}")

    def observe(self, orders: Sequence[dict[str, Any]]) -> None:
        for order in orders:
            order_id = int(order["id"])
            if order_id >= 0:
                self._observed[order_id] = dict(order)

    @property
    def observed_count(self) -> int:
        return len(self._observed)

    @property
    def observed_dynamic_count(self) -> int:
        return sum(bool(order.get("is_dynamic", False)) for order in self._observed.values())

    def _build_release_pmf(self) -> tuple[np.ndarray, np.ndarray]:
        release_horizon = max(int(self.config.get("release_horizon", 1200)), 1)
        minutes = np.arange(1, release_horizon + 1, dtype=int)
        temporal_mode = str(self.config.get("temporal_mode", "uniform")).replace("-", "_")
        if temporal_mode in {"uniform", "observed_history"}:
            return minutes, np.full(len(minutes), 1.0 / len(minutes))

        centers = [float(value) for value in self.config.get("burst_centers", [])]
        if not centers:
            burst_count = max(int(self.config.get("burst_count", 4)), 1)
            centers = np.linspace(
                release_horizon / (burst_count + 1),
                release_horizon - release_horizon / (burst_count + 1),
                burst_count,
            ).tolist()
        widths = self.config.get("burst_widths")
        if not widths:
            widths = [float(self.config.get("burst_width", 30.0))] * len(centers)
        widths = [max(float(value), 1e-3) for value in widths]
        if len(widths) < len(centers):
            widths.extend([widths[-1]] * (len(centers) - len(widths)))

        burst_strength = float(self.config.get("burst_strength", 0.70))
        component_shares = self.config.get("burst_component_shares")
        if component_shares and len(component_shares) == len(centers):
            shares = np.asarray(component_shares, dtype=float)
        else:
            shares = np.full(len(centers), burst_strength / len(centers))
        if shares.sum() > 0:
            shares *= burst_strength / shares.sum()

        background = np.ones(len(minutes), dtype=float)
        peak_windows = self.config.get("burst_peak_windows") or []
        if self.config.get("background_sampling") == "uniform_outside_peak_windows" and peak_windows:
            for lower, upper in peak_windows:
                background[(minutes >= float(lower)) & (minutes <= float(upper))] = 0.0
            if background.sum() == 0:
                background[:] = 1.0
        background /= background.sum()
        pmf = max(0.0, 1.0 - burst_strength) * background

        for idx, (center, width, share) in enumerate(zip(centers, widths, shares)):
            density = np.exp(-0.5 * ((minutes - center) / width) ** 2)
            if idx < len(peak_windows):
                lower, upper = peak_windows[idx]
                density[(minutes < float(lower)) | (minutes > float(upper))] = 0.0
            if density.sum() <= 0:
                density[np.argmin(np.abs(minutes - center))] = 1.0
            pmf += float(share) * density / density.sum()

        if pmf.sum() <= 0:
            pmf[:] = 1.0
        return minutes, pmf / pmf.sum()

    def _sample_release_times(
        self,
        rng: np.random.Generator,
        current_time: float,
        end_time: float,
    ) -> np.ndarray:
        if self.knowledge_mode == "none":
            return np.asarray([], dtype=int)
        if self.knowledge_mode == "observed_history":
            observed_dynamic = self.observed_dynamic_count
            if observed_dynamic <= 0 or current_time <= 0:
                return np.asarray([], dtype=int)
            window = max(float(end_time) - float(current_time), 0.0)
            expected = observed_dynamic / float(current_time) * window
            count = min(int(rng.poisson(expected)), self.max_sampled_requests)
            if count <= 0:
                return np.asarray([], dtype=int)
            return rng.integers(
                int(np.floor(current_time)) + 1,
                int(np.floor(end_time)) + 1,
                size=count,
            )

        future_mask = self._release_minutes > current_time
        window_mask = future_mask & (self._release_minutes <= end_time)
        remaining = max(self.dynamic_total - self.observed_dynamic_count, 0)
        future_mass = float(self._release_pmf[future_mask].sum())
        window_mass = float(self._release_pmf[window_mask].sum())
        if remaining == 0 or future_mass <= 0 or window_mass <= 0:
            return np.asarray([], dtype=int)

        probability = min(max(window_mass / future_mass, 0.0), 1.0)
        count = min(int(rng.binomial(remaining, probability)), self.max_sampled_requests)
        if count <= 0:
            return np.asarray([], dtype=int)
        candidates = self._release_minutes[window_mask]
        probabilities = self._release_pmf[window_mask]
        probabilities = probabilities / probabilities.sum()
        return rng.choice(candidates, size=count, replace=True, p=probabilities)

    def _known_cluster_ids(self) -> list[int]:
        result = {
            int(order.get("cluster_id", 0))
            for order in self._observed.values()
            if order.get("cluster_id") is not None
        }
        # Stationary instances also serialize a dormant hotspot schedule for
        # provenance.  It is not part of their active spatial process and must
        # not reveal the full cluster universe to the sampler.
        spatial_mode = str(
            self.config.get("spatial_mode", "stationary")
        ).replace("-", "_")
        if spatial_mode == "time_varying":
            for period in self.config.get("hotspot_schedule") or []:
                result.update(int(cluster) for cluster in period)
        stationary = self.config.get("stationary_cluster_weights")
        if stationary:
            result.update(range(len(stationary)))
        return sorted(result or {0})

    def _cluster_probabilities(self, release_time: float) -> tuple[list[int], np.ndarray]:
        clusters = self._known_cluster_ids()
        observed_counts = Counter(
            int(order.get("cluster_id", 0)) for order in self._observed.values()
        )
        base = np.asarray([observed_counts[cluster] + 1.0 for cluster in clusters], dtype=float)
        stationary = self.config.get("stationary_cluster_weights")
        if stationary and len(stationary) > max(clusters):
            base = np.asarray([float(stationary[cluster]) for cluster in clusters], dtype=float)
            base = np.maximum(base, 0.0)
        base = base / base.sum()

        if str(self.config.get("spatial_mode", "stationary")).replace("-", "_") != "time_varying":
            return clusters, base
        schedule = self.config.get("hotspot_schedule") or []
        if not schedule:
            return clusters, base
        boundaries = self.config.get("spatial_period_boundaries")
        if not boundaries:
            release_horizon = float(self.config.get("release_horizon", 1200))
            boundaries = np.linspace(0, release_horizon, len(schedule) + 1).tolist()
        period = int(np.searchsorted(np.asarray(boundaries[1:]), release_time, side="right"))
        period = min(max(period, 0), len(schedule) - 1)
        active = {int(cluster) for cluster in schedule[period]}
        hotspot = np.asarray([1.0 if cluster in active else 0.0 for cluster in clusters])
        if hotspot.sum() <= 0:
            return clusters, base
        hotspot /= hotspot.sum()
        strength = min(max(float(self.config.get("hotspot_strength", 0.8)), 0.0), 1.0)
        probabilities = (1.0 - strength) * base + strength * hotspot
        return clusters, probabilities / probabilities.sum()

    def _sample_location(
        self,
        rng: np.random.Generator,
        cluster_id: int,
    ) -> tuple[float, float, dict[str, Any] | None]:
        observed = list(self._observed.values())
        cluster_orders = [
            order for order in observed if int(order.get("cluster_id", 0)) == cluster_id
        ]
        pool = cluster_orders or observed
        if not pool:
            return 500.0, 500.0, None
        template = dict(pool[int(rng.integers(0, len(pool)))])
        xs = np.asarray([float(order["x"]) for order in pool])
        ys = np.asarray([float(order["y"]) for order in pool])
        global_x = np.asarray([float(order["x"]) for order in observed])
        global_y = np.asarray([float(order["y"]) for order in observed])
        fallback_x = max(float(np.std(global_x)) * 0.08, 1.0)
        fallback_y = max(float(np.std(global_y)) * 0.08, 1.0)
        jitter_x = max(float(np.std(xs)) * 0.20, fallback_x) if len(xs) > 1 else fallback_x
        jitter_y = max(float(np.std(ys)) * 0.20, fallback_y) if len(ys) > 1 else fallback_y
        x = float(template["x"]) + float(rng.normal(0.0, jitter_x))
        y = float(template["y"]) + float(rng.normal(0.0, jitter_y))
        if observed:
            x = float(np.clip(x, min(global_x) - 3 * jitter_x, max(global_x) + 3 * jitter_x))
            y = float(np.clip(y, min(global_y) - 3 * jitter_y, max(global_y) + 3 * jitter_y))
        return x, y, template

    @staticmethod
    def _integer_range(config: dict[str, Any], key: str, default: tuple[int, int]) -> tuple[int, int]:
        values = config.get(key, default)
        return int(values[0]), int(values[1])

    def sample(
        self,
        rng: np.random.Generator,
        current_time: float,
        lookahead_minutes: float,
    ) -> list[dict[str, Any]]:
        if self.knowledge_mode == "realized_future":
            visible_future = [
                order
                for order in self._realized_future
                if current_time < float(order.get("available_time", 0))
                <= current_time + lookahead_minutes
            ][: self.max_sampled_requests]
            result = []
            for order in visible_future:
                sampled = dict(order)
                sampled["id"] = -int(order["id"])
                sampled["is_sampled_scenario_request"] = True
                result.append(sampled)
            return result

        releases = self._sample_release_times(
            rng, current_time, current_time + lookahead_minutes
        )
        if len(releases) == 0:
            return []

        lead_min, lead_max = self._integer_range(
            self.config, "lead_time_range", (30, 120)
        )
        width_min, width_max = self._integer_range(
            self.config, "time_window_width_range", (60, 120)
        )
        horizon = int(self.config.get("horizon", 1440))
        observed = list(self._observed.values())
        demands = [int(order.get("demand", 1)) for order in observed] or [1]
        services = [int(order.get("service_time", 0)) for order in observed] or [0]

        sampled: list[dict[str, Any]] = []
        for pos, release in enumerate(releases):
            cluster_ids, probabilities = self._cluster_probabilities(float(release))
            cluster_id = int(rng.choice(cluster_ids, p=probabilities))
            x, y, template = self._sample_location(rng, cluster_id)
            if template is None:
                demand = int(rng.choice(demands))
                service = int(rng.choice(services))
            else:
                demand = int(template.get("demand", rng.choice(demands)))
                service = int(template.get("service_time", rng.choice(services)))
            lead = int(rng.integers(lead_min, lead_max + 1))
            width = int(rng.integers(width_min, width_max + 1))
            tw_start = min(int(release) + lead, horizon - width)
            tw_start = max(tw_start, int(release))
            tw_end = min(tw_start + width, horizon)
            if tw_end <= tw_start:
                tw_end = tw_start + 1
            sampled.append(
                {
                    "id": -(pos + 1),
                    "x": x,
                    "y": y,
                    "demand": max(demand, 1),
                    "tw_start": tw_start,
                    "tw_end": tw_end,
                    "service_time": max(service, 0),
                    "is_dynamic": True,
                    "available_time": int(release),
                    "cluster_id": cluster_id,
                    "is_sampled_scenario_request": True,
                }
            )
        return sampled

    def public_diagnostics(self) -> dict[str, Any]:
        return {
            "instance": self.instance_name,
            "knowledge": self.knowledge_mode,
            "future_customer_rows_retained": bool(self._realized_future),
            "configured_dynamic_count": self.dynamic_total,
            "observed_requests": self.observed_count,
            "observed_dynamic_requests": self.observed_dynamic_count,
            "temporal_mode": self.config.get("temporal_mode", "unknown"),
            "spatial_mode": self.config.get("spatial_mode", "unknown"),
            "uses_instance_temporal_metadata": self.knowledge_mode
            in {
                "instance_peak_centers",
                "instance_temporal",
                "instance_peak_centers_hotspot_schedule",
                "instance_peak_centers_spatial",
                "instance_temporal_hotspot_schedule",
                "instance_config",
                "realized_future",
            },
            "uses_instance_spatial_metadata": self.knowledge_mode
            in {
                "instance_hotspot_schedule",
                "instance_spatial",
                "instance_peak_centers_hotspot_schedule",
                "instance_peak_centers_spatial",
                "instance_temporal_hotspot_schedule",
                "instance_config",
                "realized_future",
            },
        }
