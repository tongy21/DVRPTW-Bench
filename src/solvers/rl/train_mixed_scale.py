#!/usr/bin/env python
"""
RL4CO 训练脚本 - 支持多种先进模型

支持的模型:
- attention (REINFORCE): 基础 Attention Model
- pomo: Policy Optimization with Multiple Optima
- am-ppo: Attention Model + PPO
- symnco: Symmetric NCO (对称性增强)
- mvmoe: Multi-Vehicle Mixture of Experts
- mdam: Multi-Decoder Attention Model
- polynet: PolyNet
"""
import argparse
import random
import math
import os
import fcntl
import json
import re
from datetime import datetime
import numpy as np
from rl4co.envs.routing import CVRPEnv
from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.models import AttentionModelPolicy, REINFORCE, POMO
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
try:
    from .train_mixed_scale_config import (
        VAL_CONFIG,
        VAL_WEIGHT_SCHEMES,
        CHECKPOINT_CONFIG,
        LOG_CONFIG,
    )
except ImportError:  # Support direct execution from this directory.
    from train_mixed_scale_config import (
        VAL_CONFIG,
        VAL_WEIGHT_SCHEMES,
        CHECKPOINT_CONFIG,
        LOG_CONFIG,
    )
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.parsing import AttributeDict
from rl4co.data.dataset import TensorDictDataset
from types import MethodType
from torch.utils.data import IterableDataset, DataLoader
import torch

# [Fix] Suppress excessive stream mismatch warnings (harmless in this context)
try:
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
except AttributeError:
    pass

# 可选的高级模型导入
try:
    from rl4co.models import PPO, SymNCO, MDAM, PolyNet, MatNet, HeterogeneousAttentionModel
    from rl4co.models.zoo.mvmoe import MVMoE_POMO
    from rl4co.models.zoo.ptrnet import PointerNetwork, PointerNetworkPolicy
    from rl4co.models.zoo.deepaco import DeepACO, DeepACOPolicy
    from rl4co.models.zoo.n2s import N2S, N2SPolicy
    ADVANCED_MODELS_AVAILABLE = True
except ImportError as e:
    ADVANCED_MODELS_AVAILABLE = False
    print(f"Warning: Some advanced models not available: {e}")

# 支持的算法列表
SUPPORTED_ALGOS = ["attention", "pomo", "am-ppo", "symnco", "mdam", "polynet", "matnet", "ham", "mvmoe", "ptrnet", "deepaco", "n2s"]

# --- Constants for TW variant ---
DEMAND_RANGE = (1, 10)
MAP_SIZE = (1000, 1000)


def infer_num_loc_from_td(td) -> int | None:
    """Infer the number of customer locations from a TensorDict batch."""
    for key in ("locs", "loc"):
        try:
            if key in td.keys():
                return int(td[key].shape[-2])
        except Exception:
            pass
    return None


class MultiScaleTensorDictBatchDataset(IterableDataset):
    """Yield already-batched TensorDict objects from several saved scale files.

    We use an IterableDataset with DataLoader(batch_size=None), so PyTorch does
    not collate TensorDict samples into ordinary dicts. Each yielded item is
    already a homogeneous-scale TensorDict batch and therefore keeps the
    .batch_size attribute required by RL4CO env.reset().
    """

    def __init__(self, scale_to_file: dict[int, str], batch_size: int):
        super().__init__()
        self.scale_to_file = dict(sorted(scale_to_file.items()))
        self.batch_size = int(batch_size)

    def __iter__(self):
        for scale, path in self.scale_to_file.items():
            td = torch.load(path, weights_only=False)
            # TensorDict normally has batch_size like torch.Size([num_samples]).
            try:
                total = int(td.batch_size[0])
            except Exception:
                total = int(len(td))

            for start in range(0, total, self.batch_size):
                end = min(start + self.batch_size, total)
                batch = td[start:end]
                try:
                    batch = batch.clone()
                    batch.set(
                        "_scale_id",
                        torch.full(batch.batch_size, int(scale), dtype=torch.long),
                    )
                except Exception:
                    pass
                yield batch

    def __len__(self):
        total_batches = 0
        for _, path in self.scale_to_file.items():
            td = torch.load(path, weights_only=False)
            try:
                total = int(td.batch_size[0])
            except Exception:
                total = int(len(td))
            total_batches += int(math.ceil(total / self.batch_size))
        return total_batches


def infer_scale_id_from_td(td):
    """Get the intended validation scale from a TensorDict batch if available."""
    try:
        if "_scale_id" in td.keys():
            return int(td["_scale_id"].reshape(-1)[0].item())
    except Exception:
        pass
    return infer_num_loc_from_td(td)


def infer_validation_segment_from_td(td):
    """Read the validation segment id attached to a random-range sample."""
    try:
        if "_val_segment_id" in td.keys():
            return int(td["_val_segment_id"].reshape(-1)[0].item())
    except Exception:
        pass
    return None


class RandomRangeValidationDataset(IterableDataset):
    """Yield one homogeneous validation batch for each sampled scale.

    Samples from different routing scales cannot be stacked into one regular
    TensorDict batch because their node dimensions differ. The saved manifest
    therefore stores one small TensorDict batch per sampled scale. In the
    default setup, each yielded batch contains the two independently generated
    instances for that scale.
    """

    def __init__(self, manifest_file: str):
        super().__init__()
        self.manifest_file = manifest_file

    def __iter__(self):
        payload = torch.load(self.manifest_file, weights_only=False)
        for scale_batch in payload["scale_batches"]:
            yield scale_batch.clone()

    def __len__(self):
        payload = torch.load(self.manifest_file, weights_only=False)
        return len(payload["scale_batches"])





class BalancedMultiScaleTrainBatchDataset(IterableDataset):
    """Yield balanced mixed-scale update groups with DDP-aware sharding.

    Two modes are supported:
      1. Legacy mode: ``samples_per_scale=None``. Each update group yields one
         batch per scale, using the provided batch size.
      2. Equal-sample mode: ``samples_per_scale=S``. Each update group yields
         exactly S samples for every scale. A large scale can use a smaller
         physical mini-batch, so it may contribute several micro-batches before
         the optimizer step. Loss weighting below makes the final update an
         equal-scale average rather than an equal-microbatch average.
    """

    def __init__(
        self,
        scale_to_file: dict[int, str],
        batch_size: int | dict[int, int],
        seed: int = 42,
        shuffle_batches: bool = True,
        shuffle_scale_order: bool = False,
        drop_last_group: bool = True,
        samples_per_scale: int | None = None,
    ):
        super().__init__()
        self.scale_to_file = dict(sorted(scale_to_file.items()))
        self.scales = list(self.scale_to_file.keys())
        self.seed = int(seed)
        self.shuffle_batches = bool(shuffle_batches)
        self.shuffle_scale_order = bool(shuffle_scale_order)
        self.drop_last_group = bool(drop_last_group)
        self.samples_per_scale = (
            int(samples_per_scale) if samples_per_scale is not None else None
        )
        if self.samples_per_scale is not None and self.samples_per_scale <= 0:
            raise ValueError("samples_per_scale must be positive when provided.")

        if isinstance(batch_size, dict):
            self.scale_to_batch_size = {
                int(scale): max(1, int(batch_size[int(scale)]))
                for scale in self.scales
            }
        else:
            shared_batch_size = max(1, int(batch_size))
            self.scale_to_batch_size = {
                int(scale): shared_batch_size for scale in self.scales
            }

        # Backward-compatible attribute for code that only prints one number.
        self.batch_size = max(self.scale_to_batch_size.values())
        self._epoch_counter = 0

    def _rank_info(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1

    def _dataset_size(self, path: str) -> int:
        td = torch.load(path, weights_only=False)
        try:
            return int(td.batch_size[0])
        except Exception:
            return int(len(td))

    def _microbatches_for_scale(self, scale: int) -> int:
        if self.samples_per_scale is None:
            return 1
        return int(math.ceil(self.samples_per_scale / self.scale_to_batch_size[int(scale)]))

    def microbatches_per_update(self) -> int:
        return sum(self._microbatches_for_scale(scale) for scale in self.scales)

    def _num_groups(self) -> int:
        """Number of complete same-index multi-scale groups before DDP padding."""
        nums = []
        for scale, path in self.scale_to_file.items():
            total = self._dataset_size(path)
            if self.samples_per_scale is None:
                bs = self.scale_to_batch_size[int(scale)]
                n = total // bs if self.drop_last_group else int(math.ceil(total / bs))
            else:
                # Equal-sample updates must be exact, so we intentionally drop
                # the tail that cannot form a full S-sample group for every scale.
                n = total // self.samples_per_scale
            nums.append(n)
        return min(nums) if nums else 0

    def _padded_group_ids_for_rank(self, epoch: int):
        rank, world_size = self._rank_info()
        num_groups = self._num_groups()
        group_ids = list(range(num_groups))

        rng = random.Random(self.seed + epoch)
        if self.shuffle_batches:
            rng.shuffle(group_ids)

        if world_size <= 1:
            return group_ids

        groups_per_rank = int(math.ceil(num_groups / world_size)) if num_groups > 0 else 0
        total_size = groups_per_rank * world_size

        if total_size > len(group_ids):
            if not group_ids:
                return []
            pad = []
            while len(group_ids) + len(pad) < total_size:
                remaining = total_size - len(group_ids) - len(pad)
                pad.extend(group_ids[:remaining])
            group_ids = group_ids + pad

        return group_ids[rank:total_size:world_size]

    def _slice_batch(self, td, scale: int, start: int, end: int, gid: int,
                     micro_idx: int, micro_count_for_scale: int, micro_idx_global: int):
        batch = td[start:end]
        try:
            batch = batch.clone()
            current_size = int(batch.batch_size[0])
            batch.set(
                "_scale_id",
                torch.full(batch.batch_size, int(scale), dtype=torch.long),
            )
            batch.set(
                "_mixed_update_group_id",
                torch.full(batch.batch_size, int(gid), dtype=torch.long),
            )
            batch.set(
                "_scale_microbatch_idx",
                torch.full(batch.batch_size, int(micro_idx), dtype=torch.long),
            )
            batch.set(
                "_scale_microbatches_in_update",
                torch.full(batch.batch_size, int(micro_count_for_scale), dtype=torch.long),
            )
            batch.set(
                "_mixed_microbatch_idx",
                torch.full(batch.batch_size, int(micro_idx_global), dtype=torch.long),
            )
            batch.set(
                "_mixed_microbatches_per_update",
                torch.full(batch.batch_size, int(self.microbatches_per_update()), dtype=torch.long),
            )
            if self.samples_per_scale is not None:
                batch.set(
                    "_scale_samples_per_update",
                    torch.full(batch.batch_size, int(self.samples_per_scale), dtype=torch.long),
                )
            batch.set(
                "_actual_microbatch_size",
                torch.full(batch.batch_size, int(current_size), dtype=torch.long),
            )
        except Exception:
            pass
        return batch

    def __iter__(self):
        epoch = self._epoch_counter
        self._epoch_counter += 1

        scale_to_td = {
            scale: torch.load(path, weights_only=False)
            for scale, path in self.scale_to_file.items()
        }

        rng = random.Random(self.seed + epoch)
        group_ids = self._padded_group_ids_for_rank(epoch)

        for gid in group_ids:
            scale_order = list(self.scales)
            if self.shuffle_scale_order:
                rng.shuffle(scale_order)

            micro_idx_global = 0
            for scale in scale_order:
                td = scale_to_td[scale]
                try:
                    total = int(td.batch_size[0])
                except Exception:
                    total = int(len(td))

                bs = self.scale_to_batch_size[int(scale)]
                if self.samples_per_scale is None:
                    start = gid * bs
                    if start >= total:
                        continue
                    end = min(start + bs, total)
                    yield self._slice_batch(
                        td, scale, start, end, gid,
                        micro_idx=0,
                        micro_count_for_scale=1,
                        micro_idx_global=micro_idx_global,
                    )
                    micro_idx_global += 1
                else:
                    group_start = gid * self.samples_per_scale
                    group_end = group_start + self.samples_per_scale
                    if group_end > total:
                        continue
                    micro_count = self._microbatches_for_scale(scale)
                    micro_idx = 0
                    start = group_start
                    while start < group_end:
                        end = min(start + bs, group_end)
                        yield self._slice_batch(
                            td, scale, start, end, gid,
                            micro_idx=micro_idx,
                            micro_count_for_scale=micro_count,
                            micro_idx_global=micro_idx_global,
                        )
                        start = end
                        micro_idx += 1
                        micro_idx_global += 1

    def __len__(self):
        _, world_size = self._rank_info()
        num_groups = self._num_groups()
        groups_per_rank = int(math.ceil(num_groups / world_size)) if world_size > 0 else num_groups
        return groups_per_rank * self.microbatches_per_update()



def patch_env_reward_normalization(env, mode: str):
    """Normalize rewards during training while preserving raw validation reward.

    The wrapper is always installed and can be disabled temporarily by setting
    ``env._reward_normalization_enabled = False``. Validation does this inside
    its own step, so training can use sqrt(N) scaling while validation and
    checkpoint selection use the original distance-based reward.
    """
    mode = (mode or "none").lower()
    original_get_reward = getattr(env, "get_reward", None)
    if original_get_reward is None:
        print("[WARN] env.get_reward not found; reward normalization is skipped.")
        return env

    env._reward_normalization_mode = mode
    env._reward_normalization_enabled = mode != "none"

    def normalized_get_reward(td, actions, *args, **kwargs):
        reward = original_get_reward(td, actions, *args, **kwargs)
        if not getattr(env, "_reward_normalization_enabled", False):
            return reward

        num_loc = infer_num_loc_from_td(td)
        if num_loc is None or num_loc <= 0:
            return reward

        active_mode = getattr(env, "_reward_normalization_mode", "none")
        if active_mode == "sqrt_n":
            denom = math.sqrt(float(num_loc))
        elif active_mode == "num_loc":
            denom = float(num_loc)
        elif active_mode == "none":
            return reward
        else:
            raise ValueError(
                f"Unsupported reward normalization mode: {active_mode}"
            )
        return reward / denom

    env.get_reward = normalized_get_reward
    if mode == "none":
        print("[INFO] Reward normalization disabled.")
    else:
        print(f"[INFO] Training reward normalization enabled: {mode}")
        print("[INFO] Validation uses the original unnormalized reward.")
    return env


class MultiWeightedValidationCallback(Callback):
    """Compute weighted means of raw validation reward.

    Validation logs exactly one raw reward metric per segment:
        val/reward_segment_0, ..., val/reward_segment_{K-1}

    This callback combines those segment rewards using every enabled weighting
    scheme and logs exactly one aggregate reward per scheme:
        val/reward_weighted_<scheme>

    No per-scheme per-segment metrics are created.
    """

    def __init__(self, num_segments: int, weight_schemes: dict):
        super().__init__()
        self.num_segments = int(num_segments)
        self.weight_schemes = validate_and_normalize_weight_schemes(
            weight_schemes,
            self.num_segments,
        )

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        segment_values = []

        for segment_id in range(self.num_segments):
            key = f"val/reward_segment_{segment_id}"
            value = metrics.get(key)
            if value is None:
                continue
            try:
                value = value.detach().float().to(pl_module.device)
            except Exception:
                value = torch.tensor(float(value), device=pl_module.device)
            segment_values.append(value)

        if len(segment_values) != self.num_segments:
            print(
                "[WARN] Random-range validation produced "
                f"{len(segment_values)}/{self.num_segments} raw segment reward "
                "metrics; weighted validation rewards were not logged."
            )
            return

        stacked = torch.stack(segment_values)
        for scheme_name, normalized_weights in self.weight_schemes.items():
            weights = torch.tensor(
                normalized_weights,
                dtype=stacked.dtype,
                device=stacked.device,
            )
            weighted_reward = torch.sum(stacked * weights)
            metric_name = f"val/reward_weighted_{scheme_name}"
            pl_module.log(
                metric_name,
                weighted_reward,
                prog_bar=(scheme_name == "uniform"),
                logger=True,
                sync_dist=True,
            )
            try:
                trainer.callback_metrics[metric_name] = weighted_reward.detach()
            except Exception:
                pass


def validate_and_normalize_weight_schemes(weight_schemes: dict, num_segments: int):
    """Validate config schemes and return normalized enabled weight vectors."""
    normalized = {}
    name_pattern = re.compile(r"^[A-Za-z0-9_]+$")

    for name, config in weight_schemes.items():
        if not config.get("enabled", True):
            continue
        if not name_pattern.fullmatch(name):
            raise ValueError(
                f"Validation scheme name '{name}' may contain only letters, "
                "numbers, and underscores."
            )
        weights = [float(x) for x in config.get("weights", [])]
        if len(weights) != num_segments:
            raise ValueError(
                f"Validation scheme '{name}' has {len(weights)} weights, "
                f"but validation uses {num_segments} segments."
            )
        if any(x < 0 for x in weights):
            raise ValueError(
                f"Validation scheme '{name}' contains a negative weight."
            )
        total = sum(weights)
        if total <= 0:
            raise ValueError(
                f"Validation scheme '{name}' must contain at least one "
                "positive weight."
            )
        normalized[name] = [x / total for x in weights]

    if not normalized:
        raise ValueError("At least one validation weight scheme must be enabled.")
    return normalized


class SaveLatestCheckpointCallback(Callback):
    """Save the true latest state on normal completion or Ctrl+C."""

    def __init__(
        self,
        last_path: str,
        interrupted_path: str,
        save_true_last: bool = True,
        save_on_keyboard_interrupt: bool = True,
    ):
        super().__init__()
        self.last_path = last_path
        self.interrupted_path = interrupted_path
        self.save_true_last = bool(save_true_last)
        self.save_on_keyboard_interrupt = bool(save_on_keyboard_interrupt)
        self._interrupted = False
        self._saved_interrupted = False

    def _save(self, trainer, path: str, label: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        trainer.save_checkpoint(path)
        if trainer.is_global_zero:
            print(
                f"[INFO] Saved {label} checkpoint: {path} "
                f"(epoch={trainer.current_epoch}, "
                f"global_step={trainer.global_step})"
            )

    def on_exception(self, trainer, pl_module, exception):
        if isinstance(exception, KeyboardInterrupt):
            self._interrupted = True
            if self.save_on_keyboard_interrupt and not self._saved_interrupted:
                self._save(trainer, self.interrupted_path, "interrupted/latest")
                self._saved_interrupted = True

    def on_train_end(self, trainer, pl_module):
        # Some Lightning flows call on_train_end after KeyboardInterrupt. Do not
        # mislabel an interrupted run as a normally completed run.
        if self._interrupted:
            return
        if self.save_true_last:
            self._save(trainer, self.last_path, "final/latest")



class MixedScaleGradientAnalysisCallback(Callback):
    """Log compact per-scale gradient geometry for mixed-scale optimizer updates.

    The callback reconstructs each micro-batch contribution by subtracting the
    previous accumulated gradient from the current accumulated gradient after
    backward. Contributions are summed by scale within one optimizer update.

    Logged metrics are intentionally compact:
      - per-scale gradient norm;
      - global gradient norm and cancellation ratio;
      - aggregate pairwise cosine statistics;
      - cosine between each scale and the sum of all other scales.
    """

    def __init__(
        self,
        scales: list[int],
        microbatches_per_update: int,
        samples_per_scale: int | None = None,
        log_every: int = 1,
        enabled: bool = True,
    ):
        super().__init__()
        self.scales = [int(s) for s in sorted(scales)]
        self.microbatches_per_update = max(1, int(microbatches_per_update))
        self.samples_per_scale = (
            int(samples_per_scale) if samples_per_scale is not None else None
        )
        self.log_every = max(1, int(log_every))
        self.enabled = bool(enabled)
        self._reset_state()

    def _reset_state(self):
        self._prev_flat_grad = None
        self._scale_grads = {}
        # Samples / microbatches are kept only for internal sanity/debug use.
        # They are intentionally not logged to keep CSV columns compact.
        self._scale_samples = {scale: 0 for scale in self.scales}
        self._scale_microbatches = {scale: 0 for scale in self.scales}
        self._current_scale = None
        self._current_batch_size = 0
        self._microbatches_seen = 0
        self._updates_seen = 0

    def on_train_start(self, trainer, pl_module):
        self._reset_state()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if not self.enabled or not trainer.is_global_zero:
            return
        self._current_scale = infer_scale_id_from_td(batch)
        try:
            self._current_batch_size = int(batch.batch_size[0])
        except Exception:
            self._current_batch_size = 1

    def _flatten_current_grads(self, pl_module):
        pieces = []
        for param in pl_module.parameters():
            if not param.requires_grad:
                continue
            grad = param.grad
            if grad is None:
                pieces.append(torch.zeros(param.numel(), dtype=torch.float32, device="cpu"))
            else:
                pieces.append(grad.detach().float().reshape(-1).cpu())
        if not pieces:
            return torch.zeros(1, dtype=torch.float32)
        return torch.cat(pieces)

    def on_after_backward(self, trainer, pl_module):
        if not self.enabled or not trainer.is_global_zero:
            return
        if self._current_scale is None:
            return

        flat_grad = self._flatten_current_grads(pl_module)
        if self._prev_flat_grad is None:
            delta = flat_grad.clone()
        else:
            delta = flat_grad - self._prev_flat_grad
        self._prev_flat_grad = flat_grad

        scale = int(self._current_scale)
        if scale not in self._scale_grads:
            self._scale_grads[scale] = delta.clone()
            self._scale_samples[scale] = 0
            self._scale_microbatches[scale] = 0
        else:
            self._scale_grads[scale].add_(delta)
        self._scale_samples[scale] += int(self._current_batch_size)
        self._scale_microbatches[scale] += 1
        self._microbatches_seen += 1

    @staticmethod
    def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
        denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
        if denom <= eps:
            return 0.0
        return float(torch.dot(a, b).item() / denom)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if not self.enabled or not trainer.is_global_zero:
            return

        should_log = (self._updates_seen % self.log_every == 0)
        if should_log and self._scale_grads:
            present_scales = [scale for scale in self.scales if scale in self._scale_grads]
            grads = {scale: self._scale_grads[scale] for scale in present_scales}

            combined = None
            for grad in grads.values():
                combined = grad.clone() if combined is None else combined.add(grad)
            if combined is None:
                combined = torch.zeros(1, dtype=torch.float32)

            combined_norm = float(torch.linalg.vector_norm(combined).item())
            metrics = {
                "grad/num_scales_present": float(len(present_scales)),
                "grad/combined_norm": combined_norm,
            }

            sum_norms = 0.0
            for scale in present_scales:
                norm = float(torch.linalg.vector_norm(grads[scale]).item())
                sum_norms += norm
                metrics[f"grad/norm_scale_{scale}"] = norm

            metrics["grad/combined_to_sum_norm_ratio"] = (
                combined_norm / sum_norms if sum_norms > 0 else 0.0
            )

            pair_cos = []
            for i, scale_i in enumerate(present_scales):
                for scale_j in present_scales[i + 1:]:
                    pair_cos.append(self._cosine(grads[scale_i], grads[scale_j]))

            if pair_cos:
                metrics["grad/mean_pairwise_cos"] = float(sum(pair_cos) / len(pair_cos))
                metrics["grad/min_pairwise_cos"] = float(min(pair_cos))
                metrics["grad/negative_pair_fraction"] = float(
                    sum(1 for x in pair_cos if x < 0.0) / len(pair_cos)
                )
            else:
                metrics["grad/mean_pairwise_cos"] = 0.0
                metrics["grad/min_pairwise_cos"] = 0.0
                metrics["grad/negative_pair_fraction"] = 0.0

            # Per-scale deviation from the rest of the mixed update.
            # Low or negative values identify which scale is pulling against the others.
            for scale in present_scales:
                rest = combined - grads[scale]
                metrics[f"grad/cos_to_rest_scale_{scale}"] = self._cosine(grads[scale], rest)

            pl_module.log_dict(
                metrics,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=False,
                batch_size=1,
            )

        self._updates_seen += 1
        # The optimizer step and zero_grad happen after this hook; reset our
        # reconstruction state so the next update starts cleanly.
        self._prev_flat_grad = None
        self._scale_grads = {}
        self._scale_samples = {scale: 0 for scale in self.scales}
        self._scale_microbatches = {scale: 0 for scale in self.scales}
        self._current_scale = None
        self._current_batch_size = 0
        self._microbatches_seen = 0


def patch_model_for_random_range_validation(model, manifest_file: str, num_segments: int):
    """Validate on fixed random scales using raw, unnormalized reward.

    Training reward normalization is temporarily disabled during validation.
    The CSV log contains only:
      - val/reward_segment_<id> for every validation segment;
      - val/reward_weighted_<scheme> from MultiWeightedValidationCallback.

    Other validation metrics emitted internally by RL4CO are intentionally not
    forwarded to the logger.
    """
    original_validation_step = model.validation_step

    def random_range_val_dataloader(self):
        dataset = RandomRangeValidationDataset(manifest_file)
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=lambda x: x,
        )

    def validation_step_with_segment_metrics(self, batch, batch_idx):
        segment_id = infer_validation_segment_from_td(batch)
        if segment_id is None:
            segment_id = "unknown"

        saved_log = self.log
        env_obj = getattr(self, "env", None)
        had_norm_flag = hasattr(env_obj, "_reward_normalization_enabled")
        previous_norm_flag = (
            getattr(env_obj, "_reward_normalization_enabled", False)
            if env_obj is not None else False
        )

        def log_raw_segment_reward_only(name, value, *args, **kwargs):
            if isinstance(name, str) and name == "val/reward":
                try:
                    current_batch_size = int(batch.batch_size[0])
                except Exception:
                    current_batch_size = 1

                log_kwargs = dict(kwargs)
                log_kwargs["add_dataloader_idx"] = False
                log_kwargs.setdefault("batch_size", current_batch_size)

                return saved_log(
                    f"val/reward_segment_{segment_id}",
                    value,
                    *args,
                    **log_kwargs,
                )

            if isinstance(name, str) and name.startswith("val/"):
                return None

            return saved_log(name, value, *args, **kwargs)

        self.log = log_raw_segment_reward_only
        try:
            if env_obj is not None and had_norm_flag:
                env_obj._reward_normalization_enabled = False
            return original_validation_step(batch, batch_idx)
        finally:
            self.log = saved_log
            if env_obj is not None and had_norm_flag:
                env_obj._reward_normalization_enabled = previous_norm_flag

    model.val_dataloader = MethodType(random_range_val_dataloader, model)
    model.validation_step = MethodType(validation_step_with_segment_metrics, model)
    print(
        "[INFO] Random-range validation enabled with raw reward-only logging: "
        f"manifest={manifest_file}, segments={num_segments}"
    )
    return model


def patch_model_for_raw_fixed_validation(model):
    """Use raw reward and log only val/reward for fixed-scale validation."""
    original_validation_step = model.validation_step

    def raw_validation_step(self, batch, batch_idx):
        saved_log = self.log
        env_obj = getattr(self, "env", None)
        had_norm_flag = hasattr(env_obj, "_reward_normalization_enabled")
        previous_norm_flag = (
            getattr(env_obj, "_reward_normalization_enabled", False)
            if env_obj is not None else False
        )

        def log_raw_reward_only(name, value, *args, **kwargs):
            if name == "val/reward":
                return saved_log("val/reward", value, *args, **kwargs)
            if isinstance(name, str) and name.startswith("val/"):
                return None
            return saved_log(name, value, *args, **kwargs)

        self.log = log_raw_reward_only
        try:
            if env_obj is not None and had_norm_flag:
                env_obj._reward_normalization_enabled = False
            return original_validation_step(batch, batch_idx)
        finally:
            self.log = saved_log
            if env_obj is not None and had_norm_flag:
                env_obj._reward_normalization_enabled = previous_norm_flag

    model.validation_step = MethodType(raw_validation_step, model)
    print("[INFO] Fixed-scale validation uses raw reward-only logging.")
    return model



def patch_reinforce_rollout_baseline_for_raw_batches(model):
    """Give an on-the-fly rollout baseline a fresh initial environment state.

    RL4CO's standard REINFORCE pipeline normally calls ``model.wrap_dataset``.
    Once rollout-baseline warmup has finished, that wrapper adds a precomputed
    ``extra`` baseline reward to every training sample. Our custom balanced
    multi-scale dataloader reads TensorDict files directly and therefore skips
    that wrapper. In that case REINFORCE falls back to ``baseline.eval(td, ...)``.

    The ``td`` passed by RL4CO has already been consumed by the actor rollout and
    is terminal (``done=True``). Reusing it makes the baseline policy perform
    zero decoding steps and raises:

        AssertionError: No logprobs were collected because all environments were done.

    When no precomputed ``extra`` value is present, rebuild the initial state
    from the untouched problem batch before evaluating the rollout baseline.
    This preserves the original rollout-baseline objective and works correctly
    with the custom DDP-sharded balanced dataloader.
    """
    if not hasattr(model, "calculate_loss"):
        print(
            "[WARN] Model has no calculate_loss method; "
            "rollout-baseline fresh-state patch was skipped."
        )
        return model

    original_calculate_loss = model.calculate_loss

    def calculate_loss_with_fresh_baseline_state(
        self,
        td,
        batch,
        policy_out,
        reward=None,
        log_likelihood=None,
    ):
        # Standard RL4CO wrapped datasets provide a cached rollout-baseline
        # reward in batch["extra"]. Raw batches from our custom iterable do not.
        try:
            extra = batch.get("extra", None)
        except Exception:
            extra = None

        baseline_td = td
        if extra is None:
            # ``batch`` contains immutable problem data, while ``td`` has been
            # advanced in-place by the actor policy. Resetting from ``batch``
            # gives the baseline policy a clean state with done=False.
            baseline_td = self.env.reset(batch)

            try:
                if bool(baseline_td["done"].all().item()):
                    raise RuntimeError(
                        "Fresh rollout-baseline state is already terminal. "
                        "Check the generated training dataset and env.reset()."
                    )
            except KeyError:
                # Some environments may not expose done until their first step.
                pass

        return original_calculate_loss(
            baseline_td,
            batch,
            policy_out,
            reward=reward,
            log_likelihood=log_likelihood,
        )

    model.calculate_loss = MethodType(
        calculate_loss_with_fresh_baseline_state,
        model,
    )
    print(
        "[INFO] Patched REINFORCE rollout baseline for raw balanced batches: "
        "baseline evaluation now starts from env.reset(batch)."
    )
    return model


def patch_model_for_mixed_loss_weighting(
    model,
    num_scales: int,
    samples_per_scale: int | None,
    microbatches_per_update: int,
):
    """Weight variable-size micro-batches so every scale contributes equally.

    Lightning divides the loss by ``accumulate_grad_batches``. If a large scale
    uses many small micro-batches, equal micro-batch weighting would accidentally
    over-weight that scale. The multiplier below makes the final accumulated
    gradient equivalent to:

        mean over scales of mean gradient over S samples from that scale.
    """
    if samples_per_scale is None:
        return model
    if not hasattr(model, "training_step"):
        print("[WARN] Model has no training_step; mixed loss weighting was skipped.")
        return model

    original_training_step = model.training_step
    num_scales = int(num_scales)
    samples_per_scale = int(samples_per_scale)
    microbatches_per_update = int(microbatches_per_update)

    def weighted_training_step(self, batch, batch_idx, *args, **kwargs):
        out = original_training_step(batch, batch_idx, *args, **kwargs)
        current_batch_size = get_batch_size_from_td(batch)
        weight = (
            microbatches_per_update
            * float(current_batch_size)
            / float(num_scales * samples_per_scale)
        )

        if abs(weight - 1.0) < 1e-12:
            return out
        if torch.is_tensor(out):
            return out * weight
        if isinstance(out, dict) and "loss" in out and torch.is_tensor(out["loss"]):
            out = dict(out)
            out["loss"] = out["loss"] * weight
            return out
        if hasattr(out, "loss") and torch.is_tensor(out.loss):
            out.loss = out.loss * weight
            return out
        return out

    model.training_step = MethodType(weighted_training_step, model)
    print(
        "[INFO] Mixed loss weighting enabled: "
        f"num_scales={num_scales}, samples_per_scale={samples_per_scale}, "
        f"microbatches_per_update={microbatches_per_update}"
    )
    return model


def patch_model_for_scale_balanced_training(
    model,
    train_files: dict[int, str],
    batch_size: int | dict[int, int],
    seed: int = 42,
    samples_per_scale: int | None = None,
):
    """Patch model.train_dataloader for balanced mixed-scale updates.

    In equal-sample mode, one optimizer step receives the same number of samples
    from every scale, while each scale can still use its own physical batch size.
    """
    scales = sorted(train_files.keys())
    preview_dataset = BalancedMultiScaleTrainBatchDataset(
        train_files,
        batch_size=batch_size,
        seed=seed,
        shuffle_batches=True,
        shuffle_scale_order=False,
        samples_per_scale=samples_per_scale,
    )
    microbatches_per_update = preview_dataset.microbatches_per_update()

    def mixed_train_dataloader(self):
        dataset = BalancedMultiScaleTrainBatchDataset(
            train_files,
            batch_size=batch_size,
            seed=seed,
            shuffle_batches=True,
            shuffle_scale_order=False,
            samples_per_scale=samples_per_scale,
        )
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            collate_fn=lambda x: x,
        )

    model.train_dataloader = MethodType(mixed_train_dataloader, model)
    model._mixed_scales = scales
    model._mixed_scale_to_batch_size = (
        dict(batch_size) if isinstance(batch_size, dict) else {s: int(batch_size) for s in scales}
    )
    model._mixed_samples_per_scale = samples_per_scale
    model._mixed_microbatches_per_update = microbatches_per_update
    patch_model_for_mixed_loss_weighting(
        model,
        num_scales=len(scales),
        samples_per_scale=samples_per_scale,
        microbatches_per_update=microbatches_per_update,
    )
    print(
        "[INFO] Scale-balanced training enabled: "
        f"scales={scales}, batch_size_by_scale={model._mixed_scale_to_batch_size}, "
        f"samples_per_scale={samples_per_scale}, "
        f"accumulate_grad_batches={microbatches_per_update}"
    )
    return model



def get_policy(env, algo: str, embed_dim: int = 128, num_encoder_layers: int = 3, num_heads: int = 8):
    """
    根据算法类型返回对应的策略网络
    """
    if algo in ["attention", "pomo", "am-ppo"]:
        return AttentionModelPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "symnco":
        # SymNCO 使用特殊的对称性策略
        from rl4co.models.zoo.symnco import SymNCOPolicy
        return SymNCOPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "mdam":
        # MDAM 使用多解码器策略
        from rl4co.models.zoo.mdam import MDAMPolicy
        return MDAMPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            num_paths=5,  # 多解码器路径数
        )
    elif algo == "polynet":
        # PolyNet 策略
        from rl4co.models.zoo.polynet.policy import PolyNetPolicy
        return PolyNetPolicy(
            k=40, # Number of strategies to learn
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "matnet":
        # MatNet 策略
        from rl4co.models.zoo.matnet import MatNetPolicy
        return MatNetPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "ham":
        # HAM (Heterogeneous Attention Model) 策略
        from rl4co.models.zoo.ham import HeterogeneousAttentionModelPolicy
        return HeterogeneousAttentionModelPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "mvmoe":
        # MVMoE: Multi-View Mixture of Experts
        # MVMoE uses AttentionModelPolicy but with specific moe_kwargs
        # Here we rely on the model to create the policy if we don't pass one,
        # or we create a standard one and let the model wrap it?
        # Actually MVMoE constructor takes a policy. If passed, it uses it.
        # But AttentionModelPolicy needs to be aware of MoE?
        # Based on MVMoE code, it updates policy_kwargs with moe_kwargs and creates policy.
        # So it's better to return None here and let get_model create the policy
        return None
    elif algo == "ptrnet":
        return PointerNetworkPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "deepaco":
        return DeepACOPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    elif algo == "n2s":
        return N2SPolicy(
            env_name=env.name,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def get_model(env, policy, algo: str, batch_size: int, train_size: int, val_size: int, test_size: int, lr: float = 1e-4):
    """
    根据算法类型返回对应的模型
    """
    common_kwargs = {
        "batch_size": batch_size,
        "train_data_size": train_size,
        "val_data_size": val_size,
        "test_data_size": test_size,
        "optimizer_kwargs": {"lr": lr}
        }

    if algo == "attention":
        return REINFORCE(
            env, policy,
            baseline="rollout",
            **common_kwargs
        )

    elif algo == "pomo":
        return POMO(
            env, policy,
            baseline="shared",
            num_starts=8,
            **common_kwargs
        )

    elif algo == "am-ppo":
        if not ADVANCED_MODELS_AVAILABLE:
            raise ImportError("PPO not available. Please upgrade rl4co.")
        return PPO(
            env, policy,
            ppo_epochs=3,
            mini_batch_size=min(batch_size // 4, 64),
            **common_kwargs
        )

    elif algo == "symnco":
        if not ADVANCED_MODELS_AVAILABLE:
            raise ImportError("SymNCO not available. Please upgrade rl4co.")
        return SymNCO(
            env, policy,
            num_augment=8,  # 对称增强数量
            **common_kwargs
        )

    elif algo == "mdam":
        if not ADVANCED_MODELS_AVAILABLE:
            raise ImportError("MDAM not available. Please upgrade rl4co.")
        return MDAM(
            env, policy,
            **common_kwargs
        )

    elif algo == "polynet":
        if not ADVANCED_MODELS_AVAILABLE:
            raise ImportError("PolyNet not available. Please upgrade rl4co.")
        return PolyNet(
            env, policy,
            **common_kwargs
        )

    elif algo == "matnet":
        if not ADVANCED_MODELS_AVAILABLE:
            raise ImportError("MatNet not available. Please upgrade rl4co.")
        return MatNet(
            env, policy,
            **common_kwargs
        )

    elif algo == "ham":
        if not ADVANCED_MODELS_AVAILABLE:
            raise ImportError("HAM not available. Please upgrade rl4co.")
        return HeterogeneousAttentionModel(
            env, policy,
            **common_kwargs
        )

    elif algo == "mvmoe":
        # Note: policy might be None here, MVMoE_POMO will create it
        return MVMoE_POMO(
            env, policy=policy,
            **common_kwargs
        )

    elif algo == "ptrnet":
        return PointerNetwork(
            env, policy,
            **common_kwargs
        )

    elif algo == "deepaco":
        return DeepACO(
            env, policy,
            baseline="rollout", # DeepACO usually uses REINFORCE/Rollout
            **common_kwargs
        )

    elif algo == "n2s":
        # N2S uses PPO-like training, might need specific critic
        return N2S(
            env, policy,
            **common_kwargs
        )

    else:
        raise ValueError(f"Unknown algorithm: {algo}")


def make_run_id(timestamp_format: str) -> str:
    """Create one run-start timestamp shared by all artifacts in this process.

    TRAIN_RUN_ID can be supplied by launch scripts to guarantee an identical ID
    across externally launched DDP workers. Otherwise, workers spawned by the
    same parent derive the timestamp from the parent process creation time.
    """
    explicit = os.environ.get("TRAIN_RUN_ID")
    if explicit:
        return explicit

    try:
        parent_ctime = os.path.getctime(f"/proc/{os.getppid()}")
        return datetime.fromtimestamp(parent_ctime).strftime(timestamp_format)
    except Exception:
        return datetime.now().strftime(timestamp_format)


def save_run_config_snapshot(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_int_list(value: str | None, name: str) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    items = []
    for raw in str(value).replace(";", ",").split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            items.append(int(raw))
        except ValueError as exc:
            raise ValueError(f"{name} must be a comma-separated int list, got: {value}") from exc
    if any(x <= 0 for x in items):
        raise ValueError(f"{name} values must be positive, got: {value}")
    return sorted(set(items))


def parse_scale_to_int(value: str | None, valid_scales: list[int], name: str) -> dict[int, int]:
    valid_scales = [int(s) for s in valid_scales]
    if value is None or str(value).strip() == "":
        return {}
    parsed = {}
    for raw_item in str(value).replace(";", ",").split(","):
        raw_item = raw_item.strip()
        if not raw_item:
            continue
        if ":" not in raw_item:
            raise ValueError(
                f"{name} must use 'scale:value' items, e.g. 20:128,500:8; got: {raw_item}"
            )
        raw_scale, raw_val = raw_item.split(":", 1)
        scale = int(raw_scale.strip())
        val = int(raw_val.strip())
        if scale not in valid_scales:
            raise ValueError(
                f"{name} contains scale {scale}, but training scales are {valid_scales}."
            )
        if val <= 0:
            raise ValueError(f"{name} values must be positive, got {raw_item}.")
        parsed[scale] = val
    missing = [scale for scale in valid_scales if scale not in parsed]
    if missing:
        raise ValueError(f"{name} is missing scales: {missing}.")
    return parsed


def get_batch_size_from_td(batch) -> int:
    try:
        return int(batch.batch_size[0])
    except Exception:
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL4CO on VRP variants (CVRP or VRPTW)")
    parser.add_argument("--num_loc", type=int, default=None,
                        help="Number of customer locations for single-scale training. Optional when using --num_loc_min/--num_loc_max.")
    parser.add_argument("--algo", type=str, choices=SUPPORTED_ALGOS, required=True,
                        help=f"Which RL algorithm to use. Choices: {SUPPORTED_ALGOS}")
    parser.add_argument("--batch_size", type=int, required=True,
                        help="Batch size for rl")
    parser.add_argument("--variant", type=str, choices=["cvrp", "twvrp"], required=True,
                        help="Problem variant to train: 'cvrp' or 'twvrp'")
    parser.add_argument("--embed_dim", type=int, default=128,
                        help="Embedding dimension for the policy network")
    parser.add_argument("--num_encoder_layers", type=int, default=3,
                        help="Number of encoder layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--num_loc_min", type=int, default=None,
                        help="Minimum scale for mixed-scale training, e.g. 20.")
    parser.add_argument("--num_loc_max", type=int, default=None,
                        help="Maximum scale for mixed-scale training, e.g. 50.")
    parser.add_argument("--num_loc_step", type=int, default=10,
                        help="Step size for --num_loc_min/--num_loc_max mixed-scale training.")
    parser.add_argument("--mixed_scale_sampling", type=str, choices=["cycle", "random"], default="cycle",
                        help="How to choose the training scale for each epoch when mixed-scale training is enabled. Used only when --mixed_train_mode=epoch_switch.")
    parser.add_argument("--mixed_train_mode", type=str, choices=["balanced_grad", "epoch_switch"], default="balanced_grad",
                        help="Mixed-scale training mode. balanced_grad interleaves one batch from each scale and accumulates their gradients into one optimizer step. epoch_switch keeps the older behavior: one scale per epoch.")
    parser.add_argument("--val_num_loc", type=int, default=None,
                        help="Fallback fixed validation/test scale. Used when random-range validation is disabled.")
    parser.add_argument("--reward_normalization", type=str, choices=["auto", "none", "sqrt_n", "num_loc"], default="auto",
                        help="Reward scaling for cross-scale training. auto = sqrt_n for mixed-scale training and none for single-scale training.")
    parser.add_argument("--disable_multiscale_val", action="store_true",
                        help="Disable random-range validation and fall back to one fixed validation scale.")
    parser.add_argument("--mixed_scales", type=str, default=None,
                        help="Explicit comma-separated mixed training scales, e.g. '20,50,100,200,500'. This overrides --num_loc_min/max/step.")
    parser.add_argument("--batch_size_by_scale", type=str, default=None,
                        help="Comma-separated per-scale batch sizes, e.g. '20:128,50:128,100:128,200:64,500:8'. Used by balanced_grad.")
    parser.add_argument("--mixed_samples_per_scale", type=int, default=None,
                        help="In balanced_grad, number of samples from each scale per optimizer update. If omitted, legacy one-batch-per-scale accumulation is used.")
    parser.add_argument("--gradient_analysis_log_every", type=int, default=1,
                        help="Log mixed-scale gradient conflict metrics every N optimizer updates.")
    parser.add_argument("--disable_gradient_analysis", action="store_true",
                        help="Disable per-scale gradient norm/cosine logging for mixed balanced training.")
    args = parser.parse_args()

    val_num_segments = int(VAL_CONFIG["num_segments"])
    val_scales_per_segment = int(VAL_CONFIG["scales_per_segment"])
    val_instances_per_scale = int(VAL_CONFIG["instances_per_scale"])
    val_scale_seed = int(VAL_CONFIG["scale_seed"])
    normalized_val_weight_schemes = validate_and_normalize_weight_schemes(
        VAL_WEIGHT_SCHEMES,
        val_num_segments,
    )

    # Resolve training scales. ``--mixed_scales`` is the preferred interface for
    # non-uniform scale sets such as 20,50,100,200,500. The older min/max/step
    # interface is kept for backward compatibility.
    explicit_mixed_scales = parse_int_list(args.mixed_scales, "--mixed_scales")
    mixed_scales = []
    if explicit_mixed_scales:
        mixed_scales = explicit_mixed_scales
        args.num_loc_min = min(mixed_scales)
        args.num_loc_max = max(mixed_scales)
    elif args.num_loc_min is not None or args.num_loc_max is not None:
        if args.num_loc_min is None or args.num_loc_max is None:
            raise ValueError("--num_loc_min and --num_loc_max must be provided together.")
        if args.num_loc_min <= 0 or args.num_loc_max < args.num_loc_min:
            raise ValueError("Training scale range must satisfy 0 < num_loc_min <= num_loc_max.")
        if args.num_loc_step <= 0:
            raise ValueError("--num_loc_step must be positive.")
        mixed_scales = list(range(args.num_loc_min, args.num_loc_max + 1, args.num_loc_step))
        if mixed_scales[-1] != args.num_loc_max:
            mixed_scales.append(args.num_loc_max)

    mixed_scales = sorted(set(mixed_scales))
    use_mixed_scale = len(mixed_scales) > 0
    if use_mixed_scale:
        args.num_loc = args.val_num_loc or max(mixed_scales)
        scale_tag = "mixed_" + "-".join(map(str, mixed_scales))
        print(f"[INFO] Mixed-scale training enabled: train_scales={mixed_scales}, val/test scale={args.num_loc}")
    else:
        if args.num_loc is None:
            raise ValueError("Please provide --num_loc, --mixed_scales, or --num_loc_min / --num_loc_max for mixed-scale training.")
        scale_tag = str(args.num_loc)

    if args.mixed_samples_per_scale is not None and args.mixed_samples_per_scale <= 0:
        raise ValueError("--mixed_samples_per_scale must be positive when provided.")

    scale_to_train_batch_size = None
    if use_mixed_scale:
        scale_to_train_batch_size = parse_scale_to_int(
            args.batch_size_by_scale,
            mixed_scales,
            "--batch_size_by_scale",
        ) if args.batch_size_by_scale else {s: int(args.batch_size) for s in mixed_scales}


    run_id = make_run_id(CHECKPOINT_CONFIG["timestamp_format"])
    experiment_name = f"{args.algo}_{scale_tag}"
    checkpoint_dir = os.path.join(
        CHECKPOINT_CONFIG["base_dir"],
        args.variant,
        experiment_name,
        run_id,
    )
    log_root = os.path.join(LOG_CONFIG["base_dir"], args.variant)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.reward_normalization == "auto":
        reward_norm_mode = "sqrt_n" if use_mixed_scale else "none"
    else:
        reward_norm_mode = args.reward_normalization

    use_random_range_val = use_mixed_scale and not args.disable_multiscale_val
    if use_random_range_val:
        # Validation covers the full continuous interval specified for training,
        # rather than introducing a separate validation-only scale range.
        val_range_min = args.num_loc_min
        val_range_max = args.num_loc_max
        if val_num_segments <= 0:
            raise ValueError("--val_num_segments must be positive.")
        if val_scales_per_segment <= 0:
            raise ValueError("--val_scales_per_segment must be positive.")
        if val_instances_per_scale <= 0:
            raise ValueError("--val_instances_per_scale must be positive.")
        if val_range_max - val_range_min < val_num_segments:
            raise ValueError(
                "Validation range is too narrow for the requested number of non-empty segments."
            )
    else:
        val_range_min = None
        val_range_max = None

    def build_generator_for_scale(num_loc: int):
        if args.variant == "cvrp":
            tmp_env = CVRPEnv(generator_params={'num_loc': num_loc})
            return tmp_env.generator
        return MTVRPGenerator(
            num_loc=num_loc,
            variant_preset="vrptw",
            capacity=500,
            max_demand=100,
            min_demand=1,
            max_time=4.6,
            map_size=MAP_SIZE,
            num_cities=max(1, 100 // 50),
            num_depots=1,
            speed=1.565,
        )

    # Environment setup
    if args.variant == "cvrp":
        env = CVRPEnv(generator_params={'num_loc': args.num_loc})
    else:
        generator = MTVRPGenerator(
            num_loc=args.num_loc,
            variant_preset="vrptw",
            capacity=500,
            max_demand=100,
            min_demand=1,
            max_time=4.6,        # [FIX] Use normalized time horizon (4.6) to match RL4CO standard & inference normalization
            map_size=MAP_SIZE,   # This param is ignored by generator logic for coords (always 0-1), only implied conceptually
            num_cities=max(1, 100 // 50),
            num_depots=1,
            speed=1.565,           # [FIX] Normalized speed 1.0 works well with max_time 4.6 and space [0,1]
        )
        env = MTVRPEnv(generator)

    patch_env_reward_normalization(env, reward_norm_mode)

    # [Fix] MatNet 需要 cost_matrix，但 MTVRPEnv/CVRPEnv 默认不提供
    # 我们在这里 monkey-patch env.reset 来计算并添加距离矩阵
    if args.algo == "matnet":
        original_reset = env.reset
        def reset_with_cost_matrix(*args, **kwargs):
            td = original_reset(*args, **kwargs)
            locs = td["locs"]
            # 计算欧氏距离矩阵 [batch, n, n]
            cost_matrix = torch.cdist(locs, locs, p=2)
            td.set("cost_matrix", cost_matrix)
            return td
        env.reset = reset_with_cost_matrix
        print("[INFO] Applied MatNet patch: calculating cost_matrix in env.reset")

    # [Fix] N2S needs cost_current in the tensordict
    if args.algo == "n2s":
        original_reset_n2s = env.reset
        original_step_n2s = env.step

        def reset_with_cost_current(*args, **kwargs):
            td = original_reset_n2s(*args, **kwargs)
            if "current_route_length" in td.keys():
                td.set("cost_current", td["current_route_length"])
            else:
                # Fallback if current_route_length is missing (e.g. init)
                batch_size = td["locs"].shape[0]
                device = td["locs"].device
                td.set("cost_current", torch.zeros(batch_size, device=device))
            return td

        def step_with_cost_current(*args, **kwargs):
            td = original_step_n2s(*args, **kwargs)
            if "current_route_length" in td.keys():
                td.set("cost_current", td["current_route_length"])
            else:
                batch_size = td["locs"].shape[0]
                device = td["locs"].device
                td.set("cost_current", torch.zeros(batch_size, device=device))
            return td

        env.reset = reset_with_cost_current
        env.step = step_with_cost_current
        print("[INFO] Applied N2S patch: aliasing cost_current to current_route_length")

    # 使用模块化的策略和模型获取函数
    print(f"[INFO] Building {args.algo} model for {args.variant} with {args.num_loc} locations...")

    try:
        policy = get_policy(
            env,
            args.algo,
            embed_dim=args.embed_dim,
            num_encoder_layers=args.num_encoder_layers,
            num_heads=args.num_heads
        )
    except Exception as e:
        # 如果算法返回 None (例如 mvmoe)，或者构建失败
        if args.algo in ["mvmoe"]:
            policy = None
            print(f"[INFO] Policy for {args.algo} will be created by the model itself.")
        else:
            print(f"[ERROR] Failed to create policy for {args.algo}: {e}")
            print("[INFO] Falling back to standard AttentionModelPolicy")
            policy = AttentionModelPolicy(
                env_name=env.name,
                embed_dim=args.embed_dim,
                num_encoder_layers=args.num_encoder_layers,
                num_heads=args.num_heads,
            )

    # Training sizes per variant
    if args.variant == "cvrp":
        train_size = 100_000
        val_size = 1_000
        test_size = 1_000
    else:
        train_size = 100000
        val_size = 200
        test_size = 200

    # [Optim] Generate persistent datasets
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    original_dataset = env.dataset

    def make_data_file(num_loc: int, phase: str, size: int):
        if use_mixed_scale and phase == "train":
            return os.path.join(data_dir, f"{args.variant}_{scale_tag}_{num_loc}_{phase}_{size}.pt")
        return os.path.join(data_dir, f"{args.variant}_{num_loc}_{phase}_{size}.pt")

    def ensure_dataset_file(num_loc: int, phase: str, size: int):
        filename = make_data_file(num_loc, phase, size)
        with open(filename + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not os.path.exists(filename):
                print(f"[INFO] Generating {phase} dataset scale={num_loc} ({size}) to {filename}...")
                generator_for_scale = build_generator_for_scale(num_loc)
                torch.save(generator_for_scale(size), filename)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        return filename


    def build_validation_segments(min_scale: int, max_scale: int, num_segments: int):
        """Build non-overlapping integer intervals over [min_scale, max_scale].

        All intervals are half-open [low, high), except the final interval whose
        effective upper bound is max_scale + 1. For 20..200 with 9 segments this
        gives [20,40), [40,60), ..., [180,201), displayed as 20-40, ..., 180-200.
        """
        raw_boundaries = np.linspace(min_scale, max_scale, num_segments + 1)
        boundaries = [int(round(x)) for x in raw_boundaries]
        boundaries[0] = min_scale
        boundaries[-1] = max_scale

        if any(boundaries[i] >= boundaries[i + 1] for i in range(num_segments)):
            raise ValueError("Validation segment boundaries are not strictly increasing.")

        segments = []
        for segment_id in range(num_segments):
            low = boundaries[segment_id]
            high_display = boundaries[segment_id + 1]
            high_exclusive = high_display if segment_id < num_segments - 1 else high_display + 1
            segments.append({
                "segment_id": segment_id,
                "low": low,
                "high_display": high_display,
                "high_exclusive": high_exclusive,
            })
        return segments

    def ensure_random_range_validation_file():
        segments = build_validation_segments(
            val_range_min,
            val_range_max,
            val_num_segments,
        )
        total_samples = (
            val_num_segments
            * val_scales_per_segment
            * val_instances_per_scale
        )
        filename = os.path.join(
            data_dir,
            f"{args.variant}_val_range_{val_range_min}-{val_range_max}_"
            f"seg{val_num_segments}_scales{val_scales_per_segment}_"
            f"rep{val_instances_per_scale}_seed{val_scale_seed}.pt",
        )

        with open(filename + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not os.path.exists(filename):
                print(
                    "[INFO] Generating random-range validation dataset: "
                    f"range={val_range_min}-{val_range_max}, "
                    f"segments={val_num_segments}, "
                    f"distinct_scales_per_segment={val_scales_per_segment}, "
                    f"instances_per_scale={val_instances_per_scale}, "
                    f"total={total_samples}, seed={val_scale_seed}"
                )

                py_state = random.getstate()
                np_state = np.random.get_state()
                try:
                    random.seed(val_scale_seed)
                    np.random.seed(val_scale_seed)
                    with torch.random.fork_rng(devices=[]):
                        torch.manual_seed(val_scale_seed)
                        scale_rng = random.Random(val_scale_seed)
                        scale_batches = []
                        sampled_scales = []

                        for segment in segments:
                            candidate_scales = list(range(
                                segment["low"],
                                segment["high_exclusive"],
                            ))
                            if len(candidate_scales) < val_scales_per_segment:
                                raise ValueError(
                                    "Validation segment "
                                    f"[{segment['low']}, {segment['high_display']}] "
                                    f"contains only {len(candidate_scales)} integer scales, "
                                    "fewer than --val_scales_per_segment="
                                    f"{val_scales_per_segment}."
                                )

                            # Sampling without replacement guarantees distinct
                            # validation scales within each segment.
                            segment_scales = sorted(scale_rng.sample(
                                candidate_scales,
                                val_scales_per_segment,
                            ))

                            for scale in segment_scales:
                                generator_for_scale = build_generator_for_scale(scale)
                                scale_batch = generator_for_scale(
                                    val_instances_per_scale
                                ).clone()
                                scale_batch.set(
                                    "_scale_id",
                                    torch.full(
                                        scale_batch.batch_size,
                                        scale,
                                        dtype=torch.long,
                                    ),
                                )
                                scale_batch.set(
                                    "_val_segment_id",
                                    torch.full(
                                        scale_batch.batch_size,
                                        segment["segment_id"],
                                        dtype=torch.long,
                                    ),
                                )
                                scale_batch.set(
                                    "_val_scale_replica_id",
                                    torch.arange(
                                        val_instances_per_scale,
                                        dtype=torch.long,
                                    ),
                                )
                                scale_batches.append(scale_batch)

                            sampled_scales.append({
                                "segment_id": segment["segment_id"],
                                "range": [segment["low"], segment["high_display"]],
                                "scales": segment_scales,
                                "instances_per_scale": val_instances_per_scale,
                            })

                    payload = {
                        "scale_batches": scale_batches,
                        "segments": segments,
                        "sampled_scales": sampled_scales,
                        "total_samples": total_samples,
                        "total_scale_batches": len(scale_batches),
                        "scales_per_segment": val_scales_per_segment,
                        "instances_per_scale": val_instances_per_scale,
                        "seed": val_scale_seed,
                    }
                    torch.save(payload, filename)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
            fcntl.flock(lock_file, fcntl.LOCK_UN)

        payload = torch.load(filename, weights_only=False)
        print("[INFO] Validation sampled scales by segment:")
        for item in payload["sampled_scales"]:
            low, high = item["range"]
            print(
                f"  segment {item['segment_id']:02d} [{low}, {high}]: "
                + ",".join(map(str, item["scales"]))
            )
        return filename, payload["segments"], int(payload["total_samples"])

    if use_mixed_scale:
        # Training size semantics:
        #   epoch_switch:
        #       each epoch trains one scale with train_size samples, matching
        #       the older one-scale-per-epoch behavior.
        #   balanced_grad:
        #       one epoch trains train_size samples in total across all scales.
        #       Therefore each scale receives roughly train_size / #scales
        #       samples. This keeps the total per-epoch training amount aligned
        #       with the original single-scale semantic.
        if args.mixed_train_mode == "balanced_grad":
            base = train_size // len(mixed_scales)
            rem = train_size % len(mixed_scales)
            train_sizes_by_scale = {
                s: base + (1 if i < rem else 0)
                for i, s in enumerate(mixed_scales)
            }
        else:
            train_sizes_by_scale = {s: train_size for s in mixed_scales}

        train_files = {
            s: ensure_dataset_file(s, "train", train_sizes_by_scale[s])
            for s in mixed_scales
        }

        effective_train_size_per_epoch = sum(train_sizes_by_scale.values())

        # Keep one fixed-scale validation/test file as a compatibility fallback.
        val_scale = args.val_num_loc or max(mixed_scales)
        val_file = ensure_dataset_file(val_scale, "val", val_size)
        test_file = ensure_dataset_file(val_scale, "test", test_size)

        if use_random_range_val:
            random_val_file, validation_segments, effective_val_size = ensure_random_range_validation_file()
        else:
            random_val_file = None
            validation_segments = None
            effective_val_size = val_size

        env._mixed_train_dataset_call_idx = 0
    else:
        train_sizes_by_scale = None
        effective_train_size_per_epoch = train_size
        effective_val_size = val_size
        random_val_file = None
        validation_segments = None
        train_file = ensure_dataset_file(args.num_loc, "train", train_size)
        val_file = ensure_dataset_file(args.num_loc, "val", val_size)
        test_file = ensure_dataset_file(args.num_loc, "test", test_size)

    # [Patch] Monkey-patch env.dataset to use the pre-generated files.
    # For mixed-scale training, each newly constructed training dataloader uses
    # one homogeneous scale. The trainer reloads dataloaders every epoch below,
    # so the model sees different scales over training without mixing shapes in
    # the same batch.
    def dataset_from_file_wrapper(batch_size=[], phase="train", filename=None):
        if phase == "train":
            if use_mixed_scale:
                if args.mixed_scale_sampling == "random":
                    scale_rng = random.Random(1234 + env._mixed_train_dataset_call_idx)
                    chosen_scale = scale_rng.choice(mixed_scales)
                else:
                    chosen_scale = mixed_scales[env._mixed_train_dataset_call_idx % len(mixed_scales)]
                env._mixed_train_dataset_call_idx += 1
                env._current_train_num_loc = chosen_scale
                print(f"[INFO] Loading mixed-scale train data: scale={chosen_scale}")
                return TensorDictDataset(torch.load(train_files[chosen_scale], weights_only=False))
            return TensorDictDataset(torch.load(train_file, weights_only=False))
        elif phase == "val":
            return TensorDictDataset(torch.load(val_file, weights_only=False))
        elif phase == "test":
            return TensorDictDataset(torch.load(test_file, weights_only=False))
        return original_dataset(batch_size, phase, filename)

    # Replace bound method on instance
    env.dataset = dataset_from_file_wrapper

    try:
        model = get_model(
            env, policy, args.algo,
            batch_size=args.batch_size,
            train_size=train_size,
            val_size=effective_val_size,
            test_size=test_size,
            lr=args.lr
        )
        print(f"[ICHECK] Created model instance of type: {type(model).__name__}")
        print(f"[ICHECK] Underlying policy type: {type(model.policy).__name__}")
        if use_mixed_scale and args.mixed_train_mode == "balanced_grad":
            patch_model_for_scale_balanced_training(
                model,
                train_files,
                scale_to_train_batch_size,
                seed=42,
                samples_per_scale=args.mixed_samples_per_scale,
            )
            if args.algo == "attention":
                patch_reinforce_rollout_baseline_for_raw_batches(model)
        if use_random_range_val:
            patch_model_for_random_range_validation(
                model,
                random_val_file,
                val_num_segments,
            )
        else:
            patch_model_for_raw_fixed_validation(model)

    except ImportError as e:
        print(f"[ERROR] Model {args.algo} not available: {e}")
        print("[INFO] Falling back to POMO")
        model = POMO(
            env, policy,
            baseline="shared",
            batch_size=args.batch_size,
            train_data_size=train_size,
            val_data_size=effective_val_size,
            optimizer_kwargs={"lr": args.lr},
        )
        if use_mixed_scale and args.mixed_train_mode == "balanced_grad":
            patch_model_for_scale_balanced_training(
                model,
                train_files,
                scale_to_train_batch_size,
                seed=42,
                samples_per_scale=args.mixed_samples_per_scale,
            )
            if args.algo == "attention":
                patch_reinforce_rollout_baseline_for_raw_batches(model)
        if use_random_range_val:
            patch_model_for_random_range_validation(
                model,
                random_val_file,
                val_num_segments,
            )
        else:
            patch_model_for_raw_fixed_validation(model)

    balanced_microbatches_per_update = (
        int(getattr(model, "_mixed_microbatches_per_update", len(mixed_scales)))
        if use_mixed_scale and args.mixed_train_mode == "balanced_grad"
        else 1
    )

    # [Added] 记录关键参数到 hparams.yaml 并强力清理
    # 这确保了在 TensorBoard 和 logs 中能看到使用了什么算法和数据规模
    if hasattr(model, "save_hyperparameters"):
        # 1. 显式保存用户关心的信息
        minimal_hparams = AttributeDict({
            "algo": args.algo,
            "num_loc": scale_tag,
            "variant": args.variant,
            "mixed_scales": mixed_scales if use_mixed_scale else None,
            "val_num_loc": args.num_loc,
            "reward_normalization": reward_norm_mode,
            "multiscale_validation": bool(use_random_range_val),
            "validation_mode": "random_range" if use_random_range_val else "fixed_scale",
            "val_range_min": val_range_min if use_random_range_val else None,
            "val_range_max": val_range_max if use_random_range_val else None,
            "val_num_segments": val_num_segments if use_random_range_val else None,
            "val_scales_per_segment": val_scales_per_segment if use_random_range_val else None,
            "val_instances_per_scale": val_instances_per_scale if use_random_range_val else None,
            "val_samples_per_segment": (
                val_scales_per_segment * val_instances_per_scale
                if use_random_range_val else None
            ),
            "val_scale_seed": val_scale_seed if use_random_range_val else None,
            "effective_val_size": effective_val_size,
            "run_id": run_id,
            "validation_metric": "raw_reward",
            "validation_weight_schemes": normalized_val_weight_schemes,
            "mixed_train_mode": args.mixed_train_mode if use_mixed_scale else "single",
            "train_size_config": train_size,
            "effective_train_size_per_epoch": effective_train_size_per_epoch,
            "train_sizes_by_scale": train_sizes_by_scale if use_mixed_scale else None,
            "batch_size_by_scale": scale_to_train_batch_size if use_mixed_scale else None,
            "mixed_samples_per_scale": args.mixed_samples_per_scale if use_mixed_scale else None,
            "mixed_microbatches_per_update": balanced_microbatches_per_update,
            "gradient_analysis_enabled": bool(use_mixed_scale and args.mixed_train_mode == "balanced_grad" and not args.disable_gradient_analysis),
        })
        model.save_hyperparameters(dict(minimal_hparams))

        # 2. [Aggressive Cleanup] 强力重置 hparams，彻底避免大对象落盘
        if hasattr(model, "hparams"):
            model._hparams = AttributeDict(dict(minimal_hparams))
        if hasattr(model, "_hparams_initial"):
            model._hparams_initial = AttributeDict(dict(minimal_hparams))

    # Checkpointing: one independent best model per validation weighting scheme,
    # plus a separately saved true final/latest checkpoint.
    best_checkpoint_callbacks = []
    if use_random_range_val:
        for scheme_name in normalized_val_weight_schemes:
            monitor_name = f"val/reward_weighted_{scheme_name}"
            callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename=(
                    f"best_{scheme_name}_{run_id}_"
                    "epoch_{epoch:03d}"
                ),
                auto_insert_metric_name=False,
                save_top_k=1,
                save_last=False,
                monitor=monitor_name,
                mode="max",
            )
            best_checkpoint_callbacks.append(callback)
            print(
                f"[INFO] Best checkpoint monitor: {monitor_name} "
                f"-> best_{scheme_name}_{run_id}_epoch_XXX.ckpt"
            )
    else:
        callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"best_fixed_{run_id}_epoch_{{epoch:03d}}",
            auto_insert_metric_name=False,
            save_top_k=1,
            save_last=False,
            monitor="val/reward",
            mode="max",
        )
        best_checkpoint_callbacks.append(callback)
        print("[INFO] Best checkpoint monitor: val/reward")

    last_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"last_{run_id}.ckpt",
    )
    interrupted_checkpoint_path = os.path.join(
        checkpoint_dir,
        f"interrupted_{run_id}.ckpt",
    )
    latest_checkpoint_callback = SaveLatestCheckpointCallback(
        last_path=last_checkpoint_path,
        interrupted_path=interrupted_checkpoint_path,
        save_true_last=CHECKPOINT_CONFIG["save_true_last"],
        save_on_keyboard_interrupt=CHECKPOINT_CONFIG[
            "save_on_keyboard_interrupt"
        ],
    )

    # Logger shares the same run-start timestamp as all checkpoints.
    logger = CSVLogger(
        save_dir=log_root,
        name=experiment_name,
        version=run_id,
    )

    callbacks = []
    if use_random_range_val:
        callbacks.append(
            MultiWeightedValidationCallback(
                val_num_segments,
                VAL_WEIGHT_SCHEMES,
            )
        )
    if use_mixed_scale and args.mixed_train_mode == "balanced_grad" and not args.disable_gradient_analysis:
        callbacks.append(
            MixedScaleGradientAnalysisCallback(
                scales=mixed_scales,
                microbatches_per_update=balanced_microbatches_per_update,
                samples_per_scale=args.mixed_samples_per_scale,
                log_every=args.gradient_analysis_log_every,
                enabled=True,
            )
        )
    callbacks.extend(best_checkpoint_callbacks)
    callbacks.append(latest_checkpoint_callback)

    # Trainer
    trainer_kwargs = {}
    if use_mixed_scale and args.mixed_train_mode == "epoch_switch":
        # Rebuild the train dataloader every epoch so dataset_from_file_wrapper
        # can switch to another homogeneous scale.
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = 1
    if use_mixed_scale and args.mixed_train_mode == "balanced_grad":
        # One optimizer step accumulates all micro-batches in one balanced group.
        # In equal-sample mode this can be larger than len(mixed_scales), because
        # large scales may need several small physical batches to reach S samples.
        trainer_kwargs["accumulate_grad_batches"] = balanced_microbatches_per_update

    trainer = RL4COTrainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        callbacks=callbacks,
        logger=logger,
        **trainer_kwargs,
    )

    run_config_snapshot = {
        "run_id": run_id,
        "variant": args.variant,
        "algo": args.algo,
        "scale_tag": scale_tag,
        "mixed_scales": mixed_scales if use_mixed_scale else None,
        "batch_size_by_scale": scale_to_train_batch_size if use_mixed_scale else None,
        "mixed_samples_per_scale": args.mixed_samples_per_scale if use_mixed_scale else None,
        "mixed_microbatches_per_update": balanced_microbatches_per_update,
        "gradient_analysis_enabled": bool(use_mixed_scale and args.mixed_train_mode == "balanced_grad" and not args.disable_gradient_analysis),
        "reward_normalization_train": reward_norm_mode,
        "validation_metric": "raw_reward",
        "validation": {
            "range_min": val_range_min,
            "range_max": val_range_max,
            "num_segments": val_num_segments,
            "scales_per_segment": val_scales_per_segment,
            "instances_per_scale": val_instances_per_scale,
            "scale_seed": val_scale_seed,
            "weight_schemes_normalized": normalized_val_weight_schemes,
        },
        "checkpoint_dir": checkpoint_dir,
        "last_checkpoint": last_checkpoint_path,
        "interrupted_checkpoint": interrupted_checkpoint_path,
    }
    save_run_config_snapshot(
        os.path.join(checkpoint_dir, "run_config.json"),
        run_config_snapshot,
    )

    print(f"[RUN] run_id={run_id}")
    print(f"[RUN] checkpoint_dir={checkpoint_dir}")
    print(f"[RUN] log_dir={os.path.join(log_root, experiment_name, run_id)}")
    print(f"[START] variant={args.variant} algo={args.algo} num_loc={scale_tag} batch_size={args.batch_size}")
    print(f"[CONFIG] embed_dim={args.embed_dim}, num_encoder_layers={args.num_encoder_layers}, num_heads={args.num_heads}")
    print(f"[CONFIG] lr={args.lr}, epochs={args.epochs}, train_size_config={train_size}, val_size={effective_val_size}")
    print(f"[CONFIG] effective_train_size_per_epoch={effective_train_size_per_epoch}")
    print(f"[CONFIG] reward_normalization={reward_norm_mode}, random_range_validation={use_random_range_val}")
    if use_random_range_val:
        print(
            "[CONFIG] validation_range="
            f"{val_range_min}-{val_range_max}, "
            f"segments={val_num_segments}, "
            f"distinct_scales_per_segment={val_scales_per_segment}, "
            f"instances_per_scale={val_instances_per_scale}, "
            f"samples_per_segment="
            f"{val_scales_per_segment * val_instances_per_scale}, "
            f"seed={val_scale_seed}, "
            f"batch_size_per_sampled_scale={val_instances_per_scale}"
        )
    if use_mixed_scale:
        print(f"[CONFIG] mixed_train_mode={args.mixed_train_mode}, balanced_accumulate_grad_batches={balanced_microbatches_per_update if args.mixed_train_mode == 'balanced_grad' else 1}")
        if args.mixed_train_mode == "balanced_grad":
            print(f"[CONFIG] balanced_train_size_semantics=total_across_scales, train_sizes_by_scale={train_sizes_by_scale}")
            print(f"[CONFIG] batch_size_by_scale={scale_to_train_batch_size}, mixed_samples_per_scale={args.mixed_samples_per_scale}")
            print(f"[CONFIG] gradient_analysis_enabled={not args.disable_gradient_analysis}, log_every={args.gradient_analysis_log_every}")
    interrupted_by_user = False
    try:
        trainer.fit(model)
    except KeyboardInterrupt:
        interrupted_by_user = True
        latest_checkpoint_callback._interrupted = True
        if (
            CHECKPOINT_CONFIG["save_on_keyboard_interrupt"]
            and not latest_checkpoint_callback._saved_interrupted
        ):
            latest_checkpoint_callback._save(
                trainer,
                interrupted_checkpoint_path,
                "interrupted/latest fallback",
            )
            latest_checkpoint_callback._saved_interrupted = True
        if trainer.is_global_zero:
            print("[INFO] Training interrupted by Ctrl+C; latest state saved.")

    if trainer.is_global_zero:
        print("[RESULT] Best checkpoints by validation criterion:")
        for callback in best_checkpoint_callbacks:
            print(
                f"  monitor={callback.monitor}, "
                f"score={callback.best_model_score}, "
                f"path={callback.best_model_path}"
            )
        if interrupted_by_user:
            print(
                f"[RESULT] Interrupted/latest checkpoint: "
                f"{interrupted_checkpoint_path}"
            )
        else:
            print(f"[RESULT] True final checkpoint: {last_checkpoint_path}")
