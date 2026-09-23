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

Stability policy for the four controlled solvers (Attention, POMO, SymNCO,
PolyNet): when the per-device micro-batch is below 128, gradient accumulation
automatically restores an exact effective per-device batch of 128.
"""
import argparse
import inspect
import time
import random
import math
import os
import fcntl
import numpy as np
from rl4co.envs.routing import CVRPEnv
from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.models import AttentionModelPolicy, REINFORCE, POMO
from rl4co.utils.trainer import RL4COTrainer
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.utilities.parsing import AttributeDict
from torch.utils.data import Dataset
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

# The four solvers used in the controlled RL comparison. For these models,
# micro-batches smaller than the target are accumulated to an exact effective
# per-device batch size, so optimizer-step semantics remain comparable.
CONTROLLED_ACCUMULATION_ALGOS = {"attention", "pomo", "symnco", "polynet"}

# --- Constants for TW variant ---
DEMAND_RANGE = (1, 10)
MAP_SIZE = (1000, 1000)


class GradientNormMonitor(Callback):
    """Log gradient norm at optimizer-step boundaries for stability diagnosis."""

    def __init__(self, log_every_n_steps: int = 50):
        super().__init__()
        self.log_every_n_steps = max(1, int(log_every_n_steps))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        squared_norm = None
        has_nonfinite = False
        for param in pl_module.parameters():
            if param.grad is None:
                continue
            grad = param.grad.detach().float()
            if not torch.isfinite(grad).all():
                has_nonfinite = True
                break
            value = grad.pow(2).sum()
            squared_norm = value if squared_norm is None else squared_norm + value

        if has_nonfinite:
            pl_module.log(
                "train/grad_nonfinite",
                torch.tensor(1.0, device=pl_module.device),
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
            return

        if squared_norm is not None:
            pl_module.log(
                "train/grad_norm",
                squared_norm.sqrt(),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                sync_dist=True,
            )


class BatchedTensorDictDataset(Dataset):
    """O(1)-construction Dataset backed directly by one TensorDict.

    Unlike RL4CO's legacy TensorDictDataset, this class does not disassemble
    every sample into a Python dictionary. PyTorch 2.x DataLoader calls
    ``__getitems__`` with a whole list of indices, so one TensorDict slice
    produces an already-batched TensorDict. This keeps startup fast even when
    every DDP rank opens the same 100k-instance dataset.
    """

    def __init__(self, td):
        self.data = td
        try:
            self.data_len = int(td.batch_size[0])
        except Exception:
            self.data_len = int(len(td))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        # Compatibility fallback for PyTorch versions that do not use
        # Dataset.__getitems__ for batched fetching.
        return self.data[index]

    def __getitems__(self, indices):
        # Batched indexing avoids constructing one Python object per sample.
        return self.data[indices]

    def add_key(self, key, value):
        """Attach/replace a per-instance field, e.g. rollout baseline reward."""
        if len(value) != self.data_len:
            raise ValueError(
                f"Data and extra field '{key}' must have the same length: "
                f"{self.data_len} != {len(value)}"
            )
        self.data.set(key, value)
        return self

    @staticmethod
    def collate_fn(batch):
        # With __getitems__, DataLoader passes an already-batched TensorDict.
        if not isinstance(batch, list):
            return batch
        # Conservative fallback for older DataLoader implementations.
        if len(batch) == 0:
            return batch
        return torch.stack(batch, dim=0)


def build_dataset(td):
    """Build a direct TensorDict view without O(num_samples) Python expansion."""
    return BatchedTensorDictDataset(td)


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


def get_model(
    env,
    policy,
    algo: str,
    batch_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    lr: float = 1e-4,
    val_batch_size: int | None = None,
    dataloader_num_workers: int = 0,
    shuffle_train_dataloader: bool = True,
    attention_baseline_kwargs: dict | None = None,
    attention_reward_scale: str | None = None,
):
    """
    根据算法类型返回对应的模型
    """
    common_kwargs = {
        "batch_size": batch_size,
        "val_batch_size": val_batch_size,
        "test_batch_size": val_batch_size,
        "train_data_size": train_size,
        "val_data_size": val_size,
        "test_data_size": test_size,
        "optimizer_kwargs": {"lr": lr},
        # We use a persistent dataset file. Unlike RL4CO's default freshly
        # generated data, it must be shuffled every epoch.
        "shuffle_train_dataloader": shuffle_train_dataloader,
        "dataloader_num_workers": dataloader_num_workers,
        }

    if algo == "attention":
        reinforce_kwargs = dict(common_kwargs)
        reinforce_signature = inspect.signature(REINFORCE.__init__).parameters

        if attention_baseline_kwargs and "baseline_kwargs" in reinforce_signature:
            reinforce_kwargs["baseline_kwargs"] = attention_baseline_kwargs

        # Recent RL4CO versions can standardize/scale the advantage internally.
        # Keep it optional so the first controlled rerun changes only batching,
        # shuffling and clipping.
        if (
            attention_reward_scale is not None
            and "reward_scale" in reinforce_signature
        ):
            reinforce_kwargs["reward_scale"] = attention_reward_scale

        return REINFORCE(
            env, policy,
            baseline="rollout",
            **reinforce_kwargs
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL4CO on VRP variants (CVRP or VRPTW)")
    parser.add_argument("--num_loc", type=int, required=True,
                        help="Number of customer locations")
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
                        help="Total target number of training epochs. When resuming, training continues until this total is reached.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible batch-size ablations.")
    parser.add_argument("--val_batch_size", type=int, default=None,
                        help="Validation and rollout-baseline batch size. Defaults to training batch size.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers per DDP process. Keep small because data are already in memory.")
    parser.add_argument("--reference_batch_size", type=int, default=128,
                        help="Target effective batch size per device for attention, POMO, SymNCO, and PolyNet. If batch_size is smaller, gradients are accumulated to reach this value exactly.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=None,
                        help="Explicit gradient-accumulation factor. If omitted, the four controlled algorithms automatically accumulate to reference_batch_size when batch_size is smaller.")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0,
                        help="Global gradient-norm clipping value. Set <=0 to disable.")
    parser.add_argument("--grad_norm_log_interval", type=int, default=50,
                        help="Log train/grad_norm every N optimizer steps.")
    parser.add_argument("--attention_reward_scale", type=str, choices=["none", "scale", "norm"], default="none",
                        help="Optional RL4CO advantage scaling for Attention. Keep 'none' for the first controlled rerun.")
    parser.add_argument("--attention_warmup_exp_beta", type=float, default=0.8,
                        help="EMA beta used by the rollout-baseline warmup at the reference batch size.")
    parser.add_argument("--attention_warmup_epochs", type=int, default=1,
                        help="Number of rollout-baseline warmup epochs.")
    parser.add_argument("--disable_train_shuffle", action="store_true",
                        help="Disable shuffling of the persistent training dataset (not recommended).")
    parser.add_argument("--disable_data_cache", action="store_true",
                        help="Reload TensorDict files on every env.dataset call instead of caching them in CPU memory.")
    parser.add_argument("--disable_data_mmap", action="store_true",
                        help="Disable torch.load(mmap=True). Memory mapping is enabled by default to avoid eight DDP ranks eagerly deserializing eight full copies of the same dataset.")
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/{variant}/{algo}_{num_loc}/last.ckpt. Fails if the checkpoint does not exist.",
    )
    resume_group.add_argument(
        "--resume_ckpt",
        type=str,
        default=None,
        help="Resume from an explicitly specified Lightning checkpoint path.",
    )
    args = parser.parse_args()
    seed_everything(args.seed, workers=True)

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.reference_batch_size <= 0:
        raise ValueError("--reference_batch_size must be positive")

    if args.accumulate_grad_batches is not None:
        if args.accumulate_grad_batches <= 0:
            raise ValueError("--accumulate_grad_batches must be positive")
        grad_accumulation = args.accumulate_grad_batches
    elif (
        args.algo in CONTROLLED_ACCUMULATION_ALGOS
        and args.batch_size < args.reference_batch_size
    ):
        # Exact matching is intentional. Using ceil() would silently turn, for
        # example, batch_size=48 into an effective batch of 144 rather than 128,
        # weakening the fairness of batch-size ablations.
        if args.reference_batch_size % args.batch_size != 0:
            raise ValueError(
                "Automatic exact gradient accumulation requires "
                "reference_batch_size to be divisible by batch_size. "
                f"Got reference_batch_size={args.reference_batch_size}, "
                f"batch_size={args.batch_size}. Choose a divisor such as "
                "16/32/64/128, or set --accumulate_grad_batches explicitly."
            )
        grad_accumulation = args.reference_batch_size // args.batch_size
    else:
        grad_accumulation = 1

    effective_batch_per_device = args.batch_size * grad_accumulation
    if (
        args.algo in CONTROLLED_ACCUMULATION_ALGOS
        and args.batch_size < args.reference_batch_size
        and effective_batch_per_device != args.reference_batch_size
    ):
        raise RuntimeError(
            "Internal error: effective batch size did not match the requested "
            "reference batch size."
        )
    resolved_val_batch_size = args.val_batch_size or args.batch_size

    # RL4CO's rollout baseline uses an exponential baseline in its warmup epoch.
    # With a smaller micro-batch, beta=0.8 would update four times as often when
    # batch_size changes from 128 to 32. Adjust beta to preserve approximately
    # the same decay per reference-sized group of samples.
    if args.algo == "attention" and args.batch_size < args.reference_batch_size:
        beta_exponent = args.batch_size / float(args.reference_batch_size)
        resolved_warmup_beta = args.attention_warmup_exp_beta ** beta_exponent
    else:
        resolved_warmup_beta = args.attention_warmup_exp_beta

    attention_baseline_kwargs = {
        "n_epochs": args.attention_warmup_epochs,
        "exp_beta": resolved_warmup_beta,
    }
    attention_reward_scale = (
        None if args.attention_reward_scale == "none"
        else args.attention_reward_scale
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
        train_size = 100_000
        val_size = 200
        test_size = 200

    # [Optim] Generate persistent datasets
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    train_file = os.path.join(data_dir, f"{args.variant}_{args.num_loc}_train_{train_size}.pt")
    val_file = os.path.join(data_dir, f"{args.variant}_{args.num_loc}_val_{val_size}.pt")
    test_file = os.path.join(data_dir, f"{args.variant}_{args.num_loc}_test_{test_size}.pt")

    # Check/Generate data with locking
    for filename, size, name in [(train_file, train_size, "train"), (val_file, val_size, "val"), (test_file, test_size, "test")]:
        with open(filename + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not os.path.exists(filename):
                print(f"[INFO] Generating {name} dataset ({size}) to {filename}...")
                torch.save(env.generator(size), filename)
            fcntl.flock(lock_file, fcntl.LOCK_UN)

    # [Patch] Monkey-patch env.dataset to use the pre-generated files
    # This avoids loading data in __init__ which can cause issues with some models (PolyNet etc)
    # and keeps distributed training efficient
    original_dataset = env.dataset
    td_cache = {}
    dataset_cache = {}

    def _process_rank() -> int:
        # Lightning's DDP subprocesses expose LOCAL_RANK/RANK before the
        # process group is initialized. The original rank-0 process may not.
        for key in ("LOCAL_RANK", "RANK"):
            value = os.environ.get(key)
            if value is not None:
                try:
                    return int(value)
                except ValueError:
                    pass
        return 0

    def _torch_load_dataset(path):
        load_kwargs = {
            "weights_only": False,
            "map_location": "cpu",
        }
        mmap_enabled = not args.disable_data_mmap
        if mmap_enabled:
            try:
                if "mmap" in inspect.signature(torch.load).parameters:
                    load_kwargs["mmap"] = True
                elif _process_rank() == 0:
                    print("[WARN] This PyTorch version does not support torch.load(mmap=True); falling back to eager loading.")
            except (TypeError, ValueError):
                # Some wrapped/built-in callables do not expose a signature.
                load_kwargs["mmap"] = True

        try:
            return torch.load(path, **load_kwargs)
        except (TypeError, RuntimeError, ValueError) as exc:
            if load_kwargs.pop("mmap", None) is not None:
                if _process_rank() == 0:
                    print(
                        f"[WARN] Memory-mapped loading failed for {path}: {exc}. "
                        "Falling back to regular CPU loading."
                    )
                return torch.load(path, **load_kwargs)
            raise

    def load_tensordict(path):
        if args.disable_data_cache:
            return _torch_load_dataset(path)
        if path not in td_cache:
            is_rank_zero = _process_rank() == 0
            mode = "memory map" if not args.disable_data_mmap else "CPU cache"
            if is_rank_zero:
                print(f"[INFO] Opening dataset with {mode}: {path}", flush=True)
            start = time.perf_counter()
            td_cache[path] = _torch_load_dataset(path)
            if is_rank_zero:
                elapsed = time.perf_counter() - start
                try:
                    samples = int(td_cache[path].batch_size[0])
                except Exception:
                    samples = len(td_cache[path])
                print(
                    f"[INFO] Opened dataset: {path} "
                    f"samples={samples}, elapsed={elapsed:.3f}s",
                    flush=True,
                )
        return td_cache[path]

    def load_dataset(path):
        # Cache both the TensorDict and its O(1) Dataset view. This avoids the
        # legacy TensorDictDataset list-comprehension in every DDP rank.
        if args.disable_data_cache:
            return build_dataset(load_tensordict(path))
        if path not in dataset_cache:
            start = time.perf_counter()
            dataset_cache[path] = build_dataset(load_tensordict(path))
            if _process_rank() == 0:
                print(
                    f"[INFO] Built BatchedTensorDictDataset: {path}, "
                    f"elapsed={time.perf_counter() - start:.3f}s",
                    flush=True,
                )
        return dataset_cache[path]

    def dataset_from_file_wrapper(batch_size=[], phase="train", filename=None):
        if phase == "train":
            return load_dataset(train_file)
        elif phase == "val":
            return load_dataset(val_file)
        elif phase == "test":
            return load_dataset(test_file)
        return original_dataset(batch_size, phase, filename)

    # Replace bound method on instance
    env.dataset = dataset_from_file_wrapper

    try:
        model = get_model(
            env, policy, args.algo,
            batch_size=args.batch_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            lr=args.lr,
            val_batch_size=resolved_val_batch_size,
            dataloader_num_workers=args.num_workers,
            shuffle_train_dataloader=not args.disable_train_shuffle,
            attention_baseline_kwargs=attention_baseline_kwargs,
            attention_reward_scale=attention_reward_scale,
        )
        print(f"[ICHECK] Created model instance of type: {type(model).__name__}")
        print(f"[ICHECK] Underlying policy type: {type(model.policy).__name__}")

    except ImportError as e:
        print(f"[ERROR] Model {args.algo} not available: {e}")
        print("[INFO] Falling back to POMO")
        model = POMO(
            env, policy,
            baseline="shared",
            batch_size=args.batch_size,
            train_data_size=train_size,
            val_data_size=val_size,
            val_batch_size=resolved_val_batch_size,
            test_batch_size=resolved_val_batch_size,
            optimizer_kwargs={"lr": args.lr},
            shuffle_train_dataloader=not args.disable_train_shuffle,
            dataloader_num_workers=args.num_workers,
        )

    # [Added] 记录关键参数到 hparams.yaml 并强力清理
    # 这确保了在 TensorBoard 和 logs 中能看到使用了什么算法和数据规模
    if hasattr(model, "save_hyperparameters"):
        # 1. 显式保存用户关心的信息
        minimal_hparams = AttributeDict({
            "algo": args.algo,
            "num_loc": args.num_loc,
            "variant": args.variant,
            "seed": args.seed,
            "embed_dim": args.embed_dim,
            "num_encoder_layers": args.num_encoder_layers,
            "num_heads": args.num_heads,
            "learning_rate": args.lr,
            "micro_batch_size_per_device": args.batch_size,
            "reference_batch_size_per_device": args.reference_batch_size,
            "automatic_reference_batch_enabled": (
                args.algo in CONTROLLED_ACCUMULATION_ALGOS
            ),
            "accumulate_grad_batches": grad_accumulation,
            "effective_batch_size_per_device": effective_batch_per_device,
            "val_batch_size": resolved_val_batch_size,
            "gradient_clip_val": args.gradient_clip_val,
            "shuffle_train_dataloader": not args.disable_train_shuffle,
            "attention_warmup_exp_beta_resolved": resolved_warmup_beta,
            "attention_reward_scale": args.attention_reward_scale,
        })
        model.save_hyperparameters(dict(minimal_hparams))

        # 2. [Aggressive Cleanup] 强力重置 hparams，彻底避免大对象落盘
        if hasattr(model, "hparams"):
            model._hparams = AttributeDict(dict(minimal_hparams))
        if hasattr(model, "_hparams_initial"):
            model._hparams_initial = AttributeDict(dict(minimal_hparams))

    # Checkpointing
    checkpoint_dir = os.path.join(
        "checkpoints", args.variant, f"{args.algo}_{args.num_loc}"
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch_{epoch:03d}",
        save_top_k=1,
        save_last=True,
        monitor="val/reward",
        mode="max",
    )

    # Resolve checkpoint restoration before constructing the trainer. Lightning
    # restores the model, optimizer, scheduler, epoch/global-step, precision,
    # and callback states when ckpt_path is passed to trainer.fit().
    resume_ckpt_path = None
    if args.resume_ckpt is not None:
        resume_ckpt_path = os.path.abspath(
            os.path.expanduser(args.resume_ckpt)
        )
    elif args.resume:
        resume_ckpt_path = os.path.abspath(
            os.path.join(checkpoint_dir, "last.ckpt")
        )

    if resume_ckpt_path is not None:
        if not os.path.isfile(resume_ckpt_path):
            raise FileNotFoundError(
                "Requested resume checkpoint does not exist: "
                f"{resume_ckpt_path}"
            )

        # Read only lightweight metadata here to provide an explicit, early
        # check. The full training state is restored later by Lightning.
        try:
            resume_metadata = torch.load(
                resume_ckpt_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Compatibility with older PyTorch versions without weights_only.
            resume_metadata = torch.load(
                resume_ckpt_path,
                map_location="cpu",
            )

        saved_epoch = resume_metadata.get("epoch")
        saved_global_step = resume_metadata.get("global_step")
        completed_epochs = (int(saved_epoch) + 1) if saved_epoch is not None else None

        saved_hparams = resume_metadata.get("hyper_parameters", {}) or {}
        expected_resume_config = {
            "algo": args.algo,
            "variant": args.variant,
            "num_loc": args.num_loc,
            "embed_dim": args.embed_dim,
            "num_encoder_layers": args.num_encoder_layers,
            "num_heads": args.num_heads,
        }
        incompatible = []
        for key, current_value in expected_resume_config.items():
            if key in saved_hparams and saved_hparams[key] != current_value:
                incompatible.append(
                    f"{key}: checkpoint={saved_hparams[key]!r}, "
                    f"current={current_value!r}"
                )
        if incompatible:
            raise ValueError(
                "The requested checkpoint is incompatible with the current "
                "training configuration:\n  - " + "\n  - ".join(incompatible)
            )

        if completed_epochs is not None and args.epochs <= completed_epochs:
            raise ValueError(
                "--epochs is the total target epoch count when resuming. "
                f"The checkpoint has already completed {completed_epochs} "
                f"epoch(s), but --epochs={args.epochs}. Set --epochs to a "
                "larger total value."
            )

        print(
            "[RESUME] "
            f"checkpoint={resume_ckpt_path}, "
            f"saved_epoch={saved_epoch}, "
            f"saved_global_step={saved_global_step}, "
            f"target_total_epochs={args.epochs}"
        )
        del resume_metadata

    # Logger
    # Structure: logs/{variant}/{num_loc}/{version}. CSVLogger deletes an
    # existing metrics.csv when a logger is opened, so resumed runs use a new
    # version directory instead of destroying the original training history.
    if resume_ckpt_path is not None:
        resume_run_id = os.environ.get("RL4CO_RESUME_RUN_ID")
        if resume_run_id is None:
            resume_run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            os.environ["RL4CO_RESUME_RUN_ID"] = resume_run_id
        logger_version = f"{args.algo}_resume_{resume_run_id}"
    else:
        logger_version = args.algo

    logger = CSVLogger(
        save_dir=f"logs/{args.variant}",
        name=str(args.num_loc),
        version=logger_version,
    )

    # Trainer
    callbacks = [
        checkpoint_callback,
        GradientNormMonitor(args.grad_norm_log_interval),
    ]

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "accelerator": "gpu",
        "callbacks": callbacks,
        "logger": logger,
        "accumulate_grad_batches": grad_accumulation,
    }
    if args.gradient_clip_val > 0:
        trainer_kwargs.update({
            "gradient_clip_val": args.gradient_clip_val,
            "gradient_clip_algorithm": "norm",
        })

    trainer = RL4COTrainer(**trainer_kwargs)

    print(f"[START] variant={args.variant} algo={args.algo} num_loc={args.num_loc} batch_size={args.batch_size}")
    print(f"[CONFIG] embed_dim={args.embed_dim}, num_encoder_layers={args.num_encoder_layers}, num_heads={args.num_heads}")
    print(f"[CONFIG] lr={args.lr}, epochs={args.epochs}, train_size={train_size}, val_size={val_size}, seed={args.seed}")
    print(f"[LOG] version={logger_version}, dir={logger.log_dir}")
    print(
        "[STABILITY] "
        f"algo={args.algo}, "
        f"micro_batch_per_device={args.batch_size}, "
        f"accumulate_grad_batches={grad_accumulation}, "
        f"effective_batch_per_device={effective_batch_per_device}, "
        f"target_effective_batch_per_device={args.reference_batch_size}, "
        f"auto_target_enabled={args.algo in CONTROLLED_ACCUMULATION_ALGOS}, "
        f"gradient_clip_val={args.gradient_clip_val}"
    )
    print(
        "[DATA] "
        f"shuffle_train={not args.disable_train_shuffle}, "
        f"num_workers_per_rank={args.num_workers}, "
        f"dataset_impl=BatchedTensorDictDataset, "
        f"cpu_cache={not args.disable_data_cache}, "
        f"mmap={not args.disable_data_mmap}"
    )
    if args.algo == "attention":
        print(
            "[ATTENTION] "
            f"rollout_warmup_beta={resolved_warmup_beta:.6f}, "
            f"reward_scale={args.attention_reward_scale}, "
            f"val_batch_size={resolved_val_batch_size}"
        )
    trainer.fit(model, ckpt_path=resume_ckpt_path)
