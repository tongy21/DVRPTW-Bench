"""Checkpoint loading shared by the dynamic RL evaluation entry point."""

from __future__ import annotations

import inspect

import torch
from rl4co.models import AttentionModelPolicy, POMO, REINFORCE

try:
    from rl4co.models import EAS, MDAM, PPO, PolyNet, SymNCO

    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    ADVANCED_MODELS_AVAILABLE = False


def load_model_by_solver(solver: str, checkpoint: str, env=None, policy=None, device=None):
    """Load an RL4CO model while ignoring non-constructor checkpoint metadata."""
    solver_to_model = {"attention": REINFORCE, "pomo": POMO}
    if ADVANCED_MODELS_AVAILABLE:
        solver_to_model.update(
            {
                "am-ppo": PPO,
                "symnco": SymNCO,
                "eas": EAS,
                "mdam": MDAM,
                "polynet": PolyNet,
            }
        )
    if solver not in solver_to_model:
        raise ValueError(
            f"Unknown or unavailable solver {solver!r}; "
            f"available choices are {sorted(solver_to_model)}"
        )

    model_cls = solver_to_model[solver]
    payload = torch.load(checkpoint, map_location=device or "cpu", weights_only=False)
    for key in ("random_state", "rng_state", "pytorch-lightning_version"):
        payload.pop(key, None)
    parameters = dict(payload.get("hyper_parameters", {}))
    try:
        allowed = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
        parameters = {key: value for key, value in parameters.items() if key in allowed}
    except (TypeError, ValueError):
        allowed = set(parameters)
    for key in ("algo", "_target_", "num_loc", "variant", "variant_preset", "map_size"):
        parameters.pop(key, None)
    if env is not None and "env" in allowed:
        parameters["env"] = env
    if policy is not None and "policy" in allowed:
        parameters["policy"] = policy
    model = model_cls(**parameters)
    model.load_state_dict(payload["state_dict"], strict=False)
    return model.to(device) if device is not None else model
