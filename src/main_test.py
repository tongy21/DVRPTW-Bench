import os
import json
import glob
import random
import numpy as np
import argparse
import sys
import torch
import warnings
import time
import math

# 屏蔽冗余的 Lightning 警告
warnings.filterwarnings("ignore", ".*Attribute 'env' is an instance of `nn.Module`.*")
warnings.filterwarnings("ignore", ".*Attribute 'policy' is an instance of `nn.Module`.*")

# RL4CO imports
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.envs.routing.mtvrp.generator import MTVRPGenerator
from rl4co.models import AttentionModelPolicy
from solvers.rl.model_loader import load_model_by_solver, ADVANCED_MODELS_AVAILABLE

# 导入求解器
from solvers.base_solver import BaseSolver
from solvers.nn_2opt_solver import NNSolver
from solvers.aco_solver import ACOSolver
from solvers.tabu_solver import TabuSolver
from solvers.ortools_solver import ORToolsSolver
from solvers.hgs_solver import HGSSolver
from solvers.lk_solver import PythonLKSolver as LKSolver
from solvers.alns_solver import ALNSSolver
from solvers.rl_solver import RLSolver
from simulation.route_validation import collect_ontime_served_orders, validate_and_normalize_routes


# === 全局配置 ===
DEBUG_MODE = False # 调试模式，只跑一个实例

# [配置] 全局求解时间限制 (秒)
SOLVER_TIME_LIMIT = 1.0
ALLOW_LATE_RETURN = True
# 是否覆盖已存在的结果
# True: 即使结果文件存在也重新跑，并覆盖（删除旧的）
# False: 如果结果文件已存在则跳过
OVERWRITE = False

# === 求解器配置 ===
RUN_ALL_SOLVERS = True # 开启所有求解器对比
SINGLE_SOLVER_CLASS = RLSolver # 仅当 RUN_ALL_SOLVERS=False 时生效

SOLVER_LIST = [
    NNSolver,
    ACOSolver,
    TabuSolver,
    ORToolsSolver,
    HGSSolver,
    LKSolver,
    RLSolver,
]
# HGSSolver,
# RL_ALGO_LIST = ["attention"]
RL_ALGO_LIST = ["pomo", "symnco", "polynet", "attention"] # 仅 RLSolver 可用，且部分算法可能因模型缺失而无法运行

# === RL 模型构建辅助函数 (适配 eval_twvrp_full.py) ===
def build_env_and_policy(algo: str, num_loc: int):
    # Configuration matching train.py for VRPTW
    generator = MTVRPGenerator(
        num_loc=num_loc,
        variant_preset="vrptw",
        capacity=500,
        max_demand=100,
        min_demand=1,
        max_time=4.6,
        map_size=1000,
        num_cities=max(1, 100 // 50), # Logic from train.py/eval_twvrp_full.py
        num_depots=1,
        speed=1.565,
    )
    env = MTVRPEnv(generator)

    if algo in ["attention", "pomo", "am-ppo", "eas"]:
        policy = AttentionModelPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=3, num_heads=8)
    elif algo == "symnco" and ADVANCED_MODELS_AVAILABLE:
        from rl4co.models.zoo.symnco import SymNCOPolicy
        policy = SymNCOPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=3, num_heads=8)
    elif algo == "polynet" and ADVANCED_MODELS_AVAILABLE:
        from rl4co.models.zoo.polynet.policy import PolyNetPolicy
        # Assuming k=40 as requested
        policy = PolyNetPolicy(env_name=env.name, embed_dim=128, num_encoder_layers=3, num_heads=8, k=40)
    else:
        raise ValueError(f"Unsupported or unavailable solver for TWVRP: {algo}")

    return env, policy

def _resolve_rl_model_tag(args, current_scale: int) -> str:
    """Choose which checkpoint tag to use for the RL solver.

    Rule:
      - If --rl_model_tag is provided, use it directly.
        Examples:
          --rl_model_tag 100                  -> polynet_100
          --rl_model_tag mixed_20-30-40-50    -> polynet_mixed_20-30-40-50

      - If --rl_model_tag is not provided, automatically use the test scale,
        with all scales >= 500 mapped to the 500-scale checkpoint.
        Examples:
          test_N=20   -> polynet_20
          test_N=100  -> polynet_100
          test_N=500  -> polynet_500
          test_N=1000 -> polynet_500

    The evaluation environment is still built with the real test scale.
    This only controls which checkpoint is loaded.
    """
    if args.rl_model_tag:
        return str(args.rl_model_tag)

    return str(min(int(current_scale), 500))



def _candidate_ckpt_roots(user_root: str | None) -> list[str]:
    """Return checkpoint roots in priority order.

    New project layout (preferred by default):
        <project>/rl/checkpoints/<variant>/<algo>_<model_tag>/<run_id>/...

    Legacy layout:
        <project>/solvers/rl/checkpoints/<variant>/<algo>_<model_tag>/...

    ``user_root`` may point to the checkpoints root, the variant directory, or
    an individual ``<algo>_<model_tag>`` experiment directory.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    cwd = os.getcwd()
    candidates = [
        user_root,
        os.path.join(project_root, "rl", "checkpoints"),
        os.path.join(project_root, "solvers", "rl", "checkpoints"),
        # Compatibility with training launched from the project root.
        os.path.join(project_root, "checkpoints"),
        os.path.join(cwd, "rl", "checkpoints"),
        os.path.join(cwd, "solvers", "rl", "checkpoints"),
        os.path.join(cwd, "checkpoints"),
    ]

    roots = []
    seen = set()
    for root in candidates:
        if not root:
            continue
        normalized = os.path.abspath(os.path.expanduser(str(root)))
        if normalized not in seen:
            seen.add(normalized)
            roots.append(normalized)
    return roots


def _validate_ckpt_selector(value: str | None, label: str) -> str | None:
    """Validate a selector that must be one directory/file-name component."""
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if value in {".", ".."} or os.path.basename(value) != value:
        raise ValueError(f"{label} must be a single name, not a path: {value!r}")
    return value


def _candidate_experiment_dirs(
    root: str,
    variant: str,
    experiment_name: str,
) -> list[str]:
    """Interpret ``root`` at several useful directory levels."""
    candidates = [
        os.path.join(root, variant, experiment_name),  # checkpoints root
        os.path.join(root, experiment_name),           # variant root
    ]

    # Exact experiment directory supplied by the user.
    if os.path.basename(os.path.normpath(root)) == experiment_name:
        candidates.insert(0, root)

    # Exact run directory supplied by the user: use its parent experiment dir.
    parent = os.path.dirname(os.path.normpath(root))
    if os.path.basename(parent) == experiment_name:
        candidates.insert(0, parent)

    result = []
    seen = set()
    for path in candidates:
        normalized = os.path.abspath(path)
        if normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _checkpoint_epoch(path: str) -> int:
    """Extract ``epoch_XXX`` for deterministic best-checkpoint ordering."""
    import re

    match = re.search(r"epoch_(\d+)\.ckpt$", os.path.basename(path))
    return int(match.group(1)) if match else -1


def _new_run_sort_key(run_dir: str):
    """Sort run directories newest first, using checkpoint mtimes when present."""
    checkpoint_files = glob.glob(os.path.join(run_dir, "*.ckpt"))
    newest_mtime = max(
        (os.path.getmtime(path) for path in checkpoint_files),
        default=os.path.getmtime(run_dir),
    )
    return newest_mtime, os.path.basename(run_dir)


def _pick_new_layout_checkpoint(
    experiment_dir: str,
    val_strategy: str | None,
    run_id: str | None,
) -> str | None:
    """Resolve checkpoints written by the current training script.

    New layout:
        experiment_dir/<run_id>/
            best_<strategy>_<run_id>_epoch_<epoch>.ckpt
            last_<run_id>.ckpt
            interrupted_<run_id>.ckpt

    Default policy:
      1. inspect runs from newest to oldest;
      2. within each run prefer ``best_uniform``;
      3. then ``best_fixed`` (single-scale validation);
      4. then any other best-validation checkpoint;
      5. finally ``last`` and ``interrupted``.

    When ``val_strategy`` is supplied, only the corresponding best checkpoint
    is accepted. When ``run_id`` is supplied, only that run is inspected.
    """
    if not os.path.isdir(experiment_dir):
        return None

    if run_id is not None:
        run_dirs = [os.path.join(experiment_dir, run_id)]
    else:
        run_dirs = [
            path
            for path in glob.glob(os.path.join(experiment_dir, "*"))
            if os.path.isdir(path) and glob.glob(os.path.join(path, "*.ckpt"))
        ]
        run_dirs.sort(key=_new_run_sort_key, reverse=True)

    for run_dir in run_dirs:
        if not os.path.isdir(run_dir):
            continue
        actual_run_id = os.path.basename(os.path.normpath(run_dir))
        escaped_run_id = glob.escape(actual_run_id)

        if val_strategy is not None:
            pattern = os.path.join(
                run_dir,
                f"best_{glob.escape(val_strategy)}_{escaped_run_id}_epoch_*.ckpt",
            )
            candidates = glob.glob(pattern)
            if candidates:
                return max(
                    candidates,
                    key=lambda path: (_checkpoint_epoch(path), os.path.getmtime(path)),
                )
            continue

        preferred_patterns = [
            f"best_uniform_{escaped_run_id}_epoch_*.ckpt",
            f"best_fixed_{escaped_run_id}_epoch_*.ckpt",
            f"best_*_{escaped_run_id}_epoch_*.ckpt",
        ]
        for filename_pattern in preferred_patterns:
            candidates = glob.glob(os.path.join(run_dir, filename_pattern))
            if candidates:
                return max(
                    candidates,
                    key=lambda path: (_checkpoint_epoch(path), os.path.getmtime(path)),
                )

        for filename in [
            f"last_{actual_run_id}.ckpt",
            f"interrupted_{actual_run_id}.ckpt",
        ]:
            candidate = os.path.join(run_dir, filename)
            if os.path.isfile(candidate):
                return candidate

    return None


def _pick_legacy_layout_checkpoint(experiment_dir: str) -> str | None:
    """Resolve a checkpoint from the pre-run-id directory layout."""
    import re

    if not os.path.isdir(experiment_dir):
        return None

    def last_version(path: str) -> int:
        name = os.path.basename(path)
        match = re.match(r"last(?:-v(\d+))?\.ckpt$", name)
        if not match:
            return -1
        return int(match.group(1) or 0)

    last_candidates = glob.glob(os.path.join(experiment_dir, "last*.ckpt"))
    if last_candidates:
        return max(
            last_candidates,
            key=lambda path: (last_version(path), os.path.getmtime(path)),
        )

    epoch_candidates = glob.glob(os.path.join(experiment_dir, "epoch_*.ckpt"))
    if epoch_candidates:
        return max(epoch_candidates, key=os.path.getmtime)

    return None


def _resolve_ckpt(
    solver: str,
    model_tag,
    ckpt_root: str | None = None,
    *,
    variant: str = "twvrp",
    val_strategy: str | None = None,
    run_id: str | None = None,
) -> str | None:
    """Resolve an RL checkpoint across both new and legacy path rules.

    Explicit ``val_strategy`` and ``run_id`` selectors apply to the new layout.
    To avoid silently loading the wrong model, legacy fallback is disabled when
    either selector is explicitly provided.
    """
    val_strategy = _validate_ckpt_selector(val_strategy, "rl_val_strategy")
    run_id = _validate_ckpt_selector(run_id, "rl_run_id")
    experiment_name = f"{solver}_{model_tag}"

    for root in _candidate_ckpt_roots(ckpt_root):
        for experiment_dir in _candidate_experiment_dirs(
            root,
            variant,
            experiment_name,
        ):
            checkpoint = _pick_new_layout_checkpoint(
                experiment_dir,
                val_strategy=val_strategy,
                run_id=run_id,
            )
            if checkpoint is not None:
                return checkpoint

            # The old layout has neither validation-strategy nor run-id fields.
            if val_strategy is None and run_id is None:
                checkpoint = _pick_legacy_layout_checkpoint(experiment_dir)
                if checkpoint is not None:
                    return checkpoint

    return None

def parse_scale_from_filename(filename: str) -> int:
    """通常文件名格式为 '100_xxx.json' 或 'scale_100_xxx.json' """
    basename = os.path.basename(filename)
    parts = basename.split('_')
    # 尝试直接解析第一部分 (e.g. '100')
    if parts[0].isdigit():
        return int(parts[0])
    # 尝试解析 'scale_100'
    for i, p in enumerate(parts):
        if p == 'scale' and i + 1 < len(parts) and parts[i+1].isdigit():
            return int(parts[i+1])
    # 默认值 (如果解析失败)
    return 100

class DynamicSimulation:
    def __init__(
        self,
        data_file: str,
        solver_class: BaseSolver,
        interval: int = 30,
        speed: float = 5.0,
        replan_strategy: str = 'fixed',
        strategy_kwargs: dict = None,
        commit_lead_time: float = 0.0,
    ):
        self.data_file_name = data_file
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)

        self.interval = interval
        self.vehicle_speed = speed
        # commit_lead_time 表示路线在物理发车前多少分钟被冻结。
        # 0 表示发车时才冻结（departure-triggered / fully flexible）。
        # 一个很大的值或 inf 表示路线一生成就冻结（computation-triggered / delayed immutable）。
        if commit_lead_time < 0:
            raise ValueError("commit_lead_time must be non-negative.")
        self.commit_lead_time = commit_lead_time

        self.replan_strategy = replan_strategy
        self.strategy_kwargs = strategy_kwargs or {}
        self.urgency_threshold = self.strategy_kwargs.get('urgency_thresh', 10.0)
        self.capacity_threshold = self.strategy_kwargs.get('capacity_thresh', 1.0)

        self.depots = self.raw_data['depots']
        self.vehicle_cap = self.raw_data['capacity']
        self.total_vehicles = self.raw_data['num_vehicles']
        self.customers = self.raw_data['customers']
        self.customer_map = {c['id']: c for c in self.customers}

        # 初始化求解器，传入时间限制
        self.solver = solver_class(
            self.depots,
            self.vehicle_cap,
            time_limit=SOLVER_TIME_LIMIT,
            allow_late_return=ALLOW_LATE_RETURN
        )

        self.vehicles = self._init_vehicles()
        self.all_executed_routes = []
        self.total_served_count = 0
        self.final_service_rate = 0.0
        self.total_assigned_count = 0
        self.assigned_service_rate = 0.0
        self.invalid_solver_route_count = 0
        self.invalid_solver_order_instances = 0
        self.invalid_solver_reason_counts = {}
        self.solve_times = [] # 记录每次 solve_batch 的耗时
        self.replan_order_counts = [] # 记录每次重规划时参与求解的订单数

        # 只记录汇总指标，不把详细事件写入结果 JSON，避免大规模实验文件膨胀。
        self.next_plan_id = 0
        self.num_planned_routes = 0
        self.num_committed_routes = 0
        self.num_departed_routes = 0

        # Gross revision: 只要 PENDING route 在冻结前被撤回，就计为一次潜在计划扰动。
        self.gross_withdrawn_routes = 0
        self.gross_withdrawn_order_instances = 0
        self.gross_withdrawn_demand = 0.0
        self.gross_withdrawn_unique_orders = set()

        # Order-level net revision: 撤回后本轮重规划结果中，订单 assignment 是否真的发生变化。
        self.net_changed_order_instances = 0
        self.net_unchanged_order_instances = 0
        self.net_vehicle_changed_order_instances = 0
        self.net_position_changed_order_instances = 0
        self.net_partner_changed_order_instances = 0
        self.net_departure_time_changed_order_instances = 0
        self.net_unassigned_after_replanning_instances = 0
        self.net_changed_unique_orders = set()

        # 记录上一次规划时的状态，用于防止无限循环重规划
        self.last_replan_state = {
            'time': -1,
            'unplanned_orders': set(),
            'available_vehicles': -1
        }
        self.replan_intervals = []
        self.replan_triggers = {
            "fixed_interval": 0,
            "new_order": 0,
            "demand_backlog": 0,
            "urgency": 0
        }

    def _init_vehicles(self):
        vehicles = []
        num_depots = len(self.depots)
        base = self.total_vehicles // num_depots
        rem = self.total_vehicles % num_depots
        vid = 0
        for d_idx in range(num_depots):
            count = base + (1 if d_idx < rem else 0)
            for _ in range(count):
                vehicles.append({
                    'id': vid, 'home_depot': d_idx, 'status': 'IDLE',
                    'available_time': 0.0, 'return_time': 0.0, 'pending_route': None
                })
                vid += 1
        return vehicles

    def _compute_commit_time(self, plan_time: float, route_info: dict) -> float:
        """Return the route-freezing time without changing physical departure time.

        Solvers are assumed to return the latest feasible departure time in
        route_info['departure_time']. The route becomes non-revisable at
        departure_time - commit_lead_time, lower-bounded by plan_time.
        """
        departure_time = float(route_info.get('departure_time', plan_time))
        if math.isinf(self.commit_lead_time):
            return float(plan_time)
        return max(float(plan_time), departure_time - float(self.commit_lead_time))

    def _normalize_route_timing(self, route_info: dict, current_time: float, vehicle_id: int) -> dict:
        """Attach plan/commit/departure metadata to a solver-produced route."""
        route_info['departure_time'] = max(float(current_time), float(route_info.get('departure_time', current_time)))
        route_info['commit_time'] = self._compute_commit_time(current_time, route_info)
        route_info['plan_time'] = float(current_time)
        route_info['vehicle_id'] = vehicle_id
        route_info['plan_id'] = self.next_plan_id
        self.next_plan_id += 1
        return route_info

    def _route_customer_ids(self, route_info: dict):
        return [cid for cid in route_info.get('customer_ids', []) if cid != 0]

    def _record_planned_route(self, route_info: dict):
        """Count a newly generated route version."""
        self.num_planned_routes += 1

    def _record_commit_route(self, vehicle: dict, current_time: float):
        """Count a route-freezing event.

        This is a commit event only: the vehicle may still wait at the depot
        until route_info['departure_time'].
        """
        self.num_committed_routes += 1

    def _route_assignment_index(self, routes_by_vehicle: dict) -> dict:
        """Build order-level assignment index for a set of routes."""
        assignment = {}
        for vid, route_info in routes_by_vehicle.items():
            customer_ids = self._route_customer_ids(route_info)
            customer_set = frozenset(customer_ids)
            for pos, cid in enumerate(customer_ids):
                assignment[cid] = {
                    'vehicle_id': vid,
                    'position': pos,
                    'route_customers': tuple(customer_ids),
                    'route_customer_set': customer_set,
                    'departure_time': float(route_info.get('departure_time', 0.0)),
                }
        return assignment

    def _record_gross_revision(self, vehicle: dict, current_time: float) -> dict:
        """Record a gross withdrawal and return old order assignments.

        Gross revision counts the act of withdrawing a PENDING route before its
        commit time, even if the next solver call recreates the same route.
        """
        route_info = vehicle['pending_route']
        customer_ids = self._route_customer_ids(route_info)
        customer_set = frozenset(customer_ids)

        self.gross_withdrawn_routes += 1
        self.gross_withdrawn_order_instances += len(customer_ids)
        self.gross_withdrawn_demand += sum(self.customer_map[cid]['demand'] for cid in customer_ids)
        self.gross_withdrawn_unique_orders.update(customer_ids)

        old_assignments = {}
        for pos, cid in enumerate(customer_ids):
            old_assignments[cid] = {
                'vehicle_id': vehicle['id'],
                'position': pos,
                'route_customers': tuple(customer_ids),
                'route_customer_set': customer_set,
                'departure_time': float(route_info.get('departure_time', 0.0)),
            }
        return old_assignments

    def _record_net_revision_for_round(self, old_assignments: dict, new_assignments: dict):
        """Compare withdrawn orders with their assignments after replanning.

        Net change is counted at order level using only route-structure changes:
        an order is net-changed only if its position in the route changes or if
        its co-routed customer set changes. Vehicle-id changes and departure-time
        changes are still reported as auxiliary diagnostics, but they do not
        determine changed_order_instances. Orders not reassigned in the current
        replanning round are tracked separately and excluded from the compared
        changed/unchanged counts.
        """
        if not old_assignments:
            return

        for cid, old in old_assignments.items():
            new = new_assignments.get(cid)
            if new is None:
                self.net_unassigned_after_replanning_instances += 1
                continue

            vehicle_changed = old['vehicle_id'] != new['vehicle_id']
            position_changed = old['position'] != new['position']
            partner_changed = old['route_customer_set'] != new['route_customer_set']
            departure_time_changed = not math.isclose(
                old['departure_time'], new['departure_time'], rel_tol=0.0, abs_tol=1e-6
            )

            if vehicle_changed:
                self.net_vehicle_changed_order_instances += 1
            if position_changed:
                self.net_position_changed_order_instances += 1
            if partner_changed:
                self.net_partner_changed_order_instances += 1
            if departure_time_changed:
                self.net_departure_time_changed_order_instances += 1

            # The net-change definition intentionally ignores vehicle-id changes
            # and departure-time changes. In a homogeneous fleet, vehicle IDs may
            # be arbitrary after shuffling, and departure time changes are timing
            # adjustments rather than route-structure changes.
            if position_changed or partner_changed:
                self.net_changed_order_instances += 1
                self.net_changed_unique_orders.add(cid)
            else:
                self.net_unchanged_order_instances += 1

    def _commit_pending_route_if_needed(self, vehicle: dict, current_time: float):
        if vehicle['status'] == 'PENDING' and current_time >= vehicle['pending_route'].get('commit_time', float('inf')):
            vehicle['status'] = 'COMMITTED'
            self._record_commit_route(vehicle, current_time)

    def _depart_route_if_needed(self, vehicle: dict, current_time: float):
        if vehicle['status'] in ('PENDING', 'COMMITTED') and current_time >= vehicle['pending_route'].get('departure_time', float('inf')):
            route_info = vehicle['pending_route']
            vehicle['status'] = 'BUSY'
            vehicle['return_time'] = route_info['return_time']
            self.all_executed_routes.append({
                'executed': True,
                'vehicle_id': vehicle['id'],
                'plan_id': route_info.get('plan_id'),
                'plan_time': route_info.get('plan_time'),
                'commit_time': route_info.get('commit_time'),
                'departure_time': route_info.get('departure_time'),
                'route_path': route_info['route_objs'],
                'times': route_info['arrival_times']
            })
            self.num_departed_routes += 1
            vehicle['pending_route'] = None

    def _should_replan(self, current_time, pending_order_ids):
        # 强制前置检查：如果当前没有任何未规划（且有效）的订单，则不需要进行重规划
        has_unplanned_available = False
        current_available_orders = set()
        for cid in pending_order_ids:
            cust = self.customer_map[cid]
            if cust['available_time'] <= current_time and current_time <= cust['tw_end']:
                has_unplanned_available = True
                current_available_orders.add(cid)

        if not has_unplanned_available:
            return False, None

        # 防止死循环：如果距离上次规划，可用订单集合一模一样（没有新订单可用），且可用空闲车辆数也没有增加（甚至可能减少了），就不重规划
        # 只有在可用订单变多，或者可用空闲车辆变多的情况下，才允许由于挤压的订单触发重规划
        current_idle_vehicles = sum(1 for v in self.vehicles if v['status'] == 'IDLE')

        # 将已分配尚未发车的车辆的订单也算入当前可用订单池中
        for v in self.vehicles:
            if v['status'] == 'PENDING':
                for cid in v['pending_route']['customer_ids']:
                    cust = self.customer_map[cid]
                    if cust['available_time'] <= current_time and current_time <= cust['tw_end']:
                        current_available_orders.add(cid)
                # 这种将要把订单退回并变为IDLE的车辆，算作待释放空闲车辆
                current_idle_vehicles += 1

        state_differs = False
        if not current_available_orders.issubset(self.last_replan_state['unplanned_orders']):
            state_differs = True # 有新出现的订单
        elif current_idle_vehicles > self.last_replan_state['available_vehicles']:
            state_differs = True # 有新空闲的车辆

        # 如果是固定间隔且刚刚重规划过，跳过
        if self.replan_strategy == 'fixed':
            if (current_time % self.interval) == 0:
                if current_time == self.last_replan_state['time']:
                    return False, None
                if not state_differs:
                    return False, None
                return True, "fixed_interval"
            return False, None

        elif self.replan_strategy == 'new_order':
            # 只要有任何未处理订单刚好在此时刻出现，即可触发重规划
            for cid in pending_order_ids:
                if self.customer_map[cid]['available_time'] == current_time:
                    return True, "new_order"
            return False, None

        elif self.replan_strategy == 'demand':
            if current_time == self.last_replan_state['time']:
                return False, None
            if not state_differs:
                return False, None

            available_demand = 0.0

            # 1. 评估未分配的可用订单
            for cid in pending_order_ids:
                cust = self.customer_map[cid]
                if cust['available_time'] <= current_time and current_time <= cust['tw_end']:
                    available_demand += cust['demand']

            # 如果需求量达到（目前设可调度所有满载车容量的一部分*阈值）
            if available_demand >= self.capacity_threshold * self.vehicle_cap and available_demand > 0:
                return True, "demand_backlog"
            return False, None

        elif self.replan_strategy == 'urgent':
            if current_time == self.last_replan_state['time']:
                return False, None
            if not state_differs:
                return False, None

            min_buffer_time = float('inf')

            # 1. 评估未分配的可用订单的时效
            for cid in pending_order_ids:
                cust = self.customer_map[cid]
                if cust['available_time'] <= current_time and current_time <= cust['tw_end']:
                    # 距离最近的配送站（保守估计发车时间）
                    min_dist_to_depot = min(math.hypot(cust['x'] - d['x'], cust['y'] - d['y']) for d in self.depots)
                    latest_departure = cust['tw_end'] - (min_dist_to_depot / self.vehicle_speed)

                    # 考虑预留的 lead_time，必须在这个时间之前完成重规划
                    if not math.isinf(self.commit_lead_time):
                        latest_departure -= self.commit_lead_time
                    else:
                        latest_departure = -float('inf')

                    buffer_time = latest_departure - current_time
                    if buffer_time < min_buffer_time:
                        min_buffer_time = buffer_time

            # 如果有即订单满足即将超时发车阈值，则触发重规划
            if min_buffer_time <= self.urgency_threshold:
                return True, "urgency"
            return False, None

        return False, None

    def run(self):
        print(f"Simulating {os.path.basename(self.data_file_name)} using {self.solver.__class__.__name__}...")

        pending_order_ids = set([c['id'] for c in self.customers])
        committed_order_ids = set()
        failed_order_ids = set()

        for current_time in range(0, 1441):

            # 1. 每分钟更新车辆状态
            for v in self.vehicles:
                # PENDING: 已规划但尚未冻结，仍可在下一次重规划时撤回。
                # COMMITTED: 已冻结但尚未发车，车辆和订单被锁定，但不等于已经出发。
                self._commit_pending_route_if_needed(v, current_time)
                self._depart_route_if_needed(v, current_time)

                if v['status'] == 'BUSY':
                    if current_time >= v['return_time']:
                        v['status'] = 'IDLE'

            # 2. 判断此分钟是否需要触发重规划
            should_replan, reason = self._should_replan(current_time, pending_order_ids)
            if not should_replan:
                continue

            if reason:
                self.replan_triggers[reason] += 1
            if self.last_replan_state['time'] != -1:
                self.replan_intervals.append(current_time - self.last_replan_state['time'])

            # 3. 触发重规划前，仅撤回尚未冻结的 PENDING 计划。
            # 已到 commit_time 的 COMMITTED 车辆不会被撤回，也不会参与本轮重优化；
            # 它们会继续等待到 departure_time 再物理发车。
            withdrawn_assignments = {}
            for v in self.vehicles:
                if v['status'] == 'PENDING':
                    withdrawn_assignments.update(self._record_gross_revision(v, current_time))
                    for cid in v['pending_route']['customer_ids']:
                        if cid in committed_order_ids:
                            committed_order_ids.remove(cid)
                            pending_order_ids.add(cid)
                    v['status'] = 'IDLE'
                    v['pending_route'] = None

            available_vehicles = [v for v in self.vehicles if v['status'] == 'IDLE']

            if available_vehicles: random.shuffle(available_vehicles)

            current_batch = []
            for cid in sorted(list(pending_order_ids)):
                cust = self.customer_map[cid]
                if current_time > cust['tw_end']:
                    pending_order_ids.remove(cid)
                    failed_order_ids.add(cid)
                    continue
                if cust['available_time'] <= current_time:
                    current_batch.append(cust)

            if not current_batch or not available_vehicles:
                self._record_net_revision_for_round(withdrawn_assignments, {})
                continue

            # 记录当前重规划时的状态，防止在没有变化时继续触发
            self.last_replan_state = {
                'time': current_time,
                'unplanned_orders': set([c['id'] for c in current_batch]),
                'available_vehicles': len(available_vehicles)
            }

            # print(f"Time {current_time}: {len(available_vehicles)} vehicles available, {len(current_batch)} pending orders")

            # 记录 solve_batch 耗时
            self.replan_order_counts.append(len(current_batch))
            start_t = time.perf_counter()
            assigned_routes, _ = self.solver.solve_batch(
                current_batch, available_vehicles, current_time, self.vehicle_speed
            )
            elapsed = time.perf_counter() - start_t
            self.solve_times.append(elapsed)

            # 求解器决定路线顺序和出发时间；环境使用统一的物理模型重放并
            # 验证路线。无效路线不会被提交，其订单保留在 pending 池中。
            for route_info in assigned_routes.values():
                route_info['departure_time'] = max(
                    float(current_time),
                    float(route_info.get('departure_time', current_time))
                )
            assigned_routes, validation_errors = validate_and_normalize_routes(
                assigned_routes=assigned_routes,
                current_batch=current_batch,
                available_vehicles=available_vehicles,
                depots=self.depots,
                vehicle_capacity=self.vehicle_cap,
                current_time=current_time,
                vehicle_speed=self.vehicle_speed,
                allow_late_return=ALLOW_LATE_RETURN,
            )
            if validation_errors:
                self.invalid_solver_route_count += len(validation_errors)
                self.invalid_solver_order_instances += sum(
                    len(set(error.get('customer_ids', []))) for error in validation_errors
                )
                for error in validation_errors:
                    reason = error['reason']
                    self.invalid_solver_reason_counts[reason] = (
                        self.invalid_solver_reason_counts.get(reason, 0) + 1
                    )
                reason_summary = ', '.join(
                    f"{reason}={count}"
                    for reason, count in sorted(self.invalid_solver_reason_counts.items())
                )
                print(
                    f"[RouteValidation] rejected {len(validation_errors)} routes at "
                    f"t={current_time}; cumulative reasons: {reason_summary}"
                )

            # 为每条新路线写入 plan_time / commit_time / departure_time。
            # 默认 solver 返回的 departure_time 视为 latest feasible departure time；
            # commit_time 只表示路线冻结时间，不会让车辆提前发车。
            for vid, route_info in assigned_routes.items():
                self._normalize_route_timing(route_info, current_time, vid)

            # Compare withdrawn orders with the newly produced assignments in this same replanning round.
            self._record_net_revision_for_round(
                withdrawn_assignments,
                self._route_assignment_index(assigned_routes)
            )

            # for assigned in assigned_routes.values():
            #     for cid in assigned['customer_ids']:
            #         if cid != 0:
            #             print(f"> id={cid}")

            for vid, route_info in assigned_routes.items():
                vehicle = next(v for v in self.vehicles if v['id'] == vid)
                self._record_planned_route(route_info)

                vehicle['pending_route'] = route_info
                if current_time >= route_info['commit_time']:
                    vehicle['status'] = 'COMMITTED'
                    self._record_commit_route(vehicle, current_time)
                else:
                    vehicle['status'] = 'PENDING'

                for cid in route_info['customer_ids']:
                    if cid in pending_order_ids:
                        pending_order_ids.remove(cid)
                        committed_order_ids.add(cid)

        # 服务率只统计实际出发路线中，在仿真时域和客户时间窗内开始服务的
        # 订单。已提交但截至 1440 尚未出发的路线不再视为已服务。
        self.total_assigned_count = len(committed_order_ids)
        self.assigned_service_rate = (
            self.total_assigned_count / len(self.customers) * 100
            if self.customers else 0.0
        )
        ontime_served_order_ids = collect_ontime_served_orders(
            self.all_executed_routes,
            self.customer_map,
            simulation_end=1440.0,
        )
        self.total_served_count = len(ontime_served_order_ids)
        self.final_service_rate = (self.total_served_count / len(self.customers)) * 100
        print(
            f"Simulation Done. On-time Service Rate: {self.final_service_rate:.2f}% "
            f"(assigned: {self.assigned_service_rate:.2f}%)"
        )
        return self.final_service_rate

    def _build_commitment_metrics(self) -> dict:
        gross_instances = self.gross_withdrawn_order_instances
        net_total = self.net_changed_order_instances + self.net_unchanged_order_instances
        return {
            'num_planned_routes': self.num_planned_routes,
            'num_committed_routes': self.num_committed_routes,
            'num_departed_routes': self.num_departed_routes,
            'num_final_routes': len(self.all_executed_routes),
            'gross_revision': {
                'withdrawn_routes': self.gross_withdrawn_routes,
                'withdrawn_order_instances': self.gross_withdrawn_order_instances,
                'withdrawn_unique_orders': len(self.gross_withdrawn_unique_orders),
                'withdrawn_demand': self.gross_withdrawn_demand,
                'withdrawn_route_rate_over_planned_routes': (
                    self.gross_withdrawn_routes / self.num_planned_routes
                    if self.num_planned_routes else 0.0
                ),
                'withdrawn_order_rate_over_served_orders': (
                    self.gross_withdrawn_order_instances / self.total_served_count
                    if self.total_served_count else 0.0
                ),
            },
            'order_level_net_revision': {
                'net_change_definition': 'position_changed_or_partner_changed',
                'changed_order_instances': self.net_changed_order_instances,
                'unchanged_order_instances': self.net_unchanged_order_instances,
                'changed_unique_orders': len(self.net_changed_unique_orders),
                'unassigned_after_replanning_instances': self.net_unassigned_after_replanning_instances,
                'vehicle_changed_order_instances': self.net_vehicle_changed_order_instances,
                'position_changed_order_instances': self.net_position_changed_order_instances,
                'partner_changed_order_instances': self.net_partner_changed_order_instances,
                'departure_time_changed_order_instances': self.net_departure_time_changed_order_instances,
                'net_change_ratio_over_withdrawn_order_instances': (
                    self.net_changed_order_instances / gross_instances
                    if gross_instances else 0.0
                ),
                'net_change_ratio_over_compared_order_instances': (
                    self.net_changed_order_instances / net_total
                    if net_total else 0.0
                ),
                'unchanged_ratio_over_compared_order_instances': (
                    self.net_unchanged_order_instances / net_total
                    if net_total else 0.0
                ),
            }
        }

    def save_and_visualize(self, output_json, solver_name=None, render_formats=None):
        problem_display_name = (
            f"Sim_{os.path.basename(self.data_file_name)}\n"
            f"Service Rate: {self.final_service_rate:.2f}%"
        )
        viz_data = {
            'locations': [], 'demands': [], 'depots': [], 'customers': [],
            'routes': [], 'route_times': [],
            'route_plan_times': [], 'route_departure_times': [],
            'service_times': [],
            'time_windows': {}, 'appear_times': {},
            'problem': problem_display_name,
            'solver': solver_name or self.solver.__class__.__name__,
            'time_limit': SOLVER_TIME_LIMIT,
            'service_rate': self.final_service_rate,
            'service_rate_definition': 'on_time_service_start_on_departed_routes',
            'total_served': self.total_served_count,
            'total_assigned': self.total_assigned_count,
            'assigned_service_rate': self.assigned_service_rate,
            'total_customers': len(self.customers),
            'invalid_solver_route_count': self.invalid_solver_route_count,
            'invalid_solver_order_instances': self.invalid_solver_order_instances,
            'invalid_solver_reason_counts': self.invalid_solver_reason_counts,
            'solve_times': self.solve_times, # 详细耗时列表
            'replan_order_counts': self.replan_order_counts,
            'avg_replan_orders': sum(self.replan_order_counts) / len(self.replan_order_counts) if self.replan_order_counts else 0,
            'avg_solve_time': sum(self.solve_times) / len(self.solve_times) if self.solve_times else 0,
            'total_solve_time': sum(self.solve_times),
            'commit_lead_time': self.commit_lead_time,

            # 新增字段：重规划原因与间隔统计
            'replan_intervals': self.replan_intervals,
            'avg_replan_interval': sum(self.replan_intervals) / len(self.replan_intervals) if self.replan_intervals else 0.0,
            'min_replan_interval': min(self.replan_intervals) if self.replan_intervals else 0.0,
            'max_replan_interval': max(self.replan_intervals) if self.replan_intervals else 0.0,
            'replan_trigger_fixed_interval': self.replan_triggers["fixed_interval"],
            'replan_trigger_new_order': self.replan_triggers["new_order"],
            'replan_trigger_demand_backlog': self.replan_triggers["demand_backlog"],
            'replan_trigger_urgency': self.replan_triggers["urgency"],

            'commitment_metrics': self._build_commitment_metrics(),
        }

        depot_coord_map = {}
        for i, d in enumerate(self.depots):
            idx = len(viz_data['locations'])
            viz_data['locations'].append([d['x'], d['y']])
            viz_data['demands'].append(0)
            viz_data['service_times'].append(0)
            viz_data['depots'].append(idx)
            depot_coord_map[(d['x'], d['y'])] = idx

        cust_id_map = {}
        for i, c in enumerate(self.customers):
            idx = len(viz_data['locations'])
            viz_data['locations'].append([c['x'], c['y']])
            viz_data['demands'].append(c['demand'])
            viz_data['service_times'].append(c['service_time'])
            viz_data['customers'].append(idx)
            cust_id_map[c['id']] = idx
            viz_data['time_windows'][idx] = (c['tw_start'], c['tw_end'])
            if c.get('is_dynamic', False): viz_data['appear_times'][idx] = c['available_time']

        for route_rec in self.all_executed_routes:
            indices = []
            for node in route_rec['route_path']:
                if 'id' in node and node['id'] in cust_id_map:
                    indices.append(cust_id_map[node['id']])
                elif (node['x'], node['y']) in depot_coord_map:
                    indices.append(depot_coord_map[(node['x'], node['y'])])
                else: indices.append(0)
            viz_data['routes'].append(indices)
            viz_data['route_times'].append(route_rec['times'])
            viz_data['route_plan_times'].append(route_rec.get('plan_time'))
            viz_data['route_departure_times'].append(route_rec.get('departure_time'))

        with open(output_json, 'w') as f: json.dump(viz_data, f, indent=4)

        render_formats = render_formats or []
        if not render_formats:
            return

        try:
            # Lazy import so JSON-only experiments do not pay visualization import/render overhead.
            from data_generation.vrp_solution_visualize import VRPSolutionVisualizer as SolutionPlotter

            output_dir = os.path.dirname(output_json)
            output_stem = os.path.splitext(os.path.basename(output_json))[0]
            plotter = SolutionPlotter(output_dir, output_dir)
            for fmt in render_formats:
                output_img = os.path.join(output_dir, f"{output_stem}.{fmt}")
                plotter.visualize_solution(viz_data, output_img)
        except Exception as e:
            print(f"Visualization failed: {e}")

def parse_render_formats(value: str | None):
    """Parse requested visualization formats.

    Accepted values:
      - none / false / 0 / no: do not render figures
      - png: render PNG only
      - svg: render SVG only
      - both or png,svg: render both PNG and SVG
    """
    if value is None:
        return []
    raw = str(value).strip().lower()
    if raw in {"", "none", "false", "0", "no", "off"}:
        return []
    if raw in {"both", "all"}:
        return ["png", "svg"]

    formats = []
    for part in raw.replace("+", ",").split(","):
        fmt = part.strip().lower()
        if not fmt:
            continue
        if fmt not in {"png", "svg"}:
            raise argparse.ArgumentTypeError(
                f"Unsupported render format '{fmt}'. Use one of: none, png, svg, both, png,svg."
            )
        if fmt not in formats:
            formats.append(fmt)
    return formats


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保一些卷积/矩阵运算在GPU上也是确定性的
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set seed to {seed}")

def main():
    set_seed(42)  # 设置固定种子
    parser = argparse.ArgumentParser(description="SDVRP Simulation")
    parser.add_argument("--scale", type=int, default=20, help="Problem scale (e.g., 20, 50, 100, 500). (default: 20)")
    parser.add_argument("--dod", type=float, default=0.0, help="Degree of dynamism (e.g., 0.0, 0.2, 0.5). (default: 0.0)")
    parser.add_argument("--filter", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="If True, only test files matching the specified scale. If False, test all files using the specified scale model. (default: True)")
    parser.add_argument("--replan_strategy", type=str, default="fixed", choices=["fixed", "new_order", "demand", "urgent"], help="Replanning strategy (fixed, new_order, demand, urgent)")
    parser.add_argument("--urgency_thresh", type=float, default=10.0, help="Urgency buffer time threshold (mins)")
    parser.add_argument("--capacity_thresh", type=float, default=0.8, help="Capacity threshold multiplier")
    parser.add_argument("--interval", type=int, default=30, help="Interval for fixed replanning strategy (mins)")
    parser.add_argument("--commit_lead_time", type=float, default=0.0, help="Minutes before physical departure when a route becomes locked. 0 = revisable until departure; inf or a very large value = locked immediately after planning while keeping the physical departure time unchanged.")
    parser.add_argument("--render_formats", type=parse_render_formats, default=[],
                        help="Visualization formats to render. Default: none. Use 'png', 'svg', 'both', or 'png,svg' when figures are needed.")
    parser.add_argument("--rl_model_tag", type=str, default=None,
                        help="Optional exact RL checkpoint tag. If omitted, checkpoint tag is selected automatically from the test scale, with N>=500 mapped to 500. Examples: 100, 500, mixed_20-30-40-50.")
    parser.add_argument("--rl_ckpt_root", type=str, default=None,
                        help="Optional checkpoint search root. It may point to checkpoints/, checkpoints/twvrp/, or a specific algorithm-tag directory. By default, new ./rl/checkpoints and legacy ./solvers/rl/checkpoints layouts are searched automatically.")
    parser.add_argument("--rl_val_strategy", type=str, default=None,
                        help="Optional validation weighting strategy for the new checkpoint layout, e.g. uniform, small_scale, large_scale, or fixed. If omitted, auto-selection prefers best_uniform, then best_fixed, any other best checkpoint, last, and interrupted.")
    parser.add_argument("--rl_run_id", type=str, default=None,
                        help="Optional exact run ID for the new checkpoint layout, e.g. 20260610_153000. If omitted, the newest matching run is selected.")
    parser.add_argument("--scenario", type=str, default="US",
                        choices=["US", "RS", "BS", "UV", "RV", "BV"],
                        help="Dataset scenario code.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Optional exact directory containing problem JSON files.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Root directory for result files.")
    args = parser.parse_args()
    try:
        args.rl_val_strategy = _validate_ckpt_selector(
            args.rl_val_strategy,
            "rl_val_strategy",
        )
        args.rl_run_id = _validate_ckpt_selector(
            args.rl_run_id,
            "rl_run_id",
        )
    except ValueError as exc:
        parser.error(str(exc))

    # 根据参数动态设置目录
    dod_percent = int(round(args.dod * 100))
    data_dir = args.data_dir or f"data/{args.scenario}/n{args.scale}/dod{dod_percent}"
    if math.isinf(args.commit_lead_time):
        commit_tag = "commit_lead_inf"
    else:
        commit_tag = f"commit_lead_{args.commit_lead_time:g}"

    output_dir = (
        f"{args.output_dir}/{args.scenario}/"
        f"{args.replan_strategy}_{args.capacity_thresh}_{args.interval}/"
        f"{commit_tag}/test_results_{args.dod}_200v"
    )

    if not args.filter:
        output_dir = (
            f"{args.output_dir}/{args.scenario}/"
            f"{commit_tag}/test_results_{args.dod}_200v"
        )

    # RLSolver 需要 GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    RLSolver.device = device

    if not os.path.exists(output_dir): os.makedirs(output_dir)

    json_files = glob.glob(os.path.join(data_dir, "*.json"))
    problem_files = [f for f in json_files if "_viz" not in f]

    # Filter by scale if specified AND filter is True
    if args.data_dir is not None and args.scale is not None and args.filter:
        pattern = f"{args.scale}_"
        problem_files = [f for f in problem_files if os.path.basename(f).startswith(pattern)]

    # Filter by dod if specified
    if args.data_dir is not None and args.dod is not None:
        dod_percent = int(args.dod * 100)
        pattern = f"_dod{dod_percent}_"
        problem_files = [f for f in problem_files if pattern in os.path.basename(f)]

    if not problem_files:
        print(f"No problem files found in {data_dir} matching criteria. Run main_gen.py first.")
        return

    files_to_process = []
    if DEBUG_MODE:
        selected = random.choice(problem_files)
        files_to_process = [selected]
        print(f"--- DEBUG MODE ON ---")
        print(f"Randomly selected 1 problem: {os.path.basename(selected)}")
    else:
        files_to_process = problem_files

    # 使用 SOLVER_LIST
    if RUN_ALL_SOLVERS:
        solvers_to_run = SOLVER_LIST
    else:
        solvers_to_run = [SINGLE_SOLVER_CLASS]

    print(f"Solvers to run: {[s.__name__ for s in solvers_to_run]}")

    # 记录当前加载的模型状态，避免重复加载。
    # 注意：mixed checkpoint 可以跨测试规模使用，但 env/policy 仍然要按当前测试规模构建，
    # 因此缓存键需要同时包含 current_scale 和 rl_model_tag。
    current_loaded_scale = -1
    current_loaded_algo = ""
    current_loaded_model_tag = ""

    for problem_file in files_to_process:
        print(f"\nProcessing Problem: {os.path.basename(problem_file)}")

        # 确定当前 Scale
        current_scale = args.scale
        if current_scale is None:
            current_scale = parse_scale_from_filename(problem_file)

        for current_solver_class in solvers_to_run:

            # 如果是 RLSolver，我们需要对列表中的每个算法都跑一遍
            if current_solver_class == RLSolver:
                algos_for_this_solver = RL_ALGO_LIST
            else:
                algos_for_this_solver = [None] # 非 RL 求解器不区分算法

            for algo in algos_for_this_solver:
                solver_name = current_solver_class.__name__
                display_solver_name = solver_name

                # === 如果是 RLSolver，处理模型加载 ===
                if current_solver_class == RLSolver:
                    try:
                        rl_model_tag = _resolve_rl_model_tag(args, current_scale)
                    except Exception as e:
                        print(f"❌ Invalid RL model selection: {e}")
                        continue

                    # 默认自动选择时保持原命名。显式指定 checkpoint tag、
                    # validation strategy 或 run ID 时，将选择器写入结果目录名，
                    # 避免不同 checkpoint 的测试结果互相覆盖。
                    display_name_parts = [solver_name, algo]
                    if args.rl_model_tag is not None:
                        display_name_parts.append(rl_model_tag)
                    if args.rl_val_strategy is not None:
                        display_name_parts.append(f"val-{args.rl_val_strategy}")
                    if args.rl_run_id is not None:
                        display_name_parts.append(f"run-{args.rl_run_id}")
                    display_solver_name = "_".join(display_name_parts)

                    # 检查是否需要重新加载模型。
                    # checkpoint tag 可能是 mixed_20-30-40-50，也可能是 500（当 test_N >= 500 时）。
                    # env/policy 仍按 current_scale 构建，用来匹配当前测试实例规模。
                    if (
                        current_scale != current_loaded_scale
                        or algo != current_loaded_algo
                        or rl_model_tag != current_loaded_model_tag
                    ):
                        ckpt_path = _resolve_ckpt(
                            algo,
                            rl_model_tag,
                            args.rl_ckpt_root,
                            val_strategy=args.rl_val_strategy,
                            run_id=args.rl_run_id,
                        )
                        if not ckpt_path:
                            print(
                                f"❌ Skipping {display_solver_name} for {os.path.basename(problem_file)}: "
                                "No checkpoint found "
                                f"(model_tag={rl_model_tag}, test_N={current_scale}, "
                                f"val_strategy={args.rl_val_strategy or 'auto'}, "
                                f"run_id={args.rl_run_id or 'latest'})"
                            )
                            continue

                        print(
                            f"\n✓ Loading RL model: {algo} "
                            f"(model_tag={rl_model_tag}, test_N={current_scale}, "
                            f"val_strategy={args.rl_val_strategy or 'auto'}, "
                            f"run_id={args.rl_run_id or 'latest'}) from {ckpt_path}"
                        )
                        try:
                            env, policy = build_env_and_policy(algo, current_scale)
                            model = load_model_by_solver(algo, ckpt_path, env=env, policy=policy, device=device)

                            RLSolver.model = model
                            RLSolver.model.policy = model.policy.to(device)
                            RLSolver.model.env = model.env

                            current_loaded_scale = current_scale
                            current_loaded_algo = algo
                            current_loaded_model_tag = rl_model_tag
                        except Exception as e:
                            print(f"❌ Error loading model: {e}")
                            continue
                # ==========================================

                print(f"  > Solver: {display_solver_name}")

                # Create solver directory if not exists
                solver_dir = os.path.join(output_dir, display_solver_name)
                if not os.path.exists(solver_dir):
                    os.makedirs(solver_dir)

                # 检查结果文件是否存在
                base_name = os.path.basename(problem_file).replace('.json', '')
                search_pattern = os.path.join(solver_dir, f"res_{base_name}_{display_solver_name}_*.json")

                existing_results = glob.glob(search_pattern)

                if existing_results:
                    if not OVERWRITE:
                        print(f"    [SKIP] Result already exists: {os.path.basename(existing_results[0])}")
                        continue
                    else:
                        print(f"    [OVERWRITE] Cleaning {len(existing_results)} old results...")
                        for old_file in existing_results:
                            try:
                                os.remove(old_file)
                                for ext in ('.png', '.svg'):
                                    fig_file = old_file.replace('.json', ext)
                                    if os.path.exists(fig_file):
                                        os.remove(fig_file)
                            except OSError as e:
                                print(f"      Warning: Could not delete {old_file}: {e}")

                # 开始求解
                try:
                    # 每个算法和问题的组合都设置一次种子，保证对比公平且可复现
                    set_seed(42)
                    sim = DynamicSimulation(
                        data_file=problem_file,
                        solver_class=current_solver_class,
                        interval=args.interval,
                        speed=5.0,
                        replan_strategy=args.replan_strategy,
                        strategy_kwargs={
                            'urgency_thresh': args.urgency_thresh,
                            'capacity_thresh': args.capacity_thresh
                        },
                        commit_lead_time=args.commit_lead_time
                    )

                    service_rate = sim.run()

                    # 生成文件名
                    result_filename_base = f"res_{base_name}_{display_solver_name}_sr{int(service_rate)}_v{sim.total_vehicles}"

                    res_json = os.path.join(solver_dir, f"{result_filename_base}.json")
                    sim.save_and_visualize(
                        res_json,
                        solver_name=display_solver_name,
                        render_formats=args.render_formats
                    )
                    if args.render_formats:
                        rendered = ", ".join(args.render_formats)
                        print(f"    [Done] Saved results to: {res_json} | rendered: {rendered}")
                    else:
                        print(f"    [Done] Saved results to: {res_json} | render: none")
                except Exception as e:
                    print(f"    [Error] Failed on solver {display_solver_name}: {e}")
                    import traceback
                    traceback.print_exc()

if __name__ == "__main__":
    main()
