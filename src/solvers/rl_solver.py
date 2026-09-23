import math
import torch
from tensordict import TensorDict
from .base_solver import BaseSolver
from rl4co.envs.routing.mtvrp.env import MTVRPEnv
from rl4co.utils.ops import gather_by_index, get_distance

class RelaxedMTVRPEnv(MTVRPEnv):
    """
    继承自 MTVRPEnv，修改了结束条件 (_step)：
    当车辆回到 Depot 且剩余所有客户都因约束（如时间窗）无法访问时，
    也被视为 Episode 结束 (Done)。
    """
    def __init__(self, allow_late_return=True, **kwargs):
        super().__init__(**kwargs)
        self.allow_late_return = allow_late_return

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        curr_node = td["current_node"]
        locs = td["locs"]
        d_ij = get_distance(
            gather_by_index(locs, curr_node)[..., None, :], locs
        )  # i (current) -> j (next)
        d_j0 = get_distance(locs, locs[..., 0:1, :])  # j (next) -> 0 (depot)

        # Time constraint (TW):
        early_tw, late_tw = (
            td["time_windows"][..., 0],
            td["time_windows"][..., 1],
        )
        arrival_time = td["current_time"] + (d_ij / td["speed"])
        # can reach in time -> only need to *start* in time
        can_reach_customer = arrival_time < late_tw

        if self.allow_late_return:
            # 如果允许晚归，那么返回 Depot 的检查默认为 True
            can_reach_depot = torch.ones_like(can_reach_customer, dtype=torch.bool)
        else:
            # we must ensure that we can return to depot in time *if* route is closed
            # i.e. start time + service time + time back to depot < late_tw
            can_reach_depot = (
                torch.max(arrival_time, early_tw) + td["service_time"] + (d_j0 / td["speed"])
            ) * ~td["open_route"] < late_tw[..., 0:1]

        # Distance limit (L): do not add distance to depot if open route (O)
        exceeds_dist_limit = (
            td["current_route_length"] + d_ij + (d_j0 * ~td["open_route"])
            > td["distance_limit"]
        )

        # Linehaul demand / delivery (C) and backhaul demand / pickup (B)
        # All linehauls are visited before backhauls
        linehauls_missing = ((td["demand_linehaul"] * ~td["visited"]).sum(-1) > 0)[
            ..., None
        ]
        is_carrying_backhaul = (
            gather_by_index(
                src=td["demand_backhaul"],
                idx=curr_node,
                dim=1,
                squeeze=False,
            )
            > 0
        )
        exceeds_cap_linehaul = (
            td["demand_linehaul"] + td["used_capacity_linehaul"] > td["vehicle_capacity"]
        )
        exceeds_cap_backhaul = (
            td["demand_backhaul"] + td["used_capacity_backhaul"] > td["vehicle_capacity"]
        )

        meets_demand_constraint = (
            linehauls_missing
            & ~exceeds_cap_linehaul
            & ~is_carrying_backhaul
            & (td["demand_linehaul"] > 0)
        ) | (~exceeds_cap_backhaul & (td["demand_backhaul"] > 0))

        # Condense constraints
        can_visit = (
            can_reach_customer
            & can_reach_depot
            & meets_demand_constraint
            & ~exceeds_dist_limit
            & ~td["visited"]
        )

        # Mask depot: don't visit depot if coming from there and there are still customer nodes I can visit
        can_visit[:, 0] = ~((curr_node == 0) & (can_visit[:, 1:].sum(-1) > 0))
        return can_visit

    def _step(self, td: TensorDict) -> TensorDict:
        # Get locations and distance
        prev_node, curr_node = td["current_node"], td["action"]
        prev_loc = gather_by_index(td["locs"], prev_node)
        curr_loc = gather_by_index(td["locs"], curr_node)
        distance = get_distance(prev_loc, curr_loc)[..., None]

        # Update current time
        service_time = gather_by_index(
            src=td["service_time"], idx=curr_node, dim=1, squeeze=False
        )
        start_times = gather_by_index(
            src=td["time_windows"], idx=curr_node, dim=1, squeeze=False
        )[..., 0]
        # we cannot start before we arrive and we should start at least at start times
        curr_time = (curr_node[:, None] != 0) * (
            torch.max(td["current_time"] + distance / td["speed"], start_times)
            + service_time
        )

        # Update current route length (reset at depot)
        curr_route_length = (curr_node[:, None] != 0) * (
            td["current_route_length"] + distance
        )

        # Linehaul (delivery) demands
        selected_demand_linehaul = gather_by_index(
            td["demand_linehaul"], curr_node, dim=1, squeeze=False
        )
        selected_demand_backhaul = gather_by_index(
            td["demand_backhaul"], curr_node, dim=1, squeeze=False
        )

        # Backhaul (pickup) demands
        # vehicles are empty once we get to the backhauls
        used_capacity_linehaul = (curr_node[:, None] != 0) * (
            td["used_capacity_linehaul"] + selected_demand_linehaul
        )
        used_capacity_backhaul = (curr_node[:, None] != 0) * (
            td["used_capacity_backhaul"] + selected_demand_backhaul
        )

        # Update visited
        visited = td["visited"].scatter(-1, curr_node[..., None], True)

        # --- 计算 Mask 以辅助判断是否该结束 ---
        # 我们需要构造一个临时的 td 来计算下一步的 mask
        # 注意: 这里并不直接更新原来的 td，直到我们计算完 done
        td_next_state = td.clone()
        td_next_state.update({
            "current_node": curr_node,
            "current_route_length": curr_route_length,
            "current_time": curr_time,
            "used_capacity_linehaul": used_capacity_linehaul,
            "used_capacity_backhaul": used_capacity_backhaul,
            "visited": visited,
        })
        new_mask = self.get_action_mask(td_next_state)

        '''
        # [DEBUG Output] 检查当车辆不在Depot时，是否还有其他可选节点
        if curr_node[0].item() != 0:
            n_reachable = new_mask[0, 1:].sum().item()
            can_return = new_mask[0, 0].item()
            print(f"DEBUG_STEP: At Node {curr_node[0].item()}. Customers Reachable: {n_reachable}. Can Return Depot: {can_return}")
            if n_reachable == 0 and not can_return:
                 print(f"WARNING: At Node {curr_node[0].item()} (Customer), but NO valid actions (neither customers nor depot)! Deadlock?")

            if n_reachable > 0:
                indices = torch.nonzero(new_mask[0, 1:], as_tuple=False).flatten() + 1
                # print(f"            Reachable Indices: {indices.tolist()}")
        '''

        # MODIFIED DONE CONDITION
        # 1. 正常完成: 所有客户节点都已访问
        # visited shape: [batch, num_nodes].
        all_visited = visited.sum(-1) == visited.size(-1)

        # 2. 提前结束: 当前处于 Depot 且 没有其他任何客户节点可达
        # curr_node == 0 表示这一步回到了 depot
        at_depot = (curr_node == 0)
        # new_mask[..., 1:] 对应所有客户节点的 mask。如果 sum 为 0，说明全都去不了
        no_customers_reachable = (new_mask[..., 1:].sum(-1) == 0)

        early_finish = at_depot & no_customers_reachable

        done = all_visited | early_finish

        # if early_finish and not all_visited:
        #     print(f"INFO: early finish, unvisited nodes remain: {(~visited).sum(-1).item()}, unvisited indices: {(~visited).nonzero(as_tuple=False)}")

        reward = torch.zeros_like(done).float()

        td.update(
            {
                "current_node": curr_node,
                "current_route_length": curr_route_length,
                "current_time": curr_time,
                "done": done,
                "reward": reward,
                "used_capacity_linehaul": used_capacity_linehaul,
                "used_capacity_backhaul": used_capacity_backhaul,
                "visited": visited,
                "action_mask": new_mask, # 更新 mask
            }
        )
        return td



def dict_to_tensordict_tw(data: dict, map_size=1000.0, max_time=1440.0, current_time=0.0, vehicle_speed=1.0) -> TensorDict:
    """
    将读取的 "Batch Data" (dict) 转换为 RL4CO 环境所需的 TensorDict。
    适应动态规划中的 Batch 数据格式，并按照 rl4co generator 的设定进行归一化。
    Generator 默认设定:
      - Coords: [0, 1]
      - Max Time (Horizon): 4.6
      - Speed: 通常为 1.0 (但在我们这里需要根据物理单位换算)

    关键修正 (Time Shifting):
      由于 RL4CO Environment 的 reset() 方法会强制将 current_time 重置为 0，
      我们需要将所有的时间窗口 (TW) 和服务时间 平移到以 current_time 为起点 (0时刻)。
      即: New_TW = (Original_TW - current_time) / Scale
    """
    # 1. 基础信息提取
    depots = data.get('depots', [])
    customers = data.get('customers', [])
    capacity_val = float(data.get('capacity', 0))

    # 动态规划中，Depot 通常就是一个 (车辆所在的 Depot)
    num_depots = len(depots)
    num_cities = len(customers)

    # 2. 构建数组
    # (1) Locations
    locs_list = [[d['x'], d['y']] for d in depots] + \
                [[c['x'], c['y']] for c in customers]

    # (2) Demands
    demands_list = [0.0] * num_depots + \
                   [float(c.get('demand', 0)) for c in customers]

    # (3) Time Windows & Service Time
    # 目标时间范围 (RL Generator Default Horizon)
    RL_A_TARGET_MAX_TIME = 4.6
    # 缩放因子: Raw(1440) / Scale -> Target(4.6)
    time_scale = max_time / RL_A_TARGET_MAX_TIME

    tw_list = []
    # shift_time = float(current_time) # 相对时间的起点

    for d in depots:
        # Depot 的时间窗:
        # Start: max(0, current - current) = 0
        # End: max_time - current
        s_res = 0.0
        e_res = float(max_time) - float(current_time)
        tw_list.append([s_res, e_res])

    for c in customers:
        # Customer TW:
        # Start: max(0, start - current)
        # End: end - current
        c_start = float(c.get('tw_start', 0))
        c_end = float(c.get('tw_end', max_time))

        s_res = max(0.0, c_start - float(current_time))
        e_res = c_end - float(current_time)
        tw_list.append([s_res, e_res])

    service_list = [0.0] * num_depots + \
                   [float(c.get('service_time', 0)) for c in customers]

    # 3. 转换为 Tensor 并增加 Batch 维度 (Batch=1)
    batch_sz = 1

    # [1, N, 2] -> 归一化到 [0, 1]
    locs_tensor = torch.tensor(locs_list, dtype=torch.float32).unsqueeze(0)
    locs_norm = locs_tensor / map_size

    # [1, N] -> 归一化需求 (Capacity=1)
    demand_linehaul = torch.tensor(demands_list, dtype=torch.float32).unsqueeze(0)
    demand_norm = demand_linehaul / capacity_val
    demand_backhaul = torch.zeros_like(demand_norm)

    # [1, N, 2] -> 归一化时间到 [0, 4.6]
    tw_tensor = torch.tensor(tw_list, dtype=torch.float32).unsqueeze(0)
    tw_norm = tw_tensor / time_scale

    # [1, N]
    service_time_tensor = torch.tensor(service_list, dtype=torch.float32).unsqueeze(0)
    service_norm = service_time_tensor / time_scale

    # [MODIFIED] 归一化速度
    # 物理速度 (e.g. 5m/min) 需要转换为模型坐标系下的速度 (units/time_unit)
    # 换算公式: speed_norm = v_phys * (time_scale / map_size)
    # 其中 time_scale = 1440/4.6, map_size = 1000
    speed_scaled_val = vehicle_speed * (time_scale / map_size)
    speed = torch.full((batch_sz, 1), speed_scaled_val, dtype=torch.float32)

    # 其他属性
    vehicle_capacity = torch.tensor([1], dtype=torch.float32).view(batch_sz, 1) # Set to 1.0
    distance_limit = torch.full((batch_sz, 1), float('inf'), dtype=torch.float32)
    open_route = torch.zeros((batch_sz, 1), dtype=torch.bool)

    # Depot pos: [1, num_depots, 2]
    depot_pos = locs_norm[:, :num_depots, :]

    return TensorDict(
        {
            'locs': locs_norm,                          # [batch, N, 2] (0-1)
            'demand_linehaul': demand_norm,           # [batch, N] (0-1)
            'demand_backhaul': demand_backhaul,
            'distance_limit': distance_limit,
            'time_windows': tw_norm,                  # [batch, N, 2] (shifted & scaled)
            'service_time': service_norm,             # [batch, N] (scaled)
            'vehicle_capacity': vehicle_capacity,     # [batch, 1] (1.0)
            'capacity_original': vehicle_capacity.clone(),
            'open_route': open_route,
            'speed': speed,                           # Normalized speed
            'depot': depot_pos,                       # [batch, num_depots, 2]
        },
        batch_size=torch.Size([batch_sz]),
    )

class RLSolver(BaseSolver):
    """
    包装 RL4CO 模型以适配 SDVRP_Bench 的 BaseSolver 接口 (支持 Rolling Horizon)
    """
    model = None         # 类变量：当前加载的 RL 模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    map_size = 1000.0

    def __init__(self, depots, capacity, time_limit, allow_late_return=True, **kwargs):
        super().__init__(depots)
        self.capacity = capacity
        self.compute_budget = time_limit  # 算法计算时间限制 (e.g. 1.0s)
        self.max_time = 1440.0           # 问题的仿真截止时间 (e.g. 1440min)
        self.allow_late_return = allow_late_return

    def _project_route_to_exact_feasibility(
        self,
        depot,
        proposed_orders,
        current_time,
        vehicle_speed,
    ):
        """Projects an RL route onto the benchmark's exact time convention.

        Neural decoding uses continuous normalized travel times, whereas the
        benchmark executes ceil(distance / speed) minutes. Customers that are
        infeasible under the executable convention are left unassigned instead
        of returning an invalid route to the environment.
        """
        route_objs = [depot]
        arrival_times = [float(current_time)]
        accepted_ids = []
        current_node = depot
        clock = float(current_time)
        load = 0.0

        for customer in proposed_orders:
            next_load = load + float(customer.get('demand', 0.0))
            if next_load > float(self.capacity) + 1e-9:
                continue
            arrival = max(
                clock + self.calculate_travel_time(current_node, customer, vehicle_speed),
                float(customer['tw_start']),
            )
            if arrival > float(customer['tw_end']) + 1e-9:
                continue

            route_objs.append(customer)
            arrival_times.append(arrival)
            accepted_ids.append(customer['id'])
            load = next_load
            clock = arrival + float(customer.get('service_time', 0.0))
            current_node = customer

        if not accepted_ids:
            return None, []

        return_time = clock + self.calculate_travel_time(current_node, depot, vehicle_speed)
        if not self.allow_late_return and return_time > self.max_time + 1e-9:
            return None, []

        route_objs.append(depot)
        arrival_times.append(return_time)
        first_travel = self.calculate_travel_time(depot, route_objs[1], vehicle_speed)
        departure_time = max(float(current_time), arrival_times[1] - first_travel)
        arrival_times[0] = departure_time

        return {
            'route_objs': route_objs,
            'arrival_times': arrival_times,
            'departure_time': departure_time,
            'return_time': return_time,
            'customer_ids': accepted_ids,
        }, accepted_ids

    def solve_batch(self, batch_orders, available_vehicles, current_time, vehicle_speed):
        """
        处理逻辑：
        1. 这是一个 Multi-Depot 环境。
        2. RL 模型通常是 Single-Depot。
        3. 策略：按照最近 Depot 将 batch_orders 分组，分别求解。
        """
        # print("RLSolver: solve_batch called with {} orders and {} vehicles at time {}".format(
        #     len(batch_orders), len(available_vehicles), current_time
        # ))
        if not RLSolver.model or not batch_orders or not available_vehicles:
            return {}, batch_orders

        # 1. 车辆按 Depot 分组
        vehs_by_depot = {}
        for v in available_vehicles:
            vehs_by_depot.setdefault(v['home_depot'], []).append(v)

        # 2. 订单分配给最近的 Depot
        # 注意：这里只分配给 "有可用车辆" 的 Depot
        # 如果最近的 Depot 没车，可能需要分配给次近的 (简化起见，这里只考虑有车的Depot中最近的)
        orders_by_depot = {}
        active_depot_indices = sorted(list(vehs_by_depot.keys()))

        if not active_depot_indices:
            return {}, batch_orders # 无车可用

        # 构建 Depot 坐标缓存
        depot_locs = {i: self.depots[i] for i in active_depot_indices}

        for order in batch_orders:
            best_d = -1
            min_dist = float('inf')
            ox, oy = order['x'], order['y']
            for d_idx, d_obj in depot_locs.items():
                dist = math.hypot(ox - d_obj['x'], oy - d_obj['y'])
                if dist < min_dist:
                    min_dist = dist
                    best_d = d_idx

            if best_d != -1:
                orders_by_depot.setdefault(best_d, []).append(order)
            else:
                pass # Should not happen

        assigned_routes = {}
        processed_order_ids = set()

        # 3. 对每个 Depot 分别求解
        model = RLSolver.model
        policy, _env = model.policy, model.env

        for d_idx, sub_orders in orders_by_depot.items():
            # print(f"RLSolver: Solving for Depot {d_idx} with {len(sub_orders)} orders and {len(vehs_by_depot[d_idx])} vehicles")
            if not sub_orders:
                continue

            sub_vehs = vehs_by_depot[d_idx]
            current_depot = self.depots[d_idx]

            # 构造 RL 输入数据
            # 必须构造假数据包含 'depots', 'customers', 'capacity'
            # 且 depots 列表中只能放这一个 depot (因为是 Single-Depot Model)
            temp_data = {
                'depots': [current_depot],
                'customers': sub_orders,
                'capacity': self.capacity
            }

            # 转 TensorDict
            try:
                td = dict_to_tensordict_tw(
                    temp_data,
                    map_size=self.map_size,
                    max_time=self.max_time,
                    current_time=current_time,
                    vehicle_speed=vehicle_speed,
                )

                # 调整 batch size 用于 inference
                current_batch_sz = td.batch_size[0] # 应该总是 1

                # [关键更改] 使用自定义的 RelaxedMTVRPEnv 替换模型原本的 env
                # 但我们需要保留原本 env 的 generator 和其他配置
                original_env = model.env
                relaxed_env = RelaxedMTVRPEnv(generator=original_env.generator, check_solution=False, allow_late_return=self.allow_late_return)
                # 确保 device 正确
                relaxed_env = relaxed_env.to(self.device)

                td_init = relaxed_env.reset(td=td.clone(), batch_size=current_batch_sz).to(self.device)

                # Inference
                with torch.no_grad():
                    # greedy decoding
                    if hasattr(policy, 'evaluate'):
                        out = policy(td_init.clone(), relaxed_env, phase="test", decode_type="greedy")
                    else:
                        out = policy(td_init.clone(), relaxed_env, phase="test", decode_type="greedy")

                # 解析结果
                # actions: [batch, max_len] -> [1, seq_len]
                # RL4CO 的 actions 是节点索引。0 是 Depot。1..N 是 customers (对应 sub_orders[i-1])
                actions = out['actions'][0].cpu().tolist()

                # 拆分 Routes (0-separated)
                # 例如: 1 2 0 3 4 5 0 ...
                routes_indices = []
                current_route = []
                for node_idx in actions:
                    if node_idx == 0:
                        if current_route:
                            routes_indices.append(current_route)
                            current_route = []
                    else:
                        current_route.append(node_idx)
                if current_route:
                    routes_indices.append(current_route)

                # 分配给车辆
                # 如果生成的路径数 > 可用车辆数，则截断 (丢单)
                # 如果生成的路径数 <= 可用车辆数，则闲置剩余车辆
                num_routes = len(routes_indices)
                num_vehs = len(sub_vehs)

                for i in range(min(num_routes, num_vehs)):
                    veh = sub_vehs[i]
                    r_indices = routes_indices[i] # 这里的 index 是相对于 sub_orders 的 (1-based)

                    proposed_orders = [sub_orders[sub_idx - 1] for sub_idx in r_indices]
                    route_info, accepted_ids = self._project_route_to_exact_feasibility(
                        current_depot,
                        proposed_orders,
                        current_time,
                        vehicle_speed,
                    )
                    if route_info is None:
                        continue
                    assigned_routes[veh['id']] = route_info
                    processed_order_ids.update(accepted_ids)

            except Exception as e:
                print(f"RL Solver Error at {current_time}: {e}")
                import traceback
                traceback.print_exc()
                # Fallback: drop these orders
                continue

        # 计算 dropped
        dropped = [o for o in batch_orders if o['id'] not in processed_order_ids]

        return assigned_routes, dropped
