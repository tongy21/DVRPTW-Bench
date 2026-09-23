import math
import random
import copy
import time
import numpy as np
from typing import List, Dict, Tuple, Any, Callable
from .base_solver import BaseSolver
from .nn_2opt_solver import NNSolver


class ALNSSolver(BaseSolver):
    """
    自适应大邻域搜索 (ALNS) 求解器

    特性:
    - 多种破坏算子: Random, Worst, Shaw, Route Removal
    - 多种修复算子: Greedy, Regret-2, Regret-3 Insertion
    - 自适应权重机制: 根据算子历史表现动态调整选择概率
    - 模拟退火接受准则
    - 局部搜索优化: 2-opt, relocate
    """

    def __init__(self, depots: List[Dict], capacity: int,time_limit: int = 2, allow_late_return: bool = True):
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = capacity

        # ==================== ALNS 核心参数 ====================
        self.max_iterations = 10000
        self.time_limit = time_limit          # 每个 batch 最多运行秒数
        # [FIX] 降低移除比例，避免破坏过度导致难以重建
        self.r_min = 0.05               # 最小移除比例 (从0.1降到0.05)
        self.r_max = 0.25               # 最大移除比例 (从0.4降到0.25)

        # 模拟退火参数
        self.start_temp = None         # 动态计算
        self.end_temp = 0.01
        # [FIX] 放缓冷却速度，给短时间运行的ALNS更多探索机会
        self.cooling_rate = 0.98       # 从0.995改为0.98，温度下降更快但仍有足够探索

        # 惩罚权重
        self.unserved_penalty = 10000  # 未服务订单的惩罚
        self.tw_violation_penalty = 1000  # 时间窗违反惩罚

        # ==================== 自适应权重参数 ====================
        self.segment_size = 50         # 每多少次迭代更新一次权重
        self.reaction_factor = 0.1     # 权重更新的反应因子

        # 算子得分参数
        self.sigma1 = 33  # 发现新的全局最优解
        self.sigma2 = 9   # 发现比当前解更好的解
        self.sigma3 = 3   # 接受了一个更差的解（多样性）

        # 初始化算子
        self._init_operators()

        # 使用 NN-2Opt 构造初始解
        self.nn_solver = NNSolver(depots, capacity, time_limit=time_limit, allow_late_return=allow_late_return)

    def _init_operators(self):
        """初始化破坏和修复算子"""
        # 破坏算子列表
        self.destroy_operators = [
            self._destroy_random,
            self._destroy_worst,
            self._destroy_shaw,
            self._destroy_route,
        ]
        self.destroy_names = ['Random', 'Worst', 'Shaw', 'Route']

        # 修复算子列表
        self.repair_operators = [
            self._repair_greedy,
            self._repair_regret2,
            self._repair_regret3,
        ]
        self.repair_names = ['Greedy', 'Regret-2', 'Regret-3']

        # 初始化权重 (均匀分布)
        self.destroy_weights = [1.0] * len(self.destroy_operators)
        self.repair_weights = [1.0] * len(self.repair_operators)

        # 算子得分和使用次数 (用于自适应更新)
        self.destroy_scores = [0.0] * len(self.destroy_operators)
        self.destroy_counts = [0] * len(self.destroy_operators)
        self.repair_scores = [0.0] * len(self.repair_operators)
        self.repair_counts = [0] * len(self.repair_operators)

    def _reset_operator_stats(self):
        """重置算子统计信息（每个segment开始时调用）"""
        self.destroy_scores = [0.0] * len(self.destroy_operators)
        self.destroy_counts = [0] * len(self.destroy_operators)
        self.repair_scores = [0.0] * len(self.repair_operators)
        self.repair_counts = [0] * len(self.repair_operators)

    def _update_weights(self):
        """根据历史表现更新算子权重"""
        # 更新破坏算子权重
        for i in range(len(self.destroy_operators)):
            if self.destroy_counts[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = (
                    self.destroy_weights[i] * (1 - self.reaction_factor) +
                    self.reaction_factor * avg_score
                )
                self.destroy_weights[i] = max(0.1, self.destroy_weights[i])

        # 更新修复算子权重
        for i in range(len(self.repair_operators)):
            if self.repair_counts[i] > 0:
                avg_score = self.repair_scores[i] / self.repair_counts[i]
                self.repair_weights[i] = (
                    self.repair_weights[i] * (1 - self.reaction_factor) +
                    self.reaction_factor * avg_score
                )
                self.repair_weights[i] = max(0.1, self.repair_weights[i])

        # [FIX] Normalize weights to prevent drift
        sum_destroy = sum(self.destroy_weights)
        if sum_destroy > 0:
            self.destroy_weights = [w / sum_destroy for w in self.destroy_weights]

        sum_repair = sum(self.repair_weights)
        if sum_repair > 0:
            self.repair_weights = [w / sum_repair for w in self.repair_weights]

    def _select_operator(self, weights: List[float]) -> int:
        """轮盘赌选择算子"""
        total = sum(weights)
        r = random.random() * total
        cumsum = 0
        for i, w in enumerate(weights):
            cumsum += w
            if r <= cumsum:
                return i
        return len(weights) - 1

    def solve_batch(self, batch_orders: List[Dict], available_vehicles: List[Dict],
                   current_time: int, vehicle_speed: float) -> Tuple[Dict, List[Dict]]:
        """
        求解当前批次的路径规划问题
        """
        print(f"[ALNSSolver][{current_time}] random.getstate()[:3] entry:", random.getstate()[1][:3])
        print(f"[ALNSSolver][{current_time}] np.random.get_state()[1][:3] entry:", np.random.get_state()[1][:3])

        start_time_global = time.time()

        if not batch_orders or not available_vehicles:
            return {}, batch_orders

        # 保存上下文信息
        self.current_time = current_time
        self.vehicle_speed = vehicle_speed
        self.all_orders = batch_orders

        # 重置算子统计
        self._reset_operator_stats()

        # 1. 构造初始解
        t0 = time.time()
        current_sol, unassigned = self._construct_initial_solution(
            batch_orders, available_vehicles, current_time, vehicle_speed
        )
        t1 = time.time()

        current_cost = self._calculate_solution_cost(current_sol, unassigned)

        # Check Remaining Time
        elapsed = time.time() - start_time_global
        remaining_time = self.time_limit - elapsed

        print(f"[ALNS DEBUG] Batch Size: {len(batch_orders)} | NN Init: {t1-t0:.4f}s | Budget Left: {remaining_time:.4f}s")
        print(f"[ALNS DEBUG] Initial Solution: {len(current_sol)} routes, {sum(len(r['orders']) for r in current_sol)} assigned, {len(unassigned)} unassigned")

        # [FIX] 移除过早退出 - 即使时间紧张也要运行ALNS主循环
        # 在主循环中会保证至少一轮迭代

        best_sol = copy.deepcopy(current_sol)
        best_unassigned = copy.deepcopy(unassigned)
        best_cost = current_cost

        # 2. 动态设置初始温度
        # [FIX] 使用订单数量和平均距离计算温度，避免动态场景温度过高
        total_dist_only = sum(self._calculate_route_cost(r) for r in current_sol)
        num_assigned = sum(len(r['orders']) for r in current_sol)

        if num_assigned > 0:
            avg_dist_per_order = total_dist_only / num_assigned
            # 温度设为能以50%概率接受平均距离10%恶化的解
            start_temp = avg_dist_per_order * 0.1 / math.log(2)
        else:
            # 如果没有分配任何订单，给一个保守的基准温度
            start_temp = 10.0

        temp = max(1.0, start_temp)

        # 3. ALNS 主循环
        # Use initial 'start_time' captured before NN if possible, but
        # usually solve_batch doesn't capture it before NN.
        # Let's verify if we want to include NN time in the budget.
        # Yes, strictly we should.
        # But 'solve_batch' logic below:

        # We capture start_time NOW.
        # Ideally, we should capture it at the very beginning of solve_batch.
        # But let's at least ensure we don't exceed remaining time.

        # ALNS Main Loop
        start_time_alns = time.time()

        iteration = 0
        no_improve_count = 0
        first_iteration = True  # [FIX] 确保至少一轮迭代

        while iteration < self.max_iterations:
            # [FIX] 只在第一轮迭代后才检查时间限制
            if not first_iteration and time.time() - start_time_global > self.time_limit:
                break

            # 检查是否需要更新权重
            if iteration > 0 and iteration % self.segment_size == 0:
                self._update_weights()
                self._reset_operator_stats()

            # 复制当前解
            new_sol = copy.deepcopy(current_sol)
            new_unassigned = copy.deepcopy(unassigned)

            # 选择算子
            destroy_idx = self._select_operator(self.destroy_weights)
            repair_idx = self._select_operator(self.repair_weights)

            # 计算移除数量
            assigned_orders = [o for route in new_sol for o in route['orders']]
            total_orders = len(assigned_orders) + len(new_unassigned)

            # [FIX] 如果初始解为空或全部unassigned，尝试修复而不是跳过
            if total_orders == 0:
                iteration += 1
                first_iteration = False  # 标记完成一轮
                continue

            # [DEBUG] 第一次迭代时打印状态
            if iteration == 0:
                print(f"[ALNS] Initial: {len(assigned_orders)} assigned, {len(new_unassigned)} unassigned")

            num_to_remove = random.randint(
                max(1, int(total_orders * self.r_min)),
                max(1, int(total_orders * self.r_max))
            )
            num_to_remove = min(num_to_remove, len(assigned_orders))

            # === Destroy ===
            removed_orders = []
            if num_to_remove > 0 and assigned_orders:
                removed_orders = self.destroy_operators[destroy_idx](new_sol, num_to_remove)
                new_unassigned.extend(removed_orders)
                self.destroy_counts[destroy_idx] += 1

            # === Repair ===
            # [FIX] 即使destroy没执行，如果有unassigned也要尝试repair
            if new_unassigned:
                self.repair_operators[repair_idx](new_sol, new_unassigned, current_time, vehicle_speed)
                self.repair_counts[repair_idx] += 1

            # === 局部搜索 (可选) ===
            if random.random() < 0.3:  # 30%概率执行局部搜索
                self._local_search(new_sol, current_time, vehicle_speed)

            # === 评估 ===
            new_cost = self._calculate_solution_cost(new_sol, new_unassigned)

            # === 接受准则 (模拟退火) ===
            accept = False
            score = 0

            if new_cost < best_cost:
                # 发现新的全局最优
                best_sol = copy.deepcopy(new_sol)
                best_unassigned = copy.deepcopy(new_unassigned)
                best_cost = new_cost
                accept = True
                score = self.sigma1
                no_improve_count = 0
            elif new_cost < current_cost:
                # 比当前解更好
                accept = True
                score = self.sigma2
            else:
                # 模拟退火接受更差的解
                delta = new_cost - current_cost
                if temp > 0:
                    prob = math.exp(-delta / temp)
                    if random.random() < prob:
                        accept = True
                        score = self.sigma3

            if accept:
                current_sol = new_sol
                unassigned = new_unassigned
                current_cost = new_cost

            # 更新算子得分
            if num_to_remove > 0:
                self.destroy_scores[destroy_idx] += score
            if new_unassigned or removed_orders:
                self.repair_scores[repair_idx] += score

            # 降温
            temp = max(self.end_temp, temp * self.cooling_rate)
            iteration += 1
            no_improve_count += 1
            first_iteration = False  # [FIX] 标记第一轮迭代完成

            # 长时间无改进时进行扰动
            if no_improve_count > 100:
                self._perturb_solution(current_sol, unassigned)
                no_improve_count = 0

        # 4. 最终局部搜索优化
        self._local_search(best_sol, current_time, vehicle_speed)

        print(f"[ALNS] Completed {iteration} iterations in {time.time()-start_time_global:.4f}s")
        print(f"[ALNS] Best cost: {best_cost:.2f}, Unassigned: {len(best_unassigned)}")

        # 5. 格式化输出
        print(f"[ALNSSolver][{current_time}] random.getstate()[:3] exit:", random.getstate()[1][:3])
        print(f"[ALNSSolver][{current_time}] np.random.get_state()[1][:3] exit:", np.random.get_state()[1][:3])
        return self._format_solution(best_sol, batch_orders, current_time, vehicle_speed)

    # ==================== 初始解构造 ====================

    def _construct_initial_solution(self, orders, vehicles, current_time, speed):
        """
        使用 NNSolver 快速构造初始解 (比 Regret-2 快得多)
        这为 ALNS 的迭代搜索留出了更多时间
        """
        # [FIX] 始终启用2-opt以保证初始解质量
        # 即使time_limit很小，好的初始解也是必要的
        do_2opt_init = True

        # 1. 调用 NN-2Opt
        nn_assigned, unassigned = self.nn_solver.solve_batch(
            orders, vehicles, current_time, speed, perform_2opt=do_2opt_init
        )

        # 2. 转换为 ALNS 内部结构 (List of Dict)
        routes = []

        for v in vehicles:
            route_struct = {
                'vehicle': v,
                'orders': [],
                'load': 0
            }

            if v['id'] in nn_assigned:
                full_route_objs = nn_assigned[v['id']]['route_objs']
                # full_route_objs: [Depot, C1, C2, ..., Cn, Depot]
                # 提取客户节点 (排除首尾Depot)
                # 使用 'tw_start' 判断是否为客户 (Depot 无 tw_start 属性，或通过 id 判断)

                cust_nodes = [n for n in full_route_objs if 'tw_start' in n]
                route_struct['orders'] = cust_nodes
                route_struct['load'] = sum(o['demand'] for o in cust_nodes)

            routes.append(route_struct)

        return routes, unassigned

    # ==================== 破坏算子 ====================

    def _destroy_random(self, routes: List[Dict], num_to_remove: int) -> List[Dict]:
        """随机移除算子"""
        removed = []
        positions = []

        for r_idx, route in enumerate(routes):
            for o_idx, order in enumerate(route['orders']):
                positions.append((r_idx, o_idx, order))

        if not positions:
            return removed

        num_to_remove = min(num_to_remove, len(positions))
        selected = random.sample(positions, num_to_remove)

        # 按路径分组，倒序移除
        to_remove_by_route = {}
        for r_idx, o_idx, order in selected:
            if r_idx not in to_remove_by_route:
                to_remove_by_route[r_idx] = []
            to_remove_by_route[r_idx].append((o_idx, order))

        for r_idx, items in to_remove_by_route.items():
            indices = sorted([x[0] for x in items], reverse=True)
            for o_idx in indices:
                order = routes[r_idx]['orders'].pop(o_idx)
                routes[r_idx]['load'] -= order['demand']
                removed.append(order)

        return removed

    def _destroy_worst(self, routes: List[Dict], num_to_remove: int) -> List[Dict]:
        """最差移除算子 - 移除插入成本最高的订单"""
        removed = []

        # 计算每个订单的移除收益（移除后成本减少量越大越优先移除）
        removal_gains = []

        for r_idx, route in enumerate(routes):
            vehicle = route['vehicle']
            depot = self.depots[vehicle['home_depot']]
            orders = route['orders']

            for o_idx, order in enumerate(orders):
                # 计算移除该订单后的成本减少
                prev_node = orders[o_idx - 1] if o_idx > 0 else depot
                next_node = orders[o_idx + 1] if o_idx < len(orders) - 1 else depot

                # 当前成本
                current_cost = (
                    self.calculate_distance(prev_node, order) +
                    self.calculate_distance(order, next_node)
                )
                # 移除后成本
                new_cost = self.calculate_distance(prev_node, next_node)

                gain = current_cost - new_cost
                removal_gains.append((gain, r_idx, o_idx, order))

        if not removal_gains:
            return removed

        # 按收益降序排序，选择前 num_to_remove 个
        removal_gains.sort(key=lambda x: -x[0])

        # 添加一些随机性，避免总是选择相同的订单
        num_to_remove = min(num_to_remove, len(removal_gains))
        # 使用随机化策略
        selected = []
        candidates = removal_gains.copy()

        while len(selected) < num_to_remove and candidates:
            # 随机化选择（偏向于排名靠前的）
            p = random.random()
            idx = int(len(candidates) * (p ** 3))  # 指数分布偏向前面
            idx = min(idx, len(candidates) - 1)
            selected.append(candidates.pop(idx))

        # 执行移除
        to_remove_by_route = {}
        for _, r_idx, o_idx, order in selected:
            if r_idx not in to_remove_by_route:
                to_remove_by_route[r_idx] = []
            to_remove_by_route[r_idx].append((o_idx, order))

        for r_idx, items in to_remove_by_route.items():
            indices = sorted([x[0] for x in items], reverse=True)
            for o_idx in indices:
                order = routes[r_idx]['orders'].pop(o_idx)
                routes[r_idx]['load'] -= order['demand']
                removed.append(order)

        return removed

    def _destroy_shaw(self, routes: List[Dict], num_to_remove: int) -> List[Dict]:
        """Shaw移除算子 - 移除相似的订单"""
        removed = []

        # 收集所有订单
        all_positions = []
        for r_idx, route in enumerate(routes):
            for o_idx, order in enumerate(route['orders']):
                all_positions.append((r_idx, o_idx, order))

        if not all_positions:
            return removed

        # 随机选择一个种子订单
        seed = random.choice(all_positions)
        seed_order = seed[2]

        # 计算所有订单与种子的相似度
        similarities = []
        for r_idx, o_idx, order in all_positions:
            if order['id'] == seed_order['id']:
                sim = 0  # 种子自己
            else:
                # 相似度基于：距离、需求、时间窗
                dist_sim = self.calculate_distance(seed_order, order)
                demand_sim = abs(seed_order['demand'] - order['demand'])
                tw_sim = abs(seed_order['tw_start'] - order['tw_start']) + \
                         abs(seed_order['tw_end'] - order['tw_end'])

                # 综合相似度（越小越相似）
                sim = 0.5 * dist_sim + 0.25 * demand_sim + 0.25 * tw_sim * 0.1

            similarities.append((sim, r_idx, o_idx, order))

        # 按相似度排序
        similarities.sort(key=lambda x: x[0])

        # 选择最相似的订单
        num_to_remove = min(num_to_remove, len(similarities))
        selected = []
        candidates = similarities.copy()

        while len(selected) < num_to_remove and candidates:
            p = random.random()
            idx = int(len(candidates) * (p ** 2))
            idx = min(idx, len(candidates) - 1)
            selected.append(candidates.pop(idx))

        # 执行移除
        to_remove_by_route = {}
        for _, r_idx, o_idx, order in selected:
            if r_idx not in to_remove_by_route:
                to_remove_by_route[r_idx] = []
            to_remove_by_route[r_idx].append((o_idx, order))

        for r_idx, items in to_remove_by_route.items():
            indices = sorted([x[0] for x in items], reverse=True)
            for o_idx in indices:
                order = routes[r_idx]['orders'].pop(o_idx)
                routes[r_idx]['load'] -= order['demand']
                removed.append(order)

        return removed

    def _destroy_route(self, routes: List[Dict], num_to_remove: int) -> List[Dict]:
        """路径移除算子 - 随机移除整条路径的订单"""
        removed = []

        # 找出非空路径
        non_empty_routes = [(i, r) for i, r in enumerate(routes) if r['orders']]

        if not non_empty_routes:
            return removed

        # 随机选择一条路径
        r_idx, route = random.choice(non_empty_routes)

        # 移除该路径上的部分或全部订单
        orders_to_remove = min(num_to_remove, len(route['orders']))

        # 随机选择移除位置
        indices_to_remove = random.sample(range(len(route['orders'])), orders_to_remove)
        indices_to_remove.sort(reverse=True)

        for o_idx in indices_to_remove:
            order = route['orders'].pop(o_idx)
            route['load'] -= order['demand']
            removed.append(order)

        return removed

    # ==================== 修复算子 ====================

    def _repair_greedy(self, routes: List[Dict], orders_to_insert: List[Dict],
                       current_time: int, speed: float):
        """贪婪修复算子 - 每次选择成本增加最小的位置"""
        # 按时间窗紧迫度排序
        orders_to_insert.sort(key=lambda x: x['tw_end'])

        inserted_ids = set()

        for order in orders_to_insert:
            best_cost = float('inf')
            best_route_idx = -1
            best_pos = -1

            for r_idx, route in enumerate(routes):
                # Critical Fix for KeyError: 'load'
                if 'load' not in route:
                    route['load'] = sum(o['demand'] for o in route['orders'])

                if route['load'] + order['demand'] > self.capacity:
                    continue

                for pos in range(len(route['orders']) + 1):
                    if self._check_feasibility(route, order, pos, current_time, speed):
                        cost = self._calculate_insertion_cost(route, order, pos)
                        if cost < best_cost:
                            best_cost = cost
                            best_route_idx = r_idx
                            best_pos = pos

            if best_route_idx != -1:
                routes[best_route_idx]['orders'].insert(best_pos, order)
                routes[best_route_idx]['load'] += order['demand']
                inserted_ids.add(order['id'])

        # 更新未插入的订单列表
        orders_to_insert[:] = [o for o in orders_to_insert if o['id'] not in inserted_ids]

    def _repair_regret2(self, routes: List[Dict], orders_to_insert: List[Dict],
                        current_time: int, speed: float):
        """Regret-2 修复算子"""
        self._repair_regret_k(routes, orders_to_insert, current_time, speed, k=2)

    def _repair_regret3(self, routes: List[Dict], orders_to_insert: List[Dict],
                        current_time: int, speed: float):
        """Regret-3 修复算子"""
        self._repair_regret_k(routes, orders_to_insert, current_time, speed, k=3)

    def _repair_regret_k(self, routes: List[Dict], orders_to_insert: List[Dict],
                         current_time: int, speed: float, k: int = 2):
        """
        Regret-k 修复算子
        选择后悔值最大的订单优先插入（即第k最优位置与最优位置成本差最大的订单）
        """
        inserted_ids = set()

        while orders_to_insert:
            best_regret = -float('inf')
            best_order = None
            best_route_idx = -1
            best_pos = -1

            for order in orders_to_insert:
                if order['id'] in inserted_ids:
                    continue

                # 计算该订单在所有位置的插入成本
                insertion_costs = []

                for r_idx, route in enumerate(routes):
                    if route['load'] + order['demand'] > self.capacity:
                        continue

                    for pos in range(len(route['orders']) + 1):
                        if self._check_feasibility(route, order, pos, current_time, speed):
                            cost = self._calculate_insertion_cost(route, order, pos)
                            insertion_costs.append((cost, r_idx, pos))

                if not insertion_costs:
                    continue

                # 按成本排序
                insertion_costs.sort(key=lambda x: x[0])

                # 计算 regret 值
                best_insertion = insertion_costs[0]
                regret = 0

                for i in range(1, min(k, len(insertion_costs))):
                    regret += insertion_costs[i][0] - best_insertion[0]

                # 如果可插入位置少于k个，增加惩罚
                if len(insertion_costs) < k:
                    regret += (k - len(insertion_costs)) * self.unserved_penalty

                if regret > best_regret:
                    best_regret = regret
                    best_order = order
                    best_route_idx = best_insertion[1]
                    best_pos = best_insertion[2]

            if best_order is None:
                break

            # 插入最佳订单
            routes[best_route_idx]['orders'].insert(best_pos, best_order)
            routes[best_route_idx]['load'] += best_order['demand']
            inserted_ids.add(best_order['id'])

        # 更新未插入的订单列表
        orders_to_insert[:] = [o for o in orders_to_insert if o['id'] not in inserted_ids]

    # ==================== 局部搜索 ====================

    def _local_search(self, routes: List[Dict], current_time: int, speed: float):
        """应用局部搜索优化"""
        improved = True
        max_iterations = 50
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            # 2-opt 优化（路径内）
            for route in routes:
                if self._two_opt(route, current_time, speed):
                    improved = True

            # relocate 优化（路径间）
            if self._relocate(routes, current_time, speed):
                improved = True

    def _two_opt(self, route: Dict, current_time: int, speed: float) -> bool:
        """2-opt 路径优化"""
        orders = route['orders']
        if len(orders) < 2:
            return False

        vehicle = route['vehicle']
        improved = False

        for i in range(len(orders) - 1):
            for j in range(i + 1, len(orders)):
                # 尝试反转 i 到 j 之间的片段
                new_orders = orders[:i] + orders[i:j+1][::-1] + orders[j+1:]

                # 检查新路径的可行性
                temp_route = {
                    'vehicle': vehicle,
                    'orders': new_orders,
                    'load': route['load']
                }

                if self._is_route_feasible(temp_route, current_time, speed):
                    # 计算成本变化
                    old_cost = self._calculate_route_cost(route)
                    new_cost = self._calculate_route_cost(temp_route)

                    if new_cost < old_cost - 1e-6:
                        route['orders'] = new_orders
                        improved = True

        return improved

    def _relocate(self, routes: List[Dict], current_time: int, speed: float) -> bool:
        """relocate 优化 - 将订单从一条路径移动到另一条"""
        improved = False

        for src_idx, src_route in enumerate(routes):
            if not src_route['orders']:
                continue

            for o_idx in range(len(src_route['orders'])):
                order = src_route['orders'][o_idx]

                # 计算移除成本
                removal_saving = self._calculate_removal_saving(src_route, o_idx)

                for dst_idx, dst_route in enumerate(routes):
                    if src_idx == dst_idx:
                        continue

                    if dst_route['load'] + order['demand'] > self.capacity:
                        continue

                    # 尝试插入到目标路径的每个位置
                    for pos in range(len(dst_route['orders']) + 1):
                        # [FIXED] Use passed-in parameter current_time
                        if self._check_feasibility(dst_route, order, pos, current_time, speed):
                            insertion_cost = self._calculate_insertion_cost(dst_route, order, pos)

                            if insertion_cost < removal_saving - 1e-6:
                                # 执行 relocate
                                src_route['orders'].pop(o_idx)
                                src_route['load'] -= order['demand']
                                dst_route['orders'].insert(pos, order)
                                dst_route['load'] += order['demand']
                                improved = True
                                break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        return improved

    def _calculate_removal_saving(self, route: Dict, o_idx: int) -> float:
        """计算移除订单后节省的成本"""
        orders = route['orders']
        vehicle = route['vehicle']
        depot = self.depots[vehicle['home_depot']]

        order = orders[o_idx]
        prev_node = orders[o_idx - 1] if o_idx > 0 else depot
        next_node = orders[o_idx + 1] if o_idx < len(orders) - 1 else depot

        current_cost = (
            self.calculate_distance(prev_node, order) +
            self.calculate_distance(order, next_node)
        )
        new_cost = self.calculate_distance(prev_node, next_node)

        return current_cost - new_cost

    # ==================== 辅助函数 ====================

    def _perturb_solution(self, routes: List[Dict], unassigned: List[Dict]):
        """当陷入局部最优时进行扰动"""
        # 随机移除 20-40% 的订单
        assigned = [o for r in routes for o in r['orders']]
        if not assigned:
            return

        num_to_remove = random.randint(
            max(1, int(len(assigned) * 0.2)),
            max(1, int(len(assigned) * 0.4))
        )

        removed = self._destroy_random(routes, num_to_remove)
        unassigned.extend(removed)

    def _check_feasibility(self, route: Dict, order: Dict, pos: int,
                          current_time: int, speed: float) -> bool:
        """检查在指定位置插入订单是否可行"""
        vehicle = route['vehicle']
        depot = self.depots[vehicle['home_depot']]

        # 构建临时路径
        temp_orders = route['orders'][:pos] + [order] + route['orders'][pos:]

        curr_time = max(current_time, vehicle['available_time'])
        curr_loc = depot

        for node in temp_orders:
            travel_t = self.calculate_travel_time(curr_loc, node, speed)
            arrival = curr_time + travel_t

            # 检查时间窗
            if arrival > node['tw_end']:
                return False

            start_service = max(arrival, node['tw_start'])
            finish_service = start_service + node['service_time']

            curr_time = finish_service
            curr_loc = node

        # 检查返程时间
        if not self.allow_late_return:
            travel_back = self.calculate_travel_time(curr_loc, depot, speed)
            if curr_time + travel_back > 1440:
                return False

        return True

    def _is_route_feasible(self, route: Dict, current_time: int, speed: float) -> bool:
        """检查整条路径是否可行"""
        vehicle = route['vehicle']
        depot = self.depots[vehicle['home_depot']]

        curr_time = max(current_time, vehicle['available_time'])
        curr_loc = depot

        for node in route['orders']:
            travel_t = self.calculate_travel_time(curr_loc, node, speed)
            arrival = curr_time + travel_t

            if arrival > node['tw_end']:
                return False

            start_service = max(arrival, node['tw_start'])
            finish_service = start_service + node['service_time']

            curr_time = finish_service
            curr_loc = node

        # 检查返程时间
        if not self.allow_late_return:
            travel_back = self.calculate_travel_time(curr_loc, depot, speed)
            if curr_time + travel_back > 1440:
                return False

        return True

    def _calculate_insertion_cost(self, route: Dict, order: Dict, pos: int) -> float:
        """计算插入成本增量"""
        vehicle = route['vehicle']
        depot = self.depots[vehicle['home_depot']]
        orders = route['orders']

        prev_node = orders[pos - 1] if pos > 0 else depot
        next_node = orders[pos] if pos < len(orders) else depot

        # 插入后的新距离
        dist_add = (
            self.calculate_distance(prev_node, order) +
            self.calculate_distance(order, next_node)
        )
        # 原来的距离
        dist_remove = self.calculate_distance(prev_node, next_node)

        return dist_add - dist_remove

    def _calculate_route_cost(self, route: Dict) -> float:
        """计算单条路径的总距离"""
        vehicle = route['vehicle']
        depot = self.depots[vehicle['home_depot']]

        if not route['orders']:
            return 0

        total_dist = 0
        curr = depot

        for node in route['orders']:
            total_dist += self.calculate_distance(curr, node)
            curr = node

        # 回到仓库
        total_dist += self.calculate_distance(curr, depot)

        return total_dist

    def _calculate_solution_cost(self, routes: List[Dict], unassigned: List[Dict]) -> float:
        """
        计算整个解的目标函数值
        目标: 最小化总距离 + 未服务订单惩罚
        """
        total_dist = 0

        for route in routes:
            total_dist += self._calculate_route_cost(route)

        # 未服务订单惩罚
        penalty = len(unassigned) * self.unserved_penalty

        return total_dist + penalty

    def _format_solution(self, routes: List[Dict], all_orders: List[Dict],
                        current_time: int, speed: float) -> Tuple[Dict, List[Dict]]:
        """将内部解格式转换为输出格式"""
        assigned_routes = {}
        assigned_ids = set()

        for route in routes:
            if not route['orders']:
                continue

            vehicle = route['vehicle']
            depot = self.depots[vehicle['home_depot']]

            route_objs = [depot] + route['orders']

            # 计算到达时间
            arrival_times = []
            curr_time = max(current_time, vehicle['available_time'])
            curr_loc = depot
            arrival_times.append(curr_time)  # Depot 出发时间

            for node in route['orders']:
                dist = self.calculate_distance(curr_loc, node)
                travel = math.ceil(dist / speed)
                arr = curr_time + travel
                arrival_times.append(arr)

                start = max(arr, node['tw_start'])
                curr_time = start + node['service_time']
                curr_loc = node

                assigned_ids.add(node['id'])

            # 计算返回时间
            dist_back = self.calculate_distance(curr_loc, depot)
            return_time = curr_time + math.ceil(dist_back / speed)

            # [FIX] Include depot at the end of the route for consistent distance calculation
            final_route_objs = route_objs + [depot]

            assigned_routes[vehicle['id']] = {
                'route_objs': final_route_objs,
                'arrival_times': arrival_times + [return_time], # Add return time
                'customer_ids': [c['id'] for c in route['orders']],
                'departure_time': max(current_time, vehicle['available_time']),
                'return_time': return_time
            }

        dropped_orders = [o for o in all_orders if o['id'] not in assigned_ids]

        return assigned_routes, dropped_orders