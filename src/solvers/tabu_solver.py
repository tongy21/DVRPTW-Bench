import math
import random
import copy
import time
import numpy as np
from typing import List, Dict, Tuple
from .base_solver import BaseSolver
from .nn_2opt_solver import NNSolver

class TabuSolver(BaseSolver):
    """
    禁忌搜索求解器
    """
    def __init__(self, depots: List[Dict], capacity: int, time_limit: int = 2, allow_late_return: bool = True):
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = capacity
        # 传递时间限制给 NNSolver
        self.nn_solver = NNSolver(depots, capacity, time_limit=time_limit, allow_late_return=allow_late_return)

        # [修改] 增加迭代次数
        self.max_iter = 100000
        self.tabu_tenure = 10
        self.neighbors_sample = 30

    def solve_batch(self, batch_orders: List[Dict], available_vehicles: List[Dict],
                   current_time: int, vehicle_speed: float) -> Tuple[Dict, List[Dict]]:

        # print(f"[TabuSolver][{current_time}] random.getstate()[:3] entry:", random.getstate()[1][:3])
        # print(f"[TabuSolver][{current_time}] np.random.get_state()[1][:3] entry:", np.random.get_state()[1][:3])

        start_time = time.time()

        # 1. 生成初始解
        # [修改] 启用 NN 中的 2-opt: Tabu 在较短时间内很难从极差的初始解中恢复，需要较好的 baseline
        initial_routes, unassigned = self.nn_solver.solve_batch(
            batch_orders, available_vehicles, current_time, vehicle_speed, perform_2opt=True
        )

        if not initial_routes:
            return {}, batch_orders

        route_list = []
        vehicle_ids = []
        for vid, info in initial_routes.items():
            route_list.append(info['route_objs'])
            vehicle_ids.append(vid)

        best_routes = [list(r) for r in route_list]
        best_cost = self._calculate_route_cost(best_routes, current_time, vehicle_speed)
        current_routes = [list(r) for r in route_list]
        tabu_list = []

        # 2. Tabu 循环
        for _ in range(self.max_iter):
            # [新增] 时间检查
            if time.time() - start_time > self.time_limit:
                break

            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None

            for _ in range(self.neighbors_sample):
                move = self._generate_random_move(current_routes)
                if not move: continue
                candidate_routes = self._apply_move(current_routes, move)
                if not candidate_routes: continue
                if not self._check_routes_feasible(candidate_routes, current_time, vehicle_speed):
                    continue

                cost = self._calculate_route_cost(candidate_routes, current_time, vehicle_speed)
                is_tabu = self._is_tabu(move, tabu_list)
                if not is_tabu or cost < best_cost:
                    if cost < best_neighbor_cost:
                        best_neighbor = candidate_routes
                        best_neighbor_cost = cost
                        best_move = move

            if best_neighbor:
                current_routes = best_neighbor
                tabu_list.append(self._get_move_signature(best_move))
                if len(tabu_list) > self.tabu_tenure: tabu_list.pop(0)
                if best_neighbor_cost < best_cost:
                    best_routes = [list(r) for r in best_neighbor]
                    best_cost = best_neighbor_cost

        # 3. 封装结果 (过滤空路径)
        final_assigned = {}
        for i, r_nodes in enumerate(best_routes):
            if len(r_nodes) <= 2:  # 过滤纯粹的 [depot, depot] 空车路线
                continue
            vid = vehicle_ids[i]
            _, times = self._calculate_route_times(r_nodes, current_time, vehicle_speed, check_feasible=False)
            depot_node = r_nodes[0]
            first_cust = r_nodes[1] if len(r_nodes)>1 else depot_node
            travel = self.calculate_travel_time(depot_node, first_cust, vehicle_speed)
            latest_departure = max(current_time, times[1] - travel) if len(times)>1 else current_time
            cust_ids = [n['id'] for n in r_nodes if 'tw_start' in n]

            final_assigned[vid] = {
                'route_objs': r_nodes, 'arrival_times': times,
                'departure_time': latest_departure, 'return_time': times[-1],
                'customer_ids': cust_ids
            }

        # print(f"[TabuSolver][{current_time}] random.getstate()[:3] exit:", random.getstate()[1][:3])
        # print(f"[TabuSolver][{current_time}] np.random.get_state()[1][:3] exit:", np.random.get_state()[1][:3])

        final_cost = self._calculate_total_distance(best_routes)
        # print(f"[TabuSolver] Batch Cost: {final_cost:.2f} (Unassigned: {len(unassigned)})")

        return final_assigned, unassigned

    # ... (其余 helper methods _generate_random_move, _apply_move, _calculate_total_distance 等保持不变) ...
    def _generate_random_move(self, routes):
        move_type = random.choice(['relocate', 'swap'])
        valid_indices = [i for i, r in enumerate(routes) if len(r) > 2]
        if len(valid_indices) < 1: return None
        r1_idx = random.choice(valid_indices)
        r1 = routes[r1_idx]
        node1_idx = random.randint(1, len(r1) - 2)
        if move_type == 'relocate':
            r2_idx = random.randint(0, len(routes) - 1)
            r2 = routes[r2_idx]
            insert_idx = random.randint(1, len(r2) - 1)
            return ('relocate', r1_idx, node1_idx, r2_idx, insert_idx)
        elif move_type == 'swap':
            if len(valid_indices) < 2: return None
            r2_idx = random.choice(valid_indices)
            while r1_idx == r2_idx: r2_idx = random.choice(valid_indices)
            r2 = routes[r2_idx]
            node2_idx = random.randint(1, len(r2) - 2)
            return ('swap', r1_idx, node1_idx, r2_idx, node2_idx)
        return None

    def _apply_move(self, routes, move):
        new_routes = [list(r) for r in routes]
        type = move[0]
        if type == 'relocate':
            _, r1, idx1, r2, ins2 = move
            node = new_routes[r1].pop(idx1)
            # fix for same-route relocation: if we popped before the insertion index, shifting occurs
            if r1 == r2 and ins2 > idx1:
                ins2 -= 1
            new_routes[r2].insert(ins2, node)
        elif type == 'swap':
            _, r1, idx1, r2, idx2 = move
            new_routes[r1][idx1], new_routes[r2][idx2] = new_routes[r2][idx2], new_routes[r1][idx1]
        return new_routes

    def _get_move_signature(self, move): return (move[0], move[1], move[3])
    def _is_tabu(self, move, tabu_list): return self._get_move_signature(move) in tabu_list

    def _calculate_route_cost(self, routes, current_time, speed):
        cost = 0
        vehicle_fixed_cost = 500  # 车辆使用惩罚
        wait_time_penalty = 1.0   # 等待时间惩罚

        for r in routes:
            if len(r) <= 2:
                continue

            cost += vehicle_fixed_cost
            curr = current_time

            for i in range(len(r)-1):
                n1, n2 = r[i], r[i+1]
                travel_dist = self.calculate_distance(n1, n2)
                travel_time = self.calculate_travel_time(n1, n2, speed)
                cost += travel_dist

                arr = curr + travel_time
                if 'tw_start' in n2:
                    if arr < n2['tw_start']:
                        wait_time = n2['tw_start'] - arr
                        cost += wait_time * wait_time_penalty
                        arr = n2['tw_start']
                    curr = arr + n2.get('service_time', 0)
                else:
                    curr = arr
        return cost

    def _calculate_total_distance(self, routes):
        dist = 0
        for r in routes:
            for i in range(len(r)-1): dist += self.calculate_distance(r[i], r[i+1])
        return dist
    def _check_routes_feasible(self, routes, current_time, speed):
        for r in routes:
            load = sum(n['demand'] for n in r if 'demand' in n)
            if load > self.capacity: return False
            valid, _ = self._calculate_route_times(r, current_time, speed)
            if not valid: return False
        return True

    def _calculate_route_times(self, route, start_time, speed, check_feasible=True):
        times = [start_time]
        curr = start_time
        for i in range(len(route)-1):
            n1, n2 = route[i], route[i+1]
            travel = self.calculate_travel_time(n1, n2, speed)
            arr = curr + travel
            if 'tw_start' in n2:
                arr = max(arr, n2['tw_start'])
                if check_feasible and arr > n2['tw_end']: return False, []
                curr = arr + n2['service_time']
            else: curr = arr
            times.append(arr)

        # 检查晚归
        if check_feasible and not self.allow_late_return:
             if times[-1] > 1440: return False, []

        return True, times