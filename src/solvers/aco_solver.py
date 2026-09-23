import math
import random
import time
import numpy as np
from typing import List, Dict, Tuple
from .base_solver import BaseSolver

class ACOSolver(BaseSolver):
    """
    蚁群算法求解器
    """
    def __init__(self, depots: List[Dict], capacity: int, time_limit: int = 2, allow_late_return: bool = True):
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = capacity

        # [修改] 增加迭代次数，依靠时间限制退出
        self.num_ants = 20
        self.max_iter = 100000
        self.alpha = 1.0
        self.beta = 2.0
        self.rho = 0.1
        self.Q = 100000.0 # [FIX] Increased from 100.0 to match distance magnitude (e.g. 20000)

    def solve_batch(self, batch_orders: List[Dict], available_vehicles: List[Dict],
                   current_time: int, vehicle_speed: float) -> Tuple[Dict, List[Dict]]:

        # print(f"[ACOSolver][{current_time}] random.getstate()[:3] entry:", random.getstate()[1][:3])
        # print(f"[ACOSolver][{current_time}] np.random.get_state()[1][:3] entry:", np.random.get_state()[1][:3])

        # [REMOVED] Hardcoded seed reset to allow external control (DynamicSimulation)
        # random.seed(42)
        # np.random.seed(42)

        if not batch_orders or not available_vehicles:
            return {}, batch_orders

        self.pheromones = {}
        best_routes_global = None
        best_cost_global = float('inf')
        best_unassigned = []

        start_time = time.time()

        # [FIX] Force at least one iteration even if time is short
        first_iteration = True

        for _ in range(self.max_iter):
            # [新增] 时间限制检查
            # Only check time limit after the first iteration is done
            if not first_iteration and time.time() - start_time > self.time_limit:
                break

            all_ant_solutions = []

            for _ in range(self.num_ants):
                routes, unassigned = self._construct_solution(
                    batch_orders, available_vehicles, current_time, vehicle_speed
                )

                cost = self._calculate_cost(routes, unassigned)
                all_ant_solutions.append((routes, cost))

                if cost < best_cost_global:
                    best_cost_global = cost
                    best_routes_global = routes
                    best_unassigned = unassigned

            self._update_pheromones(all_ant_solutions)
            first_iteration = False

        final_assigned = {}
        if best_routes_global:
            for vid, r_info in best_routes_global.items():
                final_assigned[vid] = r_info

        # print(f"[ACOSolver][{current_time}] random.getstate()[:3] exit:", random.getstate()[1][:3])
        # print(f"[ACOSolver][{current_time}] np.random.get_state()[1][:3] exit:", np.random.get_state()[1][:3])
        return final_assigned, best_unassigned

    # ... (其余方法保持不变) ...
    def _construct_solution(self, orders, vehicles, current_time, speed):
        # (保持原样)
        unassigned = orders.copy()
        assigned_routes = {}

        for vehicle in vehicles:
            if not unassigned: break

            depot = self.depots[vehicle['home_depot']]
            route = [depot]
            times = [current_time]
            load = 0
            curr_node = depot
            curr_time = current_time

            while True:
                candidates = []
                for o in unassigned:
                    if load + o['demand'] > self.capacity: continue
                    travel = self.calculate_travel_time(curr_node, o, speed)
                    arr = max(curr_time + travel, o['tw_start'])
                    if arr <= o['tw_end']:
                        # 检查晚归
                        if not self.allow_late_return:
                             finish_service_time = arr + o['service_time']
                             travel_back = self.calculate_travel_time(o, depot, speed)
                             if finish_service_time + travel_back > 1440:
                                 continue

                        candidates.append((o, arr))

                if not candidates: break
                next_cust, arr_time = self._select_next_node(curr_node, candidates)
                route.append(next_cust)
                times.append(arr_time)
                load += next_cust['demand']
                curr_node = next_cust
                curr_time = arr_time + next_cust['service_time']
                unassigned.remove(next_cust)

            if len(route) > 1:
                travel = self.calculate_travel_time(curr_node, depot, speed)
                route.append(depot)
                times.append(curr_time + travel)
                first_arr = times[1]
                t_to_first = self.calculate_travel_time(depot, route[1], speed)
                dept_time = max(current_time, first_arr - t_to_first)
                assigned_routes[vehicle['id']] = {
                    'route_objs': route, 'arrival_times': times,
                    'departure_time': dept_time, 'return_time': times[-1],
                    'customer_ids': [n['id'] for n in route if 'tw_start' in n]
                }
        return assigned_routes, unassigned

    def _select_next_node(self, curr, candidates):
        # (保持原样)
        probs = []
        for cand, _ in candidates:
            dist = self.calculate_distance(curr, cand)
            eta = 1.0 / (dist + 0.1)
            tau = self._get_pheromone(curr, cand)
            prob = (tau ** self.alpha) * (eta ** self.beta)
            probs.append(prob)
        probs = np.array(probs)
        if probs.sum() == 0: idx = random.randint(0, len(candidates)-1)
        else:
            probs = probs / probs.sum()
            idx = np.random.choice(range(len(candidates)), p=probs)
        return candidates[idx]

    def _get_pheromone(self, n1, n2):
        id1 = n1['id'] if 'tw_start' in n1 else -1
        id2 = n2['id'] if 'tw_start' in n2 else -1
        return self.pheromones.get((id1, id2), 1.0)

    def _update_pheromones(self, solutions):
        for k in self.pheromones: self.pheromones[k] *= (1.0 - self.rho)
        solutions.sort(key=lambda x: x[1])
        for routes, cost in solutions[:3]:
            delta = self.Q / (cost + 1.0)
            for r_info in routes.values():
                r = r_info['route_objs']
                for i in range(len(r)-1):
                    id1 = r[i]['id'] if 'tw_start' in r[i] else -1
                    id2 = r[i+1]['id'] if 'tw_start' in r[i+1] else -1
                    self.pheromones[(id1, id2)] = self.pheromones.get((id1, id2), 1.0) + delta

    def _calculate_cost(self, routes, unassigned):
        dist_cost = 0
        for r_info in routes.values():
            r = r_info['route_objs']
            for i in range(len(r)-1):
                dist_cost += self.calculate_distance(r[i], r[i+1])
        return dist_cost + len(unassigned) * 100000