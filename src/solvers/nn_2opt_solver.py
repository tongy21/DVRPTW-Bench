import math
import copy
import time
from typing import List, Dict, Tuple
# 注意这里使用相对导入，确保在 main.py 运行时能找到 BaseSolver
from .base_solver import BaseSolver

class NNSolver(BaseSolver):
    """
    使用 Nearest Neighbor + 2-Opt 策略求解单个批次的 VRP 问题
    """

    def __init__(self, depots: List[Dict], capacity: int, time_limit: int = 2, allow_late_return: bool = True):
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = capacity

    def solve_batch(self, batch_orders: List[Dict], available_vehicles: List[Dict],
                   current_time: int, vehicle_speed: float, perform_2opt: bool = True) -> Tuple[Dict, List[Dict]]:
        """
        实现基类的求解接口
        """
        if not batch_orders or not available_vehicles:
            return {}, batch_orders

        # Optimization: Pre-calculate distances or use optimized lookup if N is large?
        # For N=500, unassigned_orders loop is O(N) inside O(N) loop -> O(N^2).
        # With 400 orders: 400*200 iterations approx 80,000 checks.
        # Python loop overhead is significant.
        # "calculate_distance" has sqrt math call.

        # 优化: 快速缓存计算函数
        calc_dist = self.calculate_distance
        calc_time = self.calculate_travel_time
        sq = math.sqrt
        ceil = math.ceil

        # 1. 待分配订单池 (复制一份以免修改原数据)
        # Using a list is slow for removal (O(N)). Set is unordered.
        # But we iterate it.
        unassigned_orders = batch_orders.copy()
        assigned_routes = {}

        # 2. 为每辆可用车辆构建路径 (Nearest Neighbor)
        for vehicle in available_vehicles:
            if not unassigned_orders:
                break

            depot_idx = vehicle['home_depot']
            depot_node = self.depots[depot_idx]

            # 初始化路径
            route_nodes = [depot_node] # 起点
            arrival_times = [current_time] # Depot 处的虚拟到达时间
            current_load = 0

            # 当前位置和时间
            curr_node = depot_node
            # Cache coordinates to avoid dict lookup in tight loop
            curr_x, curr_y = curr_node['x'], curr_node['y']

            curr_time = current_time # 车辆从当前时间开始可用

            # 贪婪构建
            while True:
                best_customer = None
                best_distance = float('inf')
                best_arrival_time = -1
                best_idx = -1

                # 优化: 简单过滤
                # 在剩余订单中寻找最近的可行邻居
                # Python 循环慢，尝试减少循环体内操作

                for idx, order in enumerate(unassigned_orders):
                    # 1. 快速检查容量 (无函数调用)
                    if current_load + order['demand'] > self.capacity:
                        continue

                    # 2. 快速计算距离 (展开 sqrt 以减少开销? 或许没必要，但减少函数调用有帮助)
                    # dist = sq((curr_x - order['x'])**2 + (curr_y - order['y'])**2)
                    ox, oy = order['x'], order['y']
                    dx = curr_x - ox
                    dy = curr_y - oy
                    dist_sq = dx*dx + dy*dy

                    # 3. 剪枝: 如果距离平方已经大于最佳距离平方，跳过开方和后续检查
                    if dist_sq >= best_distance * best_distance and best_distance != float('inf'):
                        continue

                    dist = sq(dist_sq)

                    # 4. 时间计算
                    # Inline calculate_travel_time: int(math.ceil(dist / speed))
                    # travel_time = int(ceil(dist / vehicle_speed))
                    # Opt: if speed is 1.0 (common in benchmarks), travel_time = int(ceil(dist))
                    if vehicle_speed == 1.0:
                        travel_time = int(ceil(dist))
                    else:
                        travel_time = int(ceil(dist / vehicle_speed))

                    arrival_time = curr_time + travel_time

                    # 检查时间窗 (TW)
                    # 如果到达时间早于开始时间，需要等待
                    arrival_time = max(arrival_time, order['tw_start'])

                    # 如果到达时间晚于结束时间，不可行
                    if arrival_time > order['tw_end']:
                        continue

                    # 检查晚归约束: 必须能在 1440 前回到 Depot
                    if not self.allow_late_return:
                        leave_time = arrival_time + order['service_time']
                        # back_time = calc_time(order, depot_node, vehicle_speed)
                        # Inline back calculation
                        bdx = ox - depot_node['x']
                        bdy = oy - depot_node['y']
                        bdist = sq(bdx*bdx + bdy*bdy)
                        back_time = int(ceil(bdist / vehicle_speed))

                        if leave_time + back_time > 1440:
                            continue

                    # 找到可行解
                    # We are here, means valid and dist < best_distance (checked by dist_sq)
                    best_distance = dist
                    best_customer = order
                    best_arrival_time = arrival_time
                    best_idx = idx

                if best_customer:
                    # 加入路径
                    route_nodes.append(best_customer)
                    arrival_times.append(best_arrival_time)
                    current_load += best_customer['demand']

                    # 更新状态
                    curr_node = best_customer
                    curr_x, curr_y = curr_node['x'], curr_node['y']
                    # 离开时间 = 到达时间 + 服务时间
                    curr_time = best_arrival_time + best_customer['service_time']

                    # Remove by index is faster if we pop, but list shifts.
                    # Remove by value is unassigned_orders.remove(best_customer)
                    # pop(idx) is O(k).
                    unassigned_orders.pop(best_idx)
                else:
                    # 没有可行邻居，结束该车路径
                    break


            # 只有当路径包含客户时才保存 (len > 1 因为包含 Depot)
            if len(route_nodes) > 1:
                # 返回 Depot
                travel_back = self.calculate_travel_time(curr_node, depot_node, vehicle_speed)
                return_time = curr_time + travel_back

                route_nodes.append(depot_node)
                arrival_times.append(return_time)

                # 3. 应用 2-Opt 优化 (可选)
                if perform_2opt:
                    optimized_nodes, optimized_times = self._apply_2opt(
                        route_nodes, arrival_times, vehicle_speed
                    )
                else:
                    optimized_nodes, optimized_times = route_nodes, arrival_times

                # 计算最晚出发时间
                first_cust_arrival = optimized_times[1]
                travel_to_first = self.calculate_travel_time(depot_node, optimized_nodes[1], vehicle_speed)
                latest_departure = max(current_time, first_cust_arrival - travel_to_first)

                assigned_routes[vehicle['id']] = {
                    'route_objs': optimized_nodes,
                    'arrival_times': optimized_times,
                    'departure_time': latest_departure,
                    'return_time': optimized_times[-1],
                    # [修正点 1] 仅将含有 'tw_start' 的节点视为客户提取 ID，避免包含 Depot ID
                    'customer_ids': [n['id'] for n in optimized_nodes if 'tw_start' in n]
                }

        return assigned_routes, unassigned_orders

    def _apply_2opt(self, route_objs: List[Dict], arrival_times: List[int], speed: float):
        """
        简单的 2-opt 优化。
        """
        if len(route_objs) <= 3:
            return route_objs, arrival_times

        best_route = route_objs
        best_times = arrival_times
        improved = True

        start_time = time.time() # 记录开始时间

        def get_route_dist(r):
            d = 0
            for i in range(len(r)-1):
                d += self.calculate_distance(r[i], r[i+1])
            return d

        while improved:

            if time.time() - start_time > self.time_limit:
                break

            improved = False
            current_dist = get_route_dist(best_route)

            # 遍历所有交换可能 (保留 Depot 在首尾)
            for i in range(1, len(best_route) - 2):
                for j in range(i + 1, len(best_route) - 1):
                    # 执行 2-opt 交换
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]

                    # 检查新路径的可行性
                    is_feasible, new_times = self._check_integrity(new_route, best_times[0], speed)

                    if is_feasible:
                        new_dist = get_route_dist(new_route)
                        if new_dist < current_dist - 1e-6:
                            best_route = new_route
                            best_times = new_times
                            improved = True
                            break
                if improved:
                    break

        return best_route, best_times

    def _check_integrity(self, route_nodes: List[Dict], start_time: int, speed: float) -> Tuple[bool, List[int]]:
        """
        检查路径是否满足时间窗，并返回新的到达时间列表
        """
        times = [start_time]
        curr_time = start_time
        curr_node = route_nodes[0]

        for i in range(1, len(route_nodes)):
            next_node = route_nodes[i]
            travel = self.calculate_travel_time(curr_node, next_node, speed)
            arrival = curr_time + travel

            # [修正点 2] 使用 'tw_start' 判断是否为客户，而不是 'id'
            # 因为 Depot 也有 id 字段，但没有 tw_start
            if 'tw_start' in next_node:
                # 是客户：应用时间窗约束

                # 早到等待
                arrival = max(arrival, next_node['tw_start'])
                # 晚到不可行
                if arrival > next_node['tw_end']:
                    return False, []

                # 更新当前时间 (离开时间 = 到达 + 服务)
                curr_time = arrival + next_node['service_time']
            else:
                # 是 Depot：无时间窗约束
                # 如果不允许晚归，则需检查是否在 1440 前返回
                if not self.allow_late_return and arrival > 1440:
                    return False, []
                curr_time = arrival

            times.append(arrival)
            curr_node = next_node

        return True, times