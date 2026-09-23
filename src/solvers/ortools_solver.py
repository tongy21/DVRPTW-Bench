import math
from typing import List, Dict, Tuple
from .base_solver import BaseSolver

# 尝试导入 OR-Tools
try:
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    OR_TOOLS_AVAILABLE = True
except ImportError:
    OR_TOOLS_AVAILABLE = False

class ORToolsSolver(BaseSolver):
    """
    基于 Google OR-Tools 的 VRP 求解器
    支持：多车场、时间窗、容量限制、软约束(丢单惩罚)
    """
    def __init__(self, depots: List[Dict], capacity: int, time_limit: int = 2, allow_late_return: bool = True):
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = capacity
        if not OR_TOOLS_AVAILABLE:
            print("Warning: OR-Tools not installed. Please run 'pip install ortools'")

    def solve_batch(self, batch_orders: List[Dict], available_vehicles: List[Dict],
                   current_time: int, vehicle_speed: float) -> Tuple[Dict, List[Dict]]:

        if not OR_TOOLS_AVAILABLE or not batch_orders or not available_vehicles:
            return {}, batch_orders

        # 1. 构建数据模型
        # Node 0..D-1: Depots (D个)
        # Node D..D+N-1: Customers (N个)
        num_depots = len(self.depots)
        active_nodes = self.depots + batch_orders

        data = {}
        data['num_vehicles'] = len(available_vehicles)
        data['starts'] = [v['home_depot'] for v in available_vehicles]
        # OR-Tools 要求车辆必须有终点，VRP中通常终点就是起点
        data['ends'] = [v['home_depot'] for v in available_vehicles]

        # 构建矩阵 (距离和时间)
        dim = len(active_nodes)
        dist_matrix = [[0] * dim for _ in range(dim)]
        time_matrix = [[0] * dim for _ in range(dim)]

        for i in range(dim):
            for j in range(dim):
                if i == j: continue
                dist = self.calculate_distance(active_nodes[i], active_nodes[j])
                # OR-Tools 只接受整数，放大因子可保留精度，这里直接取整
                dist_matrix[i][j] = int(math.ceil(dist))
                # 时间 = 距离 / 速度
                travel_time = int(math.ceil(dist / vehicle_speed))
                time_matrix[i][j] = travel_time

        data['time_matrix'] = time_matrix
        data['demands'] = [0] * num_depots + [o['demand'] for o in batch_orders]
        # 服务时间: Depot 为 0
        data['service_times'] = [0] * num_depots + [o['service_time'] for o in batch_orders]

        # 时间窗: Depot (当前时间 ~ horizon), 客户 (tw_start ~ tw_end)
        # 将 horizon 设置得足够大，允许车辆晚归（只要能满足客户的时间窗即可）
        horizon = 100000 if self.allow_late_return else 1440
        time_windows = []
        for _ in range(num_depots):
            time_windows.append((current_time, horizon))
        for o in batch_orders:
            time_windows.append((o['tw_start'], o['tw_end']))

        # 2. 创建 Routing Manager
        manager = pywrapcp.RoutingIndexManager(
            len(time_matrix),
            data['num_vehicles'],
            data['starts'],
            data['ends']
        )
        routing = pywrapcp.RoutingModel(manager)

        # 3. 注册回调
        # 时间回调
        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['time_matrix'][from_node][to_node] + data['service_times'][from_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # 添加时间维度
        time_dim_name = 'Time'
        routing.AddDimension(
            transit_callback_index,
            horizon,  # 允许最大等待时间 (slack)
            horizon,  # 车辆最大行驶时间 (horizon)
            False, # start_cumul_to_zero
            time_dim_name
        )
        time_dimension = routing.GetDimensionOrDie(time_dim_name)

        # 客户节点在 RoutingIndexManager 中只有一个索引，可以直接设置。
        # 车场节点则会对应每辆车各自的 start/end 索引，不能只通过
        # manager.NodeToIndex(depot) 约束其中一个车辆起点。
        for location_idx in range(num_depots, len(time_windows)):
            start, end = time_windows[location_idx]
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(start, end)

        # 每辆车都必须从当前重规划时刻或之后出发。遗漏这里会让除第一辆
        # 车外的车辆沿用默认下界 0，从而产生“在订单出现前出发”的路线。
        for vehicle_idx in range(data['num_vehicles']):
            time_dimension.CumulVar(routing.Start(vehicle_idx)).SetRange(current_time, horizon)
            time_dimension.CumulVar(routing.End(vehicle_idx)).SetRange(current_time, horizon)

        # 添加容量维度
        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # null capacity slack
            [self.capacity] * data['num_vehicles'],
            True,
            'Capacity'
        )

        # 4. 允许丢单 (Penalty)
        # 为所有客户节点添加析取约束
        penalty = 1000000
        for node in range(num_depots, len(time_matrix)):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        # 5. 求解参数
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

        # fix: handle float time_limit (e.g. 0.2s)
        seconds_int = int(self.time_limit)
        nanos_int = int((self.time_limit - seconds_int) * 1e9)
        search_parameters.time_limit.seconds = seconds_int
        search_parameters.time_limit.nanos = nanos_int

        # 6. 求解
        solution = routing.SolveWithParameters(search_parameters)

        assigned_routes = {}
        dropped_orders = []

        if solution:
            # 提取被丢弃的订单
            for node in range(num_depots, len(time_matrix)):
                if routing.IsStart(node) or routing.IsEnd(node): continue
                if solution.Value(routing.NextVar(manager.NodeToIndex(node))) == manager.NodeToIndex(node):
                    dropped_orders.append(active_nodes[node])

            # 提取路径
            for i, vehicle in enumerate(available_vehicles):
                index = routing.Start(i)
                if routing.IsEnd(solution.Value(routing.NextVar(index))):
                    continue

                route_nodes = []
                arrival_times = []

                while not routing.IsEnd(index):
                    node_idx = manager.IndexToNode(index)
                    time_val = solution.Value(time_dimension.CumulVar(index))
                    route_nodes.append(active_nodes[node_idx])
                    arrival_times.append(time_val)
                    index = solution.Value(routing.NextVar(index))

                # 添加终点
                node_idx = manager.IndexToNode(index)
                time_val = solution.Value(time_dimension.CumulVar(index))
                route_nodes.append(active_nodes[node_idx])
                arrival_times.append(time_val)

                # 计算出发时间
                depot_node = route_nodes[0]
                first_cust = route_nodes[1]
                dist = self.calculate_distance(depot_node, first_cust)
                travel_time = int(math.ceil(dist / vehicle_speed))

                first_arrival = arrival_times[1]
                latest_departure = max(current_time, first_arrival - travel_time)

                # 以实际出发时间重新执行路线，避免仅截断 departure_time、
                # 却保留求解器中更早时间轴上的 arrival_times。
                executable_times = [latest_departure]
                route_clock = latest_departure
                for from_node, to_node in zip(route_nodes, route_nodes[1:]):
                    route_clock += int(math.ceil(self.calculate_distance(from_node, to_node) / vehicle_speed))
                    if 'tw_start' in to_node:
                        route_clock = max(route_clock, to_node['tw_start'])
                    executable_times.append(route_clock)
                    route_clock += to_node.get('service_time', 0)

                # 提取 customer_ids (排除 depot)
                c_ids = [n['id'] for n in route_nodes if 'tw_start' in n]

                assigned_routes[vehicle['id']] = {
                    'route_objs': route_nodes,
                    'arrival_times': executable_times,
                    'departure_time': latest_departure,
                    'return_time': executable_times[-1],
                    'customer_ids': c_ids
                }
        else:
            dropped_orders = batch_orders

        return assigned_routes, dropped_orders
