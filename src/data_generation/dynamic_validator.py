import json
import math
import copy
import os
import numpy as np
import random
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

# 导入可视化模块
try:
    from .vrp_solution_visualize import VRPSolutionVisualizer as SolutionPlotter
except ImportError:  # Support direct execution from this directory.
    from vrp_solution_visualize import VRPSolutionVisualizer as SolutionPlotter

class DynamicValidator:
    def __init__(self, data_file, optimization_interval=30, vehicle_speed=50.0, verbose=False):
        self.data_file_name = data_file
        with open(data_file, 'r') as f:
            self.raw_data = json.load(f)
        self.interval = optimization_interval
        self.vehicle_speed = vehicle_speed
        self.verbose = verbose

        self.depots = self.raw_data['depots']
        self.num_depots = len(self.depots)
        self.vehicle_cap = self.raw_data['capacity']
        self.total_vehicles = self.raw_data['num_vehicles']

        self.vehicles = []
        base_count = self.total_vehicles // self.num_depots
        remainder = self.total_vehicles % self.num_depots
        v_id_counter = 0
        for depot_idx in range(self.num_depots):
            count = base_count + (1 if depot_idx < remainder else 0)
            for _ in range(count):
                self.vehicles.append({
                    'id': v_id_counter, 'home_depot': depot_idx, 'status': 'IDLE',
                    'available_time': 0.0, 'return_time': 0.0, 'pending_route': None
                })
                v_id_counter += 1

        self.customers = self.raw_data['customers']
        self.customer_map = {c['id']: c for c in self.customers}
        self.all_executed_routes = []
        self.total_served_count = 0

    def calculate_distance(self, n1, n2):
        return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)

    def create_data_model(self, batch_orders, current_vehicles, current_time):
        data = {}
        # active_nodes 顺序: [Depots..., Customers...]
        active_nodes = self.depots + batch_orders
        num_depots = len(self.depots)
        data['num_vehicles'] = len(current_vehicles)

        starts = [v['home_depot'] for v in current_vehicles]
        ends = [v['home_depot'] for v in current_vehicles]
        data['starts'] = starts
        data['ends'] = ends

        AVERAGE_SPEED = self.vehicle_speed
        dim = len(active_nodes)
        transit_matrix = [[0] * dim for _ in range(dim)]
        for i in range(dim):
            for j in range(dim):
                if i == j: continue
                dist = self.calculate_distance(active_nodes[i], active_nodes[j])
                transit_matrix[i][j] = int(math.ceil(dist / AVERAGE_SPEED))
        data['distance_matrix'] = transit_matrix

        time_windows = []
        for _ in range(num_depots): time_windows.append((int(current_time), 1440))
        for o in batch_orders: time_windows.append((int(o['tw_start']), int(o['tw_end'])))
        data['time_windows'] = time_windows
        data['demands'] = [0] * num_depots + [int(o['demand']) for o in batch_orders]
        data['service_times'] = [0] * num_depots + [int(o['service_time']) for o in batch_orders]
        data['vehicle_capacities'] = [self.vehicle_cap] * data['num_vehicles']
        return data, active_nodes

    def solve_batch(self, batch_orders, current_vehicles, current_time):
        if not batch_orders or not current_vehicles:
            return {}, batch_orders

        data, active_nodes = self.create_data_model(batch_orders, current_vehicles, current_time)

        manager = pywrapcp.RoutingIndexManager(
            len(data['distance_matrix']),
            data['num_vehicles'],
            data['starts'],
            data['ends']
        )
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return data['distance_matrix'][from_node][to_node] + data['service_times'][from_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        time_dim_name = 'Time'
        routing.AddDimension(
            transit_callback_index,
            1440, 1440, False, time_dim_name)
        time_dimension = routing.GetDimensionOrDie(time_dim_name)

        for location_idx, (start, end) in enumerate(data['time_windows']):
            index = manager.NodeToIndex(location_idx)
            time_dimension.CumulVar(index).SetRange(start, end)

        def demand_callback(from_index):
            from_node = manager.IndexToNode(from_index)
            return data['demands'][from_node]

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index, 0, data['vehicle_capacities'], True, 'Capacity')

        penalty = 1000000
        num_depots = len(self.depots)
        for node in range(num_depots, len(data['distance_matrix'])):
            routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = 5
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

        solution = routing.SolveWithParameters(search_parameters)

        assigned_routes = {}
        dropped_orders = []

        if solution:
            # 识别丢单
            for node in range(num_depots, len(data['distance_matrix'])):
                if routing.IsStart(node) or routing.IsEnd(node): continue
                # 如果某个节点的 NextVar 指向它自己，说明它未被访问（被丢弃）
                if solution.Value(routing.NextVar(manager.NodeToIndex(node))) == manager.NodeToIndex(node):
                    dropped_orders.append(active_nodes[node])

            for i, vehicle in enumerate(current_vehicles):
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

                node_idx = manager.IndexToNode(index)
                time_val = solution.Value(time_dimension.CumulVar(index))
                route_nodes.append(active_nodes[node_idx])
                arrival_times.append(time_val)

                depot_node = route_nodes[0]
                first_cust = route_nodes[1]
                dist = self.calculate_distance(depot_node, first_cust)
                travel_time = int(math.ceil(dist / 50.0))

                first_arrival = arrival_times[1]
                latest_departure = max(current_time, first_arrival - travel_time)
                return_time = arrival_times[-1]

                # [修正 1] 提取客户ID时，排除 Depot 节点
                # 防止 Depot ID 与 Customer ID 冲突导致误删 pending_orders
                actual_customer_ids = []
                for n in route_nodes:
                    if 'id' in n and n not in self.depots: # 简单判断：对象不在 depot 列表中
                        actual_customer_ids.append(n['id'])

                assigned_routes[vehicle['id']] = {
                    'route_objs': route_nodes,
                    'arrival_times': arrival_times,
                    'departure_time': latest_departure,
                    'return_time': return_time,
                    'customer_ids': actual_customer_ids
                }
        else:
            dropped_orders = batch_orders

        return assigned_routes, dropped_orders

    def run_simulation(self):
        pending_order_ids = set([c['id'] for c in self.customers])
        committed_order_ids = set()
        failed_order_ids = set()

        for current_time in range(0, 1441, self.interval):
            if self.verbose:
                print(f"\n=== Sim Time: {current_time} ===")

            available_vehicles = []

            for v in self.vehicles:
                if v['status'] == 'PENDING':
                    if current_time < v['pending_route']['departure_time']:
                        if self.verbose:
                            print(f"  [Info] Vehicle {v['id']} route cancelled (Dept {v['pending_route']['departure_time']} > {current_time}). Recalculating...")
                        for cid in v['pending_route']['customer_ids']:
                            if cid in committed_order_ids:
                                committed_order_ids.remove(cid)
                                pending_order_ids.add(cid)
                        v['status'] = 'IDLE'
                        v['pending_route'] = None
                    else:
                        if self.verbose:
                            print(f"  [Info] Vehicle {v['id']} DEPARTED. (Planned: {v['pending_route']['departure_time']})")
                        v['status'] = 'BUSY'
                        v['return_time'] = v['pending_route']['return_time']
                        self.all_executed_routes.append({
                            'vehicle_id': v['id'],
                            'route_path': v['pending_route']['route_objs'],
                            'times': v['pending_route']['arrival_times']
                        })
                        v['pending_route'] = None

                if v['status'] == 'BUSY':
                    if current_time >= v['return_time']:
                        if self.verbose:
                            print(f"  [Info] Vehicle {v['id']} RETURNED to depot. Available for new tasks.")
                        v['status'] = 'IDLE'

                if v['status'] == 'IDLE':
                    available_vehicles.append(v)

            if available_vehicles:
                random.shuffle(available_vehicles)

            current_batch_orders = []
            for cid in list(pending_order_ids):
                cust = self.customer_map[cid]
                if current_time > cust['tw_end']:
                    if self.verbose:
                        print(f"  [Expire] Order {cid} expired. Marked as failed.")
                    pending_order_ids.remove(cid)
                    failed_order_ids.add(cid)
                    continue

                if cust['available_time'] <= current_time:
                    current_batch_orders.append(cust)

            if not current_batch_orders:
                if self.verbose:
                    print("  No new orders to plan.")
                continue

            if not available_vehicles:
                if self.verbose:
                    print("  No vehicles available (all busy). Orders wait.")
                continue

            if self.verbose:
                print(f"  Planning {len(current_batch_orders)} orders with {len(available_vehicles)} vehicles...")

            assigned_routes, dropped_orders_list = self.solve_batch(
                current_batch_orders, available_vehicles, current_time
            )

            if dropped_orders_list:
                if self.verbose:
                    print(f"  [Warning] {len(dropped_orders_list)} orders dropped in this batch (will retry).")

            for vid, route_info in assigned_routes.items():
                vehicle = next(v for v in self.vehicles if v['id'] == vid)
                vehicle['status'] = 'PENDING'
                vehicle['pending_route'] = route_info

                for cid in route_info['customer_ids']:
                    if cid in pending_order_ids:
                        pending_order_ids.remove(cid)
                        committed_order_ids.add(cid)

                if self.verbose:
                    print(f"    -> Vehicle {vid} assigned: {len(route_info['customer_ids'])} custs. Depart at {route_info['departure_time']}.")

        for v in self.vehicles:
            if v['status'] == 'PENDING':
                self.all_executed_routes.append({
                    'vehicle_id': v['id'],
                    'route_path': v['pending_route']['route_objs'],
                    'times': v['pending_route']['arrival_times']
                })

        self.total_served_count = len(committed_order_ids)
        total_customers = len(self.customers)
        service_rate = (self.total_served_count / total_customers) * 100

        print(f"service_rate: ({self.total_served_count}/{total_customers})={service_rate:.2f}%")

        return service_rate

    def save_results(self, output_json_path, output_img_path, time_display=True, legend_display=True):
        """
        保存结果并生成可视化

        : param output_json_path: 输出 JSON 文件路径
        : param output_img_path: 输出图片文件路径
        : param time_display: 是否显示时间标注
        : param legend_display: 是否显示图例
        """
        viz_data = {
            'locations': [], 'demands': [], 'depots': [], 'customers': [],
            'routes': [], 'route_times': [], 'service_times': [],
            'time_windows': {}, 'appear_times': {},
            'problem': f"Dynamic_Sim", 'solver': f"RH_{self.interval}"
        }

        # 1. 填充基础数据
        # 建立 Depot 坐标映射： (x, y) -> viz_index
        # 这对于多车场至关重要，因为通过 ID 无法区分 Depot 0 和 Customer 0 (如果存在)
        depot_coord_map = {}
        for i, d in enumerate(self.depots):
            viz_idx = len(viz_data['locations'])
            viz_data['locations'].append([d['x'], d['y']])
            viz_data['demands'].append(0)
            viz_data['service_times'].append(0)
            viz_data['depots'].append(viz_idx)
            depot_coord_map[(d['x'], d['y'])] = viz_idx

        cust_id_to_viz_idx = {}
        for i, c in enumerate(self.customers):
            viz_idx = len(viz_data['locations'])
            viz_data['locations'].append([c['x'], c['y']])
            viz_data['demands'].append(c['demand'])
            viz_data['service_times'].append(c['service_time'])
            viz_data['customers'].append(viz_idx)
            cust_id_to_viz_idx[c['id']] = viz_idx

            viz_data['time_windows'][viz_idx] = (c['tw_start'], c['tw_end'])
            if c.get('is_dynamic', False):
                viz_data['appear_times'][viz_idx] = c['available_time']

        # 2. 转换 Routes
        # [修正 2] 严格区分 Depot 和 Customer
        for route_record in self.all_executed_routes:
            route_indices = []
            for node in route_record['route_path']:
                # 优先级 1: 检查是否是 Depot (通过坐标匹配)
                # 因为生成器保证了坐标不重合，这是最安全的匹配方式
                if (node['x'], node['y']) in depot_coord_map:
                    route_indices.append(depot_coord_map[(node['x'], node['y'])])

                # 优先级 2: 检查是否是 Customer (通过 ID 匹配)
                elif 'id' in node and node['id'] in cust_id_to_viz_idx:
                    route_indices.append(cust_id_to_viz_idx[node['id']])

                else:
                    # 异常情况处理 (回退到 Depot 0)
                    if self.verbose:
                        print(f"Warning: Unknown node in route: {node}")
                    route_indices.append(0)

            viz_data['routes'].append(route_indices)
            viz_data['route_times'].append(route_record['times'])

        # 保存 JSON
        with open(output_json_path, 'w') as f:
            json.dump(viz_data, f, indent=4)

        # 可视化
        try:
            plotter = SolutionPlotter(os.path.dirname(output_img_path), os.path.dirname(output_img_path))
            plotter.visualize_solution(viz_data, output_img_path, time_display=time_display, legend_display=legend_display)
        except Exception as e:
            print(f"Viz Error: {e}")

if __name__ == "__main__":
    try:
        validator = DynamicValidator("dynamic_vrp_single.json", optimization_interval=30, vehicle_speed=5.0, verbose=True)
        validator.run_simulation()
        validator.save_results("dynamic_vrp_single_results.json", "result_viz/dynamic_vrp_single_solution.png", time_display=False, legend_display=False)
    except FileNotFoundError:
        print("Error: Input file 'dynamic_vrp_single.json' not found. Please run svrp_generator.py first.")
