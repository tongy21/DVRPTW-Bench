import math
import random
from typing import List, Dict, Tuple
from .base_solver import BaseSolver
import logging

try:
    from pyvrp import Model, ProblemData, VehicleType, Client, Depot
    from pyvrp.stop import MaxRuntime
    import numpy as np
    PYVRP_AVAILABLE = True
except ImportError:
    PYVRP_AVAILABLE = False


class HGSSolver(BaseSolver):
    """
    HGS Solver using PyVRP library if available.
    Falls back to a greedy heuristic otherwise.
    """

    def __init__(self, depots: List[Dict], capacity: int, time_limit: int = 2, allow_late_return: bool = True):
        super().__init__(depots, time_limit=time_limit, allow_late_return=allow_late_return)
        self.capacity = capacity

    def solve_batch(
        self,
        batch_orders: List[Dict],
        available_vehicles: List[Dict],
        current_time: int,
        vehicle_speed: float,
    ) -> Tuple[Dict, List[Dict]]:

        # Determine if we should use PyVRP or fallback
        if PYVRP_AVAILABLE:
            # print(f"  [HGS Check] PYVRP_AVAILABLE=True. Attempting solve. TimeLimit={self.time_limit}")
            try:
                return self._solve_pyvrp(batch_orders, available_vehicles, current_time, vehicle_speed)
            except Exception as e:
                logging.error(f"PyVRP execution failed: {e}. Falling back to greedy strategy.")
                print(f"  [HGS Error] PyVRP execution failed: {e}. Falling back to greedy strategy.")
                # We can print traceback for debugging but keep running
                import traceback
                traceback.print_exc()
                return self._solve_greedy(batch_orders, available_vehicles, current_time, vehicle_speed)
        else:
            print(f"  [HGS Check] PYVRP_AVAILABLE=False. Using Greedy.")
            return self._solve_greedy(batch_orders, available_vehicles, current_time, vehicle_speed)

    def _solve_pyvrp(self, batch_orders, available_vehicles, current_time, vehicle_speed):
        if not batch_orders or not available_vehicles:
            return {}, batch_orders

        # Scaling Factors (PyVRP uses integers)
        COORD_SCALE = 100
        TIME_SCALE = 10  # e.g. 0.1 min precision

        # [Adjusted] PRIZE was 1_000_000.
        # Set to a large value to encourage service (minimize drops).
        # PRIZE=0 implies mandatory in PyVRP. Use a small positive value for optional/low priority.
        PRIZE = 1
        # print(f"  [HGS Check] Inside _solve_pyvrp. PRIZE={PRIZE}")

        m = Model()

        # 1. Add Depots
        # Simulation runs until 1440 usually, but some orders require late return.
        # Extended to 10000 to allow late returns (matching ALNS behavior if it ignores max_horizon).
        SIM_END_TIME = (10000 if self.allow_late_return else 1440) * TIME_SCALE
        START_TIME_SCALED = int(current_time * TIME_SCALE)

        depot_objs = []
        depot_name_to_idx = {}

        for idx, d in enumerate(self.depots):
            d_name = f"depot_{idx}"
            dp = m.add_depot(
                x=int(d['x'] * COORD_SCALE),
                y=int(d['y'] * COORD_SCALE),
                tw_early=START_TIME_SCALED,
                tw_late=SIM_END_TIME,
                name=d_name
            )
            depot_objs.append(dp)
            depot_name_to_idx[d_name] = idx

        # 2. Add Clients (with Reachability Filtering)
        client_objs = []
        filtered_batch_orders = []
        client_name_to_order = {}

        # Identify depots that have available vehicles
        active_depot_indices = set(v['home_depot'] for v in available_vehicles)

        for i, o in enumerate(batch_orders):
            # ... (existing filtering logic) ...
            # Check if order is reachable from ANY active depot within time window
            is_reachable = False
            for d_idx in active_depot_indices:
                depot = self.depots[d_idx]

                # 1. Travel time from Depot to Customer
                t_go = self.calculate_travel_time(depot, o, vehicle_speed)
                arrival_at_cust = current_time + t_go

                # Condition A: Must arrive before customer's latest deadline
                if arrival_at_cust > o['tw_end']:
                    continue

                # 2. Service Start Time (wait if early)
                start_service = max(arrival_at_cust, o['tw_start'])

                # 3. Travel time from Customer back to Depot
                t_back = self.calculate_travel_time(o, depot, vehicle_speed)

                # 4. Finish time at Depot
                return_to_depot_time = start_service + o['service_time'] + t_back


                # Condition B: Must return to depot before extended simulation end
                # Relaxed from 1440 to 10000 to catch all orders
                limit_time = 100000 if self.allow_late_return else 1440
                if return_to_depot_time <= limit_time:
                    is_reachable = True
                    break

            if not is_reachable:
                continue

            c_name = f"client_{o['id']}_{i}"
            # ...
            c = m.add_client(
                x=int(o['x'] * COORD_SCALE),
                y=int(o['y'] * COORD_SCALE),
                delivery=int(o['demand']),
                service_duration=int(o['service_time'] * TIME_SCALE),
                tw_early=int(o['tw_start'] * TIME_SCALE),
                tw_late=int(o['tw_end'] * TIME_SCALE),
                prize=PRIZE,
                name=c_name
            )
            client_objs.append(c)
            client_name_to_order[c_name] = o
            filtered_batch_orders.append(o)

        if not client_objs:
            # If no clients are reachable, return empty assignment (or greedy fallback which will also do nothing but return dropped)
            logging.info("  [HGS Info] No reachable clients after filtering. Skipping PyVRP.")
            return {}, batch_orders


        # 3. Add Vehicle Types
        vehs_by_depot = {}
        for v in available_vehicles:
            vehs_by_depot.setdefault(v['home_depot'], []).append(v)

        vehicle_assignments = {d_idx: [] for d_idx in vehs_by_depot}

        for d_idx, vehs in vehs_by_depot.items():
            m.add_vehicle_type(
                num_available=len(vehs),
                capacity=self.capacity,
                start_depot=depot_objs[d_idx],
                end_depot=depot_objs[d_idx]
            )
            vehicle_assignments[d_idx] = vehs

        # 4. Add Edges (Distance & Duration matrix)
        # Combine all locations: Depots + Clients
        all_locs_objs = depot_objs + client_objs

        # Corresponding original data for distance calculation
        # Careful with indices.
        # depot_objs[i] correspond to self.depots[i]
        # client_objs[i] correspond to filtered_batch_orders[i]
        all_locs_data = self.depots + filtered_batch_orders

        for i, u in enumerate(all_locs_objs):
            for j, v in enumerate(all_locs_objs):
                if i == j: continue

                # Keep the optimization distance precise, but use exactly the
                # same conservative integer travel-time convention as the
                # simulation environment. The previous floor operation could
                # make PyVRP accept a route that became late when replayed with
                # ceil(distance / speed).
                distance = self.calculate_distance(all_locs_data[i], all_locs_data[j])
                d_int = int(math.ceil(distance * COORD_SCALE))
                travel_minutes = self.calculate_travel_time(
                    all_locs_data[i], all_locs_data[j], vehicle_speed
                )
                t_int = int(travel_minutes * TIME_SCALE)

                m.add_edge(u, v, distance=d_int, duration=t_int)

        # 5. Solve
        # Use existing time_limit logic
        import warnings
        import time
        t_start = time.time()
        # print(f"  [HGS Info] Starting PyVRP solve for {len(client_objs)} clients with time_limit={self.time_limit}s...")

        # Generate a deterministic seed from the current controlled random state
        # This ensures that if DynamicSimulation sets the seed, PyVRP respects it.
        pyvrp_seed = random.randint(0, 2**31 - 1)

        # Suppress warnings to avoid cluttering output
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Force max_runtime explicitly to ensure it propagates
            # Pass seed to ensure reproducibility
            res = m.solve(stop=MaxRuntime(float(self.time_limit)), seed=pyvrp_seed, display=False)

        t_dur = time.time() - t_start
        # print(f"  [HGS Info] PyVRP finished in {t_dur:.2f}s. Feasible: {res.best.is_feasible()}")

        # 6. Parse Result
        assigned_routes = {}
        served_order_ids = set()
        veh_counters = {d_idx: 0 for d_idx in vehicle_assignments}

        if not res.best.is_feasible():
            # If PyVRP failed to find a feasible solution (even with PRIZE), fallback to Greedy
            print(f"  [HGS Warning] Solver could not find a feasible solution in {self.time_limit}s. Falling back to Greedy.")
            return self._solve_greedy(batch_orders, available_vehicles, current_time, vehicle_speed)

        for route in res.best.routes():
                # Identify which depot this route belongs to
                # route.vehicle_type() returns the index of the vehicle type
                vt_idx = route.vehicle_type()
                vt = m.vehicle_types[vt_idx]

                # vt.start_depot is the index of the depot in m.locations
                depot_loc = m.locations[vt.start_depot]
                depot_idx = depot_name_to_idx[depot_loc.name]

                # Assign to one of the available vehicles at this depot
                if veh_counters[depot_idx] >= len(vehicle_assignments[depot_idx]):
                    continue # Should not happen if model is correct

                vehicle = vehicle_assignments[depot_idx][veh_counters[depot_idx]]
                veh_counters[depot_idx] += 1

                # Reconstruct Route Path
                route_nodes = [self.depots[depot_idx]] # Start

                for client_idx in route.visits():
                    # client_idx is the index in m.locations
                    client_loc = m.locations[client_idx]
                    order = client_name_to_order[client_loc.name]
                    route_nodes.append(order)
                    served_order_ids.add(order['id'])

                route_nodes.append(self.depots[depot_idx]) # End

                # Recalculate exact float times as per Simulation Standards
                curr_t = float(current_time)
                times = [curr_t]
                curr_node = route_nodes[0]

                # Forward Pass
                for i in range(1, len(route_nodes)):
                    next_node = route_nodes[i]
                    travel = self.calculate_travel_time(curr_node, next_node, vehicle_speed)
                    arr = curr_t + travel

                    if 'tw_start' in next_node:
                        arr = max(arr, next_node['tw_start'])

                    times.append(arr)

                    service = next_node.get('service_time', 0)
                    curr_t = arr + service
                    curr_node = next_node

                # Determine Departure Time (shift route strictly to fit earliest start)
                # First customer arrival is at times[1]
                # Travel to first customer is t0
                t0 = self.calculate_travel_time(route_nodes[0], route_nodes[1], vehicle_speed)
                # Maximize departure: depart as late as necessary to arrive at times[1]
                # but not earlier than current_time
                actual_start = max(float(current_time), times[1] - t0)

                assigned_routes[vehicle['id']] = {
                    'route_objs': route_nodes,
                    'arrival_times': times,
                    'departure_time': actual_start,
                    'return_time': times[-1],
                    'customer_ids': [n['id'] for n in route_nodes if 'id' in n and 'tw_start' in n],
                }

        dropped_orders = [o for o in batch_orders if o['id'] not in served_order_ids]
        return assigned_routes, dropped_orders

    def _solve_greedy(
        self,
        batch_orders: List[Dict],
        available_vehicles: List[Dict],
        current_time: int,
        vehicle_speed: float,
    ) -> Tuple[Dict, List[Dict]]:

        if not batch_orders or not available_vehicles:
            return {}, batch_orders

        # 订单复制，避免原地修改
        unassigned = list(batch_orders)

        # 按 depot 分组车辆
        vehicles_by_depot = {}
        for v in available_vehicles:
            vehicles_by_depot.setdefault(v['home_depot'], []).append(v)

        assigned_routes: Dict[int, Dict] = {}

        # 对每个 depot，逐车构造可行路径
        for d_idx, vehs in vehicles_by_depot.items():
            depot = self.depots[d_idx]

            for vehicle in vehs:
                if not unassigned:
                    break

                route_nodes = [depot]
                arrival_times = [current_time]
                load = 0
                curr_node = depot
                curr_time = current_time

                while True:
                    best = None
                    best_arrival = None

                    for o in unassigned:
                        if load + o['demand'] > self.capacity:
                            continue

                        travel = self.calculate_travel_time(curr_node, o, vehicle_speed)
                        arrival = max(curr_time + travel, o['tw_start'])
                        if arrival > o['tw_end']:
                            continue

                        # 检查晚归
                        if not self.allow_late_return:
                            finish_service = arrival + o['service_time']
                            back_time = self.calculate_travel_time(o, depot, vehicle_speed)
                            if finish_service + back_time > 1440:
                                continue

                        # 选择最早可到达的候选
                        if best is None or arrival < best_arrival:
                            best = o
                            best_arrival = arrival

                    if best is None:
                        break

                    # 插入该客户
                    route_nodes.append(best)
                    arrival_times.append(best_arrival)
                    load += best['demand']
                    curr_node = best
                    curr_time = best_arrival + best['service_time']
                    unassigned.remove(best)

                if len(route_nodes) > 1:
                    # 回到 depot
                    back_travel = self.calculate_travel_time(curr_node, depot, vehicle_speed)
                    arrival_times.append(curr_time + back_travel)
                    route_nodes.append(depot)

                    # 计算出发时间（保证不早于 current_time）
                    first_arrival = arrival_times[1]
                    dist0 = self.calculate_travel_time(route_nodes[0], route_nodes[1], vehicle_speed)
                    departure_time = max(current_time, first_arrival - dist0)

                    assigned_routes[vehicle['id']] = {
                        'route_objs': route_nodes,
                        'arrival_times': arrival_times,
                        'departure_time': departure_time,
                        'return_time': arrival_times[-1],
                        'customer_ids': [n['id'] for n in route_nodes if 'id' in n and 'tw_start' in n],
                    }

        # 剩余未服务的订单
        dropped_orders = unassigned

        return assigned_routes, dropped_orders
