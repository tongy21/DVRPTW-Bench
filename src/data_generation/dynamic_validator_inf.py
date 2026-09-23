import random

try:
    from .dynamic_validator import DynamicValidator
except ImportError:  # Support direct execution from this directory.
    from dynamic_validator import DynamicValidator


class DynamicValidatorInfCommit(DynamicValidator):
    """Service-rate validator that matches main_test.py with commit_lead_time=inf.

    Difference from DynamicValidator:
    - Original DynamicValidator cancels not-yet-departed routes at the next epoch,
      which is a flexible pre-departure revision behavior.
    - In main_test.py, commit_lead_time=inf makes commit_time == plan_time.
      Routes become COMMITTED immediately after planning and are never withdrawn,
      although their physical departure_time can still be later.

    This validator keeps the lightweight OR-Tools batch solver from DynamicValidator,
    but changes the rolling-horizon state transition to immediate commitment.
    It is intended for dataset filtering / feasibility screening, not final benchmark
    evaluation across solvers.
    """

    def run_simulation(self):
        pending_order_ids = set(c['id'] for c in self.customers)
        committed_order_ids = set()
        failed_order_ids = set()

        for current_time in range(0, 1441, self.interval):
            if self.verbose:
                print(f"\n=== Sim Time: {current_time} | immediate commit / leadtime=inf ===")

            available_vehicles = []

            for v in self.vehicles:
                # COMMITTED: route is fixed immediately after planning, but the vehicle
                # may still wait at depot until its physical departure_time.
                if v['status'] == 'COMMITTED':
                    if current_time >= v['pending_route']['departure_time']:
                        if self.verbose:
                            print(
                                f"  [Info] Vehicle {v['id']} DEPARTED. "
                                f"(Planned departure: {v['pending_route']['departure_time']})"
                            )
                        v['status'] = 'BUSY'
                        v['return_time'] = v['pending_route']['return_time']
                        self.all_executed_routes.append({
                            'vehicle_id': v['id'],
                            'route_path': v['pending_route']['route_objs'],
                            'times': v['pending_route']['arrival_times'],
                        })
                        v['pending_route'] = None

                if v['status'] == 'BUSY':
                    if current_time >= v['return_time']:
                        if self.verbose:
                            print(f"  [Info] Vehicle {v['id']} RETURNED to depot.")
                        v['status'] = 'IDLE'

                if v['status'] == 'IDLE':
                    available_vehicles.append(v)

            if available_vehicles:
                random.shuffle(available_vehicles)

            # This is the same order-pool semantics as main_test.py under leadtime=inf:
            # only orders that have never been assigned/committed remain in pending_order_ids.
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
                    print("  No available uncommitted orders to plan.")
                continue

            if not available_vehicles:
                if self.verbose:
                    print("  No vehicles available. Orders remain in the pool.")
                continue

            if self.verbose:
                print(
                    f"  Planning {len(current_batch_orders)} orders "
                    f"with {len(available_vehicles)} vehicles..."
                )

            assigned_routes, dropped_orders_list = self.solve_batch(
                current_batch_orders, available_vehicles, current_time
            )

            if dropped_orders_list and self.verbose:
                print(
                    f"  [Warning] {len(dropped_orders_list)} orders dropped in this batch "
                    f"and will be retried later."
                )

            for vid, route_info in assigned_routes.items():
                vehicle = next(v for v in self.vehicles if v['id'] == vid)
                vehicle['status'] = 'COMMITTED'
                vehicle['pending_route'] = route_info

                for cid in route_info['customer_ids']:
                    if cid in pending_order_ids:
                        pending_order_ids.remove(cid)
                        committed_order_ids.add(cid)

                if self.verbose:
                    print(
                        f"    -> Vehicle {vid} committed: "
                        f"{len(route_info['customer_ids'])} custs. "
                        f"Depart at {route_info['departure_time']}."
                    )

        # Keep routes that are committed but have not physically departed by the horizon.
        # This follows main_test.py's accounting, where committed orders are counted in SR.
        for v in self.vehicles:
            if v['status'] == 'COMMITTED' and v['pending_route'] is not None:
                self.all_executed_routes.append({
                    'vehicle_id': v['id'],
                    'route_path': v['pending_route']['route_objs'],
                    'times': v['pending_route']['arrival_times'],
                })

        self.total_served_count = len(committed_order_ids)
        total_customers = len(self.customers)
        service_rate = (self.total_served_count / total_customers) * 100 if total_customers else 0.0
        print(f"service_rate: ({self.total_served_count}/{total_customers})={service_rate:.2f}%")
        return service_rate
