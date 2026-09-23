import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import json
import math
import matplotlib.pyplot as plt

class SVRPGenerator:
    def __init__(self, num_customers=100, map_size=1000, dod=0.5, fixed_vehicle_num=None,
                 fixed_vehicle_capacity=None, depot_type='single', depot_placement='center', service_time_mode='linear',seed=None, verbose=False):
        """
        初始化生成器

        :param num_customers: 客户数量
        :param map_size: 地图尺寸 (整数范围 0 ~ map_size)
        :param dod: 动态请求比例 (0 ~ 1)
        :param fixed_vehicle_num: 固定车辆数量 (可选)
        :param fixed_vehicle_capacity: 固定车辆容量 (可选)，固定车辆数时使用，建议为250，且应确保总容量满足需求
        :param depot_type: 'single' 或 'multi'
        :param depot_placement: 'center' 或 'random' (仅对 multi depot 有效)
        :param service_time_mode: 'constant' 或 'linear' 服务时间计算模式
        :param seed: 随机种子 (可选)
        :param verbose: 是否打印详细信息
        """
        if seed is not None:
            np.random.seed(seed)

        self.n = num_customers
        self.map_size = int(map_size) # 确保地图大小是整数
        self.dod = dod
        self.depot_type = depot_type
        self.depot_placement = depot_placement
        self.service_time_mode = service_time_mode
        self.verbose = verbose


        # 论文参数设置
        self.res_prob = 0.6
        self.res_morning_mu = 480
        self.res_evening_mu = 1140
        self.res_sigma_m = 90
        self.res_sigma_e = 120
        self.com_mu = 780
        self.com_sigma = 60
        self.w_min = 60
        self.w_max = 120

        #是否固定车辆数
        self.fixed_vehicle_num = fixed_vehicle_num
        self.fixed_vehicle_capacity = fixed_vehicle_capacity

    def generate_all(self):
        """一次性执行所有生成步骤"""
        self.generate_locations()
        self.generate_demands()
        self.generate_time_windows()
        self.generate_service_times()
        self.calculate_fleet()
        self.make_dynamic()

    def generate_locations(self):
        """
        生成坐标：整数坐标 (0 ~ map_size)
        修改：增加去重逻辑，确保每个坐标点唯一 (不重合)
        """
        # 1. 确定城市数量
        num_cities = max(1, self.n // 50)

        # 2. 生成尽可能远离的城市中心
        initial_points = np.random.uniform(0, self.map_size, (num_cities * 50, 2))
        kmeans = KMeans(n_clusters=num_cities, n_init=10, random_state=42).fit(initial_points)
        self.city_centers = kmeans.cluster_centers_

        # --- Sigma 计算 ---
        diff = self.city_centers[:, np.newaxis, :] - self.city_centers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dist_matrix, np.inf)

        if num_cities > 1:
            min_dists = np.min(dist_matrix, axis=1)
            sigmas = min_dists / 4.0
            if self.verbose:
                print(f"Generated {num_cities} cities. Sigmas based on 0.25 nearest neighbor dist.")
        else:
            sigmas = np.array([self.map_size * 0.25])
            if self.verbose:
                print(f"Generated 1 city. Sigma set to default: {sigmas[0]:.2f}")
        # [新增] 用于记录已占用的坐标，防止重合
        # 使用 set 存储 tuple (x, y) 以便快速查找
        occupied_coords = set()

        # 3. 高斯采样客户 (整数 + 边界检查 + 去重)
        coords_list = []
        cust_per_city = [self.n // num_cities] * num_cities
        for i in range(self.n % num_cities): cust_per_city[i] += 1

        self.customer_city_map = []

        for i, (cx, cy) in enumerate(self.city_centers):
            current_sigma = sigmas[i]
            count = cust_per_city[i]

            # 逐个生成客户坐标
            for _ in range(count):
                while True:
                    # 采样
                    tx = np.random.normal(cx, current_sigma)
                    ty = np.random.normal(cy, current_sigma)

                    # 取整
                    ix = int(round(tx))
                    iy = int(round(ty))

                    # 检查 1: 边界
                    if not (0 <= ix <= self.map_size and 0 <= iy <= self.map_size):
                        continue # 重新采样

                    # 检查 2: 坐标重合 (去重)
                    if (ix, iy) in occupied_coords:
                        continue # 重新采样

                    # 通过检查，记录并保存
                    occupied_coords.add((ix, iy))
                    coords_list.append([ix, iy])
                    self.customer_city_map.append(i)
                    break

        self.coords = np.array(coords_list)

        # --- Depot 生成 (整数 + 边界检查 + 去重) ---
        # 注意：Depot 也不能和客户重合，也不能和其他 Depot 重合
        self.depots = []

        if self.depot_type == 'single':
            # 尝试放置在地图中心
            center_x, center_y = int(self.map_size/2), int(self.map_size/2)

            # 如果中心已经被客户占了 (虽然概率很小)，则在其附近寻找空位
            # 这里简单处理：如果在 occupied 中，就向周围螺旋搜索或随机搜索，这里用随机搜索简单处理
            search_radius = 1
            while (center_x, center_y) in occupied_coords:
                # 如果中心点冲突，就在附近随机找一个点
                offset_x = np.random.randint(-search_radius, search_radius+1)
                offset_y = np.random.randint(-search_radius, search_radius+1)
                center_x = int(self.map_size/2) + offset_x
                center_y = int(self.map_size/2) + offset_y
                search_radius += 1 # 扩大搜索范围

            occupied_coords.add((center_x, center_y))
            self.depots.append(np.array([center_x, center_y]))

        elif self.depot_type == 'multi':
            for i, (cx, cy) in enumerate(self.city_centers):
                current_sigma = sigmas[i]

                if self.depot_placement == 'center':
                    # 尝试放在城市中心整数点
                    dx, dy = int(round(cx)), int(round(cy))

                    # 同样检查冲突
                    while not (0 <= dx <= self.map_size and 0 <= dy <= self.map_size) or \
                          (dx, dy) in occupied_coords:
                        # 如果冲突，就在该城市中心附近微调
                        dx = int(round(cx + np.random.normal(0, 5))) # 小范围抖动
                        dy = int(round(cy + np.random.normal(0, 5)))

                    occupied_coords.add((dx, dy))
                    self.depots.append(np.array([dx, dy]))

                else:
                    # 随机分布 Depot
                    while True:
                        dx = np.random.normal(cx, current_sigma)
                        dy = np.random.normal(cy, current_sigma)
                        idx, idy = int(round(dx)), int(round(dy))

                        # 检查边界
                        if not (0 <= idx <= self.map_size and 0 <= idy <= self.map_size):
                            continue
                        # 检查冲突
                        if (idx, idy) in occupied_coords:
                            continue

                        occupied_coords.add((idx, idy))
                        self.depots.append(np.array([idx, idy]))
                        break

        self.depots = np.array(self.depots)

    def generate_demands(self, max_demand=100):
        self.demands = np.random.randint(1, max_demand + 1, self.n)

    def generate_time_windows(self):
        """
        生成时间窗：按分钟离散 (整数)
        """
        n_res = int(self.n * self.res_prob)
        n_com = self.n - n_res
        types = ['Residential'] * n_res + ['Commercial'] * n_com
        np.random.shuffle(types)

        tw_starts, tw_ends = [], []
        for t in types:
            # 持续时间取整
            duration = int(round(np.random.uniform(self.w_min, self.w_max)))

            if t == 'Residential':
                if np.random.rand() < 0.5:
                    start_float = np.random.normal(self.res_morning_mu, self.res_sigma_m)
                else:
                    start_float = np.random.normal(self.res_evening_mu, self.res_sigma_e)
            else:
                start_float = np.random.normal(self.com_mu, self.com_sigma)

            # 起始时间取整并限制范围
            start_int = int(round(start_float))
            start_int = max(0, min(start_int, 1440 - duration))

            tw_starts.append(start_int)
            tw_ends.append(start_int + duration)

        self.tw_starts = np.array(tw_starts, dtype=int)
        self.tw_ends = np.array(tw_ends, dtype=int)
        self.types = types

    def generate_service_times(self):
        """
        生成服务时间：整数分钟
        """
        mode=self.service_time_mode
        if mode == 'constant':
            self.service_times = np.full(self.n, 1, dtype=int)
        elif mode == 'linear':
            # 计算后四舍五入
            st_float = 1 + self.demands * 0.1
            self.service_times = np.rint(st_float).astype(int)

    def calculate_fleet(self):
        total_demand = np.sum(self.demands)
        if self.fixed_vehicle_num is None:

            vehicle_capacity = int(total_demand / (self.n / 10))

            self.capacity = vehicle_capacity
            multiplier = 1 / max(0.1,1-self.dod)
            self.num_vehicles = int(np.ceil(total_demand / vehicle_capacity * multiplier))
        else:
            self.num_vehicles = max(1,self.fixed_vehicle_num)
            self.capacity = self.fixed_vehicle_capacity  if self.fixed_vehicle_capacity is not None else int(total_demand/self.num_vehicles) # 固定容量

    def make_dynamic(self):
        """
        生成动态属性：Available Time 也是整数分钟
        """
        available_times = np.zeros(self.n, dtype=int)
        is_dynamic = np.zeros(self.n, dtype=bool)

        indices = np.arange(self.n)
        np.random.shuffle(indices)
        dynamic_count = int(self.n * self.dod)
        dynamic_indices = indices[:dynamic_count]

        for idx in dynamic_indices:
            is_dynamic[idx] = True
            reaction_buffer = 30
            # 计算最晚披露时间 (整数)
            latest_reveal = max(0, self.tw_starts[idx] - reaction_buffer)

            # 均匀分布采样并取整
            avail_float = np.random.uniform(0, latest_reveal)
            available_times[idx] = int(round(avail_float))

        self.available_times = available_times
        self.is_dynamic = is_dynamic

    def save_dataset(self, filename):
        depots_list = []
        for i in range(len(self.depots)):
            depots_list.append({
                "id": i,
                "x": int(self.depots[i][0]), # 强制 int
                "y": int(self.depots[i][1])  # 强制 int
            })

        dataset = {
            "name": f"{self.n}_{self.depot_type}_depot_dod{int(self.dod*100)}",
            "num_vehicles": int(self.num_vehicles),
            "capacity": int(self.capacity),
            "depots": depots_list,
            "customers": []
        }

        for i in range(self.n):
            dataset["customers"].append({
                "id": i + 1,
                "x": int(self.coords[i][0]), # 强制 int
                "y": int(self.coords[i][1]), # 强制 int
                "demand": int(self.demands[i]),
                "tw_start": int(self.tw_starts[i]), # 强制 int
                "tw_end": int(self.tw_ends[i]),     # 强制 int
                "service_time": int(self.service_times[i]), # 强制 int
                "is_dynamic": bool(self.is_dynamic[i]),
                "available_time": int(self.available_times[i]), # 强制 int
                "type": self.types[i]
            })

        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"Dataset saved to {filename}")

    def visualize(self, filename="instance_viz.png"):
        plt.figure(figsize=(10, 8))

        res_indices = [i for i, t in enumerate(self.types) if t == 'Residential']
        com_indices = [i for i, t in enumerate(self.types) if t == 'Commercial']

        if res_indices:
            plt.scatter(self.coords[res_indices, 0], self.coords[res_indices, 1],
                        c='skyblue', s=30, alpha=0.7, label='Residential')
        if com_indices:
            plt.scatter(self.coords[com_indices, 0], self.coords[com_indices, 1],
                        c='orange', s=30, alpha=0.7, label='Commercial')

        if hasattr(self, 'city_centers'):
            # 城市中心虽然是聚类出的浮点，但在图上仅作参考
            plt.scatter(self.city_centers[:, 0], self.city_centers[:, 1],
                        marker='x', c='black', s=50, alpha=0.5, label='Cluster Center')

        plt.scatter(self.depots[:, 0], self.depots[:, 1],
                    c='red', marker='s', s=80, edgecolors='black', label='Depot')

        plt.title(f"SVRP Locations (Integer Coords, N={self.n})")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.xlim(0, self.map_size)
        plt.ylim(0, self.map_size)
        plt.savefig(filename, dpi=150)
        print(f"Location visualization saved to {filename}")
        plt.close()

    def visualize_time_stats(self, filename="time_stats_viz.png"):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        res_starts = self.tw_starts[[t == 'Residential' for t in self.types]]
        com_starts = self.tw_starts[[t == 'Commercial' for t in self.types]]

        axes[0, 0].hist([res_starts, com_starts], bins=48, range=(0, 1440), stacked=True,
                        color=['skyblue', 'orange'], label=['Residential', 'Commercial'], alpha=0.8)
        axes[0, 0].set_title("Time Window Start (Minutes)")
        axes[0, 0].legend()

        durations = self.tw_ends - self.tw_starts
        axes[0, 1].hist(durations, bins=20, color='purple', alpha=0.7)
        axes[0, 1].set_title("Time Window Duration (Minutes)")

        dyn_indices = np.where(self.is_dynamic)[0]
        if len(dyn_indices) > 0:
            dyn_avail = self.available_times[dyn_indices]
            axes[1, 0].hist(dyn_avail, bins=48, range=(0, 1440), color='green', alpha=0.7)
            axes[1, 0].set_title(f"Dynamic Request Disclosure (Minutes)")

            dyn_starts = self.tw_starts[dyn_indices]
            axes[1, 1].scatter(dyn_starts, dyn_avail, alpha=0.6, c='teal', s=20)
            axes[1, 1].plot([0, 1440], [0, 1440], 'r--', label='y=x')
            axes[1, 1].plot([30, 1440], [0, 1410], 'k:', label='y=x-30')
            axes[1, 1].set_title("Available Time vs TW Start")
            axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Time stats visualization saved to {filename}")
        plt.close()

if __name__ == "__main__":
    print("--- Generating Multiple Depot Instance (Integer Coords) ---")
    # 生成 200 个客户，20% 动态，地图 1000x1000
    gen = SVRPGenerator(num_customers=1000, map_size=1000, dod=0.2, fixed_vehicle_num=200,
                        fixed_vehicle_capacity=250, depot_type='multi', depot_placement='center', service_time_mode='linear')
    gen.generate_all()

    output_json = "dynamic_vrp_single.json"
    output_loc_img = "dynamic_vrp_single_loc_viz.png"
    output_time_img = "dynamic_vrp_single_time_viz.png"

    gen.save_dataset(output_json)
    gen.visualize(output_loc_img)
    gen.visualize_time_stats(output_time_img)