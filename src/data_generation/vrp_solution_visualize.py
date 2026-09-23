import os
import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.font_manager as fm

# 配置字体为 Times New Roman
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

class VRPSolutionVisualizer:
    """
    Optimized VRP Visualizer (Scheme 1)
    """

    def __init__(self, results_dir: str, output_dir: str):
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _format_time(self, minutes):
        """Convert minutes to HH:MM format"""
        if minutes is None: return "N/A"
        h = int(minutes // 60)
        m = int(minutes % 60)
        return f"{h:02d}:{m:02d}"

    def visualize_solution(self, result: dict, output_file: str, time_display: bool = True, legend_display: bool = True):
        # 1. 数据预处理
        locations_raw = result.get('locations', [])
        if not locations_raw: return

        locations = np.array(locations_raw, dtype=float)

        if locations.ndim != 2 or locations.shape[1] < 2: return

        depots = result.get('depots', [])
        routes = result.get('routes', [])
        route_times = result.get('route_times', [])
        service_times = result.get('service_times', [])

        if not service_times and len(locations) > 0:
            service_times = [0] * len(locations)
        elif isinstance(service_times, dict):
            st_list = [0] * len(locations)
            for k, v in service_times.items():
                if int(k) < len(st_list): st_list[int(k)] = v
            service_times = st_list

        # 2. 设置绘图
        fig, ax = plt.subplots(figsize=(14, 10))

        x_range = locations[:, 0].max() - locations[:, 0].min()
        y_range = locations[:, 1].max() - locations[:, 1].min()
        if x_range == 0: x_range = 100
        if y_range == 0: y_range = 100

        offset_x = x_range * 0.015
        offset_y = y_range * 0.015

        # 3. 绘制节点
        # 先画客户
        ax.scatter(locations[:, 0], locations[:, 1], c='skyblue', s=40, edgecolors='k', zorder=2, label='Customer')

        # 4. 绘制路径
        cmap = plt.get_cmap('tab20')

        for r_idx, route in enumerate(routes):
            if len(route) < 2: continue

            # 补全最后一段回车库的路线，如果它丢失了的话
            if route[-1] != route[0]:
                route = list(route) + [route[0]]

            color = cmap(r_idx % 20)
            valid_route = [node for node in route if node < len(locations)]
            if len(valid_route) < 2: continue

            path_coords = locations[valid_route]

            # 路径线
            ax.plot(path_coords[:, 0], path_coords[:, 1], c=color, linewidth=1.5, alpha=0.8, zorder=1, label=f"Vehicle {r_idx}")

            # 箭头
            for i in range(len(valid_route) - 1):
                p1 = locations[valid_route[i]]
                p2 = locations[valid_route[i+1]]
                mid = (p1 + p2) / 2
                dir_vec = p2 - p1
                norm = np.linalg.norm(dir_vec)
                if norm > 0:
                    dir_vec /= norm
                    arrow_size = min(x_range, y_range) * 0.015
                    ax.arrow(mid[0], mid[1], dir_vec[0]*0.1, dir_vec[1]*0.1,
                             head_width=arrow_size, head_length=arrow_size*1.2,
                             fc=color, ec=color, zorder=1)

            # 时间标注 (仅关键点)
            if time_display and r_idx < len(route_times) and len(route_times[r_idx]) == len(route):
                times = route_times[r_idx]
                for step, node_idx in enumerate(route):
                    if node_idx >= len(locations): continue

                    is_start = (step == 0)
                    is_first_cust = (step == 1)
                    is_last_cust = (step == len(route) - 2)
                    is_end = (step == len(route) - 1)

                    if not (is_start or is_end or is_first_cust or is_last_cust):
                        continue

                    arr_time = times[step]
                    s_time = service_times[node_idx] if node_idx < len(service_times) else 0
                    dep_time = arr_time + s_time
                    time_str = f"A:{self._format_time(arr_time)}\nD:{self._format_time(dep_time)}"

                    ax.text(locations[node_idx][0] + offset_x, locations[node_idx][1] + offset_y,
                            time_str, fontsize=7, color='black',
                            fontweight='bold' if (is_start or is_end) else 'normal',
                            verticalalignment='bottom',
                            bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray', boxstyle='round,pad=0.2'),
                            zorder=10)

        # 5. 最后画 Depot，确保在所有线条之上
        valid_depots = [d for d in depots if d < len(locations)]
        if valid_depots:
            depot_locs = locations[valid_depots]
            ax.scatter(depot_locs[:, 0], depot_locs[:, 1], c='red', marker='s', s=120, edgecolors='k', zorder=20, label='Depot')

        # 设置图例
        problem_name = result.get('problem', 'Unknown')
        solver_name = result.get('solver', 'Unknown')
        ax.set_title(f"Route Visualization: {problem_name}\nSolver: {solver_name}")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        if legend_display:
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0., title="Vehicles", fontsize='small')
        ax.grid(False)
        ax.set_aspect('equal', 'box')

        plt.tight_layout()
        try:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            if output_file.endswith('.png'):
                output_svg = output_file[:-4] + '.svg'
                plt.savefig(output_svg, bbox_inches='tight')
        except Exception as e:
            print(f"Error saving image {output_file}: {e}")
        plt.close()

    def visualize_all_solutions(self):
        import glob
        files = glob.glob(os.path.join(self.results_dir, "**/*.json"), recursive=True)
        for f_path in files:
            try:
                with open(f_path, 'r') as f:
                    data = json.load(f)
                if 'routes' not in data and 'locations' not in data: continue
                solver_name = data.get('solver', 'Unknown')
                solver_dir = os.path.join(self.output_dir, solver_name)
                os.makedirs(solver_dir, exist_ok=True)
                fname = os.path.basename(f_path).replace('.json', '.png')
                out_path = os.path.join(solver_dir, fname)
                self.visualize_solution(data, out_path, time_display=False)
            except Exception as e:
                print(f"Error processing {f_path}: {e}")

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "dataset/dataset_scale_v200/dataset_final_0.0"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "viz_output"
    os.makedirs(output_dir, exist_ok=True)
    viz = VRPSolutionVisualizer(results_dir=results_dir, output_dir=output_dir)
    viz.visualize_all_solutions()
    print(f"Visualization complete. Check out the '{output_dir}' directory.")