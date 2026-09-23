import math
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class BaseSolver(ABC):
    """
    VRP 求解器基类
    定义了动态仿真中求解器必须实现的接口。
    """

    def __init__(self, depots: List[Dict], time_limit: int = 2, allow_late_return: bool = True):
        """
        初始化求解器

        :param depots: 仓库列表 [{'x':, 'y':}, ...]
        :param time_limit: 求解最大时长（秒）
        :param allow_late_return: 是否允许车辆在 1440 分钟后返回仓库
        """
        self.depots = depots
        self.time_limit = time_limit
        self.allow_late_return = allow_late_return

    def calculate_distance(self, n1: Dict, n2: Dict) -> float:
        """计算欧几里得距离"""
        return math.sqrt((n1['x'] - n2['x'])**2 + (n1['y'] - n2['y'])**2)

    def calculate_travel_time(self, n1: Dict, n2: Dict, speed: float) -> int:
        """计算行驶时间 (分钟)"""
        dist = self.calculate_distance(n1, n2)
        return int(math.ceil(dist / speed))

    @abstractmethod
    def solve_batch(self,
                    batch_orders: List[Dict],
                    available_vehicles: List[Dict],
                    current_time: int,
                    vehicle_speed: float) -> Tuple[Dict, List[Dict]]:
        """
        求解当前批次的路径规划问题

        :param batch_orders: 当前待分配的订单列表
        :param available_vehicles: 当前可用的车辆列表
        :param current_time: 当前仿真时间
        :param vehicle_speed: 车辆速度
        :return: (assigned_routes, dropped_orders)
                 assigned_routes 格式: {vehicle_id: {'route_objs': [], 'arrival_times': [], ...}}
        """
        pass