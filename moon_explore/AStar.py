"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import numpy as np
from scipy.ndimage import distance_transform_edt
from pathlib import Path

try:
    from .Utils import Setting, Pose2D, RoverPath
except:
    from Utils import Setting, Pose2D, RoverPath

from typing import Dict, List, Optional
from numpy.typing import NDArray


class AStarPlanner:
    def __init__(self, rr: float, obstacle_map: NDArray[np.bool_]) -> None:
        """
        Initialize grid map for a star planning

        rr: robot radius[m]
        obstacle_map: obstacle map (0: free, 1: obstacle)
        """

        self.resolution = 1 / Setting.MAP_SCALE  # grid resolution [m]
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 500, 500
        visible_ob = obstacle_map  # 仿真环境因为已知就提前算好了

        # 计算距离场，算出来两个不同安全度的膨胀
        distance_map: NDArray[np.float64] = distance_transform_edt(~visible_ob)  # type: ignore
        self.euclidean_dilated_least = distance_map <= rr * Setting.MAP_SCALE * 1.2  # 最小的膨胀，剪枝时使用这个
        self.euclidean_dilated_base = distance_map <= rr * Setting.MAP_SCALE * 1.5  # 这两个用于规划
        self.euclidean_dilated_safe = distance_map <= rr * Setting.MAP_SCALE * 2.0

        self.x_width, self.y_width = distance_map.shape
        self.motion = self.get_motion_model()

    class Node:
        def __init__(self, x: int, y: int, cost: float, parent_index: int) -> None:
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

        def __repr__(self):
            return str(self.x) + "," + str(self.y) + "," + str(self.cost) + "," + str(self.parent_index)

    def generate_straight_path(self, source: Pose2D, goal: Pose2D, mask: NDArray[np.bool_]) -> RoverPath:
        """
        在估计路径成本时，直接生成一条直线路径用于估计，并且对存在碰撞的路径估计加倍的时间

        Args:
            source (Pose2D): 初始位姿
            goal (Pose2D): 目标位姿
            mask (NDArray[np.bool_]): 计算碰撞使用的地图

        Returns:
            RoverPath: 返回直线路径，其中collision标识是否碰撞
        """
        path_float = np.array([[source.x, source.y], [goal.x, goal.y]])
        yaw1 = np.arctan2(goal.y - source.y, goal.x - source.x)
        pose1 = Pose2D(source.x, source.y, yaw1)
        collision = self.line_of_sight(
            mask,
            (source.x / self.resolution, source.y / self.resolution),
            (goal.x / self.resolution, goal.y / self.resolution),
        )

        return RoverPath(path_float, [pose1, goal], not collision)

    def planning(self, sx: float, sy: float, goal: Pose2D, mask: NDArray[np.bool_]) -> Optional[RoverPath]:
        """
        二阶段A*路径规划

        Args:
            sx (float): start x position [m]
            sy (float): start y position [m]
            gx (float): goal x position [m]
            gy (float): goal y position [m]
            mask (NDArray[np.bool_]): 地图掩码，True表示探明区域

        Returns:
            Optional[RoverPath]: (N, 2) 存储路径上的各点坐标(单位m)，还有一个Pose2D的列表
        """
        self.obstacle_map: NDArray[np.bool_] = self.euclidean_dilated_safe | ~mask
        self.goal = goal  # 最后生成的路径要记录这个，里面有个偏航角
        gx, gy = goal.x, goal.y
        result = self.plan_once(sx, sy, gx, gy)
        if result is None:
            self.obstacle_map = self.euclidean_dilated_base | ~mask
            result = self.plan_once(sx, sy, gx, gy)
        if result is not None:
            return self.simplify_path(result)
        else:
            return None

    def plan_once(self, sx, sy, gx, gy) -> Optional[NDArray[np.int32]]:
        # sx_idx, sy_idx = self.calc_xy_index(sx), self.calc_xy_index(sy)

        # y_min = max(sy_idx - 10, 0)
        # y_max = min(sy_idx + 10, self.obstacle_map.shape[0])
        # x_min = max(sx_idx - 10, 0)
        # x_max = min(sx_idx + 10, self.obstacle_map.shape[1])
        # self.obstacle_map[y_min:y_max, x_min:x_max] = False

        start_node = self.Node(self.calc_xy_index(sx), self.calc_xy_index(sy), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx), self.calc_xy_index(gy), 0.0, -1)

        open_set: Dict[int, AStarPlanner.Node] = dict()
        closed_set: Dict[int, AStarPlanner.Node] = dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                return None

            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node, open_set[o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(
                    current.x + self.motion[i][0], current.y + self.motion[i][1], current.cost + self.motion[i][2], c_id
                )
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        path = self.calc_final_path(goal_node, closed_set)

        return path

    @staticmethod
    def line_of_sight(grid, p1, p2):
        """
        检查p1到p2之间是否有障碍物
        传入栅格坐标
        """
        x1, y1 = p1
        x2, y2 = p2
        points = np.linspace((x1, y1), (x2, y2), num=100)  # 采样100个点
        return all(grid[round(y), round(x)] == 0 for x, y in points)  # 确保中间点无障碍物

    def simplify_once(self, path, ref_ob=None):
        """单次三角剪枝"""
        if ref_ob is None:  # 给画图程序留的接口
            ref_ob = self.euclidean_dilated_least

        simplified = [path[0]]  # 起点
        last = 0
        for i in range(1, len(path)):
            if not self.line_of_sight(ref_ob, path[last], path[i]):
                if i != 1:
                    simplified.append(path[i - 1])
                    last = i - 1

        simplified.append(path[-1])  # 终点
        return np.array(simplified)

    def simplify_path(self, path: NDArray[np.int32]) -> RoverPath:
        """
        三角剪枝，移除不必要的拐点直至收敛
        输入的路径存储的栅格坐标
        输出的路径存储的是实际的m
        """
        length_last = len(path) + 1
        cnt = 0
        while length_last != len(path) and cnt <= 5:
            length_last = len(path)
            path = self.simplify_once(path)
            cnt += 1

        # 把单位从栅格坐标变换回m
        path_float = (path * self.resolution).astype(np.float64)
        path_pose: List[Pose2D] = []
        x1, y1 = path_float[0]
        for i in range(1, path_float.shape[0]):
            x2, y2 = path_float[i]
            yaw = np.arctan2(y2 - y1, x2 - x1)
            path_pose.append(Pose2D(x1, y1, yaw))
            x1, y1 = x2, y2
        path_pose.append(self.goal)

        return RoverPath(path_float, path_pose, False)

    def calc_final_path(self, goal_node: Node, closed_set: Dict[int, Node]) -> NDArray[np.int32]:
        # generate final course
        rx, ry = [goal_node.x], [goal_node.y]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(n.x)
            ry.append(n.y)
            parent_index = n.parent_index

        path = np.array([[x, y] for x, y in zip(reversed(rx), reversed(ry))])  # (N, 2)
        return path

    @staticmethod
    def calc_heuristic(n1: Node, n2: Node) -> float:
        w = 1.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_xy_index(self, position) -> int:
        return round(position / self.resolution)

    def calc_grid_index(self, node: Node) -> int:
        """
        二维矩阵展平后的索引值

        Args:
            node (Node): 要计算的栅格

        Returns:
            int: 展平后索引值
        """
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node: Node) -> bool:
        """
        验证节点所在的位置是否合法（地图范围内/非障碍物）

        Returns:
            bool: 合法为True，否则为False
        """
        px = node.x
        py = node.y

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[int(node.y), int(node.x)]:
            return False

        return True

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [
            [1, 0, 1],
            [0, 1, 1],
            [-1, 0, 1],
            [0, -1, 1],
            [-1, -1, math.sqrt(2)],
            [-1, 1, math.sqrt(2)],
            [1, -1, math.sqrt(2)],
            [1, 1, math.sqrt(2)],
        ]

        return motion


def main():
    from Viewer import MaskViewer
    from Map import Map
    import matplotlib.pyplot as plt
    import time

    NPY_ROOT = Path(__file__).parent.parent / "resource"
    map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), god=True)
    planner = AStarPlanner(0.8, map.obstacle_mask)
    start = time.time()
    path = planner.planning(2, 2, Pose2D(12.5, 8, 0), map.mask)
    print(f"{time.time() - start:.4f} sec")

    viewer = MaskViewer(map)
    viewer.update()
    if path is None:
        print("Cannot find path")
        return
    print(path)

    viewer.show()
    viewer.plot_path(path)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
