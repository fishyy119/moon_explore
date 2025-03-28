"""

A* grid planning

author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

See Wikipedia article (https://en.wikipedia.org/wiki/A*_search_algorithm)

"""

import math
import numpy as np
import matplotlib.pyplot as plt

from Map import Map
from typing import Dict, List, Tuple, Callable
from numpy.typing import NDArray


class AStarPlanner:
    def __init__(self, rr: float, map: Map):
        """
        Initialize grid map for a star planning

        rr: robot radius[m]
        """

        self.map = map
        self.resolution = 1 / Map.MAP_SCALE  # grid resolution [m]
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 500, 500
        self.obstacle_map: NDArray[np.bool_] = map.obstacle_mask | ~map.mask  # TODO: 膨胀obmask
        self.x_width, self.y_width = self.obstacle_map.shape
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

    def planning(self, sx, sy, gx, gy) -> NDArray[np.int32]:
        """
        A star path search

        input:
            s_x: start x position [m]
            s_y: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """
        start_node = self.Node(self.calc_xy_index(sx), self.calc_xy_index(sy), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx), self.calc_xy_index(gy), 0.0, -1)

        open_set: Dict[int, AStarPlanner.Node] = dict()
        closed_set: Dict[int, AStarPlanner.Node] = dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

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
        if self.obstacle_map[node.y, node.x]:
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

    map = Map(god=True)

    planner = AStarPlanner(0.8, map)
    Path = planner.planning(2, 2, 8, 12)

    viewer = MaskViewer(map)
    viewer.update()
    viewer.plot_path(Path)
    viewer.show()


if __name__ == "__main__":
    main()
