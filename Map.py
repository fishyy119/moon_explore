from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Pose2D import Pose2D

from typing import Tuple, List, Callable
from numpy.typing import NDArray


class Map:
    MAP_SCALE = 10  # 浮点数坐标乘以这个数便对应到索引坐标
    SECTOR_SCALE = 3  # 扇形遮罩计算所用的栅格在地图基础上的进一步细分
    MAX_RANGE = 6.0  # 最远可视距离，米
    FOV = 90  # 视场角，度数

    def __init__(self) -> None:
        self.global_map: NDArray = ...
        self.visible_map: NDArray = ...
        self.mask: NDArray[np.bool_] = np.zeros((501, 501), dtype=np.bool_)
        self.obstacle_mask = np.zeros((501, 501), dtype=np.bool_)
        self.obstacle_mask[300:, 300:] = True

    def _precompute_standard_sector_mask(self, pose: Pose2D) -> NDArray[np.bool_]:
        """预生成一个偏航角为0时的扇形，在局部栅格中"""
        shape = ceil(Map.MAX_RANGE * Map.MAP_SCALE * Map.SECTOR_SCALE * 2 + 1)
        center = shape // 2

        # 创建坐标网格
        x, y = np.meshgrid(np.arange(shape), np.arange(shape))
        dx = x - center
        dy = y - center

        # 计算极坐标
        r = np.sqrt(dx**2 + dy**2)
        theta = np.round(np.degrees(np.arctan2(dy, dx))).astype(np.int32)

        # 考虑障碍物，每个角度上最大可视距离
        r_max = np.zeros_like(r)
        r_max[(-Map.FOV / 2 <= theta) & (theta <= Map.FOV / 2)] = Map.MAX_RANGE * Map.MAP_SCALE * Map.SECTOR_SCALE

        # 障碍物由全局坐标系转换到局部
        y_idxs, x_idxs = np.where(self.obstacle_mask)
        ob_points_global = np.stack([x_idxs / Map.MAP_SCALE, y_idxs / Map.MAP_SCALE, np.ones_like(x_idxs)])  # [3, N]
        # # 加密一下，不然会有空隙
        # ob_points_global = np.hstack(
        #     [
        #         ob_points_global,
        #         ob_points_global
        #         + np.array([[1 / Map.MAP_SCALE / Map.SECTOR_SCALE], [1 / Map.MAP_SCALE / Map.SECTOR_SCALE], [0]]),
        #         ob_points_global
        #         + np.array([[2 / Map.MAP_SCALE / Map.SECTOR_SCALE], [2 / Map.MAP_SCALE / Map.SECTOR_SCALE], [0]]),
        #         ob_points_global
        #         + np.array([[0.5 / Map.MAP_SCALE / Map.SECTOR_SCALE], [0.5 / Map.MAP_SCALE / Map.SECTOR_SCALE], [0]]),
        #         ob_points_global
        #         + np.array([[1.5 / Map.MAP_SCALE / Map.SECTOR_SCALE], [1.5 / Map.MAP_SCALE / Map.SECTOR_SCALE], [0]]),
        #         ob_points_global
        #         + np.array([[2.5 / Map.MAP_SCALE / Map.SECTOR_SCALE], [2.5 / Map.MAP_SCALE / Map.SECTOR_SCALE], [0]]),
        #     ]
        # )
        ob_points_local = (pose.SE2inv @ ob_points_global) * Map.MAP_SCALE * Map.SECTOR_SCALE
        x_f, y_f = ob_points_local[0, :], ob_points_local[1, :]

        i = np.round(x_f + center).astype(int)
        j = np.round(y_f + center).astype(int)
        i = np.clip(i, 0, shape - 1)
        j = np.clip(j, 0, shape - 1)
        ob_mask: NDArray[np.bool_] = np.zeros_like(r, dtype=np.bool_)
        ob_mask[j, i] = True

        ob_r = r[ob_mask]
        ob_arg = ob_r.argsort()
        ob_theta = theta[ob_mask][ob_arg]
        ob_r = ob_r[ob_arg]

        # 获取唯一的角度索引
        unique_ob_theta, unique_indices = np.unique(ob_theta, return_index=True)
        unique_ob_r_min = ob_r[unique_indices]  # 每个角度的最近障碍物距离(前面对r排序了)

        # 找到 r_max 需要更新的索引
        row_indices, col_indices = np.where(np.isin(theta, unique_ob_theta))
        theta_flat = theta[row_indices, col_indices]  # 扁平化后的角度数组

        # 获取这些角度的最小可视距离，虽然是插值，但是目标坐标都是给定点
        r_values = np.interp(theta_flat, unique_ob_theta, unique_ob_r_min)

        # 使用 `np.minimum.at` 批量更新 r_max
        np.minimum.at(r_max, (row_indices, col_indices), r_values)

        # 按照矩阵的样式，维度0对应y，维度1对应x
        mask = r <= r_max
        return mask

    def generate_sector_mask(self, pose: Pose2D) -> NDArray[np.bool_]:
        """对预生成的扇形遮罩应用SE2变换"""
        scale = Map.MAP_SCALE * Map.SECTOR_SCALE

        # 局部坐标系内的点，单位缩放到米
        local_mask = self._precompute_standard_sector_mask(pose)
        h, w = local_mask.shape
        cx, cy = w // 2, h // 2
        y_idxs, x_idxs = np.where(local_mask)
        local_points = np.stack([(x_idxs - cx) / scale, (y_idxs - cy) / scale, np.ones_like(x_idxs)])  # [3, N]

        # 变换到全局坐标系，四舍五入覆盖到全局栅格
        global_points = (pose.SE2 @ local_points) * Map.MAP_SCALE
        x_f, y_f = global_points[0, :], global_points[1, :]
        i = np.round(x_f).astype(int)
        j = np.round(y_f).astype(int)
        H, W = self.mask.shape
        i = np.clip(i, 0, H - 1)
        j = np.clip(j, 0, W - 1)
        global_mask = np.zeros((H, W), dtype=np.bool_)
        # 维度0对应y，维度1对应x
        global_mask[j, i] = True

        return global_mask

    def rover_move(self, pose: Pose2D) -> None:
        self.mask |= self.generate_sector_mask(pose)


class MaskViewer:
    def __init__(self, map_instance: Map):
        self.map = map_instance

    def show_mask(self) -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(self.map.mask, cmap="gray_r", origin="lower")

        ax.set_title("Sector Mask Viewer")
        plt.colorbar(im, ax=ax)
        plt.show()

    def show_anime(self) -> None:
        self.fig, self.ax = plt.subplots()

        # 初始化掩码显示
        self.im = self.ax.imshow(self.map.mask, cmap="gray_r", origin="lower", animated=True)
        self.ax.set_title("Sector Mask Viewer")
        self.fig.colorbar(self.im, ax=self.ax)

        # 设置 30Hz 更新频率
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=33.3, blit=True, cache_frame_data=False)
        plt.show()

    def update(self, _):
        """每帧更新 `map.mask` 可视化"""
        self.im.set_data(self.map.mask)  # 直接读取 map.mask
        return (self.im,)  # `blit=True` 需要返回可更新的对象


if __name__ == "__main__":
    map = Map()
    map.rover_move(Pose2D(40, 40, 0.3))
    map.rover_move(Pose2D(30, 30, 0.3))
    map.rover_move(Pose2D(20, 20, 0.3))
    viewer = MaskViewer(map)
    viewer.show_mask()
