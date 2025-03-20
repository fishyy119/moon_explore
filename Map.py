from math import ceil
import numpy as np
import matplotlib.pyplot as plt
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
        self.standard_sector_mask = self._precompute_standard_sector_mask()

    def _precompute_standard_sector_mask(self) -> NDArray[np.bool_]:
        """预生成一个偏航角为0时的扇形，在局部栅格中"""
        shape = ceil(Map.MAX_RANGE * Map.MAP_SCALE * Map.SECTOR_SCALE * 2 + 1)
        center = shape // 2

        # 创建坐标网格
        x, y = np.meshgrid(np.arange(shape), np.arange(shape))
        dx = x - center
        dy = y - center

        # 计算极坐标
        r = np.sqrt(dx**2 + dy**2)
        theta = np.degrees(np.arctan2(dy, dx))

        # 生成扇形掩码：半径在范围内 & 角度在 FOV 以内
        mask = (
            (r <= Map.MAX_RANGE * Map.MAP_SCALE * Map.SECTOR_SCALE) & (-Map.FOV / 2 <= theta) & (theta <= Map.FOV / 2)
        )

        return mask

    def generate_sector_mask(self, pose: Pose2D) -> NDArray[np.bool_]:
        """对预生成的扇形遮罩应用SE2变换"""
        x = pose.x
        y = pose.y
        yaw = pose.yaw_rad
        scale = Map.MAP_SCALE * Map.SECTOR_SCALE

        # 局部坐标系内的点，单位缩放到米
        local_mask = self.standard_sector_mask
        h, w = local_mask.shape
        cx, cy = w // 2, h // 2
        x_idxs, y_idxs = np.where(local_mask)
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
        global_mask[i, j] = True

        return global_mask

    @staticmethod
    def view_mask(mask: NDArray[np.bool_]) -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(mask, cmap="gray_r", origin="lower")

        ax.set_title("Sector Mask Viewer")
        plt.colorbar(im, ax=ax)
        plt.show()


if __name__ == "__main__":
    map = Map()
    map.view_mask(map.generate_sector_mask(Pose2D(20, 25, 0, True)))
