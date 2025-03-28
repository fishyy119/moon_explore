import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
import matplotlib.animation as animation
from Pose2D import Pose2D
from Map import Map

from typing import Tuple, List, Callable
from numpy.typing import NDArray


class MaskViewer:
    def __init__(self, map_instance: Map):
        self.map = map_instance
        self.fig, self.ax = plt.subplots()

    def plot_mask(self):
        # 创建一个与mask相同大小的矩阵，并根据条件设置值
        map_matrix = np.full_like(self.map.mask, -1, dtype=int)  # 默认全部设为未知区域 (-1)

        # 已知区域 (0)
        map_matrix[self.map.mask] = 0  # 将mask外的区域设为0（已知区域）

        # 障碍物区域 (1)
        map_matrix[self.map.obstacle_mask & self.map.mask] = 1  # 将障碍物区域设为1

        # 自定义颜色映射，-1为灰色，0为白色，1为黑色
        cmap = mcolors.ListedColormap(["lightgray", "white", "red"])  # 只定义三种颜色
        bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        # 使用imshow绘制
        self.ax.imshow(map_matrix, cmap=cmap, norm=norm, origin="lower")

    def plot_contours(self, plot_curvature=False):
        for contour in self.map.contours:
            curvature = self.map.curvature_discrete(contour)
            peaks_idx = self.map.detect_peaks(curvature, contour)

            self.ax.scatter(contour[:, 0], contour[:, 1], color="red", s=5, label="Contours")
            self.ax.scatter(contour[0, 0], contour[0, 1], color="yellow", marker="*", s=150, label="Start Point")
            self.ax.scatter(contour[peaks_idx, 0], contour[peaks_idx, 1], color="blue", marker="*", s=50, label="Peaks")

            if plot_curvature:
                self.plot_curvature(curvature)

    def plot_curvature(self, curvature):
        plt.figure(figsize=(8, 5))
        plt.plot(curvature, color="red", linewidth=2)
        plt.title("Curvature vs Point Position on Contour")
        plt.xlabel("Point Index Along Contour")
        plt.ylabel("Curvature")
        plt.grid(True)

    def plot_pose2d(self, pose: Pose2D) -> None:
        # 计算箭头的四个点
        yaw = pose.yaw_rad
        x = pose.x * Map.MAP_SCALE
        y = pose.y * Map.MAP_SCALE
        arrow_length = 10
        arrow_width = 10
        dx = np.cos(yaw) * arrow_length
        dy = np.sin(yaw) * arrow_length

        # 创建箭头的四个顶点
        arrow_points = np.array(
            [
                [x, y],  # 箭头的尾部
                [
                    x - dx / 2 - arrow_width * np.sin(yaw),
                    y - dy / 2 + arrow_width * np.cos(yaw),
                ],  # 箭头左边
                [x + dx, y + dy],  # 箭头顶点
                [
                    x - dx / 2 + arrow_width * np.sin(yaw),
                    y - dy / 2 - arrow_width * np.cos(yaw),
                ],  # 箭头右边
            ]
        )

        # 使用 Polygon 绘制箭头
        arrow_patch = Polygon(
            arrow_points, closed=True, facecolor="orange", edgecolor="orange", label="Pose2D", zorder=5
        )
        self.ax.add_patch(arrow_patch)

    def update(self):
        # self.ax.clear()
        self.plot_mask()
        # self.plot_obstacles()
        self.plot_contours(plot_curvature=False)
        self.ax.set_title("Sector Mask Viewer")
        self.fig.canvas.draw()

    def show(self):
        self.update()
        plt.show()

    def show_anime(self) -> None:
        self.fig, self.ax = plt.subplots()

        # 初始化掩码显示
        self.im = self.ax.imshow(self.map.mask, cmap="gray_r", origin="lower", animated=True)
        self.ax.set_title("Sector Mask Viewer")
        self.fig.colorbar(self.im, ax=self.ax)

        # 设置 30Hz 更新频率
        self.ani = animation.FuncAnimation(self.fig, self.update_a, interval=33.3, blit=True, cache_frame_data=False)
        plt.show()

    def update_a(self, _):
        """每帧更新 `map.mask` 可视化"""
        self.im.set_data(self.map.mask)  # 直接读取 map.mask
        return (self.im,)  # `blit=True` 需要返回可更新的对象
