import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.text import Annotation, Text


try:
    from .Pose2D import Pose2D
    from .Map import Map
    from .AStar import RoverPath
except:
    from Pose2D import Pose2D
    from Map import Map
    from AStar import RoverPath

from typing import Tuple, List, Callable
from enum import Enum, auto
from numpy.typing import NDArray


class MaskViewer:
    class UpdateMode(Enum):
        MOVE = auto()
        CONTOUR = auto()

    def __init__(self, map_instance: Map):
        plt.ion()
        self.map = map_instance
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)

    def plot_mask(self, god=False):
        # 创建一个与mask相同大小的矩阵，并根据条件设置值
        map_matrix = np.full_like(self.map.mask, -1, dtype=int)  # 默认全部设为未知区域 (-1)

        # 已知区域 (0)
        map_matrix[self.map.mask] = 0  # 将mask外的区域设为0（已知区域）

        # 障碍物区域 (1)
        if god:
            map_matrix[self.map.obstacle_mask] = 1  # 显示全部障碍
        else:
            map_matrix[self.map.obstacle_mask & self.map.mask] = 1  # 将障碍物区域设为1

        # 设置颜色映射
        if not hasattr(self, "_mask_cmap"):
            # 自定义颜色映射，-1为灰色，0为白色，1为黑色
            self._mask_cmap = mcolors.ListedColormap(["gray", "none", "black"])
            self._mask_bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
            self._mask_norm = mcolors.BoundaryNorm(self._mask_bounds, self._mask_cmap.N)

        # 只创建一次 imshow，后续直接更新数据
        if not hasattr(self, "_mask_image"):
            self._mask_image = self.ax.imshow(map_matrix, cmap=self._mask_cmap, norm=self._mask_norm, origin="lower")
        else:
            self._mask_image.set_data(map_matrix)

    def plot_contours(self, plot_curvature=False, show_peak=True):
        # 清除旧的 scatter 对象
        if hasattr(self, "_contour_scatters"):
            for artist in self._contour_scatters + self._peak_scatters:
                artist.remove()
        self._contour_scatters = []
        self._peak_scatters = []

        for contour, curvature, peaks_idx in self.map.contours:
            scatter = self.ax.scatter(contour[:, 0], contour[:, 1], color="orange", s=5)
            self._contour_scatters.append(scatter)

            if show_peak:
                peak_scatter = self.ax.scatter(
                    contour[peaks_idx, 0], contour[peaks_idx, 1], color="blue", marker="*", s=50
                )
                self._peak_scatters.append(peak_scatter)
            if plot_curvature:
                self.plot_curvature(curvature)

    def plot_path(self, path: RoverPath, **kwargs):
        if not hasattr(self, "_path_line"):
            kwargs.setdefault("color", "green")
            kwargs.setdefault("linewidth", 2)
            kwargs.setdefault("label", "Path")
            self._path_line = self.ax.plot(
                path.path_float[:, 0] * Map.MAP_SCALE,
                path.path_float[:, 1] * Map.MAP_SCALE,
                color="green",
                linewidth=2,
                label="Path",
            )[0]
        else:
            self._path_line.set_data(path.path_float[:, 0] * Map.MAP_SCALE, path.path_float[:, 1] * Map.MAP_SCALE)
        # .ax.scatter(path[:, 0], path[:, 1], color="blue", s=10, label="Path Points")

    @staticmethod
    def plot_curvature(curvature):
        plt.figure(figsize=(8, 5))
        plt.plot(curvature, color="red", linewidth=2)
        plt.title("Curvature vs Point Position on Contour")
        plt.xlabel("Point Index Along Contour")
        plt.ylabel("Curvature")
        plt.grid(True)

    def plot_pose2d(self, pose: Pose2D, text="", color: str = "red", scale: int = 20) -> Tuple[Annotation, Text | None]:
        x = pose.x * Map.MAP_SCALE
        y = pose.y * Map.MAP_SCALE
        yaw = pose.yaw_rad

        # 为了让线段变得不显眼
        dx = np.cos(yaw) * 0.01
        dy = np.sin(yaw) * 0.01
        # 绘制箭头
        arrow = self.ax.annotate(
            text="",
            xy=(x + dx, y + dy),  # 箭头指向方向
            xytext=(x, y),  # 箭头起点
            arrowprops=dict(
                arrowstyle="fancy",
                color=color,
                mutation_scale=scale,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=5,
        )
        # 绘制分数文本（可选）
        text_obj = None
        if text:
            text_obj = self.ax.text(
                x + dx * 100,
                y + dy * 100,
                text,
                color="black",
                fontsize=8,
                ha="center",
                va="bottom",
                zorder=6,
            )
        return arrow, text_obj

    def _old_plot_pose2d(self, pose: Pose2D, color: str = "red"):
        """旧版的绘制方法，问题在于不会自动缩放，不用了但是不想扔"""
        # 这个手动创建的多边形，不会自动缩放
        x = pose.x * Map.MAP_SCALE
        y = pose.y * Map.MAP_SCALE
        yaw = pose.yaw_rad
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
        arrow_patch = Polygon(arrow_points, closed=True, facecolor=color, edgecolor="black", zorder=5)
        self.ax.add_patch(arrow_patch)

    def update(self, mode: UpdateMode = UpdateMode.MOVE):
        if mode == MaskViewer.UpdateMode.MOVE:
            self.plot_mask()

            # 当前巡视器的箭头(多巡视器扩展不完全)
            if hasattr(self, "_pose2d_arrows_rover"):
                for arrow, text_obj in self._pose2d_arrows_rover:
                    arrow.remove()  # 移除之前绘制的箭头
                    if text_obj:
                        text_obj.remove()
            self._pose2d_arrows_rover = []  # 清空记录
            arrow = self.plot_pose2d(self.map.rover_pose, color="magenta", scale=20)
            self._pose2d_arrows_rover.append(arrow)  # 记录当前绘制的箭头
        elif mode == MaskViewer.UpdateMode.CONTOUR:
            self.plot_mask()
            self.plot_contours(plot_curvature=False, show_peak=False)

            # 候选点的箭头
            if hasattr(self, "_pose2d_arrows_canPose"):
                for arrow, text_obj in self._pose2d_arrows_canPose:
                    arrow.remove()  # 移除之前绘制的箭头
                    if text_obj:
                        text_obj.remove()
            self._pose2d_arrows_canPose = []  # 清空记录
            for point in self.map.canPoints:
                arrow = self.plot_pose2d(point.pose, color="red", scale=20, text=f"{point.path_cost:.2f}")
                self._pose2d_arrows_canPose.append(arrow)  # 记录当前绘制的箭头

        self.show()

    def show(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
