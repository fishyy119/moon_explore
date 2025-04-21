# import seaborn as sns #! 这个需要高版本的numpy，搞不懂怎么让ROS用高版本的，所以把调色板直接硬编码字符串了
import numpy as np
import imageio.v2 as iio
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Polygon
from matplotlib.text import Annotation, Text
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

try:
    from .Map import Map
    from .AStar import RoverPath
    from .Utils import Setting, Pose2D
except:
    from Map import Map
    from AStar import RoverPath
    from Utils import Setting, Pose2D

from typing import Dict, Tuple, List, Union
from enum import Enum, auto
from numpy.typing import NDArray

ColorType = Union[Tuple[float, float, float, float], Tuple[float, float, float], str]


class MaskViewer:
    class UpdateMode(Enum):
        MOVE = auto()
        CONTOUR = auto()

    def __init__(self, map_instance: Map, output_path: str):
        plt.ion()

        self.writer = iio.get_writer(output_path, fps=10)
        self.buf = io.BytesIO()

        self.map = map_instance
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 500)
        self.ax.set_ylim(0, 500)

        # self.palette = sns.color_palette("colorblind")
        self.palette = [
            "#0173b2",  # blue
            "#de8f05",  # orange
            "#029e73",  # green
            "#d55e00",  # red
            "#cc78bc",  # purple
            "#ca9161",  # brown
            "#fbafe4",  # pink
            "#949494",  # gray
            "#ece133",  # yellow
            "#56b4e9",  # cyan
        ]

        # 颜色条等高（在右边加个子图放颜色条）
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="5%", pad=0.05)

        # 用于将评分映射为颜色
        # plasm / inferno / magma 和巡视器用的品红色有冲突
        # cividis 色彩不太丰富
        # 最大值亮度过高在白色背景下不明显：加一个浅灰色边框
        self.cmap = cm.get_cmap("viridis")
        # self.cmap = LinearSegmentedColormap.from_list(
        #     "viridis_trunc", self.cmap(np.linspace(0, 1, 256))
        # )  # 截断末尾的亮色

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
            scatter = self.ax.scatter(contour[:, 0], contour[:, 1], color=self.palette[1], s=5)
            self._contour_scatters.append(scatter)

            if show_peak:
                peak_scatter = self.ax.scatter(
                    contour[peaks_idx, 0], contour[peaks_idx, 1], color="blue", marker="*", s=50
                )
                self._peak_scatters.append(peak_scatter)
            if plot_curvature:
                self.plot_curvature(curvature)

    def plot_path(self, path: RoverPath, index: int = 1, **kwargs):
        if not hasattr(self, "_path_lines"):
            self._path_lines: Dict[int, Line2D] = {}

        if index not in self._path_lines:
            kwargs.setdefault("color", "green")
            kwargs.setdefault("linewidth", 2)  #! 这个接口被改失灵了
            kwargs.setdefault("label", f"Path {index}")
            self._path_lines[index] = self.ax.plot(
                path.path_float[:, 0] * Setting.MAP_SCALE,
                path.path_float[:, 1] * Setting.MAP_SCALE,
                color=self.palette[6],
                linewidth=2,
            )[0]
        else:
            self._path_lines[index].set_data(
                path.path_float[:, 0] * Setting.MAP_SCALE, path.path_float[:, 1] * Setting.MAP_SCALE
            )

    @staticmethod
    def plot_curvature(curvature):
        plt.figure(figsize=(8, 5))
        plt.plot(curvature, color="red", linewidth=2)
        plt.title("Curvature vs Point Position on Contour")
        plt.xlabel("Point Index Along Contour")
        plt.ylabel("Curvature")
        plt.grid(True)

    def plot_pose2d(
        self,
        pose: Pose2D,
        color: ColorType,
        text="",
        scale: int = 20,
    ) -> Tuple[Annotation, Text | None]:
        x = pose.x * Setting.MAP_SCALE
        y = pose.y * Setting.MAP_SCALE
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
                path_effects=[pe.withStroke(linewidth=2, foreground="gray")],  # 添加外框
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
        x = pose.x * Setting.MAP_SCALE
        y = pose.y * Setting.MAP_SCALE
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

    def plot_voronoi(self, num: int):
        if hasattr(self, "_voronoi_graph"):
            for g in self._voronoi_graph:
                g.remove()

        if num == 1:
            return
        elif num == 2:
            p = [r.rover_pose.xy * Setting.MAP_SCALE for r in self.map.rovers]
            p1 = p[0]
            p2 = p[1]
            mid = (p1 + p2) / 2

            # 垂直方向向量（单位化）
            dir_vec = p2 - p1
            perp_vec = np.array([-dir_vec[1], dir_vec[0]])
            perp_vec = perp_vec / np.linalg.norm(perp_vec)

            # 用于延长中垂线（你也可以设为任务区边界大小）
            length = 300
            line_start = mid - perp_vec * length
            line_end = mid + perp_vec * length

            self._voronoi_graph = self.ax.plot(
                [line_start[0], line_end[0]], [line_start[1], line_end[1]], color=self.palette[0]
            )
        elif num == 3:
            raise NotImplementedError
        else:
            raise NotImplementedError

    def update(self, mode: UpdateMode = UpdateMode.MOVE, show_score_text=False):
        if mode == MaskViewer.UpdateMode.MOVE:
            self.plot_mask()

            # 当前巡视器的箭头(多巡视器扩展不完全)
            if hasattr(self, "_pose2d_arrows_rover"):
                for arrow, text_obj in self._pose2d_arrows_rover:
                    arrow.remove()  # 移除之前绘制的箭头
                    if text_obj:
                        text_obj.remove()
            self._pose2d_arrows_rover: List[Tuple[Annotation, Text | None]] = []  # 清空记录

            for rover in self.map.rovers:
                arrow = self.plot_pose2d(rover.rover_pose, color=self.palette[4], scale=20)
                self._pose2d_arrows_rover.append(arrow)  # 记录当前绘制的箭头
        elif mode == MaskViewer.UpdateMode.CONTOUR:
            self.plot_mask()
            self.plot_contours(plot_curvature=False, show_peak=False)
            self.plot_voronoi(len(self.map.rovers))

            # 候选点的箭头
            if hasattr(self, "_pose2d_arrows_canPose"):
                for arrow, text_obj in self._pose2d_arrows_canPose:
                    arrow.remove()  # 移除之前绘制的箭头
                    if text_obj:
                        text_obj.remove()
            self._pose2d_arrows_canPose: List[Tuple[Annotation, Text | None]] = []  # 清空记录

            # 将分数映射为颜色s
            scores = [p.score for p in self.map.canPoints]
            min_score, max_score = min(scores), max(scores)
            self.norm = Normalize(vmin=min_score, vmax=max_score)
            for point in self.map.canPoints:
                rgba_color = self.cmap(self.norm(point.score))
                if show_score_text:
                    arrow = self.plot_pose2d(point.pose, color=rgba_color, scale=10, text=f"{point.score:.2f}")
                else:
                    arrow = self.plot_pose2d(point.pose, color=rgba_color, scale=10)
                self._pose2d_arrows_canPose.append(arrow)  # 记录当前绘制的箭头

            # 创建伪图像对象以生成 colorbar
            sm = cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
            sm.set_array([])
            # 原先的remove方法貌似连带着吧cax干掉了（linux上貌似没有被干）
            if hasattr(self, "_pose2d_colorbar"):
                self._pose2d_colorbar.update_normal(sm)  # 重新绑定新值
            else:
                self._pose2d_colorbar = self.fig.colorbar(sm, cax=self.cax)
                self._pose2d_colorbar.set_label("Score", rotation=270, labelpad=15)

        self.show()

    def show(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # 写入视频
        self.fig.savefig(self.buf, format="png")
        self.buf.seek(0)

        img = Image.open(self.buf).convert("RGB")
        frame = np.array(img)
        self.writer.append_data(frame)

        # 使用完后清空内容
        self.buf.seek(0)
        self.buf.truncate(0)


if __name__ == "__main__":
    from Utils import MyTimer
    from Viewer import MaskViewer
    from pathlib import Path
    import time
    import matplotlib.pyplot as plt

    NPY_ROOT = Path(__file__).parent.parent / "resource"
    map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), num_rovers=1)
    viewer = MaskViewer(map, "output.mp4")

    x, y, theta = 27, 25, 90  # 初始位置
    while True:
        pose = Pose2D(x, y, theta, deg=True)
        map.rover_move(pose)  # 更新 mask
        map.step()
        viewer.update()
        viewer.update(mode=MaskViewer.UpdateMode.CONTOUR)

        # 模拟运动轨迹
        y += 0.5
        if y > 30:
            break

        time.sleep(1 / 30)  # 30Hz 更新
    viewer.writer.close()
