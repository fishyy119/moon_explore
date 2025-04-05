import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerPatch
from matplotlib.lines import Line2D
from pathlib import Path
from moon_explore.Pose2D import Pose2D
from moon_explore.Map import Map
from copy import deepcopy


from typing import Tuple, List, Callable, NamedTuple
from numpy.typing import NDArray


def plot_mask():
    # 创建一个与mask相同大小的矩阵，并根据条件设置值
    map_matrix = np.full_like(map.mask, -1, dtype=int)  # 默认全部设为未知区域 (-1)

    # 已知区域 (0)
    map_matrix[map.mask] = 0  # 将mask外的区域设为0（已知区域）

    # 障碍物区域 (1)
    map_matrix[map.obstacle_mask & map.mask] = 1  # 将障碍物区域设为1

    # 自定义颜色映射，-1为灰色，0为白色，1为黑色
    cmap = mcolors.ListedColormap(["gray", "white", "black"])  # 只定义三种颜色
    bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 使用imshow绘制
    ax.imshow(map_matrix, cmap=cmap, norm=norm, origin="lower")


def plot_contours():
    ax.clear()
    for contour in map.contours:
        curvature = map.curvature_discrete(contour)
        peaks_idx = map.detect_peaks(curvature, contour)

        ax.scatter(contour[:, 0], contour[:, 1], color="orange", s=5, label="Contours")
        ax.scatter(contour[peaks_idx, 0], contour[peaks_idx, 1], color="blue", marker="*", s=50, label="Peaks")


def plot_pose2d(pose: Pose2D):
    # 计算箭头的四个点
    yaw = pose.yaw_rad
    x = pose.x * Map.MAP_SCALE
    y = pose.y * Map.MAP_SCALE
    arrow_length = 4
    arrow_width = 4
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
    arrow_patch = Polygon(arrow_points, closed=True, facecolor="red", edgecolor="black", label="Pose2D", zorder=5)
    ax.add_patch(arrow_patch)
    return arrow_patch


NPY_ROOT = Path(__file__).parent.parent / "resource"
map = Map(map_file=str(NPY_ROOT / "map_passable.npy"))
map.rover_init(Pose2D(26, 29, 3))
map.get_contours()

fig, ax = plt.subplots()

arrow = None
plot_contours()
plot_mask()
points = map.cal_canPose()
for point in points:
    plot_pose2d(point)

# 其他图例元素
legend_elements = [
    Line2D([0], [0], marker="o", color="orange", linestyle="None", markersize=6, label="轮廓点"),
    Line2D([0], [0], marker="*", color="blue", linestyle="None", markersize=10, label="曲率峰值"),
    Polygon([[0, 0]], facecolor="red", edgecolor="black", label="候选目标"),
]


plt.rcParams["font.sans-serif"] = ["SimSun"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 10})  # 设置字体大小

# 添加图例
legend = ax.legend(handles=legend_elements, loc="upper right", frameon=True)
legend.get_frame().set_facecolor("white")
legend.get_frame().set_alpha(1.0)
legend.get_frame().set_edgecolor("black")


ax.set_xlim(185, 345)
ax.set_ylim(210, 370)

plt.tight_layout()
plt.show()
