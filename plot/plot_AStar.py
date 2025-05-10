from typing import List
from moon_explore.AStar import AStarPlanner
from moon_explore.Map import Map
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from numpy.typing import NDArray

from plot.plot_utils import *


def plot_ob_mask(mask: NDArray[np.bool_], alpha):
    # 创建一个与mask相同大小的矩阵，并根据条件设置值
    map_matrix = np.full_like(mask, 0, dtype=int)  # 默认全部设为已知知区域 (0)

    map_matrix[mask] = 1  # 障碍物区域 1

    # 自定义颜色映射，-1为灰色，0为白色，1为黑色
    cmap = mcolors.ListedColormap(["gray", "none", "black"])  # 只定义三种颜色
    bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 使用imshow绘制
    ax.imshow(map_matrix, cmap=cmap, norm=norm, alpha=alpha, origin="lower")


def plot_path(path: NDArray[np.int32], **kwargs):
    ax.plot(path[:, 0], path[:, 1], **kwargs)
    # .ax.scatter(path[:, 0], path[:, 1], color="blue", s=10, label="Path Points")


NPY_ROOT = Path(__file__).parent.parent / "resource"
map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), map_divide=str(NPY_ROOT / "map_divide.npy"), god=True)
# planner = AStarPlanner(0.8, map)
sx, sy, gx, gy = 2, 2, 12.5, 8

# 原始的规划轨迹
map.planner.obstacle_map = map.planner.euclidean_dilated_safe | ~map.mask
result = map.planner.plan_once(sx, sy, gx, gy)
if result is None:
    map.planner.obstacle_map = map.planner.euclidean_dilated_base | ~map.mask
    result = map.planner.plan_once(sx, sy, gx, gy)
    print("使用了次安全性地图")
assert result is not None

# 剪枝过程中的每个轨迹
length_last = len(result) + 1
cnt = 0
path = result
simplify_results: List[NDArray[np.int32]] = []
while length_last != len(path) and cnt <= 5:
    length_last = len(path)
    # simplify_results.append(planner.simplify_once(path, ref_ob=planner.euclidean_dilated_safe))
    simplify_results.append(map.planner.simplify_once(path))
    path = simplify_results[-1]
    cnt += 1


fig, ax = plt.subplots()

plot_ob_mask(map.planner.euclidean_dilated_safe, alpha=0.4)
plot_ob_mask(map.planner.euclidean_dilated_least, alpha=0.6)
plot_path(result, color="black", linestyle="-", linewidth=2, label="A*算法规划的路径")
# plot_path(simplify_results[1], color="green", linestyle="-", linewidth=2, label="A*算法规划的路径")
plot_path(simplify_results[0], color="black", linestyle="--", linewidth=2, label="三角剪枝后的路径")

# 创建图例元素
legend_elements = [
    mpatches.Patch(color="black", alpha=0.4, label=r"障碍物地图 ($\sigma=2$)"),
    mpatches.Patch(color="black", alpha=0.6, label=r"障碍物地图 ($\sigma=1$)"),
    plt.Line2D([0], [0], color="black", linestyle="-", linewidth=2, label="A*算法规划的路径"),  # type: ignore
    plt.Line2D([0], [0], color="black", linestyle="--", linewidth=2, label="三角剪枝后的路径"),  # type: ignore
]

ax.set_xlim(10, 130)
ax.set_ylim(10, 130)

ax_add_legend(ax, legend_elements)
ax_remove_axis(ax)
plt_tight_show()
