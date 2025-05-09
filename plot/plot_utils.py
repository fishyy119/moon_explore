import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import platform
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
from matplotlib.axes import Axes
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from scipy.spatial.transform import Rotation as R

from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable, Union
from numpy.typing import NDArray

ColorType = Union[Tuple[float, float, float, float], Tuple[float, float, float], str]

import sys

sys.path.append(str(Path(__file__).parent.parent))
from moon_explore.Utils import *


NPY_ROOT = Path(__file__).parent.parent / "resource"
MAP_PASSABLE = np.load(NPY_ROOT / "map_passable.npy")
MAP_EXPLORABLE = np.load(NPY_ROOT / "map_explorable.npy")
MAP_SLOPE = np.load(NPY_ROOT / "map_slope.npy").T
MAP_EDF: NDArray[np.float64] = distance_transform_edt(~MAP_PASSABLE) / 10  # type: ignore


def _get_default_root(cls: str) -> Path:
    system = platform.system()
    m: Callable[[str], bool] = lambda s: cls.lower() == s.lower()
    if system == "Windows":
        if m("RateCSV"):
            return Path("E:/y19/_/moon_R2023/Data/map")
        elif m("RecordCSV"):
            return Path("E:/y19/_/moon_R2023/Data/img")
        else:
            raise NotImplementedError(f"Unknown cls: {cls}")
    elif system == "Linux":
        if m("RateCSV"):
            return Path("/home/yyy/moon_R2023/Data/map")
        elif m("RecordCSV"):
            return Path("/home/yyy/moon_R2023/Data/img")
        else:
            raise NotImplementedError(f"Unknown cls: {cls}")
    else:
        raise NotImplementedError(f"Unknown platform: {system}")


class RateCSV:
    def __init__(
        self,
        file: Path | str,
        label: str | None = None,
        threshold: float = 49.1,
        root: Path = _get_default_root("RateCSV"),
        factor: float = 1,  # 忘了修改速度指令的倍数，在这里修正
    ):
        self.file = root / file
        self.label = label or self.file.stem
        self._data = pd.read_csv(self.file).astype(float)
        self.rate = self._data["view_grids"] / 501 / 501 * 100 / threshold * 100
        self.time = self._data["time"].round(1) / factor

        self.valid_indices = self.rate <= 100

    @property
    def x(self) -> Series:
        return self.time[self.valid_indices]

    @property
    def y(self) -> Series:
        return self.rate[self.valid_indices]


class RecordBase(ABC):
    def __init__(self, file: Path | str, label: str | None = None, root: Path = _get_default_root("RecordCSV")):
        self.file = root / file
        self.label = label or self.file.stem
        self._data = self._load_data()
        self.se3 = self._prepare_se3()

    @abstractmethod
    def _load_data(self) -> DataFrame:
        """加载原始数据"""
        ...

    @abstractmethod
    def _prepare_se3(self) -> DataFrame:
        """构造 se3 表(绝对坐标系下)"""
        ...

    def _compute_se3_matrix(self, row: Series) -> NDArray:
        se3_matrix = np.eye(4)
        se3_matrix[:3, :3] = row["SO3"]
        se3_matrix[:3, 3] = row["t"]
        return se3_matrix

    def _compute_so3_matrix(self, row: Series) -> NDArray:
        return R.from_quat([row["qx"], row["qy"], row["qz"], row["qw"]]).as_matrix()

    def _compute_translation_vector(self, row: Series) -> NDArray:
        return np.array([row["px"], row["py"], row["pz"]])

    def _extract_se3_components(self, row: pd.Series) -> Dict:
        T = row["SE3"]
        R_mat = T[:3, :3]
        t_vec = T[:3, 3]

        quat = R.from_matrix(R_mat).as_quat(canonical=False)  # 注意顺序是 [x, y, z, w]
        return {
            "px": t_vec[0],
            "py": t_vec[1],
            "pz": t_vec[2],
            "qx": quat[0],
            "qy": quat[1],
            "qz": quat[2],
            "qw": quat[3],
            "SO3": R_mat,
            "t": t_vec,
        }

    @property
    def time(self) -> Series:
        return self._data["time"]

    @property
    def x(self) -> Series:
        return self.se3["px"]

    @property
    def x_map(self) -> Series:
        return self.x * 10

    @property
    def y(self) -> Series:
        return self.se3["py"]

    @property
    def y_map(self) -> Series:
        return self.y * 10


class RecordCSV(RecordBase):
    def _load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file).astype(float)

    def _prepare_se3(self) -> DataFrame:
        se3_data = {
            "px": self._data["position_x"],
            "py": self._data["position_y"],
            "pz": self._data["position_z"],
            "qx": self._data["orientation_x"],
            "qy": self._data["orientation_y"],
            "qz": self._data["orientation_z"],
            "qw": self._data["orientation_w"],
        }

        se3_df = pd.DataFrame(se3_data)
        # SE3必须最后算，用到了前面的结果
        se3_df["t"] = se3_df.apply(self._compute_translation_vector, axis=1)
        se3_df["SO3"] = se3_df.apply(self._compute_so3_matrix, axis=1)
        se3_df["SE3"] = se3_df.apply(self._compute_se3_matrix, axis=1)
        return se3_df


class RecordSLAM(RecordBase):
    def __init__(
        self,
        recordAb: RecordCSV,
        file: Path | str,
        label: str | None = None,
        root: Path = _get_default_root("RecordCSV"),
    ):
        super().__init__(file, label, root)
        self.se3_relative = self.se3.copy()  # 基类在初始化时就生成了它，但是这是相对坐标系的
        self.recordAb = recordAb  # 绝对坐标系的记录
        self.align_to_absolute()  # 这里面se3被修改为绝对坐标系
        self.cal_xyz_diff()  # 计算绝对坐标系下的差值，存为self.xyz_diff

    def _load_data(self) -> DataFrame:
        return pd.read_csv(
            self.file, sep=" ", header=None, names=["time", "px", "py", "pz", "qx", "qy", "qz", "qw"]
        ).astype(float)

    def _prepare_se3(self) -> DataFrame:
        se3_data = {
            "px": self._data["px"],
            "py": self._data["py"],
            "pz": self._data["pz"],
            "qx": self._data["qx"],
            "qy": self._data["qy"],
            "qz": self._data["qz"],
            "qw": self._data["qw"],
        }

        se3_df = pd.DataFrame(se3_data)
        # SE3必须最后算，用到了前面的结果
        se3_df["t"] = se3_df.apply(self._compute_translation_vector, axis=1)
        se3_df["SO3"] = se3_df.apply(self._compute_so3_matrix, axis=1)
        se3_df["SE3"] = se3_df.apply(self._compute_se3_matrix, axis=1)
        return se3_df

    def cal_xyz_diff(self) -> None:
        slam_time: Series[float] = self.time
        true_time = self.recordAb.time

        diff_records = []

        for idx, t in enumerate(slam_time):  # 遍历 SLAM 的时间戳，使用索引 idx
            # 找到与当前 SLAM 时间戳最近的真值记录
            nearest_idx = int((true_time - t).abs().idxmin())

            # 获取 SLAM 估计的坐标（通过索引直接访问）
            slam_row = self.se3.iloc[idx]  # 使用 idx 获取对应的 SLAM 数据
            slam_px, slam_py, slam_pz = slam_row["px"], slam_row["py"], slam_row["pz"]

            # 获取真值坐标
            true_row = self.recordAb.se3.iloc[nearest_idx]  # 获取真值记录
            true_px, true_py, true_pz = true_row["px"], true_row["py"], true_row["pz"]

            # 计算差值
            dx = slam_px - true_px
            dy = slam_py - true_py
            dz = slam_pz - true_pz

            # 将时间和差值存入列表
            diff_records.append([t, dx, dy, dz])

        # 将差值记录转为 DataFrame
        diff_df = pd.DataFrame(diff_records, columns=["time", "dx", "dy", "dz"])
        self.xyz_diff = diff_df

    def align_to_absolute(self) -> None:
        # 选择一个基准帧：第一个时间戳
        ref_time: float = self._data.iloc[0]["time"]
        record_time = self.recordAb.time
        idx = int((record_time - ref_time).abs().idxmin())
        base_record = self.recordAb.se3.iloc[idx]

        # 相对坐标系的变换关系
        T_aw = base_record["SE3"]
        se3_df = pd.DataFrame()
        se3_df["SE3"] = self.se3_relative.apply(lambda row: T_aw @ row["SE3"], axis=1)
        components = se3_df.apply(self._extract_se3_components, axis=1, result_type="expand")
        se3_df = pd.concat([se3_df, components], axis=1)
        self.se3 = se3_df

    @property
    def time_diff(self) -> Series:
        return self.xyz_diff["time"]

    @property
    def x_diff(self) -> Series:
        return self.xyz_diff["dx"]

    @property
    def y_diff(self) -> Series:
        return self.xyz_diff["dy"]

    @property
    def z_diff(self) -> Series:
        return self.xyz_diff["dz"]


def plot_xyz_diff(csv: RecordSLAM, axes: List[Axes]):
    axes[0].plot(csv.time_diff, abs(csv.x_diff), label=f"{csv.label} x", linewidth=2)
    axes[0].set_ylabel("dx")
    axes[0].legend()

    axes[1].plot(csv.time_diff, abs(csv.y_diff), label=f"{csv.label} y", linewidth=2)
    axes[1].set_ylabel("dy")
    axes[1].legend()

    # axes[2].plot(csv.time_diff, abs(csv.z_diff), label=f"{csv.label} z", linewidth=2)
    # axes[2].set_ylabel("dz")
    # axes[2].legend()

    for ax in axes:
        ax.grid(True)


def plot_rate_csv(csv: RateCSV, ax: Axes):
    plt.plot(csv.x, csv.y, label=csv.label, linewidth=2)


def ax_remove_axis(ax: Axes) -> None:
    # 对于绘制地图，去除坐标轴，添加黑色边框
    ax.axis("off")
    h, w = 501, 501
    rect = patches.Rectangle((0, 0), w, h, linewidth=2, edgecolor="black", facecolor="none", transform=ax.transData)
    ax.add_patch(rect)


def ax_add_legend(ax: Axes, legend_handles=None) -> None:
    # 自动设置图例样式
    # legend = ax.legend(handles=legend_handles, loc="upper right", title="")
    legend = ax.legend(handles=legend_handles)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor("black")


def axes_add_abc(axes: List[Axes]) -> None:
    # 添加图注 (a), (b)
    for i, ax in enumerate(axes):
        ax.text(0.5, -0.05, f"({chr(97 + i)})", transform=ax.transAxes, fontsize=12, ha="center", va="top")


def plt_tight_show() -> None:
    plt.tight_layout()
    plt.show()


def plt_flat_axes(axes: List[List[Axes]]) -> List[Axes]:
    # 将二维数组展平为一维列表
    flat_axes = [ax for axs in axes for ax in axs]
    return flat_axes


def plot_path_map(csv: RecordBase, ax: Axes):
    ax.plot(csv.x_map, csv.y_map, label=csv.label, linewidth=2)

    # 添加图例
    legend = ax.legend(loc="best")

    # 设置 legend 背景为不透明白色
    legend.get_frame().set_facecolor("white")  # 设置为白底
    legend.get_frame().set_alpha(1.0)  # 设置完全不透明
    legend.get_frame().set_edgecolor("black")  # 可选：边框变为黑色


def plot_contours_map(contours: List[Contour], ax: Axes, show_peak=True):
    for contour, curvature, peaks_idx in contours:
        ax.scatter(contour[:, 0], contour[:, 1], color="#de8f05", s=5)
        if show_peak:
            ax.scatter(contour[peaks_idx, 0], contour[peaks_idx, 1], color="blue", marker="*", s=50)


def plot_pose2d_map(
    pose: Pose2D,
    ax: Axes,
    color: ColorType,
    scale: int = 20,
) -> None:
    x = pose.x * Setting.MAP_SCALE
    y = pose.y * Setting.MAP_SCALE
    yaw = pose.yaw_rad

    # 为了让线段变得不显眼
    dx = np.cos(yaw) * 0.01
    dy = np.sin(yaw) * 0.01
    # 绘制箭头
    ax.annotate(
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


def plot_canPoints_map(canPoints: List[CandidatePoint], ax: Axes):
    # 将分数映射为颜色
    cmap = cm.get_cmap("viridis")
    scores = [p.score for p in canPoints]
    min_score, max_score = min(scores), max(scores)
    norm = Normalize(vmin=min_score, vmax=max_score)
    for point in canPoints:
        rgba_color = cmap(norm(point.score))
        plot_pose2d_map(point.pose, ax=ax, color=rgba_color, scale=15)

    # 创建伪图像对象以生成 colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = ax.figure.colorbar(sm, cax=cax)  # type: ignore
    cb.set_label("Score", labelpad=15)


def plot_binary_map(map: NDArray[np.bool_], ax: Axes, visible_map: NDArray[np.bool_] | None = None, alpha: float = 1):
    map_matrix = np.full_like(map, 0, dtype=int)  # 默认全部设为已知知区域 (0)
    map_matrix[map] = 1  # 障碍物区域 1
    if visible_map is not None:
        map_matrix[~visible_map] = -1

    # 自定义颜色映射，-1为灰色，0为白色，1为黑色
    cmap = mcolors.ListedColormap(["grey", "none", "black"])  # 只定义三种颜色
    bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 使用imshow绘制
    ax.imshow(map_matrix, cmap=cmap, norm=norm, alpha=alpha, origin="lower")
    ax_remove_axis(ax)


def plot_edf_map(map: NDArray[np.float64], ax: Axes) -> None:
    im = ax.imshow(map, interpolation="nearest", origin="lower")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax, orientation="vertical", label="Distance (m)")  # type: ignore
    ax_remove_axis(ax)


def plot_slope_map(slope: NDArray, ax: Axes, passable_threshold: List[float] = [5, 15, 20]) -> None:
    """
    绘制可通行性地图

    Args:
        slope (NDArray): 坡度地图，分四档
        passable_threshold (List[float], optional): 四档的分割点，与计算所用到的参数一致
    """
    slope_deg = np.degrees(np.arctan(slope))
    passability = np.ones_like(slope_deg).astype(np.int8)
    passability = np.digitize(slope_deg, passable_threshold, right=True).astype(np.int8)

    color_list = ["darkgreen", "lightgreen", "orange", "red"]
    cmap = plt.cm.colors.ListedColormap(color_list)  # type: ignore
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)  # type: ignore

    ax.imshow(passability, cmap=cmap, norm=norm, origin="lower")

    # 创建图例
    legend_labels = [
        f"0-{passable_threshold[0]}°",
        f"{passable_threshold[0]}-{passable_threshold[1]}°",
        f"{passable_threshold[1]}-{passable_threshold[2]}°",
        f">{passable_threshold[2]}°",
    ]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(color_list, legend_labels)]

    ax_add_legend(ax, legend_handles=legend_patches)
    ax_remove_axis(ax)
