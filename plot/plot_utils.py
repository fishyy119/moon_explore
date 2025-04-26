import matplotlib.pyplot as plt
import pandas as pd
import platform
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from pandas import DataFrame, Series
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Callable
from numpy.typing import NDArray


NPY_ROOT = Path(__file__).parent.parent / "resource"
MAP_PASSABLE = np.load(NPY_ROOT / "map_passable.npy")


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
        threshold: float = 50,
        root: Path = _get_default_root("RateCSV"),
    ):
        self.file = root / file
        self.label = label or self.file.stem
        self._data = pd.read_csv(self.file).astype(float)
        self.rate = self._data["view_grids"] / 501 / 501 * 100
        self.time = self._data["time"].round(1)

        self.valid_indices = self.rate <= threshold

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
    axes[0].plot(csv.time_diff, csv.x_diff, label=f"{csv.label} x", linewidth=2)
    axes[0].set_ylabel("dx")
    axes[0].legend()

    axes[1].plot(csv.time_diff, csv.y_diff, label=f"{csv.label} y", linewidth=2)
    axes[1].set_ylabel("dy")
    axes[1].legend()

    axes[2].plot(csv.time_diff, csv.z_diff, label=f"{csv.label} z", linewidth=2)
    axes[2].set_ylabel("dz")
    axes[2].legend()

    for ax in axes:
        ax.grid(True)


def plot_rate_csv(csv: RateCSV, ax: Axes):
    plt.plot(csv.x, csv.y, label=csv.label, linewidth=2)


def plot_path_map(csv: RecordBase, ax: Axes):
    plt.plot(csv.x_map, csv.y_map, label=csv.label, linewidth=2)


def plot_ob_mask(mask: NDArray[np.bool_], ax: Axes, alpha: float = 1):
    map_matrix = np.full_like(mask, 0, dtype=int)  # 默认全部设为已知知区域 (0)
    map_matrix[mask] = 1  # 障碍物区域 1

    # 自定义颜色映射，-1为灰色，0为白色，1为黑色
    cmap = mcolors.ListedColormap(["gray", "none", "black"])  # 只定义三种颜色
    bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 使用imshow绘制
    ax.imshow(map_matrix, cmap=cmap, norm=norm, alpha=alpha, origin="lower")
