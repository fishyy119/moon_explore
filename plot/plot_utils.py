from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from typing import List
from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

from typing import List, Tuple
from numpy.typing import NDArray
from dataclasses import dataclass


@dataclass
class RATE_CSV:
    file: Path | str
    label: str | None = None
    root: Path = Path("/home/yyy/moon_R2023/Data/map")

    def __post_init__(self):
        self.file = self.root / self.file
        if self.label is None:
            self.label = self.file.stem


@dataclass
class RECORD_CSV:
    file: Path | str
    label: str | None = None
    root: Path = Path("/home/yyy/moon_R2023/Data/img")
    x: NDArray[np.float64] = np.zeros((1, 1))
    y: NDArray[np.float64] = np.zeros((1, 1))

    def __post_init__(self):
        self.file = self.root / self.file
        df = pd.read_csv(self.file)
        self.x = np.array(df["position_x"].astype(float).round(2) * 10)
        self.y = np.array(df["position_y"].astype(float).round(2) * 10)
        if self.label is None:
            self.label = self.file.parent.stem


@dataclass
class RECORD_TXT:
    file: Path | str
    label: str | None = None
    root: Path = Path("/home/yyy/moon_R2023/Data/img")
    x: NDArray[np.float64] = np.zeros((1, 1))
    y: NDArray[np.float64] = np.zeros((1, 1))

    def __post_init__(self):
        self.file = self.root / self.file
        data = np.loadtxt(self.file)  # time, x, y, z, qx, qy, qz, qw
        self.x = data[:, 1] * 10
        self.y = data[:, 2] * 10
        if self.label is None:
            self.label = self.file.parent.stem


def plot_ob_mask(mask: NDArray[np.bool_], ax: Axes, alpha: float = 1):
    map_matrix = np.full_like(mask, 0, dtype=int)  # 默认全部设为已知知区域 (0)
    map_matrix[mask] = 1  # 障碍物区域 1

    # 自定义颜色映射，-1为灰色，0为白色，1为黑色
    cmap = mcolors.ListedColormap(["gray", "none", "black"])  # 只定义三种颜色
    bounds = [-1.5, -0.5, 0.5, 1.5]  # 设置每个值的边界
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # 使用imshow绘制
    ax.imshow(map_matrix, cmap=cmap, norm=norm, alpha=alpha, origin="lower")
