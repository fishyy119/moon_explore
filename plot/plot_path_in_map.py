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

CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img")


@dataclass
class CSVLine:
    file: Path | str
    label: str | None = None

    def __post_init__(self):
        self.file = CSV_ROOT / self.file
        if self.label is None:
            self.label = self.file.parent.stem


def csv_list(*args: Tuple[str] | Tuple[str, str]) -> List[CSVLine]:
    lines = []
    for item in args:
        if len(item) == 1:
            lines.append(CSVLine(file=item[0]))
        else:
            file, label = item
            lines.append(CSVLine(file=file, label=label))
    return lines


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


def plot_csv_files(csvs: List[CSVLine]):
    # plt.figure(figsize=(10, 6))

    for csv_line in csvs:
        file_path, label = csv_line.file, csv_line.label
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                print(f"跳过 {file_path}，列数不足两列。")
                continue

            x = df["position_x"].astype(float).round(2) * 10
            y = df["position_y"].astype(float).round(2) * 10
            ax.plot(x, y, label=label)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错：{e}")

    ax.axis("equal")  # 保持比例一致
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 速度指令倍数 / beta / alpha求取使用的比例值
    csvs: List[CSVLine] = csv_list(
        ("20250422_111632/record_1.csv", "1.5/b=0/a=0.8"),
        ("20250422_091531/record_1.csv", "1.5/b=0.4/a=0.8"),
    )

    NPY_ROOT = Path(__file__).parent.parent / "resource"
    map_file = NPY_ROOT / "map_passable.npy"
    ob_mask = np.load(map_file)

    fig, ax = plt.subplots()
    plot_ob_mask(ob_mask, 1)
    plot_csv_files(csvs)
