import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from plot_utils import *

from typing import List, Tuple
from numpy.typing import NDArray


def csv_list(*args: Tuple[str] | Tuple[str, str]) -> List[RECORD_CSV]:
    lines = []
    for item in args:
        if len(item) == 1:
            lines.append(RECORD_CSV(file=item[0]))
        else:
            file, label = item
            lines.append(RECORD_CSV(file=file, label=label))
    return lines


def plot_csv_files(csvs: List[RECORD_CSV]):
    for csv_line in csvs:
        file_path, label = csv_line.file, csv_line.label
        try:
            ax.plot(csv_line.x, csv_line.y, label=label)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错：{e}")

    ax.axis("equal")  # 保持比例一致
    ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 速度指令倍数 / beta / alpha求取使用的比例值
    csvs: List[RECORD_CSV] = csv_list(
        # ("20250422_111632/record_1.csv", "1.5/b=0/a=0.8"),
        # ("20250422_091531/record_1.csv", "1.5/b=0.4/a=0.8"),
        ("20250424_162834/record_1.csv", "rover1"),
        ("20250424_162834/record_2.csv", "rover2"),
    )

    NPY_ROOT = Path(__file__).parent.parent / "resource"
    map_file = NPY_ROOT / "map_passable.npy"
    ob_mask = np.load(map_file)

    fig, ax = plt.subplots()
    plot_ob_mask(ob_mask, ax, 1)
    plot_csv_files(csvs)
