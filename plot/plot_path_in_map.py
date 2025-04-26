import matplotlib.pyplot as plt
from plot_utils import *

from typing import List


if __name__ == "__main__":
    # 速度指令倍数 / beta / alpha求取使用的比例值
    csvs: List[RecordCSV] = [
        # RecordCSV("20250422_111632/record_1.csv", "1.5/b=0/a=0.8"),
        # RecordCSV("20250422_091531/record_1.csv", "1.5/b=0.4/a=0.8"),
        RecordCSV("20250424_162834/record_1.csv", "rover1"),
        RecordCSV("20250424_162834/record_2.csv", "rover2"),
    ]

    fig, ax = plt.subplots()
    plot_ob_mask(MAP_PASSABLE, ax, 1)
    for csv in csvs:
        plot_path_map(csv, ax)

    ax.legend()
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)

    plt.tight_layout()
    plt.show()
