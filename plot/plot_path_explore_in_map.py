import matplotlib.pyplot as plt
from plot_utils import *

from typing import List


if __name__ == "__main__":
    csvs: List[RecordCSV] = [
        # RecordCSV("20250509_031046/record_1.csv", r"$\beta=0$ - 1"),
        # RecordCSV("20250508_194215/record_1.csv", r"$\beta=0.3$ - 1"),
        # RecordCSV("20250507_112308/record_1.csv", r"$\beta=0$ - 2", t_factor=1.5),
        # RecordCSV("20250507_134314/record_1.csv", r"$\beta=0.1$ - 2", t_factor=1.5),
        RecordCSV("20250424_162834/record_1.csv", "rover1"),
        RecordCSV("20250424_162834/record_2.csv", "rover2"),
    ]

    fig, axes = plt.subplots(1, 2)
    axes: List[Axes]
    for i, ax in enumerate(axes):
        plot_binary_map(MAP_PASSABLE, ax)
        plot_path_time_map(csvs[i], ax)

        # ax.set_xlim(0, 500)
        # ax.set_ylim(0, 500)

    axes_add_abc(axes)
    plt_tight_show()
