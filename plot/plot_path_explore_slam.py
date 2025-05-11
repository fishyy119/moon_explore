from pathlib import Path
from plot_utils import *


csvs: List[RecordCSV] = []
txts: List[RecordSLAM] = []


SUB = Path("20250429_102359_bag2")  # xyz: 0.1, 0.1, 0.01 / 3.9, 3.8, 3.84
csvs.append(RecordCSV(SUB / "record_1.csv", "实际运动路径", "--", "black"))
txts.append(RecordSLAM(SUB / "slam_ab_2.txt", "SLAM估计路径"))

# SUB = Path("20250425_165849")  # 将探索模块接入slam
SUB = Path("20250508_094236_doubleslam")
csvs.append(RecordCSV(SUB / "record_1.csv", "实际运动路径", "--", "black"))
txts.append(RecordSLAM(SUB / "slamab_1.txt", "SLAM估计路径"))

SUB = Path("20250508_094236_doubleslam")
csvs.append(RecordCSV(SUB / "record_2.csv", "实际运动路径", "--", "black"))
txts.append(RecordSLAM(SUB / "slamab_2.txt", "SLAM估计路径"))

# * 图一：两个路径的对比
fig, axes = plt.subplots(1, len(csvs))
axes: List[Axes]
for idx, (csv, txt, ax) in enumerate(zip(csvs, txts, axes)):
    txt.link_recordAb(csv)
    plot_path_map(csv, ax)
    plot_path_map(txt, ax)

    ax.margins(x=0.05, y=0.05)
    # 后面绘制地图会重置坐标轴范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_binary_map(MAP_PASSABLE, ax)
    ax_set_square_lim(ax, xlim, ylim, border=True)

    if idx == 0:
        ax_add_legend(ax)

# * 图二：绘制障碍物距离信息
fig_2, axes_2 = plt.subplots(1, len(csvs))
axes_2: List[Axes]
for idx, (csv, txt, ax) in enumerate(zip(csvs, txts, axes_2)):
    txt.link_recordAb(csv)
    plot_path_distance_map(csv, ax)

    ax.margins(x=0.05, y=0.05)
    # 后面绘制地图会重置坐标轴范围
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    plot_binary_map(MAP_PASSABLE, ax)
    ax_set_square_lim(ax, xlim, ylim, border=True)


plt_tight_show()
