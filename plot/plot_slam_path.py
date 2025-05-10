import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import *


SUB = Path("20250425_155457_bag1")  # xyz: 0.05, 0.05, 0.01
SUB = Path("20250425_165849")  # 将探索模块接入slam
SUB = Path("20250429_102359_bag2")  # xyz: 0.1, 0.1, 0.01 / 3.9, 3.8, 3.84

csv1 = RecordCSV(SUB / "record_1.csv", "实际运动路径")
txt1 = RecordSLAM(csv1, SUB / "slam_ab_3.txt", "SLAM估计路径")
txt2 = RecordSLAM(csv1, SUB / "slam_no.txt", "slam_no_ab")


fig, ax = plt.subplots()
plot_binary_map(MAP_PASSABLE, ax)
plot_path_map(csv1, ax)
plot_path_map(txt1, ax)
plot_path_map(txt2, ax)

ax_add_legend(ax)
plt_tight_show()
