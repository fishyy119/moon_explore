import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import *


SUB = Path("20250425_155457_bag1")  # xyz: 0.05, 0.05, 0.01
SUB = Path("20250425_165849")  # 将探索模块接入slam

csv1 = RecordCSV(SUB / "record_1.csv", "truth")
txt1 = RecordSLAM(csv1, SUB / "slam_ab.txt", "slam_ab")
# txt2 = RecordSLAM(csv1, SUB / "slam_no.txt", "slam_no_ab")


fig, ax = plt.subplots()
plot_ob_mask(MAP_PASSABLE, ax, 1)
plot_path_map(csv1, ax)
plot_path_map(txt1, ax)
# plot_path_map(txt2, ax)


plt.legend()
plt.tight_layout()
plt.show()
