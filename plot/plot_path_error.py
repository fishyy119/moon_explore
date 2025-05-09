import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import *

# xyz噪声标准差 / 修正周期
SUB = Path("20250425_155457_bag1")  # xyz: 0.05, 0.05, 0.01
SUB = Path("20250429_102359_bag2")  # xyz: 0.1, 0.1, 0.01 / 3.9, 3.8, 3.84
SUB = Path("20250508_094236_doubleslam")  # xyz: 0.1, 0.1, 0.01 /  3.84 3.8 3.82 3.82 3.78

csv1 = RecordCSV(SUB / "record_1.csv", "truth")
txt1 = RecordSLAM(csv1, SUB / "slamab_1.txt", "slam_ab")
txt2 = RecordSLAM(csv1, SUB / "slamno_1.txt", "slam_no_ab")

# csv1 = RecordCSV(SUB / "record_1.csv", "truth")
# txt1 = RecordSLAM(csv1, SUB / "slam_ab_3.txt", "slam_ab")
# txt2 = RecordSLAM(csv1, SUB / "slam_no.txt", "slam_no_ab")

fig, axes = plt.subplots(2, 1)
plot_xyz_diff(txt1, axes)
plot_xyz_diff(txt2, axes)

plt.legend()
plt.tight_layout()
plt.show()
