import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import *

# xyz噪声标准差 / 修正周期
SUB = Path("20250425_155457_bag1")  #! xyz: 0.05, 0.05, 0.01 / 这个记录错了东西
SUB = Path("20250429_102359_bag2")  # xyz: 0.1, 0.1, 0.01 / 3.9, 3.8, 3.84
# SUB = Path("20250508_094236_doubleslam")  # xyz: 0.1, 0.1, 0.01 /  3.84 3.8 3.82 3.82 3.78


SUB = Path("20250429_102359_bag2")  # xyz: 0.1, 0.1, 0.01 / 3.9, 3.8, 3.84
csv1 = RecordCSV(SUB / "record_1.csv", "真实轨迹", "--", "black")
txt1 = RecordSLAM(SUB / "slam_ab_3.txt", "融合定位框架")
txt2 = RecordSLAM(SUB / "slam_no.txt", "纯视觉SLAM")


# SUB = Path("20250508_094236_doubleslam")  # xyz: 0.1, 0.1, 0.01 /  3.84 3.8 3.82 3.82 3.78
# csv1 = RecordCSV(SUB / "record_1.csv", "truth")
# txt1 = RecordSLAM(SUB / "slamab_14.txt", "融合定位框架")
# txt2 = RecordSLAM(SUB / "slamno_1.txt", "纯视觉SLAM")


# SUB = Path("20250508_094236_doubleslam")  # xyz: 0.1, 0.1, 0.01 /  3.84 3.8 3.82 3.82 3.78
# csv1 = RecordCSV(SUB / "record_2.csv", "truth")
# txt1 = RecordSLAM(SUB / "slamab_24.txt", "融合定位框架")
# txt2 = RecordSLAM(SUB / "slamno_2.txt", "纯视觉SLAM")


txt1.link_recordAb(csv1)
txt2.link_recordAb(csv1)

fig, axes = plt.subplots(2, 1)
plot_xyz_diff(txt1, axes)
plot_xyz_diff(txt2, axes)
ax_add_legend(axes[0])

fig2, axes2 = plt.subplots(1, 2)
plot_slam_path_error(txt1, axes2[0])
plot_slam_path_error(txt2, axes2[1])
axes_add_abc(axes2)

fig3, ax3 = plt.subplots()
print_slam_report(txt1, txt2, ax3)


plt_tight_show()
