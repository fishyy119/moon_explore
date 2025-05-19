import matplotlib.pyplot as plt
from pathlib import Path
from plot_utils import *


SUB = Path("20250508_094236_doubleslam")  # xyz: 0.1, 0.1, 0.01 /  3.84 3.8 3.82 3.82 3.78
csv1 = RecordCSV(SUB / "record_1.csv", "truth")
txt11 = RecordSLAM(SUB / "slamab_14.txt", "融合定位框架").link_recordAb(csv1)
txt12 = RecordSLAM(SUB / "slamno_1.txt", "纯视觉SLAM").link_recordAb(csv1)

csv2 = RecordCSV(SUB / "record_2.csv", "truth")
txt21 = RecordSLAM(SUB / "slamab_24.txt", "融合定位框架").link_recordAb(csv2)
txt22 = RecordSLAM(SUB / "slamno_2.txt", "纯视觉SLAM").link_recordAb(csv2)


fig11, axes11 = plt.subplots(2, 1)
plot_xyz_diff(txt11, axes11)
plot_xyz_diff(txt12, axes11)
ax_add_legend(axes11[0])

fig12, axes12 = plt.subplots(2, 1)
plot_xyz_diff(txt21, axes12)
plot_xyz_diff(txt22, axes12)
ax_add_legend(axes12[0])

fig1, axes1 = plt.subplots(2, 2)
plot_xyz_diff(txt11, [axes1[0][0], axes1[1][0]])
plot_xyz_diff(txt12, [axes1[0][0], axes1[1][0]])
plot_xyz_diff(txt21, [axes1[0][1], axes1[1][1]])
plot_xyz_diff(txt22, [axes1[0][1], axes1[1][1]])
ax_add_legend(axes1[0][0])


fig2, axes2 = plt.subplots(2, 2)
plot_slam_path_error(txt11, axes2[0][0])
plot_slam_path_error(txt12, axes2[0][1])
plot_slam_path_error(txt21, axes2[1][0])
plot_slam_path_error(txt22, axes2[1][1])
axes_add_abc(plt_flat_axes(axes2))

# fig2ppt, axes2ppt = plt.subplots(1, 4)
# plot_slam_path_error(txt11, axes2ppt[0])
# plot_slam_path_error(txt12, axes2ppt[1])
# plot_slam_path_error(txt21, axes2ppt[2])
# plot_slam_path_error(txt22, axes2ppt[3])


fig3, axes3 = plt.subplots(1, 2)
print_slam_report(txt11, txt12, axes3[0])
print_slam_report(txt21, txt22, axes3[1])
axes_add_abc(axes3, -0.1)


plt_tight_show()
