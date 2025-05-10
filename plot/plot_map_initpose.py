from plot_utils import *
from moon_explore.Utils import Pose2D


fig, ax = plt.subplots()
plot_binary_map(MAP_PASSABLE, ax)
# y+2为了视觉效果，这里画的箭头顶部才是对应位置，向上偏移以下近似看做中心是对应位置（依赖于缩放尺寸）

# 单巡视器两个起点
# plot_pose2d_map(Pose2D(27, 27, 90, deg=True), ax, color="#4C72B0")
# plot_pose2d_map(Pose2D(1.85, 3.82, 90, deg=True), ax, color="#ff7f0e")

# 双巡视器两个起点
plot_pose2d_map(Pose2D(27, 27, 90, deg=True), ax, color="#4C72B0")
plot_pose2d_map(Pose2D(30, 27, 90, deg=True), ax, color="#ff7f0e")

plt_tight_show()
