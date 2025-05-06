from plot_utils import *


fig, axes = plt.subplots(1, 2)
plot_slope_map(MAP_SLOPE, axes[0])
plot_binary_map(MAP_PASSABLE, axes[1])
axes_add_abc(axes)
plt_tight_show()
