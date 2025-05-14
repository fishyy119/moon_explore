from plot_utils import *


fig, axes = plt.subplots(1, 3)
plot_scenario_map(MAP_SCENARIO, axes[0])
plot_slope_map(MAP_SLOPE, axes[1])
plot_binary_map(MAP_PASSABLE, axes[2])
axes_add_abc(axes)
plt_tight_show()
