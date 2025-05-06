from plot_utils import *

fig, axes = plt.subplots(2, 2)
axes = plt_flat_axes(axes)
plot_edf_map(MAP_EDF, axes[0])
plot_binary_map(MAP_EDF < 0.8 * 1.2, axes[1])
plot_binary_map(MAP_EDF < 0.8 * 1.5, axes[2])
plot_binary_map(MAP_EDF < 0.8 * 2.0, axes[3])
axes_add_abc(axes)
plt_tight_show()
