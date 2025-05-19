from plot_utils import *

fig, axes = plt.subplots(2, 2)
axes = plt_flat_axes(axes)
plot_edf_map(MAP_EDF, axes[0])
plot_binary_map(MAP_EDF < 0.8 * 1.2, axes[1])
plot_binary_map(MAP_EDF < 0.8 * 1.5, axes[2])
plot_binary_map(MAP_EDF < 0.8 * 2.0, axes[3])
axes_add_abc(axes)


# ==============================================
# PPT
# ==============================================
fig1, ax1 = plt.subplots()
plot_edf_map(MAP_EDF, ax1, False)

fig2, axes2 = plt.subplots(1, 3)
plot_binary_map(MAP_EDF < 0.8 * 1.2, axes2[0])
plot_binary_map(MAP_EDF < 0.8 * 1.5, axes2[1])
plot_binary_map(MAP_EDF < 0.8 * 2.0, axes2[2])


plt_tight_show()
