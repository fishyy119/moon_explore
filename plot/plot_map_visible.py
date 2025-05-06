from plot_utils import *
from moon_explore.Map import Map
from moon_explore.Utils import Pose2D

map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), map_divide=str(NPY_ROOT / "map_divide.npy"), num_rovers=1)
map.rover_move(Pose2D(27, 25.2472, 0, deg=True))
map.rover_move(Pose2D(45, 16, 240, deg=True))
map.rover_move(Pose2D(37, 12, 210, deg=True))

fig, axes = plt.subplots(1, 3)
axes: List[Axes]
plot_binary_map(MAP_PASSABLE, axes[0], visible_map=map.mask)
plot_binary_map(MAP_PASSABLE, axes[1], visible_map=map.mask)
plot_binary_map(MAP_PASSABLE, axes[2], visible_map=map.mask)

axes[0].set_xlim(380, 470)
axes[0].set_ylim(80, 170)
axes[1].set_xlim(250, 350)
axes[1].set_ylim(200, 300)
axes[2].set_xlim(290, 380)
axes[2].set_ylim(50, 140)
plt_tight_show()
