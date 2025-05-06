from plot_utils import *
from moon_explore.Map import Map
from moon_explore.Utils import Pose2D


map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), map_divide=str(NPY_ROOT / "map_divide.npy"), num_rovers=1)
map.rover_init(Pose2D(27, 25.2472, 0, deg=True))
map.step()

fig, ax = plt.subplots()
plot_binary_map(MAP_PASSABLE, ax, visible_map=map.mask)
plot_contours_map(map.contours, ax)

ax.set_xlim(200, 350)
ax.set_ylim(180, 330)
plt_tight_show()
