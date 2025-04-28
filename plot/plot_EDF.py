import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from plot_utils import *

distance_map: NDArray[np.float64] = distance_transform_edt(~MAP_PASSABLE) / 10  # type: ignore

fig, ax = plt.subplots()
im = ax.imshow(distance_map, interpolation="nearest", origin="lower")
fig.colorbar(im, ax=ax, orientation="vertical", label="Distance (m)")
plt.show()
