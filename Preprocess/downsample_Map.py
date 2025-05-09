import numpy as np
from skimage.measure import block_reduce
from plot_Map import CAL_AND_PLOT_PASSABILITY
from pathlib import Path

from typing import Tuple, List, Callable
from numpy.typing import NDArray

DownSampleFunc = Callable[[NDArray, int], NDArray]
downsample_average: DownSampleFunc = lambda image, factor: block_reduce(image, block_size=factor, func=np.mean)


def downsample_max(image, factor: int):
    return block_reduce(image, block_size=factor, func=np.max)


NPY_ROOT = Path(__file__).parent.parent / "resource"
slope_map: NDArray = np.load(NPY_ROOT / "map_slope.npy")
average = downsample_average(slope_map, 8)
max_map = downsample_max(slope_map, 8)

CAL_AND_PLOT_PASSABILITY(average, [5, 10, 20])
CAL_AND_PLOT_PASSABILITY(max_map, [5, 10, 20])


a = 1
