import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.ndimage import distance_transform_edt, label

from numpy.typing import NDArray

R = 0.8 * 10 * 1.3  # 安全半径 * 地图缩放系数 * 安全系数

NPY_ROOT = Path(__file__).parent.parent / "resource"
map = np.load(NPY_ROOT / "map_passable.npy")
map[:, 0] = 0  # 去除边界手动设置的禁行区，边界与环形山扩张后连接在一起，导致计算的不对
map[:, -1] = 0
map[0, :] = 0
map[-1, :] = 0
map[30, 495] = 1  # 右下角不做这个处理，这个角落在边界禁行的前提下巡视器进不去，现在不考虑边界禁行会有问题
edt1 = distance_transform_edt(~map)  # type: ignore
map1: NDArray[np.bool_] = edt1 > R  # type: ignore

structure = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])  # 8-邻域连通性

labeled, num_features = label(map1, structure=structure)  # type: ignore
map2 = labeled == 1  # 从图里看出来的，序号1对应了需要的连通区域
edt2: NDArray = distance_transform_edt(~map2)  # type: ignore
map_result = edt2 <= R + 2  # 可视范围模拟时，这里也是手动多看了一些
np.save(NPY_ROOT / "map_explorable.npy", ~map_result)
print(np.count_nonzero(map_result) / 501 / 501 * 100, "%")


fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
ax1.imshow(labeled, cmap="tab20", origin="lower")
ax2.imshow(~map_result, cmap="binary", origin="lower")
plt.show()
