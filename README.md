# moon_explore

## *.npy

- `map_origin.npy`: 原始的高程图，尺寸501*502
- `map.npy`: 500x500x4，貌似是处理好点颜色的场景图，但是数据只有局部
- `map_truth.npy`: 在原始的高程图基础上裁剪了多余的一列，尺寸501*501，单位m
- `map_aspect.npy`: 坡向图，尺寸同上，存储弧度
- `map_slope.npy`: 坡度图，尺寸同上，存储正切值
- `map_explorable.npy`: 估计的可探索区域，用于评价探索率
- `map_divide.npy`: 废案，本来想让巡视器分块依次探索