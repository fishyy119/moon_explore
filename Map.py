import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Pose2D import Pose2D

from typing import Tuple, List, Callable
from numpy.typing import NDArray


class Map:
    MAP_SCALE = 10  # 浮点数坐标乘以这个数便对应到索引坐标
    SECTOR_SCALE = 3  # 扇形遮罩计算所用的栅格在地图基础上的进一步细分
    MAX_RANGE = 6.0  # 最远可视距离，米
    FOV = 90  # 视场角，度数

    def __init__(self) -> None:
        self.global_map: NDArray = ...
        self.visible_map: NDArray = ...
        self.mask: NDArray[np.bool_] = np.zeros((501, 501), dtype=np.bool_)
        self.obstacle_mask = np.zeros((501, 501), dtype=np.bool_)
        self.obstacle_mask[300:, 300:] = True

    def cal_r_max(
        self,
        theta_max: NDArray[np.float64],
        theta_min: NDArray[np.float64],
        target_yaw: float,
        r: NDArray[np.float64],
        deg_type: int = 180,
    ) -> NDArray[np.float64]:
        """
        一个封装的计算各个格点理论上最大可视距离的函数，
        由于角度的环形度量会影响间断点处的大小比较，
        因此这个函数在两种不同的角度范围约定下运行两次，最后取并集

        在360的FOV下，对于正后方的一条线会有计算错误，这种整个环形全部可视的情况无法通过两次截断模拟出来
        两次计算的断点刚好在正后方，而对于覆盖的判断需要覆盖整个栅格，截断刚好漏掉这里的格子
        （不过调成400就可以模拟360度全覆盖了）

        Args:
            theta_max (NDArray[np.float64]): 每个格子对应一个角度范围，这是最大值
            theta_min (NDArray[np.float64]): 每个格子对应一个角度范围，这是最小值
            target_yaw (float): 巡视器的目标朝向，在其左右根据FOV扩展
            r (NDArray[np.float64]): 各个格点到巡视器的距离
            deg_type (int, optional): 标识使用了何种角度范围，对应处理巡视器的视角边界处角度值

        Returns:
            NDArray[np.float64]: 每个格点的理论最大可视距离（考虑FOV、障碍）
        """
        # 考虑障碍物，每个角度上最大可视距离
        r_max = np.zeros_like(r)
        range_min = target_yaw - Map.FOV / 2
        range_max = target_yaw + Map.FOV / 2

        # 设计上这个函数在两个不同的角度值范围约定下运行两次取并集，因此这里可以舍弃间断点导致的跳跃值
        if deg_type == 180:
            range_min = max(range_min, -180)
            range_max = min(range_max, 180.1)  # 感觉对于边界应该取大一点，但是没有测试
        elif deg_type == 360:
            range_min = max(range_min, 0)
            range_max = min(range_max, 360.1)
        else:
            raise NotImplementedError

        r_max[(range_min <= theta_min) & (theta_max <= range_max)] = Map.MAX_RANGE * Map.MAP_SCALE

        # 提取障碍物的角度范围和距离
        ob_r = r[self.obstacle_mask]
        ob_theta_max = theta_max[self.obstacle_mask]
        ob_theta_min = theta_min[self.obstacle_mask]

        # **先对障碍物按角度排序**
        sort_idx = np.argsort(ob_theta_min)
        ob_theta_min_sorted = ob_theta_min[sort_idx]
        ob_theta_max_sorted = ob_theta_max[sort_idx]
        ob_r_sorted = ob_r[sort_idx]

        # **对栅格的 theta_min 进行排序，便于搜索**
        theta_min_flat = theta_min.ravel()
        theta_max_flat = theta_max.ravel()
        r_max_flat = r_max.ravel()  # 注意这是引用

        theta_sort_idx = np.argsort(theta_min_flat)
        theta_min_sorted = theta_min_flat[theta_sort_idx]
        theta_max_sorted = theta_max_flat[theta_sort_idx]

        # **利用 `searchsorted` 找到受影响的索引范围**
        start_idx = np.searchsorted(theta_min_sorted, ob_theta_min_sorted, side="left")
        end_idx = np.searchsorted(theta_max_sorted, ob_theta_max_sorted, side="right")

        # **批量更新 r_max**
        for i in range(len(ob_r_sorted)):
            affected_indices = theta_sort_idx[start_idx[i] : end_idx[i]]
            np.minimum.at(r_max_flat, affected_indices, ob_r_sorted[i])

        return r_max

    def generate_sector_mask(self, pose: Pose2D) -> NDArray[np.bool_]:
        """计算扇形范围，考虑障碍"""
        if self.obstacle_mask[round(pose.y * Map.MAP_SCALE), round(pose.x * Map.MAP_SCALE)]:
            # 卡墙里就别算了
            return np.zeros_like(self.mask, dtype=np.bool_)

        # 创建坐标网格
        shape, shape = self.mask.shape
        x, y = np.meshgrid(np.arange(shape), np.arange(shape))
        dx = x - pose.x * Map.MAP_SCALE
        dy = y - pose.y * Map.MAP_SCALE

        # 计算极坐标
        r = np.sqrt(dx**2 + dy**2)

        # 对于角度，计算栅格四个顶点的角度，然后存储最大值与最小值
        theta_ur = np.degrees(np.arctan2(dy + 0.5, dx + 0.5))
        theta_ul = np.degrees(np.arctan2(dy + 0.5, dx - 0.5))
        theta_br = np.degrees(np.arctan2(dy - 0.5, dx + 0.5))
        theta_bl = np.degrees(np.arctan2(dy - 0.5, dx - 0.5))

        theta_max: NDArray[np.float64] = np.maximum.reduce([theta_ur, theta_ul, theta_br, theta_bl])
        theta_min: NDArray[np.float64] = np.minimum.reduce([theta_ur, theta_ul, theta_br, theta_bl])

        r_max1 = self.cal_r_max(theta_max, theta_min, pose.yaw_deg180, r)

        # 将角度范围变换到0~360，再运算一次，两者取并集
        theta_max_t = (theta_max + 360) % 360
        theta_min_t = (theta_min + 360) % 360
        # 新范围下原边界点处大小出现错乱
        theta_max = np.maximum(theta_max_t, theta_min_t)
        theta_min = np.minimum(theta_max_t, theta_min_t)
        r_max2 = self.cal_r_max(theta_max, theta_min, pose.yaw_deg360, r, deg_type=360)

        r_max = np.maximum(r_max1, r_max2)

        # 按照矩阵的样式，维度0对应y，维度1对应x
        mask = r <= r_max
        return mask

    def rover_move(self, pose: Pose2D) -> None:
        self.mask |= self.generate_sector_mask(pose)


class MaskViewer:
    def __init__(self, map_instance: Map):
        self.map = map_instance

    def show_mask(self) -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(self.map.mask, cmap="gray_r", origin="lower")

        ax.set_title("Sector Mask Viewer")
        plt.colorbar(im, ax=ax)
        plt.show()

    def show_anime(self) -> None:
        self.fig, self.ax = plt.subplots()

        # 初始化掩码显示
        self.im = self.ax.imshow(self.map.mask, cmap="gray_r", origin="lower", animated=True)
        self.ax.set_title("Sector Mask Viewer")
        self.fig.colorbar(self.im, ax=self.ax)

        # 设置 30Hz 更新频率
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=33.3, blit=True, cache_frame_data=False)
        plt.show()

    def update(self, _):
        """每帧更新 `map.mask` 可视化"""
        self.im.set_data(self.map.mask)  # 直接读取 map.mask
        return (self.im,)  # `blit=True` 需要返回可更新的对象


if __name__ == "__main__":
    map = Map()
    map.rover_move(Pose2D(40, 40, 0.3))
    map.rover_move(Pose2D(29, 29, 0.5))
    map.rover_move(Pose2D(20, 20, 3))
    viewer = MaskViewer(map)
    viewer.show_mask()
