import numpy as np
from scipy.signal import argrelextrema
from scipy.ndimage import binary_erosion, binary_dilation
import skimage
from Pose2D import Pose2D

from typing import Tuple, List, Callable
from numpy.typing import NDArray


class Map:
    MAP_SCALE = 10  # 浮点数坐标乘以这个数便对应到索引坐标
    MAX_RANGE = 6.0  # 最远可视距离，米
    FOV = 90  # 视场角，度数

    def __init__(self) -> None:
        self.global_map: NDArray = ...
        self.visible_map: NDArray = ...
        self.mask: NDArray[np.bool_] = np.zeros((501, 501), dtype=np.bool_)
        self.obstacle_mask = np.zeros((501, 501), dtype=np.bool_)
        self.obstacle_mask[300:, 300:] = True
        self.rover_pose = Pose2D(0, 0, 0)

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
        #! 我总感觉这里的最大值排序寻找end_idx是不太对的，但是目前还没出大问题
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
        # 这里理论上应该是<=，但是前面r_max计算索引的地方恐怕有问题，出此下策先用<判断再膨胀一下
        mask = r < r_max
        return binary_dilation(mask).astype(np.bool_)
        return mask

    def rover_move(self, pose: Pose2D) -> None:
        self.rover_pose = pose
        new_mask = self.generate_sector_mask(pose)
        self.mask |= new_mask

    def extract_grid_boundary(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """提取网格地图的边界点"""
        eroded = binary_erosion(mask).astype(np.bool_)
        boundary = mask & ~eroded
        boundary = boundary & ~self.obstacle_mask
        self.boundary = boundary
        return boundary

    def get_contours(self, boundary: NDArray[np.bool_]) -> List[NDArray[np.int32]]:
        """提取连续曲线轮廓，这里传参还是不要传边界点，直接传可视范围就好"""
        binary_mask = boundary.astype(np.uint8) * 255
        # cv2方法和ski方法都只能提闭合轮廓，如果传边界点的话，独立线段会提出双层结果
        # 但是ski方法找到了一个mask参数来进一步处理提取结果，这样就可以把障碍信息传进去，最后效果就是提出了闭合轮廓舍弃障碍的部分
        # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = skimage.measure.find_contours(
            binary_mask, level=254, fully_connected="high", positive_orientation="high", mask=~self.obstacle_mask
        )  # 这里的level，取1则提取到外侧一圈，取254则提取到内侧点上，128则是交界处
        self.contours: List[NDArray[np.int32]] = []
        for contour in contours:
            if len(contour) < 10:
                # 舍弃太短的结果，对于各种操作都不方便
                continue
            # contour = contour[:, 0, :].astype(np.int32)  # OpenCV 返回的是 (N, 1, 2) 需要展平
            contour = contour[:, ::-1]  # skimage返回的结果是(row,col)形式的，为了后面用的方便，交换一下
            self.contours.append(contour)
        return self.contours

    def curvature_discrete(self, points: NDArray[np.int32]) -> NDArray[np.float64]:
        """
        以5个点的移动窗口，计算各点离散曲率

        Args:
            points (NDArray[np.int32]): 一组二维点 (N, 2)

        Returns:
            NDArray[np.float64]: 各个中间点的曲率列表(首尾各有两个点算不到)
        """
        windows = 2  # 要计算某一点，利用前后各windows个点的信息
        # windows=1不太行，对于直线突然有一个点突出的情况不好

        # 计算两个向量求出夹角
        if windows == 2:
            v1 = points[1:-3] - points[:-4]  # P_{i-2} -> P_{i-1}
            v2 = points[4:] - points[3:-1]  # P_{i+1} -> P_{i+2}
        elif windows == 1:
            v1 = points[1:-1] - points[:-2]  # P_{i-1} -> P_{i}
            v2 = points[2:] - points[1:-1]  # P_{i} -> P_{i+1}
        else:
            raise NotImplementedError
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]  # 叉积
        dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]  # 点积
        theta = np.arctan2(cross, dot)  # 直接获得正确的角度

        # 计算近似弧长（每个点处的弧长取其左右线段各一半）
        segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)  # 所有相邻点的距离
        segment_lengths = (segment_lengths[:-1] + segment_lengths[1:]) / 2
        if windows == 2:
            ds = segment_lengths[:-2] + segment_lengths[1:-1] + segment_lengths[2:]
        else:
            ds = segment_lengths

        # 计算曲率 κ = θ / ds
        curvatures = theta / ds

        return curvatures

    def detect_peaks(self, curvature: NDArray[np.float64], contour: NDArray[np.int32], threshold=0.5) -> List[int]:
        """
        使用阈值检测曲率峰值来提取拐点，同时合并同一个峰的多个点，添加首尾点

        Args:
            curvature (NDArray[np.float64]): 曲率（需要注意其尺寸略短于contour）
            contour (NDArray[np.int32]): 轮廓点
            threshold (float, optional): 曲率阈值

        Returns:
            List[int]: 拐点在contour上对应的索引
        """
        # 1. 先找到所有局部极大值
        curvature = np.abs(curvature)  # 曲率是有正负的，这里只关心大小
        local_maxima = argrelextrema(curvature, np.greater_equal, order=1)[0]

        # 2. 过滤掉低于阈值的峰值
        peak_indices = [idx for idx in local_maxima if curvature[idx] > threshold]

        # 3. 处理平峰：查找平峰区间，并保留中间的点
        diff = np.diff(peak_indices)
        new_peaks: List[int] = []
        cnt = 0
        start = peak_indices[0]

        for i, d in enumerate(diff):
            if d == 1:
                cnt += 1  # 记录连续峰的数量
                continue
            else:
                if cnt > 0:
                    new_peaks.append((start + peak_indices[i]) // 2)  # 取中间点
                else:
                    new_peaks.append(peak_indices[i])  # 记录单独的峰
                start = peak_indices[i + 1]  # 更新新的起点
                cnt = 0

        # 处理最后一段
        if cnt > 0:
            new_peaks.append((start + peak_indices[-1]) // 2)
        else:
            new_peaks.append(peak_indices[-1])

        new_peaks = [peak + 2 for peak in new_peaks]  # 曲率没求边缘点，为了和contour中对齐，手动加2

        # 4. 添加首点和尾点
        new_peaks.append(len(contour) - 1)
        new_peaks.insert(0, 0)
        # first_point = contour[0]
        # last_point = contour[-1]
        # if np.linalg.norm(first_point - last_point) <= 2:
        #     new_peaks.insert(0, 0)
        # else:
        #     new_peaks.append(len(contour) - 1)
        #     new_peaks.insert(0, 0)

        return new_peaks

    def cal_canPose(self):
        points: List[Pose2D] = []
        for contour in self.contours:
            curvature = self.curvature_discrete(contour)
            # 前面生成这个的时候已经添加了首尾点
            peaks_idx = np.array(self.detect_peaks(curvature, contour))

            # 计算每段的中点索引
            segment_lengths = peaks_idx[1:] - peaks_idx[:-1]  # 每段的长度
            midpoints_idx = peaks_idx[:-1] + segment_lengths // 2  # 计算每段的中点索引

            neighborhood_size = 5  # 邻域大小
            half_size = neighborhood_size // 2

            for mid in midpoints_idx:
                x, y = int(contour[mid, 0]), int(contour[mid, 1])  # 中点坐标
                neighborhood = self.mask[y - half_size : y + half_size + 1, x - half_size : x + half_size + 1]

                # 获取邻域内未知区域的坐标
                unknown_points = np.array(np.where(neighborhood == False)).T  # 坐标是 (row, col)

                # 计算未知区域的质心
                centroid = np.mean(unknown_points, axis=0)
                centroid = np.array([centroid[1] + (x - half_size), centroid[0] + (y - half_size)])

                # 计算目标点到质心的向量
                vector_to_centroid = centroid - np.array([x, y])
                yaw = np.arctan2(vector_to_centroid[1], vector_to_centroid[0])  # 计算目标点的朝向

                points.append(Pose2D(contour[mid][0] / Map.MAP_SCALE, contour[mid][1] / Map.MAP_SCALE, yaw))

        return points


if __name__ == "__main__":
    from Viewer import MaskViewer

    map = Map()
    map.rover_move(Pose2D(29, 29, 0.5))
    map.rover_move(Pose2D(23, 25, 3))
    # map.rover_move(Pose2D(25, 29, 0.1))
    map.rover_move(Pose2D(26, 29, 3))
    map.rover_move(Pose2D(20, 20, 3))
    map.extract_grid_boundary(map.mask)  # 这步的作用存疑(可能在刨除障碍物边界时有作用)
    map.get_contours(map.mask)

    viewer = MaskViewer(map)
    points = map.cal_canPose()
    for point in points:
        viewer.plot_pose2d(point)
    viewer.show()
