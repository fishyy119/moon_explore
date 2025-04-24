import numpy as np
import skimage
import math
from scipy.ndimage import distance_transform_edt
from scipy.signal import argrelextrema
from pathlib import Path

try:
    from .AStar import AStarPlanner
    from .Rover import Rover
    from .Utils import Setting, Pose2D, Contour, CandidatePoint
except:
    from AStar import AStarPlanner
    from Rover import Rover
    from Utils import Setting, Pose2D, Contour, CandidatePoint

from typing import List
from numpy.typing import NDArray


class Map:
    def __init__(
        self, map_file, map_divide, num_rovers: int = 1, god=False, load_mask: NDArray[np.bool_] | None = None
    ) -> None:
        self.mask: NDArray[np.bool_] = np.zeros((501, 501), dtype=np.bool_)  # True表示已探明
        self.obstacle_mask: NDArray[np.bool_] = np.load(map_file)
        self.map_divide: NDArray[np.int32] = np.load(map_divide)
        # 计算距离场，用于舍弃距离障碍物过近的候选点
        self.distance_ob: NDArray[np.float64] = distance_transform_edt(~self.obstacle_mask)  # type: ignore
        if load_mask is not None:
            self.mask = load_mask
        if god:
            self.mask = np.ones_like(self.mask, dtype=np.bool_)

        self.planner = AStarPlanner(0.8, self.obstacle_mask)
        self.contours: List[Contour] = []

        self.rovers: List[Rover] = []
        self.num_rovers = num_rovers
        shared_mask = self.mask.view()
        shared_ob_mask = self.obstacle_mask.view()
        shared_mask.setflags(write=False)
        shared_ob_mask.setflags(write=False)
        for i in range(self.num_rovers):
            self.rovers.append(Rover(shared_mask, shared_ob_mask, self.planner))
        # 分别为每个巡视器存储分配给其的目标点
        self.rover_assignments: List[List[CandidatePoint]] = [[] for _ in range(self.num_rovers)]

    def rover_move(self, pose: Pose2D, index: int = 0) -> None:
        self.rovers[index].rover_pose = pose
        new_mask = self.rovers[index].generate_sector_mask(pose)
        self.mask |= new_mask

    def rover_init(self, pose: Pose2D, index: int = 0) -> None:
        """与`rover_move`类似，不过这个假设巡视器在最初位置原地转一圈，所以画360度的图"""
        original_FOV = Setting.FOV
        Setting.FOV = 900

        self.rovers[index].rover_pose = pose
        new_mask = self.rovers[index].generate_sector_mask(pose)
        self.mask |= new_mask
        print(f"INIT{index} at {pose}")

        Setting.FOV = original_FOV

    def get_contours(self) -> List[NDArray[np.int32]]:
        """提取连续曲线轮廓，这里传参还是不要传边界点，直接传可视范围就好"""
        # binary_mask = boundary.astype(np.uint8) * 255
        binary_mask = self.mask.astype(np.uint8) * 255
        # cv2方法和ski方法都只能提闭合轮廓，如果传边界点的话，独立线段会提出双层结果
        # 但是ski方法找到了一个mask参数来进一步处理提取结果，这样就可以把障碍信息传进去，最后效果就是提出了闭合轮廓舍弃障碍的部分
        # contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = skimage.measure.find_contours(
            binary_mask, level=254, fully_connected="high", positive_orientation="high", mask=~self.obstacle_mask
        )  # 这里的level，取1则提取到外侧一圈，取254则提取到内侧点上，128则是交界处
        result: List[NDArray[np.int32]] = []  #! 应该并不是int32，不过无伤大雅
        for contour in contours:
            if len(contour) < 20:
                # 舍弃太短的结果，对于各种操作都不方便
                continue
            # contour = contour[:, 0, :].astype(np.int32)  # OpenCV 返回的是 (N, 1, 2) 需要展平
            contour = contour[:, ::-1]  # skimage返回的结果是(row,col)形式的，为了后面用的方便，交换一下
            result.append(contour)
        return result

    @staticmethod
    def curvature_discrete(points: NDArray[np.int32]) -> NDArray[np.float64]:
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

    @staticmethod
    def detect_peaks(curvature: NDArray[np.float64], contour: NDArray[np.int32], threshold=0.5) -> List[int]:
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

        new_peaks: List[int] = []
        if len(peak_indices) > 0:
            diff = np.diff(peak_indices)
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

        return new_peaks

    def cal_canPose(self) -> None:
        ANGLE_OFFSET_RAD = Setting.canPose.ANGLE_OFFSET_RAD
        ANGLE_OFFSET_COS = Setting.canPose.ANGLE_OFFSET_COS
        HALF_NEIGHBOR = Setting.canPose.HALF_NEIGHBOR
        MAX_SEGMENT_LENGTH = Setting.canPose.MAX_SEGMENT_LENGTH

        def generate_pose2D_backward(pose: Pose2D, d: float) -> List[Pose2D]:
            """
            generate_pose2D_backward _summary_

            Args:
                pose (Pose2D): 其xy是目标观测点的坐标，yaw指示了法向
                d (float): 包围的距离

            Returns:
                List[Pose2D]: 在距离xy一定距离的地方，生成扇形包围的若干候选位姿
            """
            result: List[Pose2D] = []
            for angle in ANGLE_OFFSET_RAD:
                dx = d * math.cos(pose.yaw_rad + angle)
                dy = d * math.sin(pose.yaw_rad + angle)
                new_pose = Pose2D(pose.x - dx, pose.y - dy, pose.yaw_rad + angle)
                result.append(new_pose)
            return result

        self.canPoints: List[CandidatePoint] = []
        new_contours: List[Contour] = []
        for new_contour in self.contours:
            contour = new_contour.points
            peaks_idx = new_contour.peaks_idx
            peaks_idx_np = np.array(peaks_idx)
            segment_lengths = peaks_idx_np[1:] - peaks_idx_np[:-1]  # 每段的长度
            seg_anchors = [(start, end) for start, end in zip(peaks_idx[:-1], peaks_idx[1:])]

            cnt = 0
            for seg_idx, (start, end) in enumerate(seg_anchors):
                seg_len = segment_lengths[seg_idx]
                x1, y1 = int(contour[start, 0]), int(contour[start, 1])
                x2, y2 = int(contour[end, 0]), int(contour[end, 1])
                if math.hypot(y1 - y2, x1 - x2) < 10:
                    continue  # * 拒绝很短的空间内出现的多个点

                num_subsegments = int(np.ceil(seg_len / MAX_SEGMENT_LENGTH))
                split_point_idxs = np.linspace(start, end, 2 * num_subsegments + 1, dtype=int)
                mid_point_idxs = split_point_idxs[1::2]  # 取出split_points的偶数位（1开头）

                for mid in mid_point_idxs:
                    x, y = int(contour[mid, 0]), int(contour[mid, 1])  # 中点坐标
                    if (
                        x - HALF_NEIGHBOR < 0
                        or x + HALF_NEIGHBOR >= self.mask.shape[1]
                        or y - HALF_NEIGHBOR < 0
                        or y + HALF_NEIGHBOR >= self.mask.shape[0]
                    ):
                        continue  # 跳过边缘，避免越界

                    neighborhood = self.mask[
                        y - HALF_NEIGHBOR : y + HALF_NEIGHBOR + 1, x - HALF_NEIGHBOR : x + HALF_NEIGHBOR + 1
                    ]

                    # 获取邻域内未知区域的坐标
                    unknown_points = np.array(np.where(neighborhood == False)).T  # 坐标是 (row, col)

                    # 计算未知区域的质心
                    centroid = np.mean(unknown_points, axis=0)
                    centroid = np.array([centroid[1] + (x - HALF_NEIGHBOR), centroid[0] + (y - HALF_NEIGHBOR)])

                    # 计算目标点到质心的向量
                    vector_to_centroid = centroid - np.array([x, y])
                    yaw = np.arctan2(vector_to_centroid[1], vector_to_centroid[0])  # 计算目标点的朝向
                    canPose = Pose2D(contour[mid][0] / Setting.MAP_SCALE, contour[mid][1] / Setting.MAP_SCALE, yaw)

                    for i, p in enumerate(generate_pose2D_backward(canPose, 2)):
                        x, y = int(p.x * Setting.MAP_SCALE), int(p.y * Setting.MAP_SCALE)
                        if not (
                            x >= 0 and x < self.mask.shape[1] and y >= 0 and y < self.mask.shape[0] and self.mask[y, x]
                        ):
                            continue  # 不在可视范围内
                        if self.distance_ob[y, x] <= 0.8 * Setting.MAP_SCALE * 1.5:
                            continue  # 距离障碍物过近
                        decay_factor = ANGLE_OFFSET_COS[i]
                        subseg_factors = [1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
                        self.canPoints.append(
                            CandidatePoint(
                                p, seg_len / num_subsegments * decay_factor * subseg_factors[num_subsegments - 1]
                            )
                        )
                        cnt += 1
                # for mid in mid_point_idxs:
            # for seg_idx, (start, end) in enumerate(seg_anchors):
            if cnt != 0:
                new_contours.append(new_contour)
        # for new_contour in self.contours:
        self.contours = new_contours

    def assign_points_to_rovers(self, index: int = 0) -> None:
        """
        将候选点分配给不同的巡视器，基于距离和当前巡视器的位置。
        多巡视器：仅修改`self.rover_assignments`对应index的列表，其他区域的候选点舍弃
        单巡视器：按照预设区域划分，将候选点分配到三个列表中
        """
        if len(self.rovers) == 1:
            # 强制单巡视器在探索完一个子区域后再前往下一个区域
            self.rover_assignments = [[] for _ in range(3)]
            for point in self.canPoints:
                x, y = int(point.pose.x * Setting.MAP_SCALE), int(point.pose.y * Setting.MAP_SCALE)
                sub_area: int = self.map_divide[y, x]
                self.rover_assignments[sub_area].append(point)

        else:
            rover_positions = [rover.rover_pose for rover in self.rovers]
            self.rover_assignments[index] = []

            for point in self.canPoints:
                distances = [point.pose - rover_pose for rover_pose in rover_positions]
                closest_rover = np.argmin(distances)
                if closest_rover == index:
                    self.rover_assignments[index].append(point)

    def step(self, index: int = 0) -> None:
        self.rovers[index].targetPoint = None
        self.rovers[index].targetPoint_mask = None
        self.contours: List[Contour] = []
        for contour in self.get_contours():
            curvature = self.curvature_discrete(contour)
            peaks_idx = self.detect_peaks(curvature, contour)
            self.contours.append(Contour(contour, curvature, peaks_idx))

        self.cal_canPose()
        self.assign_points_to_rovers(index)
        # * 这些 canPoints 都是传递的引用，所以 Rover 那边进行的赋值可以在 Map 这里访问到
        if len(self.rovers) == 1:
            if True:
                self.rovers[index].evaluate_candidate_points(self.canPoints, [], [])
            else:
                rover = self.rovers[0]
                x, y = int(rover.rover_pose.x * Setting.MAP_SCALE), int(rover.rover_pose.y * Setting.MAP_SCALE)
                sub_area = self.map_divide[y, x]
                if len(self.rover_assignments[sub_area]) <= 1:
                    self.rovers[index].evaluate_candidate_points(self.canPoints, [], [])
                else:
                    self.rovers[index].evaluate_candidate_points(self.rover_assignments[sub_area], [], [])
                    self.canPoints = self.rover_assignments[sub_area]
        else:
            others_target = [r.targetPoint for r in self.rovers if r.targetPoint is not None]
            others_target_mask = [r.targetPoint_mask for r in self.rovers if r.targetPoint_mask is not None]
            self.rovers[index].evaluate_candidate_points(
                self.rover_assignments[index], others_target, others_target_mask
            )
            # * Viewer 需要访问这个变量，除了idx对应的巡视器点被更新了，其他的仍然不变（信息被保存在了assignments中）
            self.canPoints = [point for assignment in self.rover_assignments for point in assignment]


# def test_mask():
#     map.mask |= map.rovers[0].generate_sector_mask_non_ob(Pose2D(28, 25, 160, deg=True))
#     map.mask |= map.rovers[0].generate_sector_mask_non_ob(Pose2D(28, 27, 20, deg=True))
#     viewer.update()

#     plt.ioff()
#     plt.show()


if __name__ == "__main__":
    from Utils import MyTimer
    from Viewer import MaskViewer
    import matplotlib.pyplot as plt

    timer = MyTimer()
    N = 1
    NPY_ROOT = Path(__file__).parent.parent / "resource"
    map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), map_divide=str(NPY_ROOT / "map_divide.npy"), num_rovers=N)
    viewer = MaskViewer(map, "output.mp4")

    # if True:
    #     test_mask()
    #     exit(0)

    # map.rover_init(Pose2D(27, 25.2472, 88.6, deg=True), 0 % N)
    map.rover_init(Pose2D(25, 8, 270, deg=True), 1 % N)
    # map.rover_move(Pose2D(26, 29, 0.7))
    # map.rover_move(Pose2D(20, 20, 3))
    timer.checkpoint("可视计算")
    for i in range(N):
        map.step(i)
    timer.checkpoint("路径规划")

    viewer.update()
    viewer.update(mode=viewer.UpdateMode.CONTOUR, show_score_text=False)
    # viewer.plot_path(map.rovers[1].targetPoint.path)  # type: ignore

    plt.ioff()
    plt.show()
