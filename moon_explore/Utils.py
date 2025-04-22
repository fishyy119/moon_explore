import numpy as np
import math
import time
from scipy.spatial.transform import Rotation

from typing import Dict, List, Optional, Tuple, Callable, NamedTuple
from numpy.typing import NDArray
from dataclasses import dataclass, field


class Setting:
    MAP_SCALE = 10  # 浮点数坐标乘以这个数便对应到索引坐标
    MAX_RANGE = 6.0  # 最远可视距离，米
    FOV: int = 90  # 视场角，度数

    class canPose:
        NEIGHBORHOOD_SIZE = 5  # 确定朝向的邻域大小
        MAX_SEGMENT_LENGTH = 50  # 决定分裂成多少段（至少一段）
        NUM_POSE = 3  # 每个目标点生成几个包围的点（奇数）
        POSE_DEG_STEP = 30  # 生成点时的角度步长

        HALF_NEIGHBOR = NEIGHBORHOOD_SIZE // 2
        HALF_NUM_POSE = NUM_POSE // 2
        ANGLE_OFFSET_DEG = np.arange(-HALF_NUM_POSE, HALF_NUM_POSE + 1) * POSE_DEG_STEP
        ANGLE_OFFSET_RAD = np.radians(ANGLE_OFFSET_DEG)
        ANGLE_OFFSET_COS = np.cos(ANGLE_OFFSET_RAD)

    @dataclass
    class Eval:
        RATIO_THRESHOLD: float = 0.4  # 当探索比例超过这个值切换策略
        ENABLED_SWITCH: bool = True  # 开关项，控制是否开启
        NEW_BETA: float = 0.1

        D_M: float = 4  # 基准时间：直线距离m
        A_D: float = 30  # 基准时间：旋转角度deg
        L_S: float = 0.1  # 最大线速度 m/s
        A_S: float = 0.1  # 最大角速度 rad/s

        BETA: float = 0.4

        BASE_TIME: float = field(init=False)
        ALPHA: float = field(init=False)

        def __post_init__(self):
            self.BASE_TIME = self.D_M / self.L_S + np.deg2rad(self.A_D) / self.A_S
            self.ALPHA = -np.log(0.8) / self.BASE_TIME

        @property
        def T_SEG(self) -> float:
            return 100 * self.BETA

        @property
        def T_PATH(self) -> float:
            return 100 * (1 - self.BETA)

    eval = Eval()

    # class eval:
    #     RATIO_THRESHOLD = 0.4  # 当探索比例超过这个值切换策略
    #     ENABLED_SWITCH = True  # 开关项，控制是否开启

    #     D_M = 4  # 基准时间：直线距离m
    #     A_D = 30  # 基准时间：旋转角度deg
    #     L_S = 0.1  # 最大线速度 m/s
    #     A_S = 0.1  # 最大角速度 rad/s
    #     BASE_TIME = D_M / L_S + np.deg2rad(A_D) / A_S
    #     ALPHA = -np.log(0.8) / BASE_TIME
    #     # print(ALPHA)

    #     BETA = 0.4
    #     T_SEG = 100 * BETA
    #     T_PATH = 100 * (1 - BETA)


class PoseDiff(NamedTuple):
    dist: float  # 欧氏距离
    yaw_diff_deg: float  # 偏航角差绝对值(角度)
    yaw_diff_rad: float


class Pose2D:
    def __init__(self, x: float, y: float, yaw: float, deg=False) -> None:
        """
        二维的位姿

        Args:
            x (float): 全局x坐标
            y (float): 全局y坐标
            yaw (float): 弧度制偏航角
            deg (bool): 如果为True，则使用角度制定义偏航角，默认为False
        """
        self._x = x
        self._y = y
        if deg:
            self._yaw = math.radians(yaw)
        else:
            self._yaw = yaw

    @classmethod
    def from_pose_msg(cls, x, y, qx, qy, qz, qw) -> "Pose2D":
        """
        从 ROS Pose 消息创建 Pose2D 实例
        这里不处理 Pose 消息的实例，因此参数是提取后的
        坐标系是slam的图像坐标系：右下前
        要先转换回前左上

        Args:
            ROS 2 中的 geometry_msgs.msg.Pose 消息中的必要参数

        Returns:
            Pose2D: 生成的二维位姿
        """
        R_ = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])  # webots中相机系：前左上；图像的坐标系：右下前
        camera_rotation = np.dot(Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), R_.T)
        yaw = Rotation.from_matrix(camera_rotation).as_euler("xyz")[2]
        # yaw = Rotation.from_quat([qx, qy, qz, qw]).as_euler("xyz")[2]
        return cls(x, y, yaw)

    def __sub__(self, other: "Pose2D") -> float:
        return math.hypot(self._x - other._x, self._y - other._y)

    def __xor__(self, other: "Pose2D") -> PoseDiff:
        distance = self - other
        diff_abs = abs(self.yaw_deg360 - other.yaw_deg360)
        yaw_diff = min(diff_abs, 360 - diff_abs)
        return PoseDiff(dist=distance, yaw_diff_deg=yaw_diff, yaw_diff_rad=yaw_diff * math.pi / 180)

    @property
    def t(self) -> NDArray[np.float64]:
        return np.array([[self._x], [self._y]])

    @property
    def SO2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        cos_y = np.cos(self._yaw)
        sin_y = np.sin(self._yaw)
        return np.array([[cos_y, -sin_y], [sin_y, cos_y]])

    @property
    def SO2inv(self) -> NDArray[np.float64]:
        return self.SO2.T

    @property
    def SE2(self) -> NDArray[np.float64]:
        "表示在全局坐标系"
        return np.block([[self.SO2, self.t], [np.zeros(2), 1]])

    @property
    def SE2inv(self) -> NDArray[np.float64]:
        return np.block([[self.SO2inv, -self.SO2inv @ self.t], [np.zeros(2), 1]])

    @property
    def x(self) -> float:
        "全局坐标系x坐标"
        return self._x

    @x.setter
    def x(self, value: float) -> None:
        self._x = value

    @property
    def y(self) -> float:
        "全局坐标系y坐标"
        return self._y

    @y.setter
    def y(self, value: float) -> None:
        self._y = value

    @property
    def xy(self) -> NDArray[np.float64]:
        return np.array([self._x, self._y])

    @property
    def yaw_rad(self, deg=False) -> float:
        "偏航角，弧度制"
        return self._yaw

    @yaw_rad.setter
    def yaw_rad(self, value: float) -> None:
        self._yaw = value

    @property
    def yaw_deg180(self) -> float:
        """返回 [-180, 180) 的角度值"""
        deg = math.degrees(self._yaw) % 360
        return (deg + 180) % 360 - 180

    @property
    def yaw_deg360(self) -> float:
        """返回 [0, 360) 的角度值"""
        return math.degrees(self._yaw) % 360

    def __str__(self):
        return f"{self._x}, {self._y}, {self.yaw_deg360}"

    def __repr__(self):
        return f"{self._x}, {self._y}, {self.yaw_deg360}"


class RoverPath(NamedTuple):
    path_float: NDArray[np.float64]  # 这个传给绘图模块用于绘图
    path_pose: List[Pose2D]  # 这个指定若干带朝向的路径点，对其跟踪
    collision: bool  # 表示路径是否与障碍物发生碰撞（对于估计的直线路径有意义）


@dataclass
class CandidatePoint:
    pose: Pose2D  # 候选的目标位姿
    seg_len: float  # 对应弧段的长度
    path: Optional[RoverPath] = None  # 规划的路径的坐标
    path_cost: float = 0  # 运动成本
    score: float = 0  # 最终评分


class Contour(NamedTuple):
    points: NDArray[np.int32]  # 轮廓点（他的类型好像也是float）
    curvature: NDArray[np.float64]  # 曲率
    peaks_idx: List[int]  # 拐点索引


class MyTimer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = self.start_time
        self.records = []

    def checkpoint(self, label=None):
        now = time.time()
        duration = now - self.last_time
        total = now - self.start_time
        self.records.append((label, duration, total))
        if label is None:
            label = f"checkpoint{len(self.records)}"
        print(f"[{label}] {duration:.3f}s (Total: {total:.3f}s)")
        self.last_time = now
