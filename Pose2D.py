import math
import numpy as np

from numpy.typing import NDArray


class Pose2D:
    def __init__(self, x: float, y: float, yaw: float, deg=False) -> None:
        """
        二维的位姿

        Args:
            x (float): 全局x坐标
            y (float): 全局y坐标
            yaw (float): 弧度制偏航角
        """
        self._x = x
        self._y = y
        if deg:
            self._yaw = math.radians(yaw)
        else:
            self._yaw = yaw

    @classmethod
    def from3Dq(cls) -> "Pose2D":
        return cls(1, 1, 1)

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
    def yaw_rad(self, deg=False) -> float:
        "偏航角，弧度制"
        return self._yaw

    @yaw_rad.setter
    def yaw_rad(self, value: float) -> None:
        self._yaw = value

    @property
    def yaw_deg(self) -> float:
        "偏航角，角度制"
        return math.degrees(self._yaw)

    @yaw_deg.setter
    def yaw_deg(self, value: float) -> None:
        self._yaw = math.radians(value)
