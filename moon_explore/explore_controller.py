import rclpy  # type: ignore
from rclpy.node import Node  # type: ignore
from ament_index_python.packages import get_package_share_directory  # type: ignore
from geometry_msgs.msg import Twist, Pose  # type: ignore
from std_msgs.msg import Float64  # type: ignore
import sys

sys.path.append("/home/yyy/miniconda3/envs/moon_py310/lib/python3.10/site-packages")
import csv
import numpy as np
from pathlib import Path
from datetime import datetime

from moon_explore.Utils import Pose2D
from moon_explore.Map import Map
from moon_explore.Viewer import MaskViewer

from enum import Enum, auto
from typing import Callable, Dict, Optional

PROJECT_DIR = "/home/yyy/moon_R2023"


class State(Enum):
    INIT = auto()  # 初始化，自转一圈
    STOP = auto()  # 停止，进行规划
    ROTATE = auto()  # 旋转，对齐直线轨迹
    DRIVE = auto()  # 直行，内部也有反馈式方向微调


class RoverController:
    def __init__(self, node: Node, rover_id: int, map: Map, viewer: MaskViewer):
        self.id = rover_id
        self.node = node
        self.map = map
        self.viewer = viewer
        self.pose_now: Optional[Pose2D] = None
        self.fsm_state = State.STOP
        self.last_dist = 1000

        self.LOG = lambda msg: self.node.get_logger().info(str(msg))
        self.subscription = node.create_subscription(Pose, f"robot_pose_{rover_id}", self.robot_pose_callback, 10)
        self.publisher = node.create_publisher(Twist, f"cmd_vel_{rover_id}", 10)

    def robot_pose_callback(self, msg: Pose):
        last_pose = self.pose_now
        self.pose_now = Pose2D.from_pose_msg(
            msg.position.x, msg.position.y, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        if last_pose is None:
            self.map.rover_init(self.pose_now, self.id - 1)
        else:
            self.map.rover_move(self.pose_now, self.id - 1)
        self.viewer.update()

    def step(self):
        if self.pose_now is None:
            return
        twist = Twist()

        if self.fsm_state != State.DRIVE:
            self.last_dist = 1000  # DRIVE用到了这个，要初始化一个比较大的数

        m: Callable[[State], bool] = lambda s: self.fsm_state == s
        if m(State.INIT):
            pass  # 初始化自转一圈，在Map中实现，没有真的控制转一圈

        elif m(State.STOP):
            self.LOG(f"[{self.id}] 重新规划")
            self.map.step(self.id - 1)
            self.viewer.update(mode=MaskViewer.UpdateMode.CONTOUR)
            self.path = self.map.rovers[self.id - 1].targetPoint.path  # type: ignore
            assert self.path is not None
            self.viewer.plot_path(self.path, index=self.id)
            self.fsm_state = State.DRIVE
            self.targets = self.path.path_pose
            self.current_target = self.targets[0]
            self.idx_now = 0

        elif m(State.DRIVE):
            dist = self.pose_now - self.current_target
            if dist < 0.2 or dist >= self.last_dist + 0.2:
                self.fsm_state = State.ROTATE
            self.last_dist = min(dist, self.last_dist)
            twist.linear.x = 0.1 if dist >= 0.5 else 0.05
            yaw_target = np.arctan2(self.current_target.y - self.pose_now.y, self.current_target.x - self.pose_now.x)
            yaw_error = (yaw_target - self.pose_now.yaw_rad + np.pi) % (2 * np.pi) - np.pi
            if abs(yaw_error) > 0.01:
                twist.angular.z = max(-0.01, min(0.01, 0.5 * yaw_error))
            # self.LOG(f"DRIVE: {dist:.2f}")

        elif m(State.ROTATE):
            yaw_error = (self.current_target.yaw_rad - self.pose_now.yaw_rad + np.pi) % (2 * np.pi) - np.pi
            if abs(yaw_error) < 0.1:
                if self.idx_now >= len(self.targets) - 1:
                    self.fsm_state = State.STOP
                else:
                    self.idx_now += 1
                    self.current_target = self.targets[self.idx_now]
                    self.fsm_state = State.DRIVE
            else:
                twist.angular.z = max(-0.1, min(0.1, 0.5 * yaw_error))
                # self.LOG(f"ROTATE: {self.pose_now.yaw_deg360:.2f} -> {self.current_target.yaw_deg360:.2f}")

        factor = 1.5
        twist.angular.z *= factor
        twist.linear.x *= factor
        self.publisher.publish(twist)


class ExploreController(Node):
    def __init__(self):
        super().__init__("explore_controller")

        # 获取 'num_rovers' 参数的值
        self.declare_parameter("num_rovers", 1)
        num_rovers = self.get_parameter("num_rovers").get_parameter_value().integer_value

        package_share_directory = get_package_share_directory("moon_explore")
        NPY_ROOT = Path(package_share_directory) / "resource"
        self.map = Map(str(NPY_ROOT / "map_passable.npy"), num_rovers=num_rovers)
        self.timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_file = str(Path(PROJECT_DIR) / f"Data/video/output_{self.timestamp}.mp4")
        self.viewer = MaskViewer(self.map, output_file)

        # 记录区域覆盖率
        self.sim_time = None
        self.create_subscription(Float64, f"simulation_time_1", self.simulation_time_callback, 1)
        self.csv_file = str(Path(PROJECT_DIR) / f"Data/map/rate_{self.timestamp}.csv")
        with open(self.csv_file, mode="a", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [
                    "time",
                    "view_grids",
                ]
            )

        self.rovers: Dict[int, RoverController] = {}
        for i in range(num_rovers):
            self.rovers[i] = RoverController(self, i + 1, self.map, self.viewer)

    def simulation_time_callback(self, msg: Float64) -> None:
        self.sim_time = msg.data  # 当前仿真时间

    def step(self):
        for rover in self.rovers.values():
            rover.step()
        if self.sim_time is not None:
            with open(self.csv_file, mode="a", newline="") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([self.sim_time, np.count_nonzero(self.map.mask)])

    def destroy_node(self):
        np.save(str(Path(PROJECT_DIR) / f"Data/map/mask_{self.timestamp}.npy"), self.map.mask)
        self.viewer.writer.close()  # 确保关闭视频
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ExploreController()
    try:
        while rclpy.ok():
            rclpy.spin_once(node)  # 处理 ROS2 事件
            node.step()  # 执行主逻辑
            # 主要耗时在计算可视范围上，计算一次大概0.7s

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
