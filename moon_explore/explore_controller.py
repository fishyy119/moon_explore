import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, Pose
import sys, math, time
from pathlib import Path
import numpy as np

sys.path.append("/home/yyy/miniconda3/envs/moon_py310/lib/python3.10/site-packages")
from moon_explore.Pose2D import Pose2D
from moon_explore.Map import Map
from moon_explore.Viewer import MaskViewer
from moon_explore.AStar import AStarPlanner

from enum import Enum, auto
from typing import Optional


class State(Enum):
    INIT = auto()  # 初始化，自转一圈
    STOP = auto()  # 停止，进行规划
    ROTATE = auto()  # 旋转，对齐直线轨迹
    DRIVE = auto()  # 直行，内部也有反馈式方向微调


class ExploreController(Node):
    def __init__(self):
        self.last_dist = 1000
        super().__init__("explore_controller")
        self.LOG = lambda msg: self.get_logger().info(msg)
        package_share_directory = get_package_share_directory("moon_explore")
        NPY_ROOT = Path(package_share_directory) / "resource"
        self.map = Map(map_file=str(NPY_ROOT / "map_passable.npy"))
        self.viewer = MaskViewer(self.map)
        self.fsm_state: State = State.STOP
        self.pose_now: Optional[Pose2D] = None

        self.subscription = self.create_subscription(Pose, "robot_pose_1", self.robot_pose_callback, 10)
        self.publisher = self.create_publisher(Twist, "cmd_vel_1", 10)

    def robot_pose_callback(self, msg: Pose):
        last_pose = self.pose_now
        self.pose_now = Pose2D.from_pose_msg(
            msg.position.x, msg.position.y, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        if last_pose is None:
            self.map.rover_init(self.pose_now)
        else:
            self.map.rover_move(self.pose_now)
        self.viewer.update()

    def step(self, index: int) -> None:
        """每个巡视器独立进行指令发送，index表示编号"""
        twist = Twist()  # 待发送消息
        if self.pose_now is None:
            return

        if self.fsm_state != State.DRIVE:
            self.last_dist = 1000  # DRIVE用到了这个，要初始化一个比较大的数
        match self.fsm_state:
            case State.INIT:
                pass  # 初始化自转一圈，在Map中实现，没有真的控制转一圈

            case State.STOP:
                self.LOG("下一轮规划")
                self.map.step()
                self.viewer.update(mode=MaskViewer.UpdateMode.CONTOUR)
                self.path = self.map.canPoints[0].path

                assert self.path is not None
                self.viewer.plot_path(self.path)
                self.fsm_state = State.DRIVE
                self.targets = self.path.path_pose
                self.current_target = self.targets[0]
                self.idx_now = 0

            case State.DRIVE:
                diff = self.pose_now - self.current_target
                dist = diff.dist
                self.get_logger().info(f"DRIVE: {dist:.2f}")
                if dist < 0.2 or dist >= self.last_dist + 0.2:  # 允许误差
                    self.fsm_state = State.ROTATE

                self.last_dist = min(dist, self.last_dist)
                twist.linear.x = 0.1 if dist >= 0.5 else 0.05  # 行驶
                yaw_target = np.arctan2(
                    self.current_target.y - self.pose_now.y, self.current_target.x - self.pose_now.x
                )
                yaw_error = yaw_target - self.pose_now.yaw_rad
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
                if abs(yaw_error) > 0.01:
                    twist.angular.z = 0.5 * yaw_error
                    twist.angular.z = max(-0.01, min(0.01, twist.angular.z))
                else:
                    twist.angular.z = 0.0

            case State.ROTATE:
                yaw_target = self.current_target.yaw_rad
                yaw_error = yaw_target - self.pose_now.yaw_rad
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
                if abs(yaw_error) < 0.1:  # 允许误差
                    if self.idx_now >= len(self.targets) - 1:
                        self.fsm_state = State.STOP  # 到达终点，重新规划
                    else:
                        self.idx_now += 1
                        self.current_target = self.targets[self.idx_now]
                        self.fsm_state = State.DRIVE
                else:
                    twist.angular.z = 0.5 * yaw_error  # 旋转对准方向
                    twist.angular.z = max(-0.1, min(0.1, twist.angular.z))
                self.LOG(f"ROTATE: {self.pose_now.yaw_deg360:.2f} -> {self.current_target.yaw_deg360:.2f}")

        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ExploreController()
    try:
        while rclpy.ok():
            # start_time = time.time()

            rclpy.spin_once(node)  # 处理 ROS2 事件
            node.step(0)  # 执行主逻辑
            node.step(1)
            # 主要耗时在计算可视范围上，计算一次大概1.4s

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

    # rclpy.spin(node)
    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == "__main__":
    main()
