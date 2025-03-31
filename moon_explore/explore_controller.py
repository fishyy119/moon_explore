import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist, Pose
import sys, math, time
from pathlib import Path
import numpy as np


# sys.path.append(Path(__file__).resolve().parent)
sys.path.append("/home/yyy/miniconda3/envs/moon_py310/lib/python3.10/site-packages")
from moon_explore.Pose2D import Pose2D
from moon_explore.Map import Map
from moon_explore.Viewer import MaskViewer
from moon_explore.AStar import AStarPlanner


class FSM:
    STOP = 1
    ROTATE = 2
    DRIVE = 3

    def __init__(self):
        self.current = self.STOP

    def next_state(self):
        if self.current == self.STOP:
            self.current = self.ROTATE
        elif self.current == self.ROTATE:
            self.current = self.DRIVE
        elif self.current == self.DRIVE:
            self.current = self.STOP

    @property
    def state(self):
        return self.current


class ExploreController(Node):
    def __init__(self):
        self.LOG = lambda msg: self.get_logger().info(msg)
        self.last_dist = 1000
        super().__init__("explore_controller")
        package_share_directory = get_package_share_directory("moon_explore")
        NPY_ROOT = Path(package_share_directory) / "resource"
        self.map = Map(map_file=str(NPY_ROOT / "map_passable.npy"))
        self.viewer = MaskViewer(self.map)
        self.fsm = FSM()

        self.subscription = self.create_subscription(Pose, "robot_pose_1", self.robot_pose_callback, 10)
        self.publisher = self.create_publisher(Twist, "cmd_vel_1", 10)

    def robot_pose_callback(self, msg: Pose):
        self.pose_now: Pose2D = Pose2D.from_pose_msg(
            msg.position.x, msg.position.y, msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        )
        self.map.rover_move(self.pose_now)
        self.viewer.update()
        self.viewer.show()

    def plan(self):
        self.map.get_contours()
        pose_targets = self.map.cal_canPose()
        for pose in pose_targets:
            planner = AStarPlanner(1.0, self.map)
            self.get_logger().info(f"{self.pose_now.x}, {self.pose_now.y}, {pose.x}, {pose.y}")
            path = planner.planning(self.pose_now.x, self.pose_now.y, pose.x, pose.y)
            if path is not None:
                self.path = path
                self.viewer.plot_contours()
                self.viewer.update()
                self.viewer.plot_path(path)
                self.viewer.show()
                self.LOG(f"Path: {path}")
                return path

    def step(self) -> None:
        """执行状态更新"""
        if self.pose_now is None:
            return

        twist = Twist()

        if self.fsm.state == FSM.STOP:
            if self.plan() is not None:
                self.fsm.next_state()  # 切换到 ROTATE
                self.current_target = self.path[1]  # 获取路径第一个点
                self.idx_now = 1

        elif self.fsm.state == FSM.ROTATE:
            if self.current_target is not None:
                self.last_dist = 1000
                yaw_target = np.arctan2(
                    self.current_target[1] - self.pose_now.y, self.current_target[0] - self.pose_now.x
                )
                yaw_error = yaw_target - self.pose_now.yaw_rad
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
                if abs(yaw_error) < 0.1:  # 允许误差
                    self.fsm.next_state()  # 切换到 DRIVE
                else:
                    twist.angular.z = 0.5 * yaw_error  # 旋转对准方向
                    twist.angular.z = max(-0.1, min(0.1, twist.angular.z))
                self.LOG(f"target:{self.current_target},{math.degrees(yaw_target)} ; now:{self.pose_now}")

        elif self.fsm.state == FSM.DRIVE:
            if self.current_target is not None:
                dist = math.sqrt((self.path[1, 0] - self.pose_now.x) ** 2 + (self.path[1, 1] - self.pose_now.y) ** 2)
                self.get_logger().info(f"{dist}")
                if (
                    dist < 0.2 or dist >= self.last_dist + 0.2 or (self.idx_now >= len(self.path) - 1 and dist < 3)
                ):  # 允许误差
                    if self.idx_now >= len(self.path) - 1:
                        self.fsm.current = FSM.STOP  # 到达终点，重新规划
                    else:
                        self.idx_now += 1
                        self.current_target = self.path[self.idx_now]
                        self.fsm.current = FSM.ROTATE

                self.last_dist = min(dist, self.last_dist)
                twist.linear.x = 0.1 if dist >= 0.5 else 0.05  # 行驶
                yaw_target = np.arctan2(
                    self.current_target[1] - self.pose_now.y, self.current_target[0] - self.pose_now.x
                )
                yaw_error = yaw_target - self.pose_now.yaw_rad
                yaw_error = (yaw_error + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
                if abs(yaw_error) > 0.01:
                    twist.angular.z = 0.5 * yaw_error
                    twist.angular.z = max(-0.01, min(0.01, twist.angular.z))
                else:
                    twist.angular.z = 0.0

        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = ExploreController()
    try:
        while rclpy.ok():
            # start_time = time.time()

            rclpy.spin_once(node)  # 处理 ROS2 事件
            node.step()  # 执行主逻辑

            # **根据上次执行耗时调整睡眠时间**
            # sleep_time = max(0.01, node.last_execution_time)  # 最小间隔 10ms
            # elapsed = time.time() - start_time
            # if sleep_time > elapsed:
            #     time.sleep(sleep_time - elapsed)
            # time.sleep(0.01)

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
