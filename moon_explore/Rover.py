import numpy as np
import math

try:
    from .AStar import AStarPlanner
    from .Utils import Setting, Pose2D, CandidatePoint, Contour
except:
    from AStar import AStarPlanner
    from Utils import Setting, Pose2D, CandidatePoint, Contour

from typing import List
from numpy.typing import NDArray


class Rover:
    def __init__(self, shared_mask, shared_ob_mask, planner: AStarPlanner) -> None:
        # **这两个 mask 是 Map 类传递过来的视图，只读且共享内存**
        self.mask: NDArray[np.bool_] = shared_mask  # True表示已探明
        self.obstacle_mask: NDArray[np.bool_] = shared_ob_mask

        self.planner = planner
        self.rover_pose = Pose2D(0, 0, 0)
        self.targetPoint: CandidatePoint | None = None
        self.targetPoint_mask: NDArray[np.bool_] | None = None  # 其他巡视器计算重叠衰减时使用，在分配目标的时候就计算
        self.switch: bool = False  # 设置超过一定探索比例后的切换标示量

    def cal_r_max(
        self,
        pose: Pose2D,
        theta_max: NDArray[np.float64],
        theta_min: NDArray[np.float64],
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
        （补充，有些情况下调成900才没出问题，没细看）（在Pose2D(26, 29, 0.7)下测试）
        （进一步修复边界点bug后，未对其重复测试）

        Args:
            pose (Pose2D): 用于精简计算用到的障碍物，只计算周围的；从其中取target_yaw
            theta_max (NDArray[np.float64]): 每个格子对应一个角度范围，这是最大值
            theta_min (NDArray[np.float64]): 每个格子对应一个角度范围，这是最小值
            r (NDArray[np.float64]): 各个格点到巡视器的距离
            deg_type (int, optional): 标识使用了何种角度范围，对应处理巡视器的视角边界处角度值

        Returns:
            NDArray[np.float64]: 每个格点的理论最大可视距离（考虑FOV、障碍）
        """
        # 考虑障碍物，每个角度上最大可视距离
        r_max = np.zeros_like(r)
        target_yaw = pose.yaw_deg180 if deg_type == 180 else pose.yaw_deg360
        range_min = target_yaw - Setting.FOV / 2
        range_max = target_yaw + Setting.FOV / 2

        # 设计上这个函数在两个不同的角度值范围约定下运行两次取并集，因此这里可以舍弃间断点导致的跳跃值
        if deg_type == 180:
            range_min = max(range_min, -180)
            range_max = min(range_max, 180.1)  # 感觉对于边界应该取大一点，但是没有测试
        elif deg_type == 360:
            range_min = max(range_min, 0)
            range_max = min(range_max, 360.1)
        else:
            raise NotImplementedError

        r_max[(range_min <= theta_min) & (theta_max <= range_max)] = Setting.MAX_RANGE * Setting.MAP_SCALE

        # 提取障碍物的角度范围和距离，只提取附近可能用得到的障碍物，减小计算量
        H, W = self.mask.shape
        grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W))
        ob_grid_x = grid_x[self.obstacle_mask]
        ob_grid_y = grid_y[self.obstacle_mask]

        # 计算障碍物是否在 (pos_x, pos_y) 附近的 120x120 区域内
        local_mask = (
            (ob_grid_x >= (pose.x - Setting.MAX_RANGE) * Setting.MAP_SCALE)
            & (ob_grid_x <= (pose.x + Setting.MAX_RANGE) * Setting.MAP_SCALE)
            & (ob_grid_y >= (pose.y - Setting.MAX_RANGE) * Setting.MAP_SCALE)
            & (ob_grid_y <= (pose.y + Setting.MAX_RANGE) * Setting.MAP_SCALE)
        )

        # 仅保留该区域内的障碍物数据
        ob_r = r[self.obstacle_mask][local_mask]
        ob_theta_max = theta_max[self.obstacle_mask][local_mask]
        ob_theta_min = theta_min[self.obstacle_mask][local_mask]

        # **先对障碍物按角度排序**
        #! 大改之后貌似没用到这个排序的特性了
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

        # **批量更新 r_max**
        for i in range(len(ob_r_sorted)):
            affected_indices = theta_sort_idx[
                (theta_min_sorted < ob_theta_max_sorted[i]) & (theta_max_sorted > ob_theta_min_sorted[i])
            ]
            np.minimum.at(r_max_flat, affected_indices, ob_r_sorted[i])

        return r_max

    def generate_sector_mask(self, pose: Pose2D) -> NDArray[np.bool_]:
        """计算扇形范围，考虑障碍"""
        if self.obstacle_mask[round(pose.y * Setting.MAP_SCALE), round(pose.x * Setting.MAP_SCALE)]:
            # 卡墙里就别算了
            return np.zeros_like(self.mask, dtype=np.bool_)

        # 创建坐标网格
        H, W = self.mask.shape
        x, y = np.meshgrid(np.arange(H), np.arange(W))
        dx = x - pose.x * Setting.MAP_SCALE
        dy = y - pose.y * Setting.MAP_SCALE

        # 计算极坐标
        r = np.sqrt(dx**2 + dy**2)

        # 对于角度，计算栅格四个顶点的角度，然后存储最大值与最小值
        theta_ur = np.degrees(np.arctan2(dy + 0.5, dx + 0.5))
        theta_ul = np.degrees(np.arctan2(dy + 0.5, dx - 0.5))
        theta_br = np.degrees(np.arctan2(dy - 0.5, dx + 0.5))
        theta_bl = np.degrees(np.arctan2(dy - 0.5, dx - 0.5))
        theta_max_o: NDArray[np.float64] = np.maximum.reduce([theta_ur, theta_ul, theta_br, theta_bl])
        theta_min_o: NDArray[np.float64] = np.minimum.reduce([theta_ur, theta_ul, theta_br, theta_bl])

        ##########################################################################################
        # 在-180 ~ 180计算一次，这次忽略x负半轴
        theta_max = theta_max_o.copy()  # 这一次的特殊值调整不能保留到下一次
        theta_min = theta_min_o.copy()  # 这一次的特殊值调整不能保留到下一次

        # 对于跳变的异常位置，不对其进行考虑
        theta_diff = theta_max - theta_min
        mask_wrap = theta_diff > 180
        theta_max[mask_wrap] = -10000  # 设置一个不可能用到的角度值避免碰撞
        theta_min[mask_wrap] = -10000
        r_max1 = self.cal_r_max(pose, theta_max, theta_min, r)

        ##########################################################################################
        # 将角度范围变换到0~360，再运算一次，忽略x正半轴
        theta_max_t = (theta_max_o + 360) % 360
        theta_min_t = (theta_min_o + 360) % 360
        # 新范围下原边界点处大小出现错乱
        theta_max = np.maximum(theta_max_t, theta_min_t)
        theta_min = np.minimum(theta_max_t, theta_min_t)
        # 对于跳变的异常位置，不对其进行考虑
        theta_diff = theta_max - theta_min
        mask_wrap = theta_diff > 180
        theta_max[mask_wrap] = -10000  # 设置一个不可能用到的角度值避免碰撞
        theta_min[mask_wrap] = -10000
        r_max2 = self.cal_r_max(pose, theta_max, theta_min, r, deg_type=360)

        # 取并集
        r_max = np.maximum(r_max1, r_max2)

        # 按照矩阵的样式，维度0对应y，维度1对应x
        mask = r <= (r_max + 2)  # 这样能稍微把边界处的障碍物变为可视的（不过要假设障碍别太小）
        return mask

    def generate_sector_mask_non_ob(self, pose: Pose2D) -> NDArray[np.bool_]:
        """
        用来给多巡视器的信息增益衰减，不考虑障碍，用的是最一开始开销小的方法
        """
        # 创建坐标网格
        H, W = self.mask.shape
        x, y = np.meshgrid(np.arange(H), np.arange(W))
        dx = x - pose.x * Setting.MAP_SCALE
        dy = y - pose.y * Setting.MAP_SCALE

        # 计算极坐标
        r = np.hypot(dx, dy)
        theta = np.arctan2(dy, dx) * (180.0 / np.pi)
        r_max = np.zeros_like(r)

        ##################################################################################
        target_yaw = pose.yaw_deg180
        range_min = target_yaw - Setting.FOV / 2
        range_max = target_yaw + Setting.FOV / 2
        r_max[(range_min <= theta) & (theta <= range_max)] = Setting.MAX_RANGE * Setting.MAP_SCALE

        if range_min < -180:
            range_min_new = range_min + 360
            r_max[(range_min_new <= theta)] = Setting.MAX_RANGE * Setting.MAP_SCALE

        if range_max > 180:
            range_max_new = range_max - 360
            r_max[(theta <= range_max_new)] = Setting.MAX_RANGE * Setting.MAP_SCALE

        mask = r <= r_max
        return mask

    def evaluate_candidate_points(
        self,
        canPoints: List[CandidatePoint],
        others_target: List[CandidatePoint],
        others_target_mask: List[NDArray[np.bool_]],
    ) -> None:
        """
        Args:
            canPoints (List[CandidatePoint]): 由 Map 处理，根据距离分配给其的候选点
            others_target (List[CandidatePoint]): 其他巡视器的当前目标，为了避免探索同一区域，对信息进行衰减
            others_target_mask (List[NDArray[np.bool_]]): 提前计算出来的其他巡视器目标区域
        """
        ALPHA = Setting.eval.ALPHA
        T_SEG = Setting.eval.T_SEG
        T_PATH = Setting.eval.T_PATH
        if len(canPoints) == 0:
            self.targetPoint = None
            return
        self.canPoints = canPoints
        # * 0.对于其他巡视器目标附近的点，衰减其信息量
        others_union = np.zeros_like(self.mask)
        for other_p in others_target_mask:
            others_union |= other_p

        for p in self.canPoints:
            factor = 1.0
            for other_p in others_target:
                if p.pose - other_p.pose <= Setting.MAX_RANGE:
                    esti_mask = self.generate_sector_mask_non_ob(p.pose)
                    esti_sum = np.count_nonzero(esti_mask)
                    overlap = np.count_nonzero((esti_mask & others_union))
                    factor = 1 - overlap / esti_sum
                    break
            p.seg_len *= factor

        # * 1.首先假设直线可达，规划一个用于计算路径成本的路径
        for p in self.canPoints:
            # p.path = self.planner.generate_straight_path(
            #     self.rover_pose, p.pose, self.planner.euclidean_dilated_least & ~self.mask
            # )
            # 真正路径规划用到的反倒是这个。。。感觉这个更安全点，也会让结果更一致
            p.path = self.planner.generate_straight_path(self.rover_pose, p.pose, self.planner.euclidean_dilated_least)

        # * 2.根据规划的路径计算运动成本
        for point in self.canPoints:
            assert point.path is not None
            path = point.path.path_pose

            p_last = self.rover_pose
            cost = 0
            for p in path:
                diff = p ^ p_last
                p_last = p

                time = 2 * diff.yaw_diff_rad / 0.1 + diff.dist / 0.1
                cost += time

            # 存在碰撞，估计时间加倍
            if point.path.collision:
                cost *= 2

            point.path_cost = math.exp(-ALPHA * cost)
            point.path = None
            # point.path_cost = cost
        path_costs = np.array([p.path_cost for p in self.canPoints])

        # * 3.目标区域的信息增益
        seg_lens = np.array([p.seg_len for p in self.canPoints])
        max_seg_len = np.max(seg_lens)

        # * 4.计算分数
        # 高于一定探索比例切换为就近探索
        if not self.switch:
            if Setting.eval.ENABLED_SWITCH:
                x, y = self.mask.shape
                rate = np.count_nonzero(self.mask) / x / y
                if rate > Setting.eval.RATIO_THRESHOLD:
                    self.switch = True
                    Setting.eval.BETA = Setting.eval.NEW_BETA

        # 要等到下一轮，才会用新的beta计算系数
        score_segs = T_SEG * seg_lens / max_seg_len
        score_paths = T_PATH * path_costs
        scores = score_segs + score_paths

        for point, score in zip(self.canPoints, scores):
            point.score = score

        # * 5.排序，随后规划真正可行的路径
        true_path_cnt = 0
        self.canPoints.sort(key=lambda p: p.score, reverse=True)
        for p in self.canPoints:
            p.path = self.planner.planning(self.rover_pose.x, self.rover_pose.y, p.pose, self.mask)
            if p.path is not None:
                true_path_cnt += 1
                if true_path_cnt >= 4:
                    break
        self.canPoints = [point for point in self.canPoints if point.path is not None]
        min_canPoints = self.canPoints[:true_path_cnt]
        for point in min_canPoints:
            assert point.path is not None
            path = point.path.path_pose

            p_last = self.rover_pose
            cost = 0
            for p in path:
                diff = p ^ p_last
                p_last = p

                time = 2 * diff.yaw_diff_rad / 0.1 + diff.dist / 0.1
                cost += time

            path_cost = math.exp(-ALPHA * cost)
            point.score = T_SEG * point.seg_len / max_seg_len + T_PATH * path_cost
        # 规划四个真的路径，再重新评估一下路径成本
        min_canPoints.sort(key=lambda p: p.score, reverse=True)
        self.targetPoint = min_canPoints[0]
        self.targetPoint_mask = self.generate_sector_mask_non_ob(self.targetPoint.pose)
        # 这里提前计算好预估的遮罩，其他巡视器衰减时直接取用
