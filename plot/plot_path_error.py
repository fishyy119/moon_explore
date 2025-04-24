import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from plot_utils import plot_ob_mask

# ! 这些记录里面都不是真值
# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250424_175450_ab")  # xyzstd: 0.5 0.5 0.01
# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_124817_ab")  # xyzstd: 0.2 0.2 0.01
# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_141515_ab_2")  # xyzstd: 0.1 0.1 0 / 1.9min
# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_150330_ab_2")  # xyzstd: 0.05 0.05 0 / 1.9min

# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_123200")
# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_142736_2")
# !

CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_155457_bag1")  # xyz: 0.05, 0.05, 0.01
# CSV_ROOT = Path("/home/yyy/moon_R2023/Data/img/20250425_165849")
csv1 = CSV_ROOT / "record_1.csv"
txt1 = CSV_ROOT / "SLAM_ab.txt"
txt1 = CSV_ROOT / "KeyFrameTrajectory_1.txt"

# csv2 = CSV_ROOT / "record_1.csv"
# txt2 = CSV_ROOT / "SLAM_no.txt"


def func(csv1, txt1):
    # 读取真值（带四元数）CSV
    truth_df = pd.read_csv(csv1)

    # 读取估计数据（空格分隔）
    estimate_data = np.loadtxt(txt1)  # shape: (N, 8)
    estimate_timestamps = estimate_data[:, 0]
    estimate_positions = estimate_data[:, 1:4]

    t0_est = estimate_timestamps[0]  # 起始时间
    origin_relative = estimate_positions[0]  # 应该是 [0, 0, 0]

    def find_closest_idx(arr, val):
        return np.argmin(np.abs(arr - val))

    t_truth = truth_df["time"].values
    idx_closest = find_closest_idx(t_truth, t0_est)

    # 提取齐次变换矩阵 T_abs0
    def get_transform(row):
        t = np.array([row["position_x"], row["position_y"], row["position_z"]])
        q = np.array([row["orientation_x"], row["orientation_y"], row["orientation_z"], row["orientation_w"]])
        rot = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        return T

    def get_transform_txt(row):
        t = np.array([row[1], row[2], row[3]])
        q = np.array([row[4], row[5], row[6], row[7]])
        rot = R.from_quat(q).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = t
        return T

    T_aw = get_transform(truth_df.iloc[idx_closest])
    T_wa = np.linalg.inv(T_aw)

    positions_truth_relative = []
    positions_truth_absolute = []
    for _, row in truth_df.iterrows():
        T_abs = get_transform(row)
        T_rel = T_wa @ T_abs
        pos_rel = T_rel[:3, 3]  # 取平移部分
        positions_truth_relative.append(pos_rel)
        positions_truth_absolute.append(T_abs[:3, 3])
    positions_truth_relative = np.array(positions_truth_relative)  # Nx3
    positions_truth_absolute = np.array(positions_truth_absolute)

    positions_esti_absolute = []
    for row in estimate_data:
        T_rel = get_transform_txt(row)
        T_ac = T_aw @ T_rel
        pos = T_ac[:3, 3]
        positions_esti_absolute.append(pos)
    positions_esti_absolute = np.array(positions_esti_absolute)

    # 提取真值时间和相对xy
    truth_xy_time = truth_df["time"].values
    # truth_x = positions_truth_relative[:, 0]
    # truth_y = positions_truth_relative[:, 2]
    truth_x = positions_truth_absolute[:, 0]
    truth_y = positions_truth_absolute[:, 1]

    esti_x = positions_esti_absolute[:, 0]
    esti_y = positions_esti_absolute[:, 1]  # 绝对坐标系没有SLAM坐标系那个问题

    # 构造插值函数
    fx = interp1d(truth_xy_time, truth_x, kind="linear")
    fy = interp1d(truth_xy_time, truth_y, kind="linear")
    mask = estimate_timestamps < max(truth_xy_time)
    estimate_timestamps = estimate_timestamps[mask]

    # 在估计时间上插值
    x_interp = fx(estimate_timestamps)
    y_interp = fy(estimate_timestamps)

    # 误差计算
    xy_est = estimate_positions[:, :3]  # 只用 x, y, SLAM用的坐标系，要取13位？
    xy_true_interp = np.stack([x_interp, y_interp], axis=1)
    # errors = np.linalg.norm(xy_est - xy_true_interp, axis=1)

    # return estimate_timestamps, esti_x - x_interp, esti_y - y_interp
    return estimate_timestamps, x_interp, y_interp, esti_x[mask], esti_y[mask]


fig, ax = plt.subplots()

error1 = func(csv1, txt1)
# error2 = func(csv2, txt2)


NPY_ROOT = Path(__file__).parent.parent / "resource"
map_file = NPY_ROOT / "map_passable.npy"
ob_mask = np.load(map_file)
plot_ob_mask(ob_mask, alpha=1, ax=ax)

ax.plot(error1[3] * 10, error1[4] * 10, label="slam")
ax.plot(error1[1] * 10, error1[2] * 10, label="truth")

# ax.plot(estimate_timestamps, xy_est, label="XY Error")
# ax.plot(estimate_timestamps, xy_true_interp, label="XY Error")

# ax.plot(error1[0], error1[1], label="truth 1")
# ax.plot(error1[0], error1[3] - error1[1], label="ab")
# ax.plot(error1[0], error1[4] - error1[2], label="ab")

# ax.plot(error2[0], error2[1], label="truth 2")
# ax.plot(error2[0], error2[3] - error2[1], label="no-ab")
# ax.plot(error2[0], error2[4] - error2[2], label="no-ab")

# ax.xlabel("Time")
# ax.ylabel("Position Error (m)")
# plt.title("XY Position Error Over Time")
# plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
