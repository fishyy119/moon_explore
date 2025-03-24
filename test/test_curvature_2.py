import enum
import numpy as np
import matplotlib.pyplot as plt


def curvature_discrete(points):
    """
    计算离散曲率
    :param points: 一组二维点 (N, 2)
    :return: 各个中间点的曲率列表
    """
    curvatures = []
    w = 2

    points = np.asarray(points)  # 确保输入是 NumPy 数组
    n = len(points)

    # 计算向量（P_{i-2} → P_i）和（P_{i+2} → P_i）
    v1 = points[1 : -w - 1] - points[: -w - 2]  # P_{i-2} -> P_i
    v2 = points[w + 2 :] - points[w + 1 : -1]  # P_{i+2} -> P_i

    # 计算夹角 θ（使用点积公式）
    dot_product = np.sum(v1 * v2, axis=1)  # 向量点积
    norm_v1 = np.linalg.norm(v1, axis=1)  # |v1|
    norm_v2 = np.linalg.norm(v2, axis=1)  # |v2|
    cos_theta = dot_product / (norm_v1 * norm_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 避免浮点误差
    theta = np.arccos(cos_theta)  # 转角

    # 计算近似弧长（取 4 条线段的总长度）
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)  # 所有相邻点的距离
    segment_lengths = (segment_lengths[:-1] + segment_lengths[1:]) / 2
    # ds = (segment_lengths[:-3] + segment_lengths[1:-2] + segment_lengths[2:-1] + segment_lengths[3:]) / 2
    ds = segment_lengths[:-2] + segment_lengths[1:-1] + segment_lengths[2:]

    # 计算曲率 κ = θ / ds
    curvatures = theta / ds

    return curvatures

    for i in range(2, len(points) - 2):  # 遍历每个三连点
        A, B, C, D = points[i - 2], points[i - 1], points[i + 1], points[i + 2]

        # 计算向量
        v1 = A - B
        v2 = C - D

        # 计算夹角
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差导致 acos 越界
        theta = np.arccos(cos_theta)  # 转角

        # 计算步长（近似弧长）
        ds = np.linalg.norm(C - A)

        # 计算曲率
        kappa = theta / ds
        curvatures.append(kappa)

    return np.array(curvatures)


# 测试点（可修改）
points = np.array([[1, 2], [2, 3], [3, 3.5], [4, 3], [5, 2]])
# points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 4]])
points = np.array([[1, 0], [2, 0], [3, 0], [4, 1], [5, 0], [6, 0], [7, 0]])

# 计算曲率
curvatures = curvature_discrete(points)

# 绘制点
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Given Points")

# 在点之间绘制折线
plt.plot(points[:, 0], points[:, 1], linestyle="-", color="black", alpha=0.5)

# 显示曲率值
for i, c in enumerate(curvatures):
    plt.text(points[i + 2, 0], points[i + 2, 1] + 0.2, f"{c:.3f}", fontsize=10, color="red")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Discrete Curvature Computation")
plt.legend()
plt.axis("equal")
plt.grid()
plt.show()
