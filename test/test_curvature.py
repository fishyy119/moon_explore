import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import least_squares


def compute_curvature(self, contour):
    """计算曲率"""
    x = contour[:, 0]
    y = contour[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (np.power(dx**2 + dy**2, 1.5) + 1e-8)
    return curvature


def geometric_circle_fitting(points):
    """
    基于几何误差的圆拟合（最小化点到圆的距离）
    :param points: 5 个 (x, y) 坐标点
    :return: (xc, yc, R) 拟合圆的圆心和半径
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # 计算点的均值作为初始圆心
    x_m, y_m = np.mean(x), np.mean(y)
    # x_m, y_m = 0, 0

    # 目标函数：最小化每个点到圆的距离偏差
    def residuals(params):
        xc, yc, R = params
        return np.sqrt((x - xc) ** 2 + (y - yc) ** 2) - R

    # 初始估计值
    R0 = np.mean(np.sqrt((x - x_m) ** 2 + (y - y_m) ** 2))
    res = least_squares(residuals, [x_m, y_m, R0])

    return res.x  # 返回优化后的 (xc, yc, R)


def fit_circle(points):
    """
    最小二乘法拟合圆
    :param points: 5 个 (x, y) 坐标点
    :return: (xc, yc, R) 拟合圆的圆心和半径
    """
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # 构造矩阵 A 和向量 B
    A = np.column_stack((x, y, np.ones_like(x)))
    B = x**2 + y**2

    # 使用最小二乘法求解 D, E, F
    C, residuals, rank, s = np.linalg.lstsq(A, -B, rcond=None)
    D, E, F = C

    # 计算圆心坐标
    xc = -D / 2
    yc = -E / 2

    # 计算半径
    R = np.sqrt(xc**2 + yc**2 - F)

    return xc, yc, R


# 硬编码 5 个测试点
points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 4]])

# points = np.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 4], [6, 5], [7, 6]])

# points = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])

points = np.array([[1, 0], [2, 0], [3, 1], [4, 0], [5, 0]])

# 拟合圆
# xc, yc, R = fit_circle(points)
xc, yc, R = geometric_circle_fitting(points)
# 绘制结果
plt.figure(figsize=(6, 6))
plt.scatter(points[:, 0], points[:, 1], color="blue", label="Given Points")  # 绘制输入点
plt.scatter([xc], [yc], color="red", marker="x", s=100, label="Fitted Center")  # 绘制拟合圆心

# 绘制拟合圆
theta = np.linspace(0, 2 * np.pi, 100)
x_circle = xc + R * np.cos(theta)
y_circle = yc + R * np.sin(theta)
plt.plot(x_circle, y_circle, linestyle="--", color="green", label="Fitted Circle")

# 显示图例
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Fitted Circle from Given 5 Points")
plt.axis("equal")
plt.grid()
plt.show()
