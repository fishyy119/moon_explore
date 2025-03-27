import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Pose2D import Pose2D
from Map import Map

from typing import Tuple, List, Callable
from numpy.typing import NDArray


class MaskViewer:
    def __init__(self, map_instance: Map):
        self.map = map_instance

    def show_mask(self) -> None:
        fig, ax = plt.subplots()
        im = ax.imshow(self.map.mask, cmap="gray_r", origin="lower")
        boundary_points = np.argwhere(self.map.boundary)
        # ax.scatter(
        #     boundary_points[:, 1], boundary_points[:, 0], color="cyan", s=5, label="Boundary Points"
        # )  # 画出边界点

        ob_points = np.argwhere(self.map.obstacle_mask)
        ax.scatter(ob_points[:, 1], ob_points[:, 0], color="cyan", s=5, label="Boundary Points")  # 障碍点

        contours = self.map.contours

        for contour in contours:
            curvature = self.map.curvature_discrete(contour)
            peaks_idx = self.map.detect_peaks(curvature, contour)
            ax.scatter(contour[:, 1], contour[:, 0], color="red", s=5, label="Boundary Points")  # 画出边界点
            ax.scatter(
                contour[0, 1], contour[0, 0], color="red", marker="*", s=150, label="Boundary Points"
            )  # 画出头部
            # for i, (x, y) in enumerate(contour):
            #     ax.text(x, y, str(i), color="red", fontsize=8, ha="center", va="center")
            ax.scatter(contour[peaks_idx, 1], contour[peaks_idx, 0], color="blue", marker="*", s=50, label="Peaks")
            # contour = loess_smooth(contour)
            # curvature = self.map.compute_curvature(contour)  # 计算曲率
            plt.figure(figsize=(8, 5))
            plt.plot(curvature, color="red", linewidth=2)
            plt.title("Curvature vs Point Position on Contour")
            plt.xlabel("Point Index Along Contour")
            plt.ylabel("Curvature")
            plt.grid(True)

            # 绘制平滑曲线，并用曲率颜色表示
            # points = contour.reshape(-1, 2)
            # curvature_normalized = (curvature - curvature.min()) / (curvature.ptp() + 1e-8)  # 归一化
            # for i in range(len(points) - 1):
            #     segment = points[i : i + 2]  # 曲线段
            #     color = plt.cm.jet(curvature_normalized[i])  # type: ignore # 曲率颜色
            #     ax.plot(segment[:, 0], segment[:, 1], color=color, linewidth=2)
            # 绘制每条轮廓
            # ax.plot(contour[:, 0], contour[:, 1], linewidth=1, color="cyan")  # 画出轮廓线

        ax.set_title("Sector Mask Viewer")
        # plt.colorbar(im, ax=ax)
        plt.show()

    def show_anime(self) -> None:
        self.fig, self.ax = plt.subplots()

        # 初始化掩码显示
        self.im = self.ax.imshow(self.map.mask, cmap="gray_r", origin="lower", animated=True)
        self.ax.set_title("Sector Mask Viewer")
        self.fig.colorbar(self.im, ax=self.ax)

        # 设置 30Hz 更新频率
        self.ani = animation.FuncAnimation(self.fig, self.update, interval=33.3, blit=True, cache_frame_data=False)
        plt.show()

    def update(self, _):
        """每帧更新 `map.mask` 可视化"""
        self.im.set_data(self.map.mask)  # 直接读取 map.mask
        return (self.im,)  # `blit=True` 需要返回可更新的对象
