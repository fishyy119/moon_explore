import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import threading
import time
from Map import Map, MaskViewer
from Pose2D import Pose2D

if __name__ == "__main__":
    map = Map()  # 你的 Map 类
    viewer = MaskViewer(map)

    threading.Thread(target=viewer.show_anime, daemon=True).start()

    x, y, theta = 20, 30, 0  # 初始位置
    while True:
        pose = Pose2D(x, y, theta)
        map.rover_move(pose)  # 更新 mask

        # 模拟运动轨迹
        x += 0.4
        if x > 50:  # 到达边界后重置
            x = 20
            y += 5
            if y > 50:
                break

        time.sleep(1 / 30)  # 30Hz 更新
