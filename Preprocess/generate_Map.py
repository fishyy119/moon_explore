import numpy as np
from pathlib import Path

from typing import Tuple, List
from numpy.typing import NDArray


def calculate_slope_aspect(dem: NDArray) -> Tuple[NDArray, NDArray]:
    """
    计算坡度和坡向

    Args:
        dem (NDArray): 数字高程图

    Returns:
        Tuple[NDArray, NDArray]: 坡度图和坡向图
    """
    cell_size: float = 0.1
    # 在外围扩充一圈为0的边缘
    dem_padded = np.pad(dem, pad_width=1, mode="constant", constant_values=0)
    rows, cols = dem_padded.shape
    grad_x = np.zeros((rows - 2, cols - 2))
    grad_y = np.zeros((rows - 2, cols - 2))

    # 遍历每个 3x3 窗口
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 获取 3x3 窗口
            window = dem_padded[i - 1 : i + 2, j - 1 : j + 2]

            grad_x[i - 1, j - 1] = (
                (window[0, 2] + 2 * window[1, 2] + window[2, 1]) - (window[0, 0] + 2 * window[1, 0] + window[0, 1])
            ) / (8 * cell_size)

            grad_y[i - 1, j - 1] = (
                (window[2, 0] + 2 * window[2, 1] + window[2, 2]) - (window[0, 0] + 2 * window[0, 1] + window[0, 2])
            ) / (8 * cell_size)

    # 计算坡度
    slope = np.sqrt(grad_x**2 + grad_y**2)

    # 计算坡向
    aspect = np.arctan2(-grad_y, grad_x)

    return slope, aspect


def calculate_passability(slope: NDArray, passable_threshold: List[float] = [5, 15, 20]) -> NDArray:
    """
    根据坡度计算可通行性，分四档

    Args:
        slope (NDArray): 坡度图，存储正切值
        passable_threshold (List[float], optional): 四档的三个中间点，递增，左开右闭

    Returns:
        NDArray: 可通行性地图，四档分别为0/1/2/3
    """
    slope_deg = np.degrees(np.arctan(slope))
    passability = np.ones_like(slope_deg).astype(np.int8)
    passability = np.digitize(slope_deg, passable_threshold, right=True).astype(np.int8)

    return passability


def main() -> None:
    """
    主函数，读取DEM文件，计算坡度和坡向，并保存结果
    """
    NPY_ROOT = Path(__file__).parent.parent / "resource"
    input_file: str = str(NPY_ROOT / "map_truth.npy")
    slope_output_file: str = str(NPY_ROOT / "map_slope.npy")
    aspect_output_file: str = str(NPY_ROOT / "map_aspect.npy")
    passable_output_file: str = str(NPY_ROOT / "map_passable.npy")

    # 加载DEM数据
    dem: NDArray = np.load(input_file)

    # 计算坡度和坡向
    slope, aspect = calculate_slope_aspect(dem)
    passable = calculate_passability(slope)
    passable = (passable >= 1.5).astype(np.bool_)

    # 保存结果
    np.save(slope_output_file, slope)
    np.save(aspect_output_file, aspect)
    np.save(passable_output_file, passable.T)

    print(f"坡度图已保存到 {slope_output_file}")
    print(f"坡向图已保存到 {aspect_output_file}")
    print("可通行性地图已保存到", passable_output_file)


if __name__ == "__main__":
    main()
