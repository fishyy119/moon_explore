import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from generate_Map import calculate_passability

from GLOABL import *
from typing import Tuple, List, Callable
from numpy.typing import NDArray


CAL_AND_PLOT_PASSABILITY: Callable[[NDArray, List[float]], None] = lambda slope, threshold: plot_passability(
    calculate_passability(slope, threshold), threshold
)


def plot_maps(dem: NDArray, slope: NDArray, aspect: NDArray) -> None:
    """
    绘制DEM、坡度图和坡向图

    Args:
        dem (NDArray): 数字高程图
        slope (NDArray): 坡度图
        aspect (NDArray): 坡向图
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(dem, cmap="terrain")
    axes[0].set_title("DEM")
    axes[0].axis("on")

    slope_deg = np.degrees(np.arctan(slope))
    slope_deg_clap = np.clip(slope_deg, 0, 45)
    im = axes[1].imshow(slope_deg_clap, cmap="viridis")
    axes[1].set_title("Slope (degrees)")
    axes[1].axis("on")
    fig.colorbar(im, ax=axes[1], orientation="vertical", label="Slope (degrees)")

    aspect_deg = np.degrees(aspect)
    axes[2].imshow(aspect_deg, cmap="twilight")
    axes[2].set_title("Aspect")
    axes[2].axis("on")

    plt.tight_layout()
    plt.show()


def plot_passability(passability: NDArray, passable_threshold: List[float] = [5, 10, 20]) -> None:
    """
    绘制可通行性地图

    Args:
        passability (NDArray): 可通行性地图，分四档
        passable_threshold (List[float], optional): 四档的分割点，与计算所用到的参数一致
    """
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    color_list = ["darkgreen", "lightgreen", "orange", "red"]
    cmap = plt.cm.colors.ListedColormap(color_list)  # type: ignore
    bounds = [0, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)  # type: ignore

    ax2.imshow(passability, cmap=cmap, norm=norm, origin="lower")
    ax2.set_title("Passability (Slope)")

    # 创建图例
    legend_labels = [
        f"0-{passable_threshold[0]}°",
        f"{passable_threshold[0]}-{passable_threshold[1]}°",
        f"{passable_threshold[1]}-{passable_threshold[2]}°",
        f">{passable_threshold[2]}°",
    ]
    legend_patches = [Patch(color=color, label=label) for color, label in zip(color_list, legend_labels)]
    ax2.legend(handles=legend_patches, loc="upper right", title="Slope Categories")

    plt.tight_layout()
    plt.show()


def main() -> None:
    dem: NDArray = np.load(NPY_ROOT / "map_truth.npy")
    slope_map: NDArray = np.load(NPY_ROOT / "map_slope.npy")
    aspect_map: NDArray = np.load(NPY_ROOT / "map_aspect.npy")

    # plot_maps(dem, slope_map, aspect_map)
    CAL_AND_PLOT_PASSABILITY(slope_map, [5, 15, 20])


if __name__ == "__main__":
    main()
