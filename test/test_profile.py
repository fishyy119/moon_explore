from line_profiler import LineProfiler
from moon_explore.Map import Map
from moon_explore.Utils import Pose2D
from pathlib import Path

if __name__ == "__main__":
    NPY_ROOT = Path(__file__).parent.parent / "resource"
    map = Map(map_file=str(NPY_ROOT / "map_passable.npy"), num_rovers=1)
    map.rover_init(Pose2D(27, 25.2472, 88.6, deg=True), 0)

    lp = LineProfiler()
    # lp_wrapper = lp(map.rovers[0].generate_sector_mask_non_ob)

    lp.add_function(map.rovers[0].cal_r_max)
    lp_wrapper = lp(map.rovers[0].generate_sector_mask)
    lp_wrapper(Pose2D(28, 25, 160, deg=True))
    lp.print_stats()
