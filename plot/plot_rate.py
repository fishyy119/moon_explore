import matplotlib.pyplot as plt
from plot_utils import RateCSV, plot_rate_csv

from typing import List


if __name__ == "__main__":
    csvs: List[RateCSV] = [
        RateCSV("rate_0422_111633.csv", r"$\beta = 0$"),  # "1.5/b=0/a=0.8"
        # RateCSV("rate_0423_171945.csv", "1.5/b=0/a=0.8 - a"),
        # RateCSV("rate_0423_133631.csv", "1.5/b=0/a=0.8 - 2"),
        # RateCSV("rate_0423_154438.csv", "1.5/b=0.3/a=0.8"),
        # RateCSV("rate_0423_163538.csv", "1.5/b=0.3/a=0.8 - a"),
        RateCSV("rate_0422_091532.csv", r"$\beta = 0.4$"),  # "1.5/b=0.4/a=0.8"
        # RateCSV("rate_0422_143957.csv", "1.5/b=0.4/a=0.8/new_b=40/0"),  # "1.5/b=0.4/a=0.8/new_b=40/0"
        RateCSV("rate_0422_162350.csv", r"$\beta = 0.4$"),  # "1.5/b=0.4/a=0.8/new_b=40/0.1"
        RateCSV("rate_0424_162835.csv", r"$\beta = 0.4$ 双巡视器"),  # "1.5/b=0.4/a=0.8/double"
    ]

    fig, ax = plt.subplots()
    for csv in csvs:
        plot_rate_csv(csv, ax)

    plt.rcParams["font.sans-serif"] = ["SimSun"]  # 设置中文字体为宋体
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams.update({"font.size": 10})  # 设置字体大小

    plt.xlabel("时间 (s)")
    plt.ylabel("探索率 (%)")
    plt.legend()
    plt.grid(True, axis="both")
    plt.tight_layout()
    plt.show()
