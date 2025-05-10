from plot_utils import *

from typing import List


if __name__ == "__main__":
    csvs: List[RateCSV] = [
        # RateCSV("rate_0422_111633.csv", r"$\beta = 0$"),  # "1.5/b=0/a=0.8"
        # RateCSV("rate_0423_171945.csv", "1.5/b=0/a=0.8 - a"),
        # RateCSV("rate_0423_133631.csv", "1.5/b=0/a=0.8 - 2"),
        # RateCSV("rate_0423_154438.csv", "1.5/b=0.3/a=0.8"),
        # RateCSV("rate_0423_163538.csv", "1.5/b=0.3/a=0.8 - a"),
        # RateCSV("rate_0422_091532.csv", r"$\beta = 0.4$"),  # "1.5/b=0.4/a=0.8"
        # RateCSV("rate_0422_143957.csv", "1.5/b=0.4/a=0.8/new_b=40/0"),  # "1.5/b=0.4/a=0.8/new_b=40/0"
        # RateCSV("rate_0422_162350.csv", r"$\beta = 0.4$"),  # "1.5/b=0.4/a=0.8/new_b=40/0.1"
        # RateCSV("rate_0424_162835.csv", r"$\beta = 0.4$ 双巡视器"),  # "1.5/b=0.4/a=0.8/double"
        # RateCSV("rate_0507_091436.csv"),
        # RateCSV("rate_0507_112309.csv"),
        # RateCSV("rate_0507_153709.csv"),
        # RateCSV("rate_0507_134315.csv"),
        # RateCSV("rate_0507_183100.csv"),
        # RateCSV("rate_0507_205930.csv", factor=1.5),
        # RateCSV("rate_0507_225836.csv", factor=1.5),
        # RateCSV("rate_0508_131952.csv"),
    ]

    # 起始位置1
    csvs1: List[RateCSV] = [
        # RateCSV("rate_0508_142851.csv", r"$\beta=0$"),
        RateCSV("rate_0509_031047.csv", r"$\beta=0$", "--", "black"),
        RateCSV("rate_0508_204418.csv", r"$\beta=0.1$"),
        RateCSV("rate_0508_183110.csv", r"$\beta=0.2$"),
        RateCSV("rate_0508_194216.csv", r"$\beta=0.3$"),
        RateCSV("rate_0508_223134.csv", r"$\beta=0.4$"),
        # RateCSV("rate_0509_015148.csv", r"$\beta=0.4$"),
    ]

    # 起始位置2
    csvs2: List[RateCSV] = [
        RateCSV("rate_0507_112309.csv", r"$\beta=0$", "--", "black", factor=1.5),
        RateCSV("rate_0507_134315.csv", r"$\beta=0.1$", factor=1.5),
        RateCSV("rate_0507_153709.csv", r"$\beta=0.2$", factor=1.5),
        RateCSV("rate_0507_183100.csv", r"$\beta=0.3$", factor=1.5),
        RateCSV("rate_0507_091436.csv", r"$\beta=0.4$", factor=1.5),
    ]

    # 多巡视器与单巡视器对比
    csvs3: List[RateCSV] = [
        RateCSV("rate_0508_223134.csv", r"单巡视器（时间轴压缩后）", factor=2),
        # RateCSV("rate_0422_162350.csv", r"$\beta = 0.4$"),
        RateCSV("rate_0424_162835.csv", r"双巡视器"),
    ]

    fig, ax = plt.subplots()
    for csv in csvs2:
        plot_rate_csv(csv, ax)
        print(csv.label, csv.x.iloc[-1])

    ax.set_xlabel("时间 (s)", fontsize=10.5)
    ax.set_ylabel("探索率 (%)", fontsize=10.5)
    ax.grid(True, axis="both")

    ax_add_legend(ax)
    plt_tight_show()
