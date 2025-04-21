import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from typing import List, Tuple
from dataclasses import dataclass

CSV_ROOT = Path("/home/yyy/moon_R2023/Data/map")


@dataclass
class CSVLine:
    file: Path | str
    label: str | None = None

    def __post_init__(self):
        self.file = CSV_ROOT / self.file
        if self.label is None:
            self.label = self.file.stem


def csv_list(*args: Tuple[str] | Tuple[str, str]) -> List[CSVLine]:
    lines = []
    for item in args:
        if len(item) == 1:
            lines.append(CSVLine(file=item[0]))
        else:
            file, label = item
            lines.append(CSVLine(file=file, label=label))
    return lines


def plot_csv_files(csvs: List[CSVLine]):
    plt.figure(figsize=(10, 6))

    for csv_line in csvs:
        file_path, label = csv_line.file, csv_line.label
        try:
            df = pd.read_csv(file_path)
            if df.shape[1] < 2:
                print(f"跳过 {file_path}，列数不足两列。")
                continue

            x = df.iloc[:, 0][::2]
            x = x.astype(float).round(1)
            y = df.iloc[:, 1][::2]
            y = y.astype(float) / 501 / 501 * 100
            plt.plot(x, y, label=label, linewidth=2)
        except Exception as e:
            print(f"读取文件 {file_path} 时出错：{e}")

    plt.xlabel("Time (s)")
    plt.ylabel("Rate (%)")
    plt.legend()
    plt.grid(True, axis="both")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    # 速度指令倍数 / beta / alpha求取使用的比例值 / 达到一定比例的比例与新beta
    csvs: List[CSVLine] = csv_list(
        ("rate_0422_091532.csv", "1.5/b=0.4/a=0.8"),
        ("rate_0422_111633.csv", "1.5/b=0/a=0.8"),
        ("rate_0422_143957.csv", "1.5/b=0.4/a=0.8/new_b=40/0"),
        ("rate_0422_162350.csv", "1.5/b=0.4/a=0.8/new_b=40/0.1"),
    )

    plot_csv_files(csvs)
