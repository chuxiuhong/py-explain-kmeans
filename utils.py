import numpy as np


def read_csv(path) -> (np.ndarray, np.ndarray):
    x, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            line = list(map(float, line.split(",")))
            x.append(line[:-1])
            y.append(line[-1])
    return np.array(x), np.array(y)


if __name__ == "__main__":
    pass
