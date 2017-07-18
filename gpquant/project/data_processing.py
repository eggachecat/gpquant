import pandas as pd
import numpy as np


def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data


def generate_random_data(n_data, dim, file_path, _range=None):
    if _range is None:
        _range = [-1, 1]

    data = np.random.uniform(_range[0], _range[1], n_data * dim).reshape(n_data, 3)
    np.savetxt(file_path, data, delimiter=",", fmt="%.5f")


def main():
    generate_random_data(20000, 3, "../tests/_test.txt")


if __name__ == "__main__":
    main()
