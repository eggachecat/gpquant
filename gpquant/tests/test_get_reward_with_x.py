from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

CONFIG_N_DIM = 3


def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data


CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_REWARD_FUNC_KEY = "?get_reward_with_x@BackTesting@GPQuant@@SANPEAHPEANHPEAPEANH@Z"

file_path = "test.txt"

mid = Middleware(CONFIG_DLL_PATH)

get_reward_func = mid.get_function(CONFIG_REWARD_FUNC_KEY)
get_reward_func.restype = ctypes.POINTER(ctypes.c_double)

x_data = read_data(file_path)


def convert_list_to_c_array(x_list, n_data, n_dim):
    x_arr = (ctypes.c_double * n_dim * n_data)()

    for i in range(n_data):
        print(x_list[i])
        for j in range(n_dim):
            print(x_list[i][j])
            x_arr[i][j] = 0

    return x_arr


x_list = [[2, 3, 1], [1, 2, 3]]
x_arr = convert_list_to_c_array(x_list, 2, 3)
print(x_arr)
exit()


def explicit_fitness(y, y_pred, sample_weight):
    n_data = len(y)

    y = [ctypes.c_int(int(_)) for _ in y]
    indices = (ctypes.c_int * n_data)(*y)

    y_pred_arr = (ctypes.c_double * n_data)(*y_pred)
    x_arr = (ctypes.c_double * n_data * CONFIG_N_DIM)()

    res = get_reward_func(indices, ctypes.c_int(n_data), y_pred_arr)

    return res[0]


function_set = ['add', 'sub', 'mul', 'div', 'sin']
est_gp = SymbolicRegressor(population_size=500,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           metric=make_fitness(explicit_fitness, False),
                           function_set=function_set,
                           max_samples=0.1, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

_ = [i for i in range(x_data.shape[0])]
est_gp.fit(x_data, _)

y_truth = np.sin(x_data[:, 0] * x_data[:, 1]) - x_data[:, 2] * x_data[:, 2] + x_data[:, 2]
canvas = gp_plot.GPCanvas()
canvas.draw_line_chart_2d(range(0, len(y_truth)), y_truth, color="blue", label="y_truth", line_style="solid")

y_pred = est_gp.predict(x_data)
canvas.draw_line_chart_2d(range(0, len(y_pred)), y_pred, color="red", label="y_pred")
canvas.set_legend()
canvas.froze()

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_png("outputs/gp.png")
