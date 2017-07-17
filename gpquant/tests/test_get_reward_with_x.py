from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np



CONFIG_N_DIM = 3
CONFIG_FILE_PATH = "test.txt"

CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_REWARD_FUNC_KEY = "?get_reward_with_x@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CONFIG_CHEATING_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEANHH@Z"


def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data


mid = Middleware(CONFIG_DLL_PATH)

get_reward_func = mid.get_function(CONFIG_REWARD_FUNC_KEY)
get_reward_func.restype = ctypes.c_double

cheating_func = mid.get_function(CONFIG_CHEATING_FUNC_KEY)
cheating_func.restype = ctypes.POINTER(ctypes.c_double)

def explicit_fitness(y, y_pred, sample_weight):
    _indices = [int(i) for i in range(len(y)) if sample_weight[i]]
    _y_pred_arr = [y_pred[i] for i in range(len(y)) if sample_weight[i]]
    n_data = len(_y_pred_arr)

    indices = (ctypes.c_int * n_data)(*_indices)
    y_pred_arr = (ctypes.c_double * n_data)(*_y_pred_arr)

    res = get_reward_func(indices, y_pred_arr, n_data, x_arr, CONFIG_N_DIM, x_len)

    return res

x_data = read_data(CONFIG_FILE_PATH)
_x_data = x_data.flatten().tolist()
x_arr = (ctypes.c_double * len(_x_data))(*_x_data)
x_len = len(x_arr)


function_set = ['add', 'sub', 'mul', 'div', 'sin', 'log']
est_gp = SymbolicRegressor(population_size=500,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           metric=make_fitness(explicit_fitness, False),
                           function_set=function_set,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

_ = [i for i in range(x_data.shape[0])]
est_gp.fit(x_data, _)

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_png("outputs/gp.png")

n_data = int(x_len / CONFIG_N_DIM)
res = cheating_func(x_arr, CONFIG_N_DIM, x_len)
y_truth = [float(res[i]) for i in range(n_data)]
canvas = gp_plot.GPCanvas()
canvas.draw_line_chart_2d(range(0, n_data), y_truth, color="blue", label="y_truth", line_style="solid")

y_pred = est_gp.predict(x_data)
canvas.draw_line_chart_2d(range(0, n_data), y_pred, color="red", label="y_pred")
canvas.set_legend()
canvas.froze()

