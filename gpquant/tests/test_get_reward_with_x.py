from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

CONFIG_N_DIM = 3
CONFIG_FILE_PATH = "_test.txt"

CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_REWARD_FUNC_KEY = "?get_reward_with_x@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CONFIG_CHEATING_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEANHH@Z"
CONFIG_DOUBLE_GC_FUNC_KEY = "?delete_double_pointer@BackTesting@GPQuant@@SAXPEAN@Z"
CONFIG_INT_GC_FUNC_KEY = "?delete_int_pointer@BackTesting@GPQuant@@SAXPEAH@Z"


def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data


mid = Middleware(CONFIG_DLL_PATH)

get_reward_func = mid.get_function(CONFIG_REWARD_FUNC_KEY)
get_reward_func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
get_reward_func.restype = ctypes.c_double

cheating_func = mid.get_function(CONFIG_CHEATING_FUNC_KEY)
cheating_func.restype = ctypes.POINTER(ctypes.c_double)

gc_d_func = mid.get_function(CONFIG_DOUBLE_GC_FUNC_KEY)
gc_d_func.argtypes = [ctypes.POINTER(ctypes.c_double)]
gc_d_func.restypes = None

gc_i_func = mid.get_function(CONFIG_INT_GC_FUNC_KEY)
gc_i_func.argtypes = [ctypes.POINTER(ctypes.c_int)]
gc_i_func.restypes = None
#
# y = [1, 2, 3, 4, 5]
# sample_weight = [1, 0, 0, 1, 1]
#
# _indices = np.array([int(i) for i in range(len(y)) if sample_weight[i]], dtype=int)
# n_data = len(_indices)
# indices_pointer = _indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
#
# print(indices_pointer)
#
# gc_i_func(indices_pointer)
# print("indices_pointer has been deleted")
#
# exit()

x_data = read_data(CONFIG_FILE_PATH)
_x_data = x_data.flatten()
x_arr_pointer = _x_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x_len = len(_x_data)

n_data = int(x_len / CONFIG_N_DIM)


def explicit_fitness(y, _y_pred, sample_weight):

    _indices = np.array([i for i in range(len(y)) if sample_weight[i]], dtype=int)
    _y_pred_arr = np.array([_y_pred[i] for i in range(len(y)) if sample_weight[i]], dtype=float)

    _n_data = len(_indices)

    indices_pointer = _indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    y_pred_arr_pointer = _y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    res = get_reward_func(indices_pointer, y_pred_arr_pointer, _n_data, x_arr_pointer, CONFIG_N_DIM, x_len)

    gc_i_func(indices_pointer)
    print("indices_pointer has been deleted")
    gc_d_func(y_pred_arr_pointer)
    print("y_pred_arr_pointer has been deleted")

    return res


function_set = ['add', 'sub', 'mul', 'div', 'sin']
est_gp = SymbolicRegressor(population_size=500,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           metric=make_fitness(explicit_fitness, False),
                           function_set=function_set,
                           max_samples=0.5, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

_ = [i for i in range(x_data.shape[0])]
est_gp.fit(x_data, _)

import time

ts = int(time.time())

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_png("outputs/gp-{suffix}.png".format(suffix=ts))

res = cheating_func(x_arr_pointer, CONFIG_N_DIM, x_len)

y_truth = [float(res[i]) for i in range(n_data)]
canvas = gp_plot.GPCanvas()
canvas.draw_line_chart_2d(range(0, n_data), y_truth, color="blue", label="y_truth", line_style="solid")

y_pred = est_gp.predict(x_data)
canvas.draw_line_chart_2d(range(0, n_data), y_pred, color="red", label="y_pred")

mse = ((np.array(y_truth) - np.array(y_pred)) ** 2).mean()

canvas.set_x_label("Indices")
canvas.set_y_label("Values")
canvas.set_title("Fitting plot with MSE={:5f}".format(mse))
canvas.set_legend()
canvas.set_axis_invisible()

canvas.froze()
