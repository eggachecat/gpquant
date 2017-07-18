from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

import time

CONFIG_N_DIM = 3
CONFIG_FILE_PATH = "./data/_test.txt"

CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_FUNC_KEY_REWARD = "?get_reward_with_x@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CONFIG_FUNC_KEY_CHEAT = "?cheating@BackTesting@GPQuant@@SAPEANPEANHH@Z"
CONFIG_FUNC_KEY_CONVERT = "?convert_1d_array_to_2d_array@BackTesting@GPQuant@@SAPEAPEANPEANHH@Z"


def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data


mid = Middleware(CONFIG_DLL_PATH)

get_reward_func = mid.get_function(CONFIG_FUNC_KEY_REWARD)
get_reward_func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                            ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
get_reward_func.restype = ctypes.c_double

cheating_func = mid.get_function(CONFIG_FUNC_KEY_CHEAT)
cheating_func.restype = ctypes.POINTER(ctypes.c_double)

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

    result = get_reward_func(indices_pointer, y_pred_arr_pointer, _n_data, x_arr_pointer, CONFIG_N_DIM, x_len)

    # explicit_fitness.counter += 1
    #
    # if explicit_fitness.counter % 100 == 0:
    #     print(explicit_fitness.counter)
    #     print(explicit_fitness.res)
    #     print(explicit_fitness.res / explicit_fitness.counter)
    #
    # explicit_fitness.res += result

    return result


explicit_fitness.counter = 0
explicit_fitness.res = 0

function_set = ['add', 'sub', 'mul', 'div', 'sin']
est_gp = SymbolicRegressor(population_size=5000,
                           generations=10, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           metric=make_fitness(explicit_fitness, False),
                           function_set=function_set,
                           max_samples=0.8, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

_ = [i for i in range(x_data.shape[0])]
est_gp.fit(x_data, _)

ts = int(time.time())

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_png("outputs/gp-{suffix}.png".format(suffix=ts))

res = cheating_func(x_arr_pointer, CONFIG_N_DIM, x_len)
y_truth = np.array([float(res[i]) for i in range(n_data)])
y_pred = np.array(est_gp.predict(x_data))

n_data_plot = 200
indicies_plot = sorted(np.random.choice(n_data, n_data_plot, replace=False))

canvas = gp_plot.GPCanvas()
canvas.draw_line_chart_2d(range(0, n_data_plot), y_truth[indicies_plot], color="blue", label="y_truth",
                          line_style="solid")

canvas.draw_line_chart_2d(range(0, n_data_plot), y_pred[indicies_plot], color="red", label="y_pred")

mse = ((np.array(y_truth) - np.array(y_pred)) ** 2).mean()

canvas.set_x_label("Indices")
canvas.set_y_label("Values")
canvas.set_title("Fitting plot with MSE={:5f}".format(mse))
canvas.set_legend()
canvas.set_axis_invisible()

canvas.froze()
