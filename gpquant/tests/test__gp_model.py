from gpquant.gp_model import *

from gpquant.gp_middleware import Middleware
import ctypes

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

import time


def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data


CONFIG_N_DIM = 3
CONFIG_FILE_PATH = "./data/_test.txt"

CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_FUNC_KEY_REWARD = "?get_reward_with_x@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CONFIG_FUNC_KEY_CHEAT = "?cheating@BackTesting@GPQuant@@SAPEANPEANHH@Z"

mid = Middleware(CONFIG_DLL_PATH)
cheating_func = mid.get_function(CONFIG_FUNC_KEY_CHEAT)
cheating_func.restype = ctypes.POINTER(ctypes.c_double)

x_data = read_data(CONFIG_FILE_PATH)
ts = int(time.time())
_x_data = x_data.flatten()
x_arr_pointer = _x_data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
x_len = len(_x_data)

gpquant = GPQuant(CONFIG_N_DIM, _x_data, CONFIG_DLL_PATH, CONFIG_FUNC_KEY_REWARD)
est_gp = gpquant.fit(x_data)

n_data = int(x_len / CONFIG_N_DIM)
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
