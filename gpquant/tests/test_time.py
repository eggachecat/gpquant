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
CONFIG_FILE_PATH = "_test.txt"

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

    return result


y_truth = np.array([np.random.rand() for i in range(n_data)])
y_pred = np.array([np.random.rand() for i in range(n_data)])
sample_weight = [np.random.randint(0, 2) for _ in range(n_data)]

RUN_TIMES = 100
start = time.time()
for _ in range(RUN_TIMES):
    explicit_fitness(y_truth, y_pred, sample_weight)
end = time.time()
t = end - start
print("Number of test cases: {rt}, consuming time: {t}(s), avg time per case: {a}(s)".format(rt=RUN_TIMES, t=t,
                                                                                        a=t / RUN_TIMES))
