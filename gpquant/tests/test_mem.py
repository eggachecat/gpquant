from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

CONFIG_N_DIM = 3
CONFIG_FILE_PATH = "./data/_test.txt"

CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_REWARD_FUNC_KEY = "?get_reward_with_x@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"
CONFIG_CHEATING_FUNC_KEY = "?cheating@BackTesting@GPQuant@@SAPEANPEANHH@Z"
CONFIG_DOUBLE_GC_FUNC_KEY = "?delete_double_pointer@BackTesting@GPQuant@@SAXPEAN@Z"
CONFIG_INT_GC_FUNC_KEY = "?delete_int_pointer@BackTesting@GPQuant@@SAXPEAH@Z"
CONFIG_TEST_MEM_FUNC_KEY = "?test_mem@BackTesting@GPQuant@@SANPEAHPEANH@Z"


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

test_mem_func = mid.get_function(CONFIG_TEST_MEM_FUNC_KEY)

total_data = 1000000
y = range(total_data)
sample_weight = [np.random.randint(0, 2) for _ in range(total_data)]

indices = np.array([int(i) for i in range(total_data) if sample_weight[i]], dtype=int)
n_data = len(indices)
indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))


class Foo:
    def __init__(self, func):
        self.func = func

    def fit(self):
        _indices = np.array([i for i in range(len(y)) if sample_weight[i]], dtype=int)
        _y_pred_arr = np.array([np.random.rand() for i in range(len(y)) if sample_weight[i]], dtype=float)

        test_mem_func(_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                  _y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), len(_indices))


def f_foo(x, y):
    x


foo = Foo(test_mem_func)
while True:
    foo.fit()
