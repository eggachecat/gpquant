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

y = [1, 2, 3, 4, 5]
sample_weight = [1, 0, 0, 1, 1]

_indices = np.array([int(i) for i in range(len(y)) if sample_weight[i]], dtype=int)
n_data = len(_indices)
indices_pointer = _indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

print(indices_pointer)
print(ctypes.addressof(indices_pointer))

gc_i_func(indices_pointer)
print("indices_pointer has been deleted")
print(ctypes.addressof(indices_pointer))

_indices = np.array([int(i) for i in range(len(y)) if sample_weight[i]], dtype=int)
n_data = len(_indices)
indices_pointer = _indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

print(indices_pointer)
print(ctypes.addressof(indices_pointer))

gc_i_func(indices_pointer)
print("indices_pointer has been deleted")
print(ctypes.addressof(indices_pointer))

_indices = np.array([int(i) for i in range(len(y)) if sample_weight[i]], dtype=int)
n_data = len(_indices)
indices_pointer = _indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

print(indices_pointer)
print(ctypes.addressof(indices_pointer))

gc_i_func(indices_pointer)
print("indices_pointer has been deleted")
print(ctypes.addressof(indices_pointer))