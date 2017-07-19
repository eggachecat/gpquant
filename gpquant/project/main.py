from gpquant.gp_middleware import Middleware
import ctypes

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

import time

from gpquant import gp_model
from gpquant import gp_io

x_data, price_table = gp_io.read_data("./data/data.csv")
CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
CONFIG_FUNC_KEY_REWARD = "?get_reward@BackTesting@GPQuant@@SANPEAHPEANH1HH@Z"

gp = gp_model.GPQuant(56, price_table, CONFIG_DLL_PATH, CONFIG_FUNC_KEY_REWARD)
est_gp = gp.fit(x_data)
y_pred = est_gp.predict(x_data)

print("---------------------------------------------")
indices = np.array(range(len(x_data)), dtype=int)

indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
y_pred_arr_pointer = y_pred.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
price_table_ptr = price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

mid = Middleware(CONFIG_DLL_PATH)
reward_func = mid.get_function(CONFIG_FUNC_KEY_REWARD)
result = reward_func(indices_pointer, y_pred_arr_pointer, len(x_data), price_table_ptr, 0, -1)
print(result)
print(est_gp._program.fitness_)

