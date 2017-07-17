import sys
sys.path.append("D:/sunao/workspace/python/gpquant/")

from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd



def read_data(file_path, header=None):
    x_data = pd.read_csv(file_path, header=header)
    x_data = x_data.as_matrix()
    return x_data

### something wrong with this dll
CONFIG_DLL_PATH = "./libs/Power_API.dll"
CONFIG_REWARD_FUNC_KEY = "?AdvanceGP@@YAPEANPEAHHPEAN@Z"

# CONFIG_DLL_PATH = "D:/sunao/workspace/cpp/GPQuant/x64/Release/GPQuant.dll"
# CONFIG_REWARD_FUNC_KEY = "?get_ten@BackTesting@GPQuant@@SAPEANXZ"

file_path = "test.txt"

mid = Middleware(CONFIG_DLL_PATH)

get_reward_func = mid.get_function(CONFIG_REWARD_FUNC_KEY)
get_reward_func.restype = ctypes.POINTER(ctypes.c_double)

x_data = read_data(file_path)


def explicit_fitness(y, y_pred, sample_weight):
    n_data = len(y)

    y = [ctypes.c_int(int(_)) for _ in y]
    indices = (ctypes.c_int * n_data)(*y)

    arr = (ctypes.c_double * n_data)(*y_pred)
    # print(y)

    # start = time.time()
    res = get_reward_func(indices, ctypes.c_int(n_data), arr)
    # end = time.time()
    # print(end - start)

    return res[0]


function_set = ['add', 'sub', 'mul', 'div', 'sin']
est_gp = SymbolicRegressor(population_size=500,
                           generations=1, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           metric=make_fitness(explicit_fitness, False),
                           function_set=function_set,
                           max_samples=0.1, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)

_ = [i for i in range(x_data.shape[0])]
est_gp.fit(x_data, _)
import numpy as np

y_truth = np.sin(x_data[:, 0] * x_data[:, 0]) - x_data[:, 1] + 3 * x_data[:, 2]

canvas = gp_plot.GPCanvas()

canvas.draw_line_chart_2d(range(0, len(y_truth)), y_truth, color="blue", label="y_truth", line_style="solid")

y_pred = est_gp.predict(x_data)
canvas.draw_line_chart_2d(range(0, len(y_pred)), y_pred, color="red", label="y_pred")

canvas.set_legend()
canvas.froze()

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_png("outputs/gp.png")


