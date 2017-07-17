import ctypes
from numpy.ctypeslib import ndpointer
from gpquant.gp_dynamic import *
from gpquant.gp_fitness import *
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from ctypes import Structure
import pydotplus


class DataPackage(Structure):
    _fields_ = [('n_data', ctypes.c_int), ('n_dim', ctypes.c_int),
                ('data', ctypes.POINTER(ctypes.POINTER(ctypes.c_double)))]


mid = Middleware("GPQuant.dll")
get_data_func = mid.get_function("?get_data@BackTesting@GPQuant@@SA?AUTestDataPackage@2@XZ")
get_data_func.restype = DataPackage

get_reward_func = mid.get_function("?get_reward@BackTesting@GPQuant@@SANPEAHPEAN@Z")
get_reward_func.restype = ctypes.c_double

package = get_data_func()
n_dim = int(package.n_dim)
n_data = int(package.n_data)

x_data = []
for i in range(n_dim):
    _x_data = []
    for j in range(n_data):
        _x_data.append(package.data[i][j])
    x_data.append(_x_data)
x_data = np.transpose(np.array(x_data))
print(x_data)


#
# class DataPackage(Structure):
#     _fields_ = [('n_data', ctypes.c_int), ('data', ctypes.POINTER(ctypes.c_double))]
#
#
# mid = Middleware("GPQuant_struct.dll")
# get_data_func = mid.get_function("?get_data@BackTesting@gpquant@@SA?AUTestDataPackage@2@XZ")
# get_data_func.restype = DataPackage
#
#
# package = get_data_func()
# n_data = int(package.n_data)

#
# mid = Middleware("gpquant.dll")
# get_data_func = mid.get_function("?get_data@BackTesting@gpquant@@SAPEANXZ")
# get_data_func.restype = ndpointer(dtype=ctypes.c_double, shape=(10,))
# x_data = get_data_func()
#
# get_reward_func = mid.get_function("?get_reward@BackTesting@gpquant@@SANPEAHPEAN@Z")
# get_reward_func.restype = ctypes.c_double
#
#
def explicit_fitness(y, y_pred, sample_weight):
    n_data = len(y)
    y = [int(_) for _ in y]
    indices = (ctypes.c_int * n_data)(*y)
    arr = (ctypes.c_double * n_data)(*y_pred)
    res = get_reward_func(indices, arr)
    # print(res)
    return res


# metric_gp = DynamicSymbolicRegressor.make_explict_fitness(get_reward_func, y_as_fitness, False)


# x_data = x_data.reshape(10, 1)
est_gp = SymbolicRegressor(population_size=50,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           metric=make_fitness(explicit_fitness, False),
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
_ = [i for i in range(x_data.shape[0])]
est_gp.fit(x_data, _)
from PIL import Image

graph = pydotplus.graphviz.graph_from_dot_data(est_gp._program.export_graphviz())
graph.write_png("tree.png")
# print([method for method in dir(graph) if callable(getattr(graph, method))])
# Image.open(graph.create_png())
