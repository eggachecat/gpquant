from gpquant.gp_middleware import Middleware
import ctypes

from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness

from gpquant import gp_plot

import pydotplus

import pandas as pd
import numpy as np

import time


class GPQuant:
    def __init__(self, n_dim, price_table, dll_path, func_key_reward, func_reward_config=None, function_set=None):

        self.n_dim = n_dim
        self.price_table = price_table
        self.price_table_ptr = self.price_table.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        self.len_price_table = len(price_table)

        self.mid = Middleware(dll_path)
        self.reward_func = self.mid.get_function(func_key_reward)

        if func_reward_config is None:
            func_reward_config = {
                "argtypes": [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                             ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int],
                "restype": ctypes.c_double
            }

        if function_set is None:
            function_set = ['add', 'sub', 'mul', 'div', 'sin']

        self.function_set = function_set

        self.reward_func.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                     ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]
        self.reward_func.restype = ctypes.c_double

    def make_explict_func(self):

        n_dim = self.n_dim
        price_table_ptr = self.price_table_ptr
        len_price_table = self.len_price_table

        def explicit_fitness(y, y_pred, sample_weight):
            """

            :param y: as indicies correspondint to _y_pred
                    see fit() below
                e.g.
                    y = [2,5,7] and y_pred = [1.23, 2.34, 8.12]
                    means:
                        f(x[2]) = 1.23
                        f(x[5]) = 2.34
                        f(x[7]) = 8.12
            :param y_pred:
            :param sample_weight:
            :return:
            """

            bool_sample_weight = np.array(sample_weight, dtype=bool)
            indices = y[bool_sample_weight]
            y_pred_arr = y_pred[bool_sample_weight]

            indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            y_pred_arr_pointer = y_pred_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

            result = self.reward_func(indices_pointer, y_pred_arr_pointer, len(indices), price_table_ptr, n_dim, 0)

            return result

        return explicit_fitness

    def fit(self, x_data):
        est_gp = SymbolicRegressor(population_size=500,
                                   generations=10, stopping_criteria=0.0001,
                                   p_crossover=0.7, p_subtree_mutation=0.1,
                                   p_hoist_mutation=0.05, p_point_mutation=0.1,
                                   metric=make_fitness(self.make_explict_func(), False),
                                   function_set=self.function_set,
                                   verbose=1, parsimony_coefficient=0.01)

        indicies = np.arange(x_data.shape[0])
        est_gp.fit(x_data, indicies)

        return est_gp
