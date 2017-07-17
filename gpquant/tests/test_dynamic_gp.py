from gpquant.gp_dynamic import *
from gpquant.gp_fitness import *
from sklearn.utils import check_random_state

rng = check_random_state(0)
X_train = rng.uniform(-1, 1, 100).reshape(50, 2)
y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1


def get_score_from_ff(y):
    return [y_train[i] for i in y]


metric_gp = DynamicSymbolicRegressor.make_explict_fitness(get_score_from_ff, mean_absolute_error, False,
                                                          use_raw_y=True)
est_gp = DynamicSymbolicRegressor(metric_gp, population_size=5000,
                                  generations=20, stopping_criteria=0.01,
                                  p_crossover=0.7, p_subtree_mutation=0.1,
                                  p_hoist_mutation=0.05, p_point_mutation=0.1,
                                  max_samples=0.9, verbose=1,
                                  parsimony_coefficient=0.01, random_state=0)
est_gp.dynamic_fit(X_train)
print(est_gp._program)
