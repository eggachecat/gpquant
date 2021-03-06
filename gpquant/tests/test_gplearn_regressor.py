import numpy as np
import pylab as plt
from gplearn.genetic import SymbolicRegressor
from mpl_toolkits.mplot3d import Axes3D
from sklearn.utils import check_random_state

CONFIG_N_DATA = 100
CONFIG_N_DIM = 2

x_0 = np.arange(-1, 1, 1 / 10.)
x_1 = np.arange(-1, 1, 1 / 10.)
x_0, x_1 = np.meshgrid(x_0, x_1)
y_truth = x_0 ** 2 - x_1 ** 2 + x_1 - 1

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
surf = ax.plot_surface(x_0, x_1, y_truth, rstride=1, cstride=1,
                       color='green', alpha=0.5)
# plt.show()

rng = check_random_state(0)

# Training samples
X_train = rng.uniform(-1, 1, CONFIG_N_DATA * CONFIG_N_DIM).reshape(CONFIG_N_DATA, CONFIG_N_DIM)
y_train = X_train[:, 0] ** 2 - X_train[:, 1] ** 2 + X_train[:, 1] - 1

# Testing samples
X_test = rng.uniform(-1, 1, CONFIG_N_DATA * CONFIG_N_DIM).reshape(CONFIG_N_DATA, CONFIG_N_DIM)
y_test = X_test[:, 0] ** 2 - X_test[:, 1] ** 2 + X_test[:, 1] - 1

function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']

est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           function_set=function_set,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0)
est_gp.fit(X_train, y_train)
print(est_gp._program)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

est_tree = DecisionTreeRegressor()
est_tree.fit(X_train, y_train)
est_rf = RandomForestRegressor()
est_rf.fit(X_train, y_train)

y_gp = est_gp.predict(np.c_[x_0.ravel(), x_1.ravel()]).reshape(x_0.shape)
score_gp = est_gp.score(X_test, y_test)
y_tree = est_tree.predict(np.c_[x_0.ravel(), x_1.ravel()]).reshape(x_0.shape)
score_tree = est_tree.score(X_test, y_test)
y_rf = est_rf.predict(np.c_[x_0.ravel(), x_1.ravel()]).reshape(x_0.shape)
score_rf = est_rf.score(X_test, y_test)

fig = plt.figure(figsize=(8, 6))

for i, (y, score, title) in enumerate([(y_truth, None, "Ground Truth"),
                                       (y_gp, score_gp, "SymbolicRegressor"),
                                       (y_tree, score_tree, "DecisionTreeRegressor"),
                                       (y_rf, score_rf, "RandomForestRegressor")]):

    ax = fig.add_subplot(2, 2, i + 1, projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    surf = ax.plot_surface(x_0, x_1, y, rstride=1, cstride=1, color='green', alpha=0.5)
    points = ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
    if score is not None:
        score = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score, 'x', fontsize=14)
    plt.title(title)

plt.show()
