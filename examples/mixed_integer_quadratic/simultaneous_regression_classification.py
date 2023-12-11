"""
InvOpt package example: simultaneous regression and classification.

Dataset: Breast Cancer Wisconsin Prognostic (BCWP)
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(Prognostic)

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np
import gurobipy as gp
import matplotlib.pyplot as plt
import invopt as iop
from sklearn import svm
from sklearn import linear_model
from sklearn import neighbors
from sklearn import gaussian_process
from sklearn import tree
from sklearn import ensemble
from sklearn import neural_network
from sklearn import kernel_ridge

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import colors, mean_percentiles, L1

np.random.seed(0)


def FOP_MIQP(theta, s):
    """Forward optimization problem."""
    _, _, _, w = s
    u = len(w)
    cQ = u + 1 + u + 1
    Qyy = theta[0]
    Q = theta[1:(1 + cQ)]
    q = theta[(1 + cQ):]

    if len(theta) != (1 + cQ + cQ):
        raise Exception('Dimentions do not match!')

    mdl = gp.Model('MIQP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVar(vtype=gp.GRB.CONTINUOUS, name='y')
    z = mdl.addVar(vtype=gp.GRB.BINARY, name='z')

    phi1_wz = [wi for wi in w] + [z] + [z*wi for wi in w] + [1]
    phi2_wz = [wi for wi in w] + [z] + [z*wi for wi in w] + [1]

    mdl.setObjective(
        Qyy*y**2 + y*gp.quicksum(Q[i]*phi1_wz[i] for i in range(cQ))
        + gp.quicksum(q[i]*phi2_wz[i] for i in range(cQ)), gp.GRB.MINIMIZE
    )

    mdl.optimize()

    y_opt = np.array([y.X])
    z_opt = np.array([round(z.X)])

    return y_opt, z_opt


def load_data(train_test_slip):
    """Load and preprosses BCWP data."""
    # dataset = np.genfromtxt(path_to_invopt + r'\examples\mixed_integer_quadratic\breast-cancer-wisconsin-data\wpbc_data.csv',   # nopep8
    #                         delimiter=',')

    dataset = np.genfromtxt(
        r'breast-cancer-wisconsin-data\wpbc_data.csv', delimiter=','
    )

    # Signal-response data
    S = dataset[:, 2:].copy()
    X = dataset[:, :2].copy()
    X[:, [1, 0]] = X[:, [0, 1]]

    N, m = S.shape
    N_train = round(N*(1-train_test_slip))
    # N_test = round(N*train_test_slip)

    train_idx = np.random.choice(N, N_train, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True

    # Split data into train/test
    S_train = S[train_mask, :].copy()
    X_train = X[train_mask, :].copy()
    S_test = S[~train_mask, :].copy()
    X_test = X[~train_mask, :].copy()

    return S_train, X_train, S_test, X_test


def IO_preprocessing(S_train, X_train, S_test, X_test):
    """Preprocess data for IO."""
    A = -np.eye(1)
    B = np.zeros((1, 1))
    c = np.zeros((1))

    N_train = len(S_train)
    N_test = len(S_test)

    # Create dataset for IO
    dataset_train = []
    for i in range(N_train):
        s_hat = (A, B, c, S_train[i, :])
        x_hat = (np.array([X_train[i, 0]]), np.array([X_train[i, 1]]))
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        s_hat = (A, B, c, S_test[i, :])
        x_hat = (np.array([X_train[i, 0]]), np.array([X_test[i, 1]]))
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def phi1(w, z):
    """Feature function."""
    return np.concatenate((w, z, w*z, [1]))


def phi2(w, z):
    """Feature function."""
    return np.concatenate((w, z, w*z, [1]))


def phi(s, x):
    """Transform phi1 and phi2 into phi."""
    _, _, _, w = s
    y, z = x
    return np.concatenate((y*phi1(w, z), phi2(w, z)))


def dist_y(x1, x2):
    """Distance function for continous partof decision vector."""
    y1, z1 = x1
    y2, z2 = x2
    dy = L1(y1, y2)
    return dy


def dist_z(x1, x2):
    """Distance function for discrete part of decision vector."""
    y1, z1 = x1
    y2, z2 = x2
    dz = L1(z1, z2)
    return dz


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

train_test_slip = 0.1
runs = 3
add_y = False
kappa = 10**3

print('')
print(f'train_test_slip = {train_test_slip}')
print(f'runs = {runs}')
print(f'add_y = {add_y}')
print(f'kappa = {kappa}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
y_diff_train_hist = np.empty(runs)
y_diff_test_hist = np.empty(runs)
z_diff_train_hist = np.empty(runs)
z_diff_test_hist = np.empty(runs)

y_diff_train_sk_hist = np.empty(runs)
y_diff_test_sk_hist = np.empty(runs)
z_diff_train_sk_hist = np.empty(runs)
z_diff_test_sk_hist = np.empty(runs)

tic = time.time()
for run in range(runs):
    np.random.seed(run)  # Make sure the same random slipt is used
    S_train, X_train, S_test, X_test = load_data(train_test_slip)
    dataset_train, dataset_test = IO_preprocessing(
        S_train, X_train, S_test, X_test
    )

    theta_IO = iop.mixed_integer_quadratic(dataset_train,
                                           ('binary', 1, None),
                                           phi1=phi1,
                                           phi2=phi2,
                                           dist_func_z=L1,
                                           reg_param=kappa,
                                           add_dist_func_y=add_y)

    y_diff_train = iop.evaluate(theta_IO, dataset_train, FOP_MIQP, dist_y)
    y_diff_test = iop.evaluate(theta_IO, dataset_test, FOP_MIQP, dist_y)
    z_diff_train = iop.evaluate(theta_IO, dataset_train, FOP_MIQP, dist_z)
    z_diff_test = iop.evaluate(theta_IO, dataset_test, FOP_MIQP, dist_z)

    y_diff_train_hist[run] = y_diff_train
    y_diff_test_hist[run] = y_diff_test
    z_diff_train_hist[run] = z_diff_train
    z_diff_test_hist[run] = z_diff_test

    # Scikit-learn regressors
    # reg = svm.SVR()
    # reg = linear_model.LinearRegression()
    reg = kernel_ridge.KernelRidge()
    # reg = neural_network.MLPRegressor(max_iter=3000)
    # reg = neighbors.KNeighborsRegressor()
    # reg = gaussian_process.GaussianProcessRegressor()
    # reg = tree.DecisionTreeRegressor()

    reg.fit(S_train, X_train[:, 0])
    y_diff_train_sk = np.mean(np.abs(reg.predict(S_train) - X_train[:, 0]))
    y_diff_test_sk = np.mean(np.abs(reg.predict(S_test) - X_test[:, 0]))
    y_diff_train_sk_hist[run] = y_diff_train_sk
    y_diff_test_sk_hist[run] = y_diff_test_sk

    # Scikit-learn classifiers
    clf = svm.SVC()
    # clf = linear_model.LogisticRegression(max_iter=2000)
    # clf = neighbors.KNeighborsClassifier()
    # clf = gaussian_process.GaussianProcessClassifier()
    # clf = tree.DecisionTreeClassifier()
    # clf = ensemble.RandomForestClassifier()
    # clf = neural_network.MLPClassifier(max_iter=1000)
    # clf = ensemble.AdaBoostClassifier()

    clf.fit(S_train, X_train[:, 1])
    z_diff_train_sk = np.mean(np.abs(clf.predict(S_train) - X_train[:, 1]))
    z_diff_test_sk = np.mean(np.abs(clf.predict(S_test) - X_test[:, 1]))
    z_diff_train_sk_hist[run] = z_diff_train_sk
    z_diff_test_sk_hist[run] = z_diff_test_sk

    print(f'{round(100*(run+1)/runs)}%')

toc = time.time()
print(f"Simulation time = {round(toc-tic,2)} seconds")
print('')

print(f'y_diff train error = {mean_percentiles(y_diff_train_hist)[0]}')
print(f'y_diff test error = {mean_percentiles(y_diff_test_hist)[0]}')
print(f'z_diff train error = {mean_percentiles(z_diff_train_hist)[0]}')
print(f'z_diff test error = {mean_percentiles(z_diff_test_hist)[0]}')
print('')
print(f'SK y_diff train error = {mean_percentiles(y_diff_train_sk_hist)[0]}')
print(f'SK y_diff test error = {mean_percentiles(y_diff_test_sk_hist)[0]}')
print(f'SK z_diff train error = {mean_percentiles(z_diff_train_sk_hist)[0]}')
print(f'SK z_diff test error = {mean_percentiles(z_diff_test_sk_hist)[0]}')
