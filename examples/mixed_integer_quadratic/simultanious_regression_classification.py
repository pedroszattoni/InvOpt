"""
InvOpt package example: simultanious regression and classification.

Dataset: Breast Cancer Wisconsin Prognostic (BCWP)
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(Prognostic)

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
import time
import numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt
from utils_examples import colors, mean_percentiles, L1
import invopt as iop

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

    mdl = Model('MIQP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVar(vtype=GRB.CONTINUOUS, name='y')
    z = mdl.addVar(vtype=GRB.BINARY, name='z')

    phi1_wz = [wi for wi in w] + [z] + [z*wi for wi in w] + [1]
    phi2_wz = [wi for wi in w] + [z] + [z*wi for wi in w] + [1]

    mdl.setObjective(Qyy*y**2
                     + y*quicksum(Q[i]*phi1_wz[i] for i in range(cQ))
                     + quicksum(q[i]*phi2_wz[i] for i in range(cQ)),
                     GRB.MINIMIZE)

    mdl.optimize()

    y_opt = np.array([y.X])
    z_opt = np.array([round(z.X)])

    return y_opt, z_opt


def load_data(train_test_slip):
    """Load and preprosses BCWP data."""
    dataset = np.genfromtxt(r'breast-cancer-wisconsin-data\wpbc_data.csv',
                            delimiter=',')

    # Signal-response data
    S = dataset[:, 2:].copy()
    X = dataset[:, :2].copy()
    X[:, [1, 0]] = X[:, [0, 1]]

    N, m = S.shape
    N_train = round(N*(1-train_test_slip))
    N_test = round(N*train_test_slip)

    train_idx = np.random.choice(N, N_train, replace=False)
    train_mask = np.zeros(N, dtype=bool)
    train_mask[train_idx] = True

    # Split data into train/test
    S_train = S[train_mask, :].copy()
    X_train = X[train_mask, :].copy()
    S_test = S[~train_mask, :].copy()
    X_test = X[~train_mask, :].copy()

    A = -np.eye(1)
    B = np.zeros((1, 1))
    c = np.zeros((1))

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

print('')
print(f'train_test_slip = {train_test_slip}')
print(f'runs = {runs}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

kappa_list = np.logspace(-4, 1, 5).tolist()
reg_size = len(kappa_list)

y_diff_train_hist = np.empty((runs, reg_size))
y_diff_test_hist = np.empty((runs, reg_size))
z_diff_train_hist = np.empty((runs, reg_size))
z_diff_test_hist = np.empty((runs, reg_size))

tic = time.time()
for run in range(runs):
    dataset_train, dataset_test = load_data(train_test_slip)

    for kappa in kappa_list:
        r_index = kappa_list.index(kappa)

        theta_IO = iop.mixed_integer_quadratic(dataset_train,
                                               ('binary', 1, None),
                                               phi1=phi1,
                                               phi2=phi2,
                                               dist_func_z=L1,
                                               reg_param=kappa)

        y_diff_train = iop.evaluate(theta_IO,
                                    dataset_train,
                                    FOP_MIQP,
                                    dist_y)
        y_diff_test = iop.evaluate(theta_IO,
                                   dataset_test,
                                   FOP_MIQP,
                                   dist_y)
        z_diff_train = iop.evaluate(theta_IO,
                                    dataset_train,
                                    FOP_MIQP,
                                    dist_z)
        z_diff_test = iop.evaluate(theta_IO,
                                   dataset_test,
                                   FOP_MIQP,
                                   dist_z)

        y_diff_train_hist[run, r_index] = y_diff_train
        y_diff_test_hist[run, r_index] = y_diff_test
        z_diff_train_hist[run, r_index] = z_diff_train
        z_diff_test_hist[run, r_index] = z_diff_test

    print(f'{round(100*(run+1)/runs)}%')

toc = time.time()
print(f"Simulation time = {round(toc-tic,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

y_diff_train_mean, y_diff_train_p5, y_diff_train_p95 = \
    mean_percentiles(y_diff_train_hist)
y_diff_test_mean, y_diff_test_p5, y_diff_test_p95 = \
    mean_percentiles(y_diff_test_hist)
z_diff_train_mean, z_diff_train_p5, z_diff_train_p95 = \
    mean_percentiles(z_diff_train_hist)
z_diff_test_mean, z_diff_test_p5, z_diff_test_p95 = \
    mean_percentiles(z_diff_test_hist)

plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['font.family'] = 'serif'

plt.figure(1)
plt.plot(kappa_list, y_diff_train_mean, c=colors[0])
plt.fill_between(kappa_list, y_diff_train_p5, y_diff_train_p95, alpha=0.3,
                 facecolor=colors[0])
plt.xscale('log')
plt.ylabel(r'$| y_{\mathrm{IO}} - y_{\mathrm{true}} |$', fontsize=18)
plt.xlabel(r'$\kappa$', fontsize=18)
plt.grid(visible=True)
plt.tight_layout()

plt.figure(2)
plt.plot(kappa_list, z_diff_train_mean, c=colors[1])
plt.fill_between(kappa_list, z_diff_train_p5, z_diff_train_p95, alpha=0.3,
                 facecolor=colors[1])
plt.xscale('log')
plt.ylabel(r'$| z_{\mathrm{IO}} - z_{\mathrm{true}} |$', fontsize=18)
plt.xlabel(r'$\kappa$', fontsize=18)
plt.grid(visible=True)
plt.tight_layout()

plt.figure(3)
plt.plot(kappa_list, y_diff_test_mean, c=colors[0])
plt.fill_between(kappa_list, y_diff_test_p5, y_diff_test_p95, alpha=0.3,
                 facecolor=colors[0])
plt.xscale('log')
plt.ylabel(r'$| y_{\mathrm{IO}} - y_{\mathrm{true}} |$', fontsize=18)
plt.xlabel(r'$\kappa$', fontsize=18)
plt.grid(visible=True)
plt.tight_layout()

plt.figure(4)
plt.plot(kappa_list, z_diff_test_mean, c=colors[1])
plt.fill_between(kappa_list, z_diff_test_p5, z_diff_test_p95, alpha=0.3,
                 facecolor=colors[1])
plt.xscale('log')
plt.ylabel(r'$| z_{\mathrm{IO}} - z_{\mathrm{true}} |$', fontsize=18)
plt.xlabel(r'$\kappa$', fontsize=18)
plt.grid(visible=True)
plt.tight_layout()
