"""
InvOpt package example: mixed-integer quadratic program.

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np
from gurobipy import Model, GRB, quicksum

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import L2, plot_results

import invopt as iop

np.random.seed(0)


def create_datasets(theta, FOP, m1, m2, n, N_train, N_test):
    """Create datasets for the IO problem."""
    dataset_train = []
    for i in range(N_train):
        flag = False
        while not flag:
            A1 = -np.random.rand(m1, m2)
            B1 = -np.random.rand(m1, n)
            c1 = -2*np.random.rand(m1)
            flag = all(np.sum(A1, axis=1) + np.sum(B1, axis=1) <= c1)

        A2 = np.vstack((np.eye(m2), -np.eye(m2)))
        c2 = np.hstack((np.ones(m2), np.zeros(m2)))

        A = np.vstack((A1, A2))
        B = np.vstack((B1, np.zeros((2*m2, n))))
        c = np.hstack((c1, c2))

        s_hat = (A, B, c, 0)
        x_hat = FOP(theta, s_hat)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        flag = False
        while not flag:
            A1 = -np.random.rand(m1, m2)
            B1 = -np.random.rand(m1, n)
            c1 = -2*np.random.rand(m1)
            flag = all(np.sum(A1, axis=1) + np.sum(B1, axis=1) <= c1)

        A2 = np.vstack((np.eye(m2), -np.eye(m2)))
        c2 = np.hstack((np.ones(m2), np.zeros(m2)))

        A = np.vstack((A1, A2))
        B = np.vstack((B1, np.zeros((2*m2, n))))
        c = np.hstack((c1, c2))

        s_hat = (A, B, c, 0)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def FOP_MIQP(theta, s):
    """Forward optimization approach: mixed-integer quadratic program."""
    A, B, c, _ = s
    m1, m2 = A.shape
    _, n = B.shape
    Qyy = theta[:m2**2].reshape((m2, m2))
    Qyz = theta[m2**2:(m2**2 + m2*n)].reshape((m2, n))
    qy = theta[(m2**2 + m2*n):(m2**2 + m2*n + m2)]
    qz = theta[(m2**2 + m2*n + m2):]

    if len(theta) != (m2**2 + m2*n + m2 + n):
        raise Exception('Dimentions do not match!')

    mdl = Model('MIQP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVars(m2, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='y')
    z = mdl.addVars(n, vtype=GRB.BINARY, name='z')

    mdl.setObjective(
        quicksum(Qyy[i, j]*y[i]*y[j] for i in range(m2) for j in range(m2))
        + quicksum(Qyz[i, j]*y[i]*z[j] for i in range(m2) for j in range(n))
        + quicksum(qy[i]*y[i] for i in range(m2))
        + quicksum(qz[i]*z[i] for i in range(n)), GRB.MINIMIZE
    )

    mdl.addConstrs(
        quicksum(A[k, i]*y[i] for i in range(m2))
        + quicksum(B[k, j]*z[j] for j in range(n)) <= c[k] for k in range(m1)
    )

    mdl.optimize()

    if mdl.status == 2:
        y_opt = np.array([y[k].X for k in range(m2)])
        z_opt = np.array([z[k].X for k in range(n)])
    else:
        raise Exception(
            f'Optimal solution not found. Gurobi status code = {mdl.status}.'
        )

    return (y_opt, z_opt)


def phi1(w, z):
    """Feature mapping."""
    return np.concatenate((z, [1]))


def phi2(w, z):
    """Feature mapping."""
    return z


def phi(s, x):
    """Transform phi1 and phi2 into phi for mixed_integer_quadratic case."""
    _, _, _, w = s
    y, z = x
    return np.concatenate((np.kron(y, y), np.kron(phi1(w, z), y), phi2(w, z)))


def dist_func(x1, x2):
    """Distance function."""
    y1, z1 = x1
    y2, z2 = x2
    dy = L2(y1, y2)
    dz = L2(z1, z2)
    return dy+dz


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 10
N_test = 10
n = 3
m = 2
reg_param = 0.001
Z = ('binary', n, None)
Theta = 'nonnegative'
resolution = 5
runs = 3

print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'n = {n}')
print(f'm = {m}')
print(f'reg_param = {reg_param}')
print(f'Z = {Z}')
print(f'Theta = {Theta}')
print(f'resolution = {resolution}')
print(f'runs = {runs}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Create IO datasets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

print("Creating datasets...")
dataset_train_runs = []
dataset_test_runs = []
theta_true_runs = []
grb_models_train_runs = []
grb_models_test_runs = []

tic_dataset = time.time()
for run in range(runs):
    Q_temp = 0.5*(2*np.random.rand(n, n)-1)
    Qyy_true = np.matmul(Q_temp, Q_temp.T)
    Qyz_true = np.random.rand(n, n)
    qy_true = np.random.rand(n)
    qz_true = np.random.rand(n)
    theta_true = np.concatenate(
        (Qyy_true.flatten('F'), Qyz_true.flatten('F'), qy_true, qz_true)
    )
    theta_true_runs.append(theta_true)

    dataset_train, dataset_test = create_datasets(
        theta_true, FOP_MIQP, m, n, n, N_train, N_test
    )

    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Tested approaches
# approaches = ['SL-MIQP', 'ASL-MIQP-z', 'ASL-MIQP-yz']
approaches = ['SL-MIQP', 'ASL-MIQP-z']
len_prob = len(approaches)

theta_diff_hist = np.empty((len_prob, runs, resolution))
x_diff_train_hist = np.empty((len_prob, runs, resolution))
x_diff_test_hist = np.empty((len_prob, runs, resolution))
obj_diff_train_hist = np.empty((len_prob, runs, resolution))
obj_diff_test_hist = np.empty((len_prob, runs, resolution))

gap = round(N_train/resolution)
N_list = np.linspace(gap, N_train, resolution, dtype=int).tolist()
for p_index, approach in enumerate(approaches):
    print(f'Approach: {approach}')

    if approach == 'SL-MIQP':
        dist_func_z = None
        add_y = False
    elif approach == 'ASL-MIQP-z':
        dist_func_z = L2
        add_y = False
    elif approach == 'ASL-MIQP-yz':
        dist_func_z = L2
        add_y = True

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N_index, N in enumerate(N_list):
            theta_IO = iop.mixed_integer_quadratic(dataset_train[:N],
                                                   Z,
                                                   Theta=Theta,
                                                   phi1=phi1,
                                                   phi2=phi2,
                                                   dist_func_z=dist_func_z,
                                                   add_dist_func_y=add_y,
                                                   reg_param=reg_param)

            x_diff_train, obj_diff_train, theta_diff = iop.evaluate(
                theta_IO, dataset_train[:N], FOP_MIQP, dist_func,
                theta_true=theta_true, phi=phi
            )

            x_diff_test, obj_diff_test, _ = iop.evaluate(
                theta_IO, dataset_test, FOP_MIQP, dist_func,
                theta_true=theta_true, phi=phi
            )

            x_diff_train_hist[p_index, run, N_index] = x_diff_train
            obj_diff_train_hist[p_index, run, N_index] = obj_diff_train
            x_diff_test_hist[p_index, run, N_index] = x_diff_test
            obj_diff_test_hist[p_index, run, N_index] = obj_diff_test
            theta_diff_hist[p_index, run, N_index] = theta_diff

        print(f'{round(100*(run+1)/runs)}%')

    toc = time.time()
    print(f"Simulation time = {round(toc-tic,2)} seconds")
    print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

results = {}
results['approaches'] = approaches
results['N_list'] = N_list
results['theta_diff_hist'] = theta_diff_hist
results['x_diff_train_hist'] = x_diff_train_hist
results['x_diff_test_hist'] = x_diff_test_hist
results['obj_diff_train_hist'] = obj_diff_train_hist
results['obj_diff_test_hist'] = obj_diff_test_hist
plot_results(results)
