"""
InvOpt package example: quadratic program.

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


def theta_to_Qq(theta, n):
    """Extract Q and q from cost vector theta."""
    Q = theta[:n**2].reshape((n, n))
    q = theta[n**2:]
    return Q, q


def Qq_to_theta(Q, q):
    """Vectorize Q and q to create cost vector theta."""
    theta = np.concatenate((Q.flatten('F'), q))
    return theta


def create_datasets(theta, FOP, n, t, N_train, N_test, noise_level):
    """Create dataset for the IO problem."""
    dataset_train = []
    for i in range(N_train):
        flag = False
        while not flag:
            A = 1 - 2*np.random.rand(t, n)
            b = -np.random.rand(t)
            for k in range(2**n):
                x_bin = iop.dec_to_bin(k, n)
                if (A @ x_bin <= b).all():
                    flag = True
                    break

        A2 = np.vstack((np.eye(n), -np.eye(n)))
        b2 = np.hstack((np.ones(n), np.zeros(n)))
        A = np.vstack((A, A2))
        b = np.hstack((b, b2))

        Q_noise_temp = 0.5*(2*np.random.rand(n, n)-1)
        Q_noise = np.matmul(Q_noise_temp, Q_noise_temp.T)
        q_noise = np.random.rand(n)
        noise = Qq_to_theta(Q_noise, q_noise)

        theta_noise = theta + noise_level*noise
        s_hat = (A, b, 0)
        x_hat = FOP(theta_noise, s_hat)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        flag = False
        while not flag:
            A = 1 - 2*np.random.rand(t, n)
            b = -np.random.rand(t)
            for k in range(2**n):
                x_bin = iop.dec_to_bin(k, n)
                if (A @ x_bin <= b).all():
                    flag = True
                    break

        A2 = np.vstack((np.eye(n), -np.eye(n)))
        b2 = np.hstack((np.ones(n), np.zeros(n)))
        A = np.vstack((A, A2))
        b = np.hstack((b, b2))

        s_hat = (A, b, 0)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def quadratic_FOP(theta, s):
    """Forward optimization approach: quadratic program."""
    A, b, _ = s
    t, n = A.shape
    Qxx, qx = theta_to_Qq(theta, n)

    if len(theta) != (n**2 + n):
        raise Exception('Dimentions do not match!')

    mdl = Model('QP')
    mdl.setParam('OutputFlag', 0)
    x = mdl.addVars(n, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x')

    mdl.setObjective(
        quicksum(Qxx[i, j]*x[i]*x[j] for i in range(n) for j in range(n))
        + quicksum(qx[i]*x[i] for i in range(n)),
        GRB.MINIMIZE
    )

    mdl.addConstrs(
        quicksum(A[k, i]*x[i] for i in range(n)) <= b[k] for k in range(t)
    )

    mdl.optimize()

    if mdl.status == 2:
        x_opt = np.array([x[k].X for k in range(n)])
    else:
        raise Exception(
            f'Optimal solution not found. Gurobi status code = {mdl.status}.'
        )

    return x_opt


def phi1(w):
    """Feature mapping."""
    return np.array([1])


def phi(s, x):
    """Transform phi1 into phi for continuous quadratic case."""
    _, _, w = s
    return np.concatenate((np.kron(x, x), np.kron(phi1(w), x)))


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 30
N_test = 30
n = 3
t = 2
noise_level = 0
kappa = 0
resolution = 10
runs = 3

print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'n = {n}')
print(f't = {t}')
print(f'noise_level = {noise_level}')
print(f'kappa = {kappa}')
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
    Qxx_true = np.matmul(Q_temp, Q_temp.T)
    qx_true = np.random.rand(n)
    theta_true = Qq_to_theta(Qxx_true, qx_true)
    theta_true_runs.append(theta_true)

    dataset_train, dataset_test = create_datasets(
        theta_true, quadratic_FOP, n, t, N_train, N_test, noise_level
    )
    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print('Done!')
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Tested approaches
approaches = ['SL', 'ASL']
len_appr = len(approaches)

# Initialize arrays to store results
theta_diff_hist = np.empty((len_appr, runs, resolution))
x_diff_train_hist = np.empty((len_appr, runs, resolution))
x_diff_test_hist = np.empty((len_appr, runs, resolution))
obj_diff_train_hist = np.empty((len_appr, runs, resolution))
obj_diff_test_hist = np.empty((len_appr, runs, resolution))

gap = round(N_train/resolution)
N_list = np.linspace(gap, N_train, resolution, dtype=int).tolist()
for p_index, approach in enumerate(approaches):
    print(f'Approach: {approach}')

    if approach == 'SL':
        add_y = False
    else:
        add_y = True

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N_index, N in enumerate(N_list):
            theta_IO = iop.continuous_quadratic(
                dataset_train[:N], phi1, add_dist_func_y=add_y, reg_param=kappa
            )

            x_diff_train, obj_diff_train, theta_diff = iop.evaluate(
                theta_IO, dataset_train[:N], quadratic_FOP, L2,
                theta_true=theta_true, phi=phi
            )

            x_diff_test, obj_diff_test, _ = iop.evaluate(
                theta_IO, dataset_test, quadratic_FOP, L2,
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
