"""
InvOpt package example: mixed-integer linear program.

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
import time
import numpy as np
from gurobipy import Model, GRB, quicksum
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


def FOP_MILP(theta, s):
    """Forward optimization approach."""
    A, B, c, _ = s
    m1, m2 = A.shape
    _, n = B.shape
    qy = theta[:m2]
    qz = theta[m2:]

    mdl = Model('MILP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVars(m2, vtype=GRB.CONTINUOUS, name='y')
    z = mdl.addVars(n, vtype=GRB.BINARY, name='z')

    mdl.setObjective(quicksum(qy[i]*y[i] for i in range(m2))
                     + quicksum(qz[i]*z[i] for i in range(n)), GRB.MINIMIZE)

    mdl.addConstrs(quicksum(A[k, i]*y[i] for i in range(m2))
                   + quicksum(B[k, j]*z[j] for j in range(n))
                   <= c[k] for k in range(m1))
    mdl.addConstrs(y[i] <= 1 for i in range(m2))

    mdl.optimize()

    if mdl.status == 2:
        y_opt = np.array([y[k].X for k in range(m2)])
        z_opt = np.array([z[k].X for k in range(n)])
    else:
        raise Exception('Optimal solution not found. Gurobi status code '
                        f'= {mdl.status}.')

    return (y_opt, z_opt)


def phi1(w, z):
    """Feature mapping."""
    return np.array([1])


def phi2(w, z):
    """Feature mapping."""
    return z


def phi(s, x):
    """Transform phi1 and phi2 into phi for MIP_linear case."""
    _, _, _, w = s
    y, z = x
    return np.concatenate((np.kron(phi1(w, z), y), phi2(w, z)))


def dist_func(x1, x2):
    """Distance function."""
    y1, z1 = x1
    y2, z2 = x2
    dy = L2(y1, y2)
    dz = L2(z1, z2)
    return dy+dz


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 30
N_test = 30
n = 4
m = 3
decision_space = ('binary', n)
Theta = 'nonnegative'
resolution = 10
runs = 3

print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'n = {n}')
print(f'm = {m}')
print(f'decision_space = {decision_space}')
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
    qy_true = np.random.rand(n)
    qz_true = np.random.rand(n)
    theta_true = np.concatenate((qy_true, qz_true))
    theta_true_runs.append(theta_true)

    dataset_train, dataset_test = create_datasets(theta_true,
                                                  FOP_MILP,
                                                  m, n, n,
                                                  N_train, N_test)

    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Tested approaches
approaches = ['SL-MILP', 'ASL-MILP']
len_prob = len(approaches)

theta_diff_hist = np.empty((len_prob, runs, resolution))
x_diff_train_hist = np.empty((len_prob, runs, resolution))
x_diff_test_hist = np.empty((len_prob, runs, resolution))
obj_diff_train_hist = np.empty((len_prob, runs, resolution))
obj_diff_test_hist = np.empty((len_prob, runs, resolution))

gap = round(N_train/resolution)
N_list = np.linspace(gap, N_train, resolution, dtype=int).tolist()
for approach in approaches:
    p_index = approaches.index(approach)
    print(f'Approach: {approach}')

    if approach == 'SL-MILP':
        sub_loss = True
    else:
        sub_loss = False

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N in N_list:
            theta_IO = iop.MIP_linear(dataset_train[:N],
                                      decision_space,
                                      phi1=phi1,
                                      phi2=phi2,
                                      Theta=Theta,
                                      dist_func_z=L2,
                                      sub_loss=sub_loss)

            x_diff_train, obj_diff_train, theta_diff = \
                iop.evaluate(theta_IO,
                             dataset_train[:N],
                             FOP_MILP,
                             dist_func,
                             theta_true=theta_true,
                             phi=phi)

            x_diff_test, obj_diff_test, _ = iop.evaluate(theta_IO,
                                                         dataset_test,
                                                         FOP_MILP,
                                                         dist_func,
                                                         theta_true=theta_true,
                                                         phi=phi)

            N_index = N_list.index(N)
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
