"""
InvOpt package example: binary LP with consistent data.

Author: Pedro Zattoni Scroccaro
"""
from os.path import dirname, abspath
import sys
sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
import time
import numpy as np
from gurobipy import Model, GRB, quicksum
import polytope as pc
from utils_examples import (binary_linear_FOP, linear_X, linear_phi, L2,
                            plot_results)
import invopt as iop


np.random.seed(1)


def create_datasets(theta, FOP, n, m, N_train, N_test):
    """Create dataset for the IO problem."""
    dataset_train = []
    for i in range(N_train):
        flag = False
        while not flag:
            A = -np.random.rand(m, n)
            b = -np.random.rand(m)
            flag = (np.sum(A, axis=1) <= b).all()

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        flag = False
        while not flag:
            A = -np.random.rand(m, n)
            b = -np.random.rand(m)
            flag = (np.sum(A, axis=1) <= b).all()

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def circumcenter(dataset):
    """
    Solves IO problem using circumcenter strategy.

    Reference: Omar Besbes, Yuri Fonseca, Ilan Lobel, "Contextual Inverse
    Optimization: Offline and Online Learning", 2023.
    """
    N = len(dataset)
    n = len(dataset[0][1])

    mdl = Model('Circumcenter')
    mdl.setParam('OutputFlag', 0)
    theta = mdl.addVars(n, vtype=GRB.CONTINUOUS)
    r = mdl.addVar(vtype=GRB.CONTINUOUS)

    C = []
    d = []
    # Set of consistent vector
    for t in range(N):
        s_hat, x_hat = dataset[t]
        A, b = s_hat

        for k in range(2**n):
            x_bin = np.binary_repr(k).zfill(n)
            x_bin = np.array([int(x_bin[i]) for i in range(n)])
            if linear_X(s_hat, x_bin):
                C.append(x_hat - x_bin)
                d.append(0)

    # Theta is assumed to be nonnegative
    for i in range(n):
        C.append([-1 if i == j else 0 for j in range(n)])
        d.append(0)

    # Add sum(theta) = 1, i.e., the nonnegative faced of the L1 sphere
    C.append([1]*n)
    d.append(1)

    # Compute the extreme points of the set of consistent cost vector, and
    # exclude theta=0
    poly = pc.Polytope(np.array(C), np.array(d))
    extreme_p = pc.extreme(poly)
    z_i = np.where(np.all(extreme_p <= 1e-10, axis=1))
    extreme_p = np.delete(extreme_p, z_i, 0)

    mdl.setObjective(r, GRB.MINIMIZE)

    for w in extreme_p:
        # Rescale extreme points so that they lie on the L2-sphere
        w = w/np.linalg.norm(w, 2)
        mdl.addConstr(quicksum((w[i] - theta[i])**2 for i in range(n)) <= r)

    mdl.setParam('NonConvex', 2)
    mdl.addConstr(quicksum(theta[i]**2 for i in range(n)) == 1)

    mdl.optimize()

    theta_opt = np.array([theta[i].X for i in range(n)])

    return theta_opt


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 50
N_test = 50
n = 5
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
    theta_true = np.random.rand(n)
    theta_true_runs.append(theta_true)

    dataset_train, dataset_test = create_datasets(theta_true,
                                                  binary_linear_FOP,
                                                  n, m,
                                                  N_train, N_test)
    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print('Done!')
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Tested approaches
approaches = ['Feasibility', 'Incenter', 'Circumcenter']
len_appr = len(approaches)

# Initialize arrays to store results
theta_diff_hist = np.empty((len_appr, runs, resolution))
x_diff_train_hist = np.empty((len_appr, runs, resolution))
x_diff_test_hist = np.empty((len_appr, runs, resolution))
obj_diff_train_hist = np.empty((len_appr, runs, resolution))
obj_diff_test_hist = np.empty((len_appr, runs, resolution))

gap = round(N_train/resolution)
N_list = np.linspace(gap, N_train, resolution, dtype=int).tolist()
for approach in approaches:
    p_index = approaches.index(approach)
    print(f'Approach: {approach}')

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N in N_list:
            if approach == 'Circumcenter':
                theta_IO = circumcenter(dataset_train[:N])
            else:
                feas = (approach == 'Feasibility')
                theta_IO = iop.discrete_model_consistent(dataset_train[:N],
                                                         linear_phi,
                                                         decision_space,
                                                         X=linear_X,
                                                         dist_func=L2,
                                                         Theta=Theta,
                                                         feasibility=feas)

            x_diff_train, obj_diff_train, theta_diff = \
                iop.evaluate(theta_IO,
                             dataset_train[:N],
                             binary_linear_FOP,
                             L2,
                             theta_true=theta_true,
                             phi=linear_phi)

            x_diff_test, obj_diff_test, _ = iop.evaluate(theta_IO,
                                                         dataset_test,
                                                         binary_linear_FOP,
                                                         L2,
                                                         theta_true=theta_true,
                                                         phi=linear_phi)

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
