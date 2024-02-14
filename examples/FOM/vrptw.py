"""
InvOpt package example: vehicle routing problem with time-windows.

Author: Pedro Zattoni Scroccaro
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
import pyvrp as pv
import gurobipy as gp
from vrptw_utils import (
    create_new_instance, get_routes, solve_VRPTW, routes_to_vec
)
import invopt as iop

np.random.seed(0)


# %%%%%%%%%%%%%%%%%%%%%%%%%%% IO functions %%%%%%%%%%%%%%%%%%%%%%%%%%%

def step_size_func(t):
    """Step-size function."""
    if step_size_type == '1/t':
        ss = step_size_constant/(1+t)
    elif step_size_type == '1/sqrt(t)':
        ss = step_size_constant/np.sqrt(1+t)
    return ss


def phi(s, x):
    """Feature mapping."""
    x_vec = routes_to_vec(x, n_total)
    return x_vec


def FOP(theta, s):
    """Forward optimization problem."""
    new_instance = create_new_instance(instance, theta, s, n_total, M)
    result = solve_VRPTW(new_instance, iter_limit)
    routes_list = get_routes(result)

    return routes_list


def FOP_approx(theta, s):
    """Approximate forward optimization problem."""
    new_instance = create_new_instance(instance, theta, s, n_total, M)
    result = solve_VRPTW(new_instance, iter_limit_approx)
    routes_list = get_routes(result)

    return routes_list


def FOP_aug_approx(theta, s, x):
    """Approximate augmented FOP."""
    x_vec = routes_to_vec(x, n_total)

    theta_aug = theta + (2*x_vec - 1)
    x_aug = FOP_approx(theta_aug, s)

    return x_aug


def callback(theta):
    """Store iterate."""
    return theta


def L1_dist(x1, x2):
    """L1 distance between two sets of routes."""
    x1_vec = routes_to_vec(x1, n_total)
    x2_vec = routes_to_vec(x2, n_total)
    return np.linalg.norm(x1_vec - x2_vec, 1)


def cutting_plane(dataset, T):
    """
    Solve IO problem using a cutting plane method.

    Parameters
    ----------
    dataset : list
        Dataste containing training examples.
    T : int
        Maximum number of cuts per example.
    tol : float
        Optimal solution tolerance.

    Returns
    -------
    theta_IO : list
        List of cost vectors, one for each iteration of the CP algorithm.

    References
    ----------
    [1] Lizhi Wang. "Cutting plane algorithms for the inverse mixed integer
    linear programming problem." Operations research letters (2009).
    [2] Merve Bodur, Timothy Chan, and Ian Yihang Zhu. "Inverse mixed
    integer optimization: Polyhedral insights and trust region methods."
    INFORMS Journal on Computing (2022).
    """
    theta_IO = theta_0
    theta_IO_list = [theta_IO]
    N = len(dataset)
    n = n_total**2
    X_cuts = [[] for _ in range(N)]
    mdl = gp.Model()
    mdl.setParam('OutputFlag', 0)
    theta = mdl.addVars(n, vtype=gp.GRB.CONTINUOUS)
    theta_abs = mdl.addVars(n, vtype=gp.GRB.CONTINUOUS)

    mdl.setObjective(
        gp.quicksum(theta_abs[j] for j in range(n)), gp.GRB.MINIMIZE
    )

    mdl.addConstrs(theta_0[j] - theta[j] <= theta_abs[j] for j in range(n))
    mdl.addConstrs(theta_0[j] - theta[j] >= -theta_abs[j] for j in range(n))

    for _ in range(T):
        # Generate cuts. If no cuts are added, an optimal solution was found,
        # up to the tolerance error
        for i in range(N):
            s_hat, x_hat = dataset[i]
            x_i = FOP_approx(theta_IO, s_hat)
            cost = theta_IO @ (phi(s_hat, x_hat) - phi(s_hat, x_i))
            if cost > 0:
                X_cuts[i].append(x_i)

        for i in range(N):
            s_hat, x_hat = dataset[i]
            for x in X_cuts[i]:
                phi_x_hat = phi(s_hat, x_hat)
                phi_x = phi(s_hat, x)
                mdl.addConstr(
                    gp.quicksum(
                        theta[j]*(phi_x_hat[j] - phi_x[j]) for j in range(n)
                    ) <= 0
                )

        mdl.optimize()
        theta_IO = np.array([theta[i].X for i in range(n)])
        theta_IO_list.append(theta_IO)

    return theta_IO_list


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%
N = 10
iter_limit = 50
iter_limit_approx = 50
batch_type = 'reshuffled'
step_size_constant = 0.2
step_size_type = '1/t'
epochs = 3
approach = 'exponentiated'
resolution = 1
theta_0_type = 'euclidean'
dataset = '200_10000'

# Path to data folder
data_folder = str(Path(__file__).parent) + r'\vrptw_data\\'

file_original = 'ORTEC-VRPTW-ASYM-cc05bba4-d1-n200-k15.txt'

instance = pv.read(data_folder + file_original, round_func="round")
theta_true = instance.distance_matrix().astype(float).flatten()

M = 1e8
n_clients = instance.num_clients
n_total = n_clients + 1

if theta_0_type == 'euclidean':
    theta_0 = np.ones((n_total, n_total))
    for i in range(n_total):
        for j in range(n_total):
            xi = instance.client(i).x
            yi = instance.client(i).y
            xj = instance.client(j).x
            yj = instance.client(j).y
            theta_0[i, j] = np.sqrt((xi - xj)**2 + (yi - yj)**2)
    theta_0 = theta_0.flatten()
    theta_0 = (M/sum(theta_0))*theta_0
elif theta_0_type == 'uniform':
    theta_0 = np.ones((n_total, n_total))
    np.fill_diagonal(theta_0, 0)
    theta_0 = theta_0.flatten()
    theta_0 = (M/sum(theta_0))*theta_0


print('')
print(f'N = {N}')
print(f'iter_limit = {iter_limit}')
print(f'iter_limit_approx = {iter_limit_approx}')
print(f'batch_type = {batch_type}')
print(f'step_size_type = {step_size_type}')
print(f'step_size_constant = {step_size_constant}')
print(f'epochs = {epochs}')
print(f'approach = {approach}')
print(f'resolution = {resolution}')
print(f'theta_0_type = {theta_0_type}')
print(f'dataset = {dataset}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Load IO datasets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset_ID = 0  # 0 to 4
dataset_train_name = (
    'dataset_train_' + dataset + '_v' + str(dataset_ID) + '.p'
)
dataset_test_name = (
    'dataset_test_' + dataset + '_v' + str(dataset_ID) + '.p'
)
dataset_train = pickle.load(open(data_folder + dataset_train_name, "rb"))
dataset_test = pickle.load(open(data_folder + dataset_test_name, "rb"))

dataset_train = [dataset_train[i] for i in range(N)]
dataset_test = [dataset_test[i] for i in range(N)]


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Initialize arrays to store results
theta_diff_hist = np.empty(epochs + 1)
x_diff_train_hist = np.empty(epochs + 1)
x_diff_test_hist = np.empty(epochs + 1)
obj_diff_train_hist = np.empty(epochs + 1)
obj_diff_test_hist = np.empty(epochs + 1)

if batch_type == 'reshuffled':
    T = epochs
    resolution_N = resolution
else:
    batch_type = float(batch_type)
    T = epochs*N
    resolution_N = resolution*N


tic = time.time()
if approach == 'cutting_plane':
    theta_IO_list = cutting_plane(dataset_train, epochs)
else:
    theta_IO_list = iop.FOM(
        dataset_train, phi, theta_0, FOP_aug_approx, step_size_func, T,
        Theta='nonnegative',
        step=approach,
        batch_type=batch_type,
        callback=callback,
        callback_resolution=resolution_N,
    )


for theta_idx, theta_IO in enumerate(theta_IO_list):
    x_diff_train, obj_diff_train, theta_diff = iop.evaluate(
        theta_IO, dataset_train, FOP, L1_dist, theta_true=theta_true, phi=phi
    )

    x_diff_test, obj_diff_test, _ = iop.evaluate(
        theta_IO, dataset_test, FOP, L1_dist, theta_true=theta_true, phi=phi
    )

    x_diff_train_hist[theta_idx] = x_diff_train
    obj_diff_train_hist[theta_idx] = obj_diff_train
    x_diff_test_hist[theta_idx] = x_diff_test
    obj_diff_test_hist[theta_idx] = obj_diff_test
    theta_diff_hist[theta_idx] = theta_diff

toc = time.time()
print(f"Simulation time = {round(toc-tic,2)} seconds")
print('')

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

plt.rcParams["mathtext.fontset"] = 'cm'
plt.rcParams['font.family'] = 'serif'

idxs = np.arange(epochs + 1)

plt.figure(1)
plt.plot(idxs, theta_diff_hist, label=approach)
plt.ylabel(
    r'$\| \theta_{\mathrm{IO}} - \theta_{\mathrm{true}} \|_2$',
    fontsize=18
)
plt.xlabel('Epoch', fontsize=14)
plt.grid(visible=True)
plt.legend(fontsize='12')
plt.tight_layout()

plt.figure(2)
plt.plot(idxs, x_diff_train_hist, label=approach)
plt.yscale('log')
plt.ylabel(
    r'$\| x_{\mathrm{IO}} - x_{\mathrm{true}} \|_2$', fontsize=18
)
plt.xlabel('Epoch', fontsize=14)
plt.grid(visible=True)
plt.legend(fontsize='12')
plt.tight_layout()

plt.figure(3)
plt.plot(idxs, obj_diff_train_hist, label=approach)
plt.yscale('log')
plt.ylabel(
    (r'$\frac{\mathrm{Cost}_\mathrm{IO} - \mathrm{Cost}_\mathrm{true}}'
     r'{\mathrm{Cost}_\mathrm{true}}$'),
    fontsize=20
)
plt.xlabel('Epoch', fontsize=14)
plt.grid(visible=True)
plt.legend(fontsize='12')
plt.tight_layout()

plt.figure(4)
plt.plot(idxs, x_diff_test_hist, label=approach)
plt.yscale('log')
plt.ylabel(
    r'$\| x_{\mathrm{IO}} - x_{\mathrm{true}} \|_2$', fontsize=18
)
plt.xlabel('Epoch', fontsize=14)
plt.grid(visible=True)
plt.legend(fontsize='12')
plt.tight_layout()

plt.figure(5)
plt.plot(idxs, obj_diff_test_hist, label=approach)
plt.yscale('log')
plt.ylabel(
    (r'$\frac{\mathrm{Cost}_\mathrm{IO} - \mathrm{Cost}_\mathrm{true}}'
     r'{\mathrm{Cost}_\mathrm{true}}$'),
    fontsize=20
)
plt.xlabel('Epoch', fontsize=14)
plt.grid(visible=True)
plt.legend(fontsize='12')
plt.tight_layout()
