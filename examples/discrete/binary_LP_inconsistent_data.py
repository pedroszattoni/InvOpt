"""
InvOpt package example: binary LP with inconsistent data.

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import (binary_linear_FOP, linear_ind_func, linear_phi, L2,
                            plot_results)

import invopt as iop

np.random.seed(0)


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

        theta_noise = theta + noise_level*np.random.randn(n)
        s_hat = (A, b)
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

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 50
N_test = 50
n = 5
t = 3
noise_level = 0.05
kappa = 0.001
X = ('binary', n, linear_ind_func)
resolution = 10
runs = 3

print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'n = {n}')
print(f't = {t}')
print(f'noise_level = {noise_level}')
print(f'kappa = {kappa}')
print(f'X = {X}')
print(f'resolution = {resolution}')
print(f'runs = {runs}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Create IO datasets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

dataset_train_runs = []
dataset_test_runs = []
theta_true_runs = []
grb_models_train_runs = []
grb_models_test_runs = []

print("Creating datasets...")
tic_dataset = time.time()
for run in range(runs):
    theta_true = 2*np.random.rand(n)-1
    theta_true_runs.append(theta_true)

    dataset_train, dataset_test = create_datasets(theta_true,
                                                  binary_linear_FOP, n, t,
                                                  N_train, N_test, noise_level)
    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Tested approaches
approaches = ['SL', 'ASL']
len_prob = len(approaches)

# Initialize arrays to store results
theta_diff_hist = np.empty((len_prob, runs, resolution))
x_diff_train_hist = np.empty((len_prob, runs, resolution))
x_diff_test_hist = np.empty((len_prob, runs, resolution))
obj_diff_train_hist = np.empty((len_prob, runs, resolution))
obj_diff_test_hist = np.empty((len_prob, runs, resolution))

gap = round(N_train/resolution)
N_list = np.linspace(gap, N_train, resolution, dtype=int).tolist()
for a_index, approach in enumerate(approaches):
    print(f'Approach: {approach}')

    if approach == 'SL':
        dist_func = None
        reg_param = 0
    else:
        dist_func = L2
        reg_param = kappa

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N_index, N in enumerate(N_list):
            theta_IO = iop.discrete(dataset_train[:N],
                                    X,
                                    linear_phi,
                                    reg_param=reg_param,
                                    dist_func=dist_func)

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

            x_diff_train_hist[a_index, run, N_index] = x_diff_train
            obj_diff_train_hist[a_index, run, N_index] = obj_diff_train
            x_diff_test_hist[a_index, run, N_index] = x_diff_test
            obj_diff_test_hist[a_index, run, N_index] = obj_diff_test
            theta_diff_hist[a_index, run, N_index] = theta_diff

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
