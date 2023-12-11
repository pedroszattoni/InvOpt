"""
InvOpt package example: first-order methods.

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import (
    binary_linear_FOP, linear_phi, L2, linear_ind_func, L1, mean_percentiles,
    colors
)

import invopt as iop

np.random.seed(0)


def create_datasets(theta, FOP, n, t, N_train, N_test):
    """Create dataset for the IO approach."""
    dataset_train = []
    for i in range(N_train):
        flag = False
        while not flag:
            A = -np.random.rand(t, n)
            b = -(n/3)*np.random.rand(t)
            flag = all(np.sum(A, axis=1) <= b)

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        flag = False
        while not flag:
            A = -np.random.rand(t, n)
            b = -(n/3)*np.random.rand(t)
            flag = all(np.sum(A, axis=1) <= b)

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def FOP_aug(theta, s_hat, x_hat):
    """Augmented FOP for binary FOP with d(x,x_hat) = || x - x_hat ||_1."""
    theta_aug = theta + 2*x_hat - 1
    x_aug = binary_linear_FOP(theta_aug, s_hat)

    return x_aug


def FOP_aug_approx(theta, s_hat, x_hat):
    """Approximated (i.e., time limited) augmented FOP."""
    theta_aug = theta + 2*x_hat - 1
    gurobi_params = [('TimeLimit', time_limit)]
    x_aug = binary_linear_FOP(theta_aug, s_hat, gurobi_params)

    return x_aug


def step_size(t):
    """Step size function."""
    return step_size_constant/np.sqrt(t+1)


def callback(theta):
    """Store input and current time."""
    return theta, time.time()


def linear_interpolation(values, x_new):
    """Linear interpolation between two points."""
    x1, y1, x2, y2 = values
    try:
        t = (x_new - x1)/(x2 - x1)
    except ZeroDivisionError:
        t = 0
    y_new = (1 - t)*y1 + t*y2
    return y_new


def interpolate(t_old, values_old, t_new):
    """Linear interpolation of list of values according to new timestamps."""
    t_list = t_new.copy()
    values_new_list = []
    t_new_list = []

    current_index = 0
    current_points = (
        t_old[current_index], values_old[current_index],
        t_old[current_index+1], values_old[current_index+1]
    )
    while t_list:
        t_new = t_list[0]
        if t_new <= current_points[2]:
            t_new_list.append(t_new)
            v_new = linear_interpolation(current_points, t_new)
            values_new_list.append(v_new)
            t_list.pop(0)
        else:
            current_index += 1
            current_points = (
                t_old[current_index], values_old[current_index],
                t_old[current_index+1], values_old[current_index+1]
            )

    return np.array(values_new_list)


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 30
N_test = 30
n = 5
t = 3
X = ('binary', n, linear_ind_func)
Theta = 'nonnegative'
regularizer = 'L1'
reg_param = 0.01
theta_0 = np.random.rand(n)
new_interval = 0.01
runs = 3
epoch_max = 5
normalize_grad = True
batch_ratio = 0.1
time_limit_approx = 0.01
step_size_constant = 0.5


print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'n = {n}')
print(f't = {t}')
print(f'X = {X}')
print(f'regularizer = {regularizer}')
print(f'reg_param = {reg_param}')
print(f'Theta = {Theta}')
print(f'theta_0 = {theta_0}')
print(f'runs = {runs}')
print(f'epoch_max = {epoch_max}')
print(f'step_size_constant = {step_size_constant}')
print(f'new_interval = {new_interval}')
print(f'batch_ratio = {batch_ratio}')
print(f'time_limit_approx = {time_limit_approx}')
print(f'normalize_grad = {normalize_grad}')
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Create IO datasets %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

theta_true_list = []
dataset_train_list = []
dataset_test_list = []
theta_opt_list = []
loss_opt_list = []

print('Creating datasets...')
tic_dataset = time.time()
for run in range(runs):
    theta_true = np.random.rand(n)
    theta_true_list.append(theta_true)

    dataset_train, dataset_test = create_datasets(
        theta_true, binary_linear_FOP, n, t, N_train, N_test
    )
    dataset_train_list.append(dataset_train)
    dataset_test_list.append(dataset_test)

    # Optimizer and optimal value
    theta_opt = iop.discrete(
        dataset_train, X, linear_phi, dist_func=L1, Theta=Theta,
        regularizer=regularizer, reg_param=reg_param
    )
    theta_opt_list.append(theta_opt)
    loss_opt = iop.ASL(
        theta_opt, dataset_train, FOP_aug, linear_phi, L1,
        regularizer=regularizer, reg_param=reg_param
    )
    loss_opt_list.append(loss_opt)

toc_dataset = time.time()
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO approach %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# SM: subgradient method.
# MD: mirror-descent. For this case, it corresponds to an exponentiated
#   subgradient method.
# SSM: stochastic subgradient method.
# ASM: approximate subgradient method.
# SMD: stochastic mirror descent
# AMD: approximate mirror descent
# SASM: stochastic approximate subgradient method
# SAMD: stochastic approximate mirror descent
approaches = ['SM', 'MD', 'SSM', 'ASM', 'SMD', 'AMD', 'SASM', 'SAMD']

x_diff_train_hist = []
obj_diff_train_hist = []
x_diff_test_hist = []
obj_diff_test_hist = []
theta_diff_hist = []
loss_diff_hist = []
time_list_hist = []

for approach in approaches:
    print(f'approach: {approach}')

    if approach == 'SM':
        step = 'standard'
        batch = 1
        time_limit = 600
    elif approach == 'MD':
        step = 'exponentiated'
        batch = 1
        time_limit = 600
    elif approach == 'SSM':
        step = 'standard'
        batch = batch_ratio
        time_limit = 600
    elif approach == 'ASM':
        step = 'standard'
        batch = 1
        time_limit = time_limit_approx
    elif approach == 'SMD':
        step = 'exponentiated'
        batch = batch_ratio
        time_limit = 600
    elif approach == 'AMD':
        step = 'exponentiated'
        batch = 1
        time_limit = time_limit_approx
    elif approach == 'SASM':
        step = 'standard'
        batch = batch_ratio
        time_limit = time_limit_approx
    elif approach == 'SAMD':
        step = 'exponentiated'
        batch = batch_ratio
        time_limit = time_limit_approx

    x_diff_train_appro = []
    obj_diff_train_appro = []
    x_diff_test_appro = []
    obj_diff_test_appro = []
    theta_diff_appro = []
    loss_diff_appro = []
    time_list_appro = []

    # Adjust number of iterations according to batch size
    T_max_alg = int(epoch_max/batch)

    tic = time.time()
    for run in range(runs):
        x_diff_train_runs = []
        obj_diff_train_runs = []
        x_diff_test_runs = []
        obj_diff_test_runs = []
        theta_diff_runs = []
        loss_diff_runs = []

        theta_true = theta_true_list[run]
        dataset_train = dataset_train_list[run]
        dataset_test = dataset_test_list[run]
        theta_opt = theta_opt_list[run]
        loss_opt = loss_opt_list[run]

        if step == 'exponentiated':
            reg_param_alg = 1/np.linalg.norm(theta_opt, 1)
        else:
            reg_param_alg = reg_param

        callback_results = iop.FOM(dataset_train, linear_phi, theta_0,
                                   FOP_aug_approx, step_size, T_max_alg,
                                   Theta=Theta,
                                   step=step,
                                   regularizer=regularizer,
                                   reg_param=reg_param_alg,
                                   batch_type=batch,
                                   callback=callback,
                                   normalize_grad=normalize_grad)

        # Split callback results into two lists
        theta_IO_list, time_list = map(list, zip(*callback_results))

        # Evaluate theta's from first-order algorithm
        for theta_IO in theta_IO_list:
            x_diff_train, obj_diff_train, theta_diff = iop.evaluate(
                theta_IO, dataset_train, binary_linear_FOP, L2,
                theta_true=theta_true, phi=linear_phi
            )

            x_diff_test, obj_diff_test, _ = iop.evaluate(
                theta_IO, dataset_test, binary_linear_FOP, L2,
                theta_true=theta_true, phi=linear_phi
            )

            loss = iop.ASL(theta_IO, dataset_train, FOP_aug, linear_phi, L1)

            x_diff_train_runs.append(x_diff_train)
            obj_diff_train_runs.append(obj_diff_train)
            x_diff_test_runs.append(x_diff_test)
            obj_diff_test_runs.append(obj_diff_test)
            theta_diff_runs.append(theta_diff)
            loss_diff_runs.append(loss - loss_opt)

        # Interpolated results according to a standard time interval. This way,
        # we can compare the performance of the algorithm in terms of time
        # instead of number of iterations.
        t_list_new = np.arange(
            time_list[0], time_list[-1], new_interval
        ).tolist()
        x_diff_train_appro.append(
            interpolate(time_list, x_diff_train_runs, t_list_new)
        )
        obj_diff_train_appro.append(
            interpolate(time_list, obj_diff_train_runs, t_list_new)
        )
        x_diff_test_appro.append(
            interpolate(time_list, x_diff_test_runs, t_list_new)
        )
        obj_diff_test_appro.append(
            interpolate(time_list, obj_diff_test_runs, t_list_new)
        )
        theta_diff_appro.append(
            interpolate(time_list, theta_diff_runs, t_list_new)
        )
        loss_diff_appro.append(
            interpolate(time_list, loss_diff_runs, t_list_new)
        )
        time_list_appro.append(t_list_new)

        print(f'{round(100*(run+1)/runs)}%')

    x_diff_train_hist.append(x_diff_train_appro)
    obj_diff_train_hist.append(obj_diff_train_appro)
    x_diff_test_hist.append(x_diff_test_appro)
    obj_diff_test_hist.append(obj_diff_test_appro)
    theta_diff_hist.append(theta_diff_appro)
    loss_diff_hist.append(loss_diff_appro)
    time_list_hist.append(time_list_appro)

    toc = time.time()
    print(f"Simulation time = {round(toc-tic,2)} seconds")
    print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Plot results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for a_index, approach in enumerate(approaches):
    # Retrieve results
    x_diff_train_appro = x_diff_train_hist[a_index]
    obj_diff_train_appro = obj_diff_train_hist[a_index]
    x_diff_test_appro = x_diff_test_hist[a_index]
    obj_diff_test_appro = obj_diff_test_hist[a_index]
    theta_diff_appro = theta_diff_hist[a_index]
    loss_diff_appro = loss_diff_hist[a_index]
    time_list_appro = time_list_hist[a_index]

    # Shift timesteps so that they start at 0
    timestamps = min(time_list_appro, key=len)
    timestamps = [t-timestamps[0] for t in timestamps]
    len_data = len(timestamps)

    # Trim data according to shortest run
    x_diff_train_appro = np.array([x[:len_data] for x in x_diff_train_appro])
    obj_diff_train_appro = np.array(
        [x[:len_data] for x in obj_diff_train_appro]
    )
    x_diff_test_appro = np.array([x[:len_data] for x in x_diff_test_appro])
    obj_diff_test_appro = np.array([x[:len_data] for x in obj_diff_test_appro])
    theta_diff_appro = np.array([x[:len_data] for x in theta_diff_appro])
    loss_diff_appro = np.array([x[:len_data] for x in loss_diff_appro])

    # Compute mean and percentiles
    x_diff_train_mean, x_diff_train_p5, x_diff_train_p95 = mean_percentiles(
        x_diff_train_appro
    )
    obj_diff_train_mean, obj_diff_train_p5, obj_diff_train_p95 = \
        mean_percentiles(obj_diff_train_appro)
    x_diff_test_mean, x_diff_test_p5, x_diff_test_p95 = mean_percentiles(
        x_diff_test_appro
    )
    obj_diff_test_mean, obj_diff_test_p5, obj_diff_test_p95 = mean_percentiles(
        obj_diff_test_appro
    )
    theta_diff_mean, theta_diff_p5, theta_diff_p95 = mean_percentiles(
        theta_diff_appro
    )
    loss_diff_mean, loss_diff_p5, loss_diff_p95 = mean_percentiles(
        loss_diff_appro
    )

    color = colors[a_index]

    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.rcParams['font.family'] = 'serif'

    plt.figure(1)
    plt.plot(timestamps, theta_diff_mean, c=color, label=approach)
    plt.fill_between(
        timestamps, theta_diff_p5, theta_diff_p95, alpha=0.3, facecolor=color
    )
    plt.ylabel(
        r'$\| \theta_{\mathrm{IO}} - \theta_{\mathrm{true}} \|_2$', fontsize=18
    )
    plt.xlabel(r'Time (s)', fontsize=14)
    plt.grid(visible=True)
    plt.legend(fontsize='14', loc='upper right')
    plt.tight_layout()

    plt.figure(2)
    plt.plot(timestamps, x_diff_train_mean, c=color, label=approach)
    plt.fill_between(
        timestamps, x_diff_train_p5, x_diff_train_p95, alpha=0.3,
        facecolor=color
    )
    plt.yscale('log')
    plt.ylabel(r'$\| x_{\mathrm{IO}} - x_{\mathrm{true}} \|_2$', fontsize=18)
    plt.xlabel(r'Time (s)', fontsize=14)
    plt.grid(visible=True)
    plt.legend(fontsize='14', loc='upper right')
    plt.tight_layout()

    plt.figure(3)
    plt.plot(timestamps, obj_diff_train_mean, c=color, label=approach)
    plt.fill_between(
        timestamps, obj_diff_train_p5, obj_diff_train_p95, alpha=0.3,
        facecolor=color
    )
    plt.yscale('log')
    plt.ylabel(
        (r'$\frac{\mathrm{Cost}_\mathrm{IO} - \mathrm{Cost}_\mathrm{true}}'
         r'{\mathrm{Cost}_\mathrm{true}}$'), fontsize=20
    )
    plt.xlabel(r'Time (s)', fontsize=14)
    plt.grid(visible=True)
    plt.legend(fontsize='14', loc='upper right')
    plt.tight_layout()

    plt.figure(4)
    plt.plot(timestamps, x_diff_test_mean, c=color, label=approach)
    plt.fill_between(
        timestamps, x_diff_test_p5, x_diff_test_p95, alpha=0.3, facecolor=color
    )
    plt.yscale('log')
    plt.ylabel(r'$\| x_{\mathrm{IO}} - x_{\mathrm{true}} \|_2$', fontsize=18)
    plt.xlabel(r'Time (s)', fontsize=14)
    plt.grid(visible=True)
    plt.legend(fontsize='14', loc='upper right')
    plt.tight_layout()

    plt.figure(5)
    plt.plot(timestamps, obj_diff_test_mean, c=color, label=approach)
    plt.fill_between(
        timestamps, obj_diff_test_p5, obj_diff_test_p95, alpha=0.3,
        facecolor=color
    )
    plt.yscale('log')
    plt.ylabel(
        (r'$\frac{\mathrm{Cost}_\mathrm{IO} - \mathrm{Cost}_\mathrm{true}}' +
         r'{\mathrm{Cost}_\mathrm{true}}$'), fontsize=20
    )
    plt.xlabel(r'Time (s)', fontsize=14)
    plt.grid(visible=True)
    plt.legend(fontsize='14', loc='upper right')
    plt.tight_layout()

    plt.figure(6)
    plt.plot(timestamps, loss_diff_mean, c=color, label=approach)
    plt.fill_between(
        timestamps, loss_diff_p5, loss_diff_p95, alpha=0.3, facecolor=color
    )
    plt.yscale('log')
    plt.ylabel(r'Training loss gap', fontsize=14)
    plt.xlabel(r'Time (s)', fontsize=14)
    plt.grid(visible=True)
    plt.legend(fontsize='14', loc='upper right')
    plt.tight_layout()
