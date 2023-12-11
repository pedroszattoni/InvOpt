"""
Utils module for IO experiments.

Author: Pedro Zattoni Scroccaro
"""


import numpy as np
from gurobipy import Model, GRB, quicksum
import matplotlib.pyplot as plt

# colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33',
#           '#a65628', '#f781bf']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#808080',
          '#a65628', '#FFD700']


def linear_phi(s, x):
    """Linear feature mapping."""
    return x


def L1(x1, x2):
    """L1 distance function."""
    return np.linalg.norm(x1-x2, 1)


def L2(x1, x2):
    """L2 distance function."""
    return np.linalg.norm(x1-x2)


def linear_ind_func(s, x):
    """Check if x satisfies constraint set inequality."""
    A, b = s
    return (A @ x <= b).all()


def binary_linear_FOP(theta, s, gurobi_params=None):
    """Forward optimization problem: binary linear program."""
    A, b = s
    m, n = A.shape
    p = len(theta)

    mdl = Model('FOP')
    mdl.setParam('OutputFlag', 0)

    x = mdl.addVars(n, vtype=GRB.BINARY, name='x')

    mdl.setObjective(quicksum(theta[i]*x[i] for i in range(p)), GRB.MINIMIZE)

    mdl.addConstrs(
        quicksum(A[j, i]*x[i] for i in range(n)) <= b[j] for j in range(m)
    )

    if gurobi_params is not None:
        for param, value in gurobi_params:
            mdl.setParam(param, value)

    mdl.optimize()

    if mdl.status == 2:
        x_opt = np.array([x[k].X for k in range(n)])
    elif mdl.status == 9:
        # Time limit reched. Return vector a all ones
        x_opt = np.ones(n)
    else:
        raise Exception(
            f'Optimal solution not found. Gurobi status code = {mdl.status}.'
        )

    return x_opt


def mean_percentiles(data, p1=5, p2=95):
    """
    Compute mean, p1th and p2th percentiles of the data along the first axis.

    For instance, if data is a (n, m), it returns a vectors of dimention (m,).

    Parameters
    ----------
    data : ndarray
        2D numpy array containing the data.
    p1 : float, optional
        First percentile. Default is 5.
    p2 : float, optional
        Second percentile. Default is 95.

    Returns
    -------
    mean : ndarray
        1D numpy array with means.
    perc1 : ndarray
        1D numpy array with p1th percentiles.
    perc2 : ndarray
        1D numpy array with p2th percentiles.

    """
    mean = np.mean(data, axis=0)
    perc1 = np.percentile(data, p1, axis=0)
    perc2 = np.percentile(data, p2, axis=0)

    return mean, perc1, perc2


def plot_results(results):
    """Plot results."""
    approaches = results['approaches']
    N_list = results['N_list']
    theta_diff_hist = results['theta_diff_hist']
    x_diff_train_hist = results['x_diff_train_hist']
    x_diff_test_hist = results['x_diff_test_hist']
    obj_diff_train_hist = results['obj_diff_train_hist']
    obj_diff_test_hist = results['obj_diff_test_hist']

    for a_index, approach in enumerate(approaches):
        theta_diff_runs = theta_diff_hist[a_index]
        x_diff_train_runs = x_diff_train_hist[a_index]
        x_diff_test_runs = x_diff_test_hist[a_index]
        obj_diff_train_runs = obj_diff_train_hist[a_index]
        obj_diff_test_runs = obj_diff_test_hist[a_index]

        x_diff_train_mean, x_diff_train_p5, x_diff_train_p95 = \
            mean_percentiles(x_diff_train_runs)
        obj_diff_train_mean, obj_diff_train_p5, obj_diff_train_p95 = \
            mean_percentiles(obj_diff_train_runs)
        x_diff_test_mean, x_diff_test_p5, x_diff_test_p95 = \
            mean_percentiles(x_diff_test_runs)
        obj_diff_test_mean, obj_diff_test_p5, obj_diff_test_p95 = \
            mean_percentiles(obj_diff_test_runs)
        theta_diff_mean, theta_diff_p5, theta_diff_p95 = \
            mean_percentiles(theta_diff_runs)

        color = colors[a_index]

        plt.rcParams["mathtext.fontset"] = 'cm'
        plt.rcParams['font.family'] = 'serif'

        plt.figure(1)
        plt.plot(N_list, theta_diff_mean, c=color, label=approach)
        plt.fill_between(
            N_list, theta_diff_p5, theta_diff_p95, alpha=0.3, facecolor=color
        )
        plt.ylabel(
            r'$\| \theta_{\mathrm{IO}} - \theta_{\mathrm{true}} \|_2$',
            fontsize=18
        )
        plt.xlabel(r'Number of training examples', fontsize=14)
        plt.grid(visible=True)
        # plt.legend(fontsize='14', loc='upper right')
        plt.legend(fontsize='12', loc='upper left')
        plt.tight_layout()

        plt.figure(2)
        plt.plot(N_list, x_diff_train_mean, c=color, label=approach)
        plt.fill_between(
            N_list, x_diff_train_p5, x_diff_train_p95, alpha=0.3,
            facecolor=color
        )
        plt.yscale('log')
        plt.ylabel(
            r'$\| x_{\mathrm{IO}} - x_{\mathrm{true}} \|_2$', fontsize=18
        )
        plt.xlabel(r'Number of training examples', fontsize=14)
        plt.grid(visible=True)
        # plt.legend(fontsize='14', loc='upper right')
        plt.legend(fontsize='12', loc='lower left', ncol=1)
        plt.tight_layout()

        plt.figure(3)
        plt.plot(N_list, obj_diff_train_mean, c=color, label=approach)
        plt.fill_between(
            N_list, obj_diff_train_p5, obj_diff_train_p95, alpha=0.3,
            facecolor=color
        )
        plt.yscale('log')
        plt.ylabel(
            (r'$\frac{\mathrm{Cost}_\mathrm{IO} - \mathrm{Cost}_\mathrm{true}}'
             r'{\mathrm{Cost}_\mathrm{true}}$'), fontsize=20
        )
        plt.xlabel(r'Number of training examples', fontsize=14)
        plt.grid(visible=True)
        # plt.legend(fontsize='14', loc='upper right')
        plt.legend(fontsize='12', loc='lower left', ncol=1)
        plt.tight_layout()

        plt.figure(4)
        plt.plot(N_list, x_diff_test_mean, c=color, label=approach)
        plt.fill_between(
            N_list, x_diff_test_p5, x_diff_test_p95, alpha=0.3, facecolor=color
        )
        plt.yscale('log')
        plt.ylabel(
            r'$\| x_{\mathrm{IO}} - x_{\mathrm{true}} \|_2$', fontsize=18
        )
        plt.xlabel(r'Number of training examples', fontsize=14)
        plt.grid(visible=True)
        # plt.legend(fontsize='14', loc='upper right')
        plt.legend(fontsize='12', loc='lower left', ncol=1)
        plt.tight_layout()

        plt.figure(5)
        plt.plot(N_list, obj_diff_test_mean, c=color, label=approach)
        plt.fill_between(
            N_list, obj_diff_test_p5, obj_diff_test_p95, alpha=0.3,
            facecolor=color
        )
        plt.yscale('log')
        plt.ylabel(
            (r'$\frac{\mathrm{Cost}_\mathrm{IO} - \mathrm{Cost}_\mathrm{true}}'
             r'{\mathrm{Cost}_\mathrm{true}}$'), fontsize=20
        )
        plt.xlabel(r'Number of training examples', fontsize=14)
        plt.grid(visible=True)
        # plt.legend(fontsize='14', loc='upper right')
        plt.legend(fontsize='12', loc='lower left', ncol=1)
        plt.tight_layout()
