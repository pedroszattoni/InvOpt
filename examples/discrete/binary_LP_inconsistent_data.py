"""
InvOpt package example: binary LP with inconsistent data.

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np
import cvxpy as cp
import gurobipy as gp
import invopt as iop

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import (
    binary_linear_FOP, linear_ind_func, linear_phi, L2, plot_results
)

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


def ellipsoidal_ASL(N):
    """
    Solve IO problem using an ellpsoidal version of the ASL.

    Reference
    ----------
    [1] Pedro Zattoni Scroccaro, Bilge Atasoy and Peyman Mohajerin Esfahani.
    "Learning in Inverse Optimization: Incenter Cost, Augmented Suboptimality
    Loss, and Algorithms." arXiv:2305.07730 (2023).
    """
    theta = cp.Variable(n)
    A = cp.Variable((n, n), PSD=True)
    beta = cp.Variable(N)
    constraints = [cp.SOC(1, theta)]
    constraints += [A - 1000*np.eye(n) << 0]  # added to avoid ill-posedness

    for i in range(N):
        s_hat, x_hat = dataset_train[i]
        for k in range(2**n):
            x = iop.candidate_action(k, decision_space, n)
            if ind_func(s_hat, x):
                x_diff = x - x_hat
                constraints += [
                    cp.SOC(theta @ x_diff + beta[i], A @ x_diff)
                ]

    obj = cp.Minimize(-kappa*cp.log_det(A) + (1/N)*cp.sum(beta))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print(
            f'Optimal solution not found. CVXPY status code = {prob.status}.'
            'Set the flag verbose=True for more details.'
        )

    theta_IO = theta.value
    return theta_IO


def cutting_plane(N, T, tol):
    """
    Solve IO problem using a cutting plane method.

    Parameters
    ----------
    N : int
        Number of traning examples.
    T : int
        Maximum number of cuts per example.
    tol : float
        Optimal solution tolerance.

    Returns
    -------
    theta_IO : 1D ndarray
        An optimal cost vector.

    References
    ----------
    [1] Lizhi Wang. "Cutting plane algorithms for the inverse mixed integer
    linear programming problem." Operations research letters (2009).
    [2] Merve Bodur, Timothy Chan, and Ian Yihang Zhu. "Inverse mixed
    integer optimization: Polyhedral insights and trust region methods."
    INFORMS Journal on Computing (2022).
    """
    X_cuts = [[] for _ in range(N)]
    for _ in range(T):
        mdl = gp.Model()
        mdl.setParam('OutputFlag', 0)

        theta = mdl.addVars(n, lb=-gp.GRB.INFINITY,  vtype=gp.GRB.CONTINUOUS)

        obj = 0
        for i in range(N):
            s_hat, x_hat = dataset_train[i]

            theta_i = mdl.addVars(
                n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS,
                name='theta'+str(i)
            )
            theta_abs = mdl.addVars(n, vtype=gp.GRB.CONTINUOUS)

            obj += gp.quicksum(theta_abs[j] for j in range(n))
            mdl.addConstrs(
                theta_i[j] - theta[j] <= theta_abs[j] for j in range(n)
            )
            mdl.addConstrs(
                theta_i[j] - theta[j] >= -theta_abs[j] for j in range(n)
            )

            for x in X_cuts[i]:
                mdl.addConstr(
                    gp.quicksum(theta_i[j]*(x_hat[j] - x[j])
                                for j in range(n)) <= 0
                )

        mdl.setObjective((1/N)*obj, gp.GRB.MINIMIZE)
        # Search over facets of unit L-infinity sphere for the
        # solution with the lowest objective value.
        best_obj = np.inf
        for i in range(n):
            for j in [-1, 1]:
                cons = mdl.addConstr(theta[i] == j)
                mdl.optimize()
                mdl.remove(cons)
                # If an optimal solution was found
                if mdl.status == 2:
                    obj_val = mdl.objVal
                    if obj_val < best_obj:
                        best_obj = obj_val
                        theta_IO = np.array([theta[i].X for i in range(n)])

        # Generate cuts. If not cuts are added, an optimal solution was found,
        # up to the tolerance error
        flag = 1
        for i in range(N):
            s_hat, x_hat = dataset_train[i]

            theta_i = [
                mdl.getVarByName('theta' + str(i)+'['+str(j)+']').X
                for j in range(n)
            ]
            theta_i = np.array(theta_i)

            x_i = binary_linear_FOP(theta_i, s_hat)
            cost = theta_i @ (x_hat - x_i)
            if cost > tol:
                flag = 0
                X_cuts[i].append(x_i)

        if flag:
            break
    return theta_IO


def predictability_loss(N):
    """
    Solve IO problem using the predictability loss.

    References
    ----------
    [1] Anil Aswani, Zuo-Jun Shen, and Auyon Siddiq. "Inverse optimization
    with noisy data." Operations Research (2018).
    [2] Peyman Mohajerin Esfahani, Soroosh Shafieezadeh-Abadeh, Grani
    Hanasusanto and Daniel Kuhn. "Data-driven inverse optimization with
    imperfect information." Operations Research (2018).
    """
    mdl = gp.Model()
    mdl.setParam('OutputFlag', 0)

    theta = mdl.addVars(
        n, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='theta'
    )

    obj = 0
    for i in range(N):
        s_hat, x_hat = dataset_train[i]
        A_hat, b_hat = s_hat
        y = mdl.addVars(n, vtype=gp.GRB.BINARY, name='y'+str(i))
        mdl.addConstrs(
            gp.quicksum(A_hat[j, i]*y[i] for i in range(n)) <= b_hat[j]
            for j in range(t)
        )

        obj += gp.quicksum((y[i] - x_hat[i])**2 for i in range(n))

        for k in range(2**n):
            x = iop.candidate_action(k, decision_space, n)
            if ind_func(s_hat, x):
                mdl.addConstr(
                    gp.quicksum(theta[i]*(x[i] - y[i]) for i in range(n)) >= 0
                )

    mdl.setObjective((1/N)*obj, gp.GRB.MINIMIZE)
    mdl.setParam('TimeLimit', 10)

    # Search over facets of unit L-infinity sphere for the solution
    # with the lowest objective value.
    best_obj = np.inf
    theta_IO = 2*np.random.rand(n)-1
    for i in range(n):
        for j in [-1, 1]:
            cons = mdl.addConstr(theta[i] == j)
            mdl.optimize()
            mdl.remove(cons)
            if (mdl.status == 2) or (mdl.status == 9):
                try:
                    obj_val = mdl.objVal
                except AttributeError:
                    obj_val = np.inf
                if obj_val < best_obj:
                    best_obj = obj_val
                    theta_IO = np.array([theta[i].X for i in range(n)])

    return theta_IO

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%


N_train = 50
N_test = 50
n = 5
t = 3
noise_level = 0.05
kappa = 0.001
decision_space = 'binary'
ind_func = linear_ind_func
X = (decision_space, n, ind_func)
resolution = 5
runs = 3

approaches = ['SL', 'ASL']
# approaches = [
#     'SL', 'ASL', 'Ellipsoidal ASL', 'Cutting plane', 'Predictability loss'
# ]

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

    dataset_train, dataset_test = create_datasets(
        theta_true, binary_linear_FOP, n, t, N_train, N_test, noise_level
    )
    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Number of IO approaches
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

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N_index, N in enumerate(N_list):
            if approach == 'SL':
                theta_IO = iop.discrete(dataset_train[:N], X, linear_phi)
            elif approach == 'ASL':
                theta_IO = iop.discrete(
                    dataset_train[:N], X, linear_phi, reg_param=kappa,
                    dist_func=L2
                )
            elif approach == 'Ellipsoidal ASL':
                theta_IO = ellipsoidal_ASL(N)
            elif approach == 'Cutting plane':
                T = 100
                tol = 1e-10
                theta_IO = cutting_plane(N, T, tol)
            elif approach == 'Predictability loss':
                theta_IO = predictability_loss(N)

            x_diff_train, obj_diff_train, theta_diff = iop.evaluate(
                theta_IO, dataset_train[:N], binary_linear_FOP, L2,
                theta_true=theta_true, phi=linear_phi
            )

            x_diff_test, obj_diff_test, _ = iop.evaluate(
                theta_IO, dataset_test, binary_linear_FOP, L2,
                theta_true=theta_true, phi=linear_phi
            )

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
