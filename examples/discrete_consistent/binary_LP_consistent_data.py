"""
InvOpt package example: binary LP with consistent data.

Author: Pedro Zattoni Scroccaro
"""
from os.path import dirname, abspath
import sys
import time
import numpy as np
from gurobipy import Model, GRB, quicksum
import polytope as pc
import cvxpy as cp
import gurobipy as gp
import invopt as iop

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import (
    binary_linear_FOP, linear_ind_func, linear_phi, L2, plot_results
)

np.random.seed(1)


def create_datasets(theta, FOP, n, t, N_train, N_test):
    """Create dataset for the IO problem."""
    dataset_train = []
    for i in range(N_train):
        flag = False
        while not flag:
            A = -np.random.rand(t, n)
            b = -np.random.rand(t)
            flag = (np.sum(A, axis=1) <= b).all()

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        flag = False
        while not flag:
            A = -np.random.rand(t, n)
            b = -np.random.rand(t)
            flag = (np.sum(A, axis=1) <= b).all()

        s_hat = (A, b)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def compute_extreme_points(N):
    """Compute extreme points of the set of consistent cost vectors."""
    C = []
    d = []
    # Set of consistent vector
    for t in range(N):
        s_hat, x_hat = dataset_train[t]
        A, b = s_hat
        for k in range(2**n):
            x_bin = iop.candidate_action(k, decision_space, n)
            if linear_ind_func(s_hat, x_bin):
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
    return extreme_p


def circumcenter(N):
    """
    Solve IO problem using circumcenter strategy.

    Reference
    ----------
    [1] Omar Besbes, Yuri Fonseca and Ilan Lobel.
    "Contextual Inverse Optimization: Offline and Online Learning."
    Operations research (2023).
    """
    extreme_p = compute_extreme_points(N)

    mdl = Model('Circumcenter')
    mdl.setParam('OutputFlag', 0)
    theta = mdl.addVars(n, vtype=GRB.CONTINUOUS)
    r = mdl.addVar(vtype=GRB.CONTINUOUS)

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


def ellip_incenter(N):
    """
    Solve IO problem using an ellpsoidal version of the incenter.

    Reference
    ----------
    [1] Pedro Zattoni Scroccaro, Bilge Atasoy and Peyman Mohajerin Esfahani.
    "Learning in Inverse Optimization: Incenter Cost, Augmented Suboptimality
    Loss, and Algorithms." arXiv:2305.07730 (2023).
    """
    theta = cp.Variable(n)
    A = cp.Variable((n, n), PSD=True)
    constraints = [theta >= 0]
    constraints += [cp.SOC(1, theta)]
    constraints += [A - 1000*np.eye(n) << 0]  # added to avoid ill-posedness

    # Add constraints
    for i in range(N):
        s_hat, x_hat = dataset_train[i]
        for k in range(2**n):
            x = iop.candidate_action(k, decision_space, n)
            if ind_func(s_hat, x):
                x_diff = x - x_hat
                constraints += [cp.SOC(theta @ x_diff, A @ x_diff)]

    obj = cp.Maximize(cp.log_det(A))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print(f'Optimal solution not found. CVXPY status code = {prob.status}.'
              ' Set the flag verbose=True for more details.')

    theta_IO = theta.value
    return theta_IO


def ellip_circumcenter(N):
    """
    Solves IO problem using the circumcenter of an ellipsoidal cone.

    References
    ----------
    [1] Omar Besbes, Yuri Fonseca and Ilan Lobel. "Contextual Inverse
    Optimization: Offline and Online Learning." Operations research (2023).
    [2] Pedro Zattoni Scroccaro, Bilge Atasoy and Peyman Mohajerin Esfahani.
    "Learning in Inverse Optimization: Incenter Cost, Augmented Suboptimality
    Loss, and Algorithms." arXiv:2305.07730 (2023).
    """
    extreme_p = compute_extreme_points(N)

    # Define optimization problem
    b = cp.Variable(n)
    A = cp.Variable((n, n), PSD=True)

    constraints = [A - 1000*np.eye(n) << 0]  # added to avoid ill-posedness

    for w in extreme_p:
        # Rescale extreme points so that they lie on the L2-sphere
        w = w/np.linalg.norm(w, 2)
        constraints += [cp.SOC(1, A @ w + b)]

    obj = cp.Maximize(cp.log_det(A))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    if prob.status != 'optimal':
        print(
            f'Optimal solution not found. CVXPY status code = {prob.status}.'
            'Set the flag verbose=True for more details.'
        )

    if b.value is None:
        b_opt = np.zeros(n)
    else:
        b_opt = b.value
    A_opt = A.value
    theta_IO = -np.linalg.inv(A_opt) @ b_opt

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
    """
    X_cuts = [[] for _ in range(N)]
    for _ in range(T):
        mdl = gp.Model()
        mdl.setParam('OutputFlag', 0)

        theta = mdl.addVars(n, vtype=gp.GRB.CONTINUOUS)
        mdl.addConstr(gp.quicksum(theta) == 1)

        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            for x in X_cuts[i]:
                mdl.addConstr(
                    gp.quicksum(theta[j]*(x_hat[j] - x[j]) for j in range(n))
                    <= 0
                )

        mdl.optimize()
        theta_IO = np.array([theta[i].X for i in range(n)])

        # Generate cuts. If not cuts are added, an optimal solution was found,
        # up to the tolerance error
        flag = 1
        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            x_i = binary_linear_FOP(theta_IO, s_hat)
            cost = theta_IO @ (x_hat - x_i)
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

    theta = mdl.addVars(n, vtype=gp.GRB.CONTINUOUS, name='theta')
    mdl.addConstr(gp.quicksum(theta) == 1)

    # Initialize objective function
    obj = 0

    # Add constraints
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
    mdl.optimize()

    if mdl.status != 2:
        print(f'Optimal solution not found. Gurobi status code={mdl.status}.')

    theta_IO = np.array([theta[i].X for i in range(n)])

    return theta_IO


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%

N_train = 50
N_test = 50
n = 5
t = 3
decision_space = 'binary'
ind_func = linear_ind_func
X = (decision_space, n, ind_func)
Theta = 'nonnegative'
resolution = 5
runs = 3

# Approaches to be tested
# approaches = [
#     'Feasibility', 'Incenter', 'Circumcenter', 'Ellip. incenter',
#     'Ellip. circumcenter', 'Cutting plane', 'Predictability loss'
# ]
approaches = ['Feasibility', 'Incenter', 'Circumcenter']

print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'n = {n}')
print(f't = {t}')
print(f'X = {X}')
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

    dataset_train, dataset_test = create_datasets(
        theta_true, binary_linear_FOP, n, t, N_train, N_test
    )
    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print('Done!')
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N_index, N in enumerate(N_list):
            if approach == 'Feasibility':
                theta_IO = iop.discrete_consistent(
                    dataset_train[:N], X, linear_phi, Theta=Theta
                )
            elif approach == 'Incenter':
                theta_IO = iop.discrete_consistent(
                    dataset_train[:N], X, linear_phi, Theta=Theta, dist_func=L2
                )
            elif approach == 'Circumcenter':
                theta_IO = circumcenter(N)
            elif approach == 'Ellip. incenter':
                theta_IO = ellip_incenter(N)
            elif approach == 'Ellip. circumcenter':
                theta_IO = ellip_circumcenter(N)
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
