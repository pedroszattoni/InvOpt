"""
InvOpt package example: mixed-integer linear program.

Author: Pedro Zattoni Scroccaro
"""

from os.path import dirname, abspath
import sys
import time
import numpy as np
import gurobipy as gp
import invopt as iop

sys.path.append(dirname(dirname(abspath(__file__))))  # nopep8
from utils_examples import L2, plot_results

np.random.seed(0)


def create_datasets(theta, FOP, t, u, v, N_train, N_test):
    """Create datasets for the IO problem."""
    dataset_train = []
    for i in range(N_train):
        flag = False
        while not flag:
            A1 = -np.random.rand(t, u)
            B1 = -np.random.rand(t, v)
            c1 = -2*np.random.rand(t)
            flag = all(np.sum(A1, axis=1) + np.sum(B1, axis=1) <= c1)

        A2 = np.vstack((np.eye(u), -np.eye(u)))
        c2 = np.hstack((np.ones(u), np.zeros(u)))

        A = np.vstack((A1, A2))
        B = np.vstack((B1, np.zeros((2*u, v))))
        c = np.hstack((c1, c2))

        s_hat = (A, B, c, 0)
        x_hat = FOP(theta, s_hat)
        dataset_train.append((s_hat, x_hat))

    dataset_test = []
    for i in range(N_test):
        flag = False
        while not flag:
            A1 = -np.random.rand(t, u)
            B1 = -np.random.rand(t, v)
            c1 = -2*np.random.rand(t)
            flag = all(np.sum(A1, axis=1) + np.sum(B1, axis=1) <= c1)

        A2 = np.vstack((np.eye(u), -np.eye(u)))
        c2 = np.hstack((np.ones(u), np.zeros(u)))

        A = np.vstack((A1, A2))
        B = np.vstack((B1, np.zeros((2*u, v))))
        c = np.hstack((c1, c2))

        s_hat = (A, B, c, 0)
        x_hat = FOP(theta, s_hat)
        dataset_test.append((s_hat, x_hat))

    return dataset_train, dataset_test


def FOP_MILP(theta, s):
    """Forward optimization approach: mixed-integer linear program."""
    A, B, c, _ = s
    m1, m2 = A.shape
    _, n = B.shape
    qy = theta[:m2]
    qz = theta[m2:]

    mdl = gp.Model('MILP')
    mdl.setParam('OutputFlag', 0)
    y = mdl.addVars(m2, vtype=gp.GRB.CONTINUOUS, name='y')
    z = mdl.addVars(n, vtype=gp.GRB.BINARY, name='z')

    mdl.setObjective(
        gp.quicksum(qy[i]*y[i] for i in range(m2))
        + gp.quicksum(qz[i]*z[i] for i in range(n)), gp.GRB.MINIMIZE
    )

    mdl.addConstrs(
        gp.quicksum(A[k, i]*y[i] for i in range(m2))
        + gp.quicksum(B[k, j]*z[j] for j in range(n)) <= c[k]
        for k in range(m1)
    )
    mdl.addConstrs(y[i] <= 1 for i in range(m2))

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
    return np.array([1])


def phi2(w, z):
    """Feature mapping."""
    return z


def phi(s, x):
    """Transform phi1 and phi2 into phi for mixed_integer_linear function."""
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


def circumcenter_cut(N, T, tol, time_limit):
    """
    Solves IO problem using the circumcenter strategy.

    Due to the hardness of computing the circumcenter for mixed-integer IO
    problems, an approximate (cutting-plane) method is used.

    Reference
    ----------
    [1] Omar Besbes, Yuri Fonseca and Ilan Lobel.
    "Contextual Inverse Optimization: Offline and Online Learning."
    Operations research (2023).
    """
    extreme_p = []
    for _ in range(T):
        mdl = gp.Model('Circumcenter')
        mdl.setParam('OutputFlag', 0)
        qy_u = mdl.addVars(u, vtype=gp.GRB.CONTINUOUS, name='qy_u')
        qz_u = mdl.addVars(v, vtype=gp.GRB.CONTINUOUS, name='qy_u')
        r = mdl.addVar(vtype=gp.GRB.CONTINUOUS, name='r')
        mdl.setObjective(r, gp.GRB.MINIMIZE)

        for qy_cut, qz_cut in extreme_p:
            # Rescale extreme points so that they lie on the L2-sphere
            mdl.addConstr(
                gp.quicksum((qy_cut[i] - qy_u[i])**2 for i in range(u))
                + gp.quicksum((qz_cut[i] - qz_u[i])**2 for i in range(v)) <= r
            )

        mdl.setParam('NonConvex', 2)
        mdl.addConstr(
            gp.quicksum(qy_u[i]**2 for i in range(u))
            + gp.quicksum(qz_u[i]**2 for i in range(v)) == 1
        )

        mdl.optimize()

        qy_upper = np.array([qy_u[i].X for i in range(u)])
        qz_upper = np.array([qz_u[i].X for i in range(v)])

        # Lower (constraint) optmization problem
        qy_lower, qz_lower, opt_val = circumcenter_lower_opt(
            N, qy_upper, qz_upper, T, tol, time_limit
        )

        if opt_val > r.X + tol:
            extreme_p.append((qy_lower, qz_lower))
        else:
            break

    theta_upper = np.concatenate((qy_upper, qz_upper))
    return theta_upper


def circumcenter_lower_opt(N, qy_upper, qz_upper, T, tol, time_limit):
    """
    Solve circumcenter lower optimization problem.

    The lower optimization problem is also solved approximately using a
    cutting-plane approach.
    """
    X_cuts = [[] for _ in range(N)]
    for _ in range(T):
        # Initialize Gurobi model
        mdl = gp.Model('max_constraint')
        mdl.setParam('OutputFlag', 0)
        qy_l = mdl.addVars(u, vtype=gp.GRB.CONTINUOUS, name='qy_l')
        qz_l = mdl.addVars(v, vtype=gp.GRB.CONTINUOUS, name='qz_l')

        obj = gp.quicksum((qy_l[i] - qy_upper[i])**2 for i in range(u))
        obj += gp.quicksum((qz_l[i] - qz_upper[i])**2 for i in range(v))

        mdl.setObjective(obj, gp.GRB.MAXIMIZE)

        mdl.setParam('NonConvex', 2)
        mdl.addConstr(
            gp.quicksum(qy_l[i]**2 for i in range(u)) +
            gp.quicksum(qz_l[i]**2 for i in range(v)) == 1
        )

        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            y_hat, z_hat = x_hat
            for x in X_cuts[i]:
                y, z = x
                mdl.addConstr(
                    gp.quicksum(qy_l[j]*(y_hat[j] - y[j]) for j in range(u))
                    + gp.quicksum(qz_l[j]*(z_hat[j] - z[j]) for j in range(v))
                    <= 0
                )

        time_flag = 1
        extra_time = 0
        while time_flag == 1:
            mdl.setParam('TimeLimit', time_limit + extra_time)
            mdl.optimize()
            try:
                opt_val = mdl.objVal
                time_flag = 0
            except AttributeError:
                extra_time += time_limit

        if mdl.status != 2:
            print(
                'Optimal solution not found. Gurobi status code ='
                f'{mdl.status}.'
            )

        qy_lower = np.array([qy_l[i].X for i in range(u)])
        qz_lower = np.array([qz_l[i].X for i in range(v)])
        theta_lower = np.concatenate((qy_lower, qz_lower))

        flag = 1
        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            y_hat, z_hat = x_hat
            y_lower, z_lower = FOP_MILP(theta_lower, s_hat)
            cost = qy_lower @ (y_hat - y_lower) + qz_lower @ (z_hat - z_lower)
            if cost > tol:
                flag = 0
                X_cuts[i].append((y_lower, z_lower))

        if flag:
            break

    return qy_lower, qz_lower, opt_val


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

        qy = mdl.addVars(u, vtype=gp.GRB.CONTINUOUS)
        qz = mdl.addVars(v, vtype=gp.GRB.CONTINUOUS)
        mdl.addConstr(gp.quicksum(qy) + gp.quicksum(qz) == 1)

        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            y_hat, z_hat = x_hat

            for x in X_cuts[i]:
                y, z = x

                mdl.addConstr(
                    gp.quicksum(qy[j]*(y_hat[j] - y[j]) for j in range(u))
                    + gp.quicksum(qz[j]*(z_hat[j] - z[j]) for j in range(v))
                    <= 0
                )
        mdl.optimize()

        if mdl.status != 2:
            print(
                'Optimal solution not found. Gurobi status code ='
                f'{mdl.status}.'
            )

        qy_IO = np.array([qy[i].X for i in range(u)])
        qz_IO = np.array([qz[i].X for i in range(v)])
        theta_IO = np.concatenate((qy_IO, qz_IO))

        # Generate cuts. If not cuts are added, an optimal solution was found,
        # up to the tolerance error
        flag = 1
        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            y_hat, z_hat = x_hat

            y_IO, z_IO = FOP_MILP(theta_IO, s_hat)
            cost = qy_IO @ (y_hat - y_IO) + qz_IO @ (z_hat - z_IO)
            if cost > tol:
                flag = 0
                X_cuts[i].append((y_IO, z_IO))

        if flag:
            break
    return theta_IO


def predictability_loss(N, T, tol):
    """
    Solve IO problem using the predictability loss.

    Due to the hardness of using the predictability loss for mixed-integer IO
    problems, an approximate (cutting-plane) method is used.

    References
    ----------
    [1] Anil Aswani, Zuo-Jun Shen, and Auyon Siddiq. "Inverse optimization
    with noisy data." Operations Research (2018).
    [2] Peyman Mohajerin Esfahani, Soroosh Shafieezadeh-Abadeh, Grani
    Hanasusanto and Daniel Kuhn. "Data-driven inverse optimization with
    imperfect information." Operations Research (2018).
    """
    X = [[] for _ in range(N)]
    for _ in range(T):
        mdl = gp.Model()
        mdl.setParam('OutputFlag', 0)

        qy = mdl.addVars(u, vtype=gp.GRB.CONTINUOUS)
        qz = mdl.addVars(v, vtype=gp.GRB.CONTINUOUS)
        mdl.addConstr(gp.quicksum(qy) + gp.quicksum(qz) == 1)

        obj = 0
        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            y_hat, z_hat = x_hat
            A, B, c, _ = s_hat
            m1, m2 = A.shape
            _, n = B.shape

            y_p = mdl.addVars(u, vtype=gp.GRB.CONTINUOUS, name='y'+str(i))
            z_p = mdl.addVars(v, vtype=gp.GRB.BINARY, name='z'+str(i))

            obj += gp.quicksum((y_p[i] - y_hat[i])**2 for i in range(u))
            obj += gp.quicksum((z_p[i] - z_hat[i])**2 for i in range(v))

            mdl.addConstrs(
                gp.quicksum(A[k, i]*y_p[i] for i in range(m2))
                + gp.quicksum(B[k, j]*z_p[j] for j in range(n)) <= c[k]
                for k in range(m1)
            )
            mdl.addConstrs(y_p[i] <= 1 for i in range(m2))

            for x in X[i]:
                y, z = x
                mdl.addConstr(
                    gp.quicksum(qy[j]*(y_p[j] - y[j]) for j in range(u))
                    + gp.quicksum(qz[j]*(z_p[j] - z[j]) for j in range(v)) <= 0
                )

        mdl.setParam('NonConvex', 2)
        mdl.setParam('TimeLimit', 1*60)
        mdl.setObjective((1/N)*obj, gp.GRB.MINIMIZE)
        mdl.optimize()

        if mdl.status != 2:
            print('Optimal solution not found. Gurobi status '
                  f'code = {mdl.status}.')

        qy_IO = np.array([qy[i].X for i in range(u)])
        qz_IO = np.array([qz[i].X for i in range(v)])
        theta_IO = np.concatenate((qy_IO, qz_IO))

        flag = 1
        for i in range(N):
            s_hat, x_hat = dataset_train[i]
            y_IO, z_IO = FOP_MILP(theta_IO, s_hat)
            y_p = [
                mdl.getVarByName('y'+str(i)+'['+str(j)+']').X for j in range(u)
            ]
            z_p = [
                mdl.getVarByName('z'+str(i)+'['+str(j)+']').X for j in range(v)
            ]
            cost = (
                qy_IO @ (np.array(y_p) - y_IO) + qz_IO @ (np.array(z_p) - z_IO)
            )
            if cost > tol:
                flag = 0
                X[i].append((y_IO, z_IO))

        if flag:
            break
    return theta_IO

# %%%%%%%%%%%%%%%%%%%%%%%%%%% Simulation parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%


N_train = 30
N_test = 30
u = 4
v = u
t = 3
decision_space = 'binary'
Z = (decision_space, v, None)
Theta = 'nonnegative'
resolution = 5
runs = 3

# approaches = ['SL-MILP', 'ASL-MILP-z', 'ASL-MILP-yz', 'Circumcenter',
#               'Cutting plane', 'Predictability loss']
approaches = ['SL-MILP', 'ASL-MILP-z']

print('')
print(f'N_train = {N_train}')
print(f'N_test = {N_test}')
print(f'u = {u}')
print(f't = {t}')
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
    qy_true = np.random.rand(u)
    qz_true = np.random.rand(v)
    theta_true = np.concatenate((qy_true, qz_true))
    theta_true_runs.append(theta_true)

    dataset_train, dataset_test = create_datasets(
        theta_true, FOP_MILP, t, u, v, N_train, N_test
    )

    dataset_train_runs.append(dataset_train)
    dataset_test_runs.append(dataset_test)

toc_dataset = time.time()
print(f"Create dataset time = {round(toc_dataset-tic_dataset,2)} seconds")
print('')


# %%%%%%%%%%%%%%%%%%%%%%%%%%% Solve IO problem %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

    tic = time.time()
    for run in range(runs):
        dataset_train = dataset_train_runs[run]
        dataset_test = dataset_test_runs[run]
        theta_true = theta_true_runs[run]

        for N_index, N in enumerate(N_list):

            if approach == 'SL-MILP':
                theta_IO = iop.mixed_integer_linear(
                    dataset_train[:N], Z, phi1=phi1, phi2=phi2, Theta=Theta,
                    dist_func_z=None, add_dist_func_y=False
                )
            elif approach == 'ASL-MILP-z':
                theta_IO = iop.mixed_integer_linear(
                    dataset_train[:N], Z, phi1=phi1, phi2=phi2, Theta=Theta,
                    dist_func_z=L2, add_dist_func_y=False
                )
            elif approach == 'ASL-MILP-yz':
                theta_IO = iop.mixed_integer_linear(
                    dataset_train[:N], Z, phi1=phi1, phi2=phi2, Theta=Theta,
                    dist_func_z=L2, add_dist_func_y=True
                )
            elif approach == 'Circumcenter':
                T = 100
                tol = 1e-5
                time_limit = 1
                theta_IO = circumcenter_cut(N, T, tol, time_limit)
            elif approach == 'Cutting plane':
                T = 100
                tol = 1e-10
                theta_IO = cutting_plane(N, T, tol)
            elif approach == 'Predictability loss':
                T = 100
                tol = 1e-10
                theta_IO = predictability_loss(N, T, tol)

            x_diff_train, obj_diff_train, theta_diff = iop.evaluate(
                theta_IO, dataset_train[:N], FOP_MILP, dist_func,
                theta_true=theta_true, phi=phi
            )

            x_diff_test, obj_diff_test, _ = iop.evaluate(
                theta_IO, dataset_test, FOP_MILP, dist_func,
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
