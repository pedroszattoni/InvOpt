"""
invopt: Inverse Optimization with Python.

Author: Pedro Zattoni Scroccaro
"""

import numpy as np
import warnings


def check_Theta(Theta):
    """Check if Theta is valid."""
    if Theta not in [None, 'nonnegative']:
        raise Exception('Invalid Theta. Accepted values are: None (default) '
                        'and \'nonnegative\'.')


def check_decision_space(decision_space):
    """Check if decision_space is valid."""
    if decision_space[0] != 'binary':
        raise Exception('Invalid decision space. Accepted values are: ' +
                        ' tuple(\'binary\', n) (default).')


def check_regularizer(regularizer):
    """Check if regularizer is valid."""
    if regularizer not in ['L2_squared', 'L1']:
        raise Exception('Invalid regularizer. Accepted values are:' +
                        ' \'L2_squared\' (default) and \'L1\'.')


def check_reg_parameter(reg_param):
    """Check if reg_param is valid."""
    if reg_param < 0:
        raise Exception('reg_param must be nonnegative.')


def check_dist_func(dist_func, sub_loss):
    """Check if dist_func is given when sub_loss=False."""
    if (not sub_loss) and (dist_func is None):
        raise Exception('dist_func required when sub_loss=False.')


def warning_large_decision_space(decision_space):
    """Warn user if decision space is binary and high-dimensional."""
    if (decision_space[0] == 'binary') and (decision_space[1] > 15):
        warnings.warn('Attention! Using this IO method for models with binary '
                      'decision variables requires solving an optimization '
                      'problem with potentially O(N*2^n) constraints, '
                      'where N is the number of training examples and n is '
                      'the dimension of the binary decision vector.')


def warning_dist_func_reg_param_sub_loss(dist_func, reg_param, sub_loss):
    """Warn user dist_func / reg_param are not necessary when sub_loss=True."""
    if sub_loss:
        if dist_func is not None:
            warnings.warn('dist_func not used when sub_loss=True.')

        if reg_param > 0:
            warnings.warn('reg_param not used when sub_loss=True.')


def normalize(vec, norm):
    """
    Normalize nonzero array according to some norm.

    Parameters
    ----------
    vec : 1D ndarray
        Array to be normalized.
    norm : {non-zero int, inf, -inf},
        Order of the norm. See numpy.linalg.norm documentation for more
        details.

    Returns
    -------
    vec : 1D ndarray
        Normalized array.

    """
    norm_vec = np.linalg.norm(vec, norm)
    if norm_vec > 0:
        vec = vec/norm_vec
    return vec


def ASL(theta, dataset, FOP_aug, phi, dist_func,
        regularizer='L2_squared',
        reg_param=0,
        theta_hat=0):
    """
    Evaluate augmented suboptimality loss.

    Parameters
    ----------
    theta : 1D ndarray
        Cost vector.
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    FOP_aug : callable
        Augmented forward optimization problem. Takes as input a cost vector
        theta, a signal s, and the respective response x. Returns the optimal
        augmented response. The distance function used to augment the FOP
        should be the same (or equivalent) to dist_func.
        Syntax: FOP_aug(theta, s, x).
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    dist_func : callable
        Distance function. Given two responses x1 and x2, returns the distance
        between them according to some distance metric.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector. The default is
        None.

    Raises
    ------
    Exception
        If invalid regularization parameter. If negative regularization
        parameter.

    Returns
    -------
    float
        Augmented suboptimality loss value.

    """
    # Check if inputs are valid
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)

    if theta_hat is None:
        theta_hat = 0

    if regularizer == 'L2_squared':
        reg_term = (reg_param/2)*np.linalg.norm(theta - theta_hat)**2
    elif regularizer == 'L1':
        reg_term = reg_param*np.linalg.norm(theta - theta_hat, 1)

    N = len(dataset)
    loss = 0
    for i in range(N):
        s_hat, x_hat = dataset[i]
        x = FOP_aug(theta, s_hat, x_hat)
        phi_diff = phi(s_hat, x_hat) - phi(s_hat, x)
        loss += theta @ phi_diff + dist_func(x_hat, x)

    return reg_term + (1/N)*loss


def dec_to_bin(decimal_value, num_bits):
    """
    Decimal to binary conversion.

    Transform decimal_value into its binary representation as an 1D numpy array
    of dimension num_bits.

    Parameters
    ----------
    decimal_value : int
        Decimal value to be converted.
    num_bits : int
        dimension of the converted binary number.

    Returns
    -------
    x_bin : 1D ndarray
        Binary representation of decimal_value as a binary array of dimension
        num_bits.

    """
    x_bin = np.array(list(format(decimal_value, f"0{num_bits}b")), dtype=int)

    return x_bin


def evaluate(theta, dataset, FOP, dist_func,
             theta_true=None,
             phi=None,
             scale_obj_diff=True):
    """
    Evaluate cost vector theta.

    Computes the average difference of the expert responses x with the one
    returned by the FOP using the cost vector theta. If true_theta is given,
    also computes the average objective value difference and cost vector
    difference.

    Parameters
    ----------
    theta : 1D ndarray
        Cost vector.
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    FOP : callable
        Forward optimization problem. Takes as input a cost vector
        theta and a signal s. Returns an optimal response for the problem.
        Syntax: FOP(theta, s).
    dist_func : callable
        Distance function. Given two responses x1 and x2, returns the distance
        between them according to some distance metric. Syntax:
        dist_func(x1, x2).
    theta_true : {1D ndarray, None}, optional
        Cost vector used to generate the data. The default is None.
    phi : {callable, None}, optional
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x). The default is None.
    scale_obj_diff : bool, optional
        When theta_true is given, scale objective value difference using
        theta_true. The default is True.

    Raises
    ------
    Exception
        If theta_true is given but phi function is not.

    Returns
    -------
    results : {float, tuple(float, float, float)}
        Returns the average response difference. If true_theta is given, also
        returns the average objective value difference and cost vector
        difference as a tuple.

    """
    if (theta_true is not None) and (phi is None):
        raise Exception('Feature function phi is required to compute '
                        'evaluation metrics using theta_true.')

    N = len(dataset)

    x_diff = 0
    theta_diff = 0
    obj_diff = 0
    for i in range(N):
        s_hat, x_hat = dataset[i]
        x_IO = FOP(theta, s_hat)

        # Response difference
        x_diff += dist_func(x_hat, x_IO)

        if theta_true is not None:
            obj_IO = np.inner(theta_true, phi(s_hat, x_IO))
            obj_true = np.inner(theta_true, phi(s_hat, x_hat))

            # Objetive value difference
            if scale_obj_diff:
                obj_diff += (obj_IO - obj_true)/abs(obj_true)
            else:
                obj_diff += obj_IO - obj_true

    x_diff_avg = x_diff/N
    obj_diff_avg = obj_diff/N

    if theta_true is not None:
        theta_norm = theta/np.linalg.norm(theta)
        theta_true_norm = theta_true/np.linalg.norm(theta_true)
        # Cost vector difference
        theta_diff = np.linalg.norm(theta_norm - theta_true_norm)
        results = (x_diff_avg, obj_diff_avg, theta_diff)
    else:
        results = x_diff_avg

    return results


def discrete_model_consistent(dataset, phi, decision_space,
                              X=None,
                              dist_func=None,
                              Theta=None,
                              regularizer='L2_squared',
                              theta_hat=None,
                              feasibility=False,
                              verbose=False,
                              gurobi_params=None):
    """
    Inverse optimization for discrete models with consistent data.

    Uses incenter (default) or feasibility strategy. See an example usage at
    https://github.com/pedroszattoni/invopt/tree/main/examples

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    decision_space : {tuple('binary', n)}
        Tuple containing type and dimension of the decision space.
    X : {callable, None}, optional
        Constraint set. Given a signal s and response x, returns True if x is a
        feasible response, and False otherwise. Syntax: X(s, x). If None, it
        will be defined as "def X(s, x): return True". The default is None.
    dist_func : {callable, None}, optional
        Distance function. Given two responses x1 and x2, returns the distance
        between them according to some distance metric. Not required when
        using the feasibility strategy. Syntax: dist_func(x1, x2). The default
        is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector. When not None, the
        cost vector returned will be the feasible vector closest to theta_hat.
        The default is None.
    feasibility : bool, optional
        If True, solve problem as a feasibility problem. Namely, searches
        over the facets of the unit L-infinity sphere until a feasible
        cost vector is found. If Theta='nonnegative', only searches over the
        nonnegative facet of the L-1 sphere. If False, solve the problem
        using the Incenter strategy. The default is False.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution. If feasibility=True and theta_hat is not
        None. If feasibility=False, theta_hat is None and dist_func is None.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector according to the chosen strategy.

    """
    import gurobipy as gp

    # Check if inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)

    warning_large_decision_space(decision_space)

    if (theta_hat is not None) and feasibility:
        raise Exception('Either set feasibility=True or set theta_hat not '
                        'None. When feasibility=True, returns any feasible '
                        'solution. When theta_hat is not None, returns the '
                        'feasible solution closest to theta_hat, that is, '
                        'regularizer(theta - theta_hat) is minimized.')

    if dist_func is None:
        if (not feasibility) and (theta_hat is None):
            raise Exception('dist_func required when feasibility=False and '
                            'theta_hat is not given.')
    else:
        if feasibility or (theta_hat is not None):
            warnings.warn('dist_func not used when a theta_hat is given or '
                          'feasibility=True.')

    if X is None:
        def X(s, x): return True

    N = len(dataset)

    # Sample signal and response to get dimension of the cost vector
    s_test, x_test = dataset[0]
    p = len(phi(s_test, x_test))

    # Initialize Gurobi model
    mdl = gp.Model()
    if not verbose:
        mdl.setParam('OutputFlag', 0)
    if gurobi_params is not None:
        for param, value in gurobi_params:
            mdl.setParam(param, value)

    theta = mdl.addVars(p, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)

    if Theta == 'nonnegative':
        mdl.addConstrs(theta[i] >= 0 for i in range(p))

    # Add constraints
    for i in range(N):
        s_hat, x_hat = dataset[i]
        if decision_space[0] == 'binary':
            n = decision_space[1]
            for k in range(2**n):
                x = dec_to_bin(k, n)
                if X(s_hat, x):
                    phi_1 = phi(s_hat, x)
                    phi_2 = phi(s_hat, x_hat)
                    if (theta_hat is not None) or feasibility:
                        dist = 0
                    else:
                        dist = dist_func(x_hat, x)
                    mdl.addConstr(gp.quicksum(theta[j]*(phi_1[j] - phi_2[j])
                                              for j in range(p)) >= dist)

    if feasibility:
        if Theta == 'nonnegative':
            mdl.addConstr(gp.quicksum(theta) == 1)
            mdl.optimize()
        else:
            # Search over facets of unit L-infinity sphere for a feasible
            # solution.
            for i in range(p):
                for j in [-1, 1]:
                    cons = mdl.addConstr(theta[i] == j)
                    mdl.optimize()
                    mdl.remove(cons)
                    if mdl.status == 2:
                        break
                if mdl.status == 2:
                    break
    else:
        if theta_hat is None:
            theta_hat = np.zeros(p)

        if regularizer == 'L2_squared':
            obj = 0.5*gp.quicksum((theta[i] - theta_hat[i])**2
                                  for i in range(p))
        elif regularizer == 'L1':
            t = mdl.addVars(p, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            obj = gp.quicksum(t)
            mdl.addConstrs(theta[i] - theta_hat[i] <= t[i] for i in range(p))
            mdl.addConstrs(theta_hat[i] - theta[i] <= t[i] for i in range(p))

        mdl.setObjective(obj, gp.GRB.MINIMIZE)
        mdl.optimize()

    theta_opt = np.array([theta[i].X for i in range(p)])

    if mdl.status != 2:
        raise Exception('Optimal solution not found. Gurobi status code '
                        f'= {mdl.status}. Set the flag verbose=True for more '
                        'details. The optimization problem will '
                        'always be infeasible if the data is not consistent.')

    return theta_opt


def discrete_model(dataset, phi, decision_space,
                   X=None,
                   dist_func=None,
                   Theta=None,
                   regularizer='L2_squared',
                   reg_param=0,
                   theta_hat=None,
                   sub_loss=False,
                   verbose=False,
                   gurobi_params=None):
    """
    Inverse optimization for discrete models.

    See an example usage at
    https://github.com/pedroszattoni/invopt/tree/main/examples

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    decision_space : {tuple('binary', n)}
        Tuple containing type and dimension of the decision space.
    X : {callable, None}, optional
        Constraint set. Given a signal s and response x, returns True if x is a
        feasible response, and False otherwise. Syntax: X(s, x). If None, it
        will be defined as "def X(s, x): return True". The default is None.
    dist_func : {callable, None}, optional
        Distance function. Given two responses x1 and x2, returns the distance
        between them according to some distance metric. Not required when
        sub_loss=True. Syntax: dist_func(x1, x2). The default is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector. When not None,
        theta_hat is defined as the vector of zeros. The default is None.
    sub_loss : bool, optional
        If True, solve the problem using the Suboptimality loss. Namely,
        searches over the facts of the unit L-infinity sphere for the cost
        vector with the smallest loss. If Theta='nonnegative', only searches
        over the nonnegative facet of the L-1 sphere. If False, solve the
        problem using the Augmented Suboptimality loss. The default is False.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution. If sub_loss=False and dist_func is None.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector according to the chosen strategy.

    """
    import gurobipy as gp

    # Check if inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)
    check_dist_func(dist_func, sub_loss)

    # Warnings
    warning_large_decision_space(decision_space)
    warning_dist_func_reg_param_sub_loss(dist_func, reg_param, sub_loss)

    if X is None:
        def X(s, x): return True

    N = len(dataset)

    # Sample signal and response to get dimension of the cost vector
    s_test, x_test = dataset[0]
    p = len(phi(s_test, x_test))

    # Initialize Gurobi model
    mdl = gp.Model()
    if not verbose:
        mdl.setParam('OutputFlag', 0)
    if gurobi_params is not None:
        for param, value in gurobi_params:
            mdl.setParam(param, value)

    theta = mdl.addVars(p, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    beta = mdl.addVars(N, vtype=gp.GRB.CONTINUOUS)
    sum_beta = (1/N)*gp.quicksum(beta)

    if Theta == 'nonnegative':
        mdl.addConstrs(theta[i] >= 0 for i in range(p))

    # Add constraints
    for i in range(N):
        s_hat, x_hat = dataset[i]
        if decision_space[0] == 'binary':
            n = decision_space[1]
            for k in range(2**n):
                x = dec_to_bin(k, n)
                if X(s_hat, x):
                    phi_1 = phi(s_hat, x)
                    phi_2 = phi(s_hat, x_hat)
                    if sub_loss:
                        dist = 0
                    else:
                        dist = dist_func(x_hat, x)
                    mdl.addConstr(
                        gp.quicksum(theta[j]*(phi_1[j] - phi_2[j])
                                    for j in range(p)) >= dist - beta[i])

    if sub_loss:
        mdl.setObjective(sum_beta, gp.GRB.MINIMIZE)

        if Theta is None:
            # Search over facets of unit L-infinity sphere for the solution
            # with the lowest objective value.
            best_obj = np.inf
            for i in range(p):
                for j in [-1, 1]:
                    cons = mdl.addConstr(theta[i] == j)
                    mdl.optimize()
                    obj_val = mdl.objVal
                    mdl.remove(cons)
                    if (mdl.status == 2) and (obj_val < best_obj):
                        best_obj = obj_val
                        theta_opt = np.array([theta[i].X for i in range(p)])
        elif Theta == 'nonnegative':
            mdl.addConstr(gp.quicksum(theta) == 1)
            mdl.optimize()
            theta_opt = np.array([theta[i].X for i in range(p)])
    else:
        if theta_hat is None:
            theta_hat = np.zeros(p)

        if regularizer == 'L2_squared':
            reg_term = (reg_param/2)*gp.quicksum((theta[i] - theta_hat[i])**2
                                                 for i in range(p))
        elif regularizer == 'L1':
            t = mdl.addVars(p, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            reg_term = reg_param*gp.quicksum(t)
            mdl.addConstrs(theta[i] - theta_hat[i] <= t[i] for i in range(p))
            mdl.addConstrs(theta_hat[i] - theta[i] <= t[i] for i in range(p))

        mdl.setObjective(reg_term + sum_beta, gp.GRB.MINIMIZE)
        mdl.optimize()
        theta_opt = np.array([theta[i].X for i in range(p)])

    if mdl.status != 2:
        raise Exception('Optimal solution not found. Gurobi status code '
                        f'= {mdl.status}. Set the flag verbose=True for more '
                        'details.')

    return theta_opt


def MIP_linear(dataset, decision_space,
               Z=None,
               phi1=None,
               phi2=None,
               dist_func=None,
               Theta=None,
               regularizer='L2_squared',
               reg_param=0,
               sub_loss=False,
               verbose=False,
               gurobi_params=None):
    """
    Inverse optimization for linear models with mixed-integer feasible sets.

    See an example usage at
    https://github.com/pedroszattoni/invopt/tree/main/examples

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    decision_space : {tuple('binary', n)}
        Tuple containing type and dimension of the decision space.
    Z : {callable, None}, optional
        Constraint set of the integer part of the decision vector. Given a
        signal s = (A, B, c, w) and response x = (y, z), returns True if z
        (i.e., the integer part of x) is a feasible response, and False
        otherwise. Syntax: Z(w, z). If None, it will be defined as
        "def Z(w, x): return True". The default is None.
    phi1 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi1(w, z): return np.array([0])". The default is None.
    phi2 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi2(w, z): return np.array([0])". The default is None.
    dist_func : {callable, None}, optional
        Distance function. Given two responses x1=(y1,z1) and x2=(y2,z2),
        returns the distance of their integer parts according to some distance
        metric. Not required when sub_loss=True. Syntax: dist_func(z1, z2).
        The default is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    sub_loss : bool, optional
        If True, solve the problem using the Suboptimality loss. Namely,
        searches over the facts of the unit L-infinity sphere for the cost
        vector with the smallest loss. If Theta='nonnegative', only searches
        over the nonnegative facet of the L-1 sphere. If False, solve the
        problem using the Augmented Suboptimality loss. The default is False.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution. If sub_loss=False and dist_func is None.
        If neither phi1 nor phi2 are given.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector according to the chosen strategy.

    """
    import gurobipy as gp

    # Check if inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)
    check_dist_func(dist_func, sub_loss)

    # Warnings
    warning_large_decision_space(decision_space)
    warning_dist_func_reg_param_sub_loss(dist_func, reg_param, sub_loss)

    if (phi1 is None) and (phi2 is None):
        raise Exception('Either phi1 or phi2 have to be given.')

    N = len(dataset)

    if Z is None:
        def Z(s, z): return True

    if phi1 is None:
        def phi1(w, z): return np.array([0])
    if phi2 is None:
        def phi2(w, z): return np.array([0])

    # Sample signal and response to get dimensions of problem
    s_test, x_test = dataset[0]
    A_test, _, _, w_test = s_test
    y_test, z_test = x_test
    u1 = len(phi1(w_test, z_test))
    u2 = len(phi2(w_test, z_test))
    m1, m2 = A_test.shape

    # Initialize Gurobi model
    mdl = gp.Model()
    if not verbose:
        mdl.setParam('OutputFlag', 0)
    if gurobi_params is not None:
        for param, value in gurobi_params:
            mdl.setParam(param, value)

    Q = mdl.addVars(m2, u1, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    q = mdl.addVars(u2, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    beta = mdl.addVars(N, vtype=gp.GRB.CONTINUOUS)
    sum_beta = (1/N)*gp.quicksum(beta)

    if Theta == 'nonnegative':
        mdl.addConstrs(Q[i, j] >= 0 for i in range(m2) for j in range(u1))
        mdl.addConstrs(q[i] >= 0 for i in range(u2))

    for i in range(N):
        s_hat, x_hat = dataset[i]
        y_hat, z_hat = x_hat
        A, B, c, w_hat = s_hat
        if decision_space[0] == 'binary':
            n = decision_space[1]
            for k in range(2**n):
                z = dec_to_bin(k, n)
                if Z(w_hat, z):
                    lamb = mdl.addVars(m1, vtype=gp.GRB.CONTINUOUS)

                    if sub_loss:
                        dist = 0
                    else:
                        dist = dist_func(z_hat, z)

                    phi1_hat = phi1(w_hat, z_hat)
                    phi2_hat = phi2(w_hat, z_hat)
                    Qphi1 = [gp.quicksum(Q[i, j]*phi1_hat[j]
                                         for j in range(u1))
                             for i in range(m2)]
                    yQphi1 = gp.quicksum(y_hat[j]*Qphi1[j] for j in range(m2))
                    qphi2_hat = gp.quicksum(q[j]*phi2_hat[j]
                                            for j in range(u2))
                    theta_phi = yQphi1 + qphi2_hat

                    Bz = B @ z
                    lambcBz = gp.quicksum(lamb[j]*(c[j] - Bz[j])
                                          for j in range(m1))
                    ph2 = phi2(w_hat, z)
                    qphi2 = gp.quicksum(q[j]*ph2[j] for j in range(u2))

                    mdl.addConstr(theta_phi + lambcBz - qphi2 + dist
                                  <= beta[i])

                    ph1 = phi1(w_hat, z)
                    mdl.addConstrs(gp.quicksum(Q[i, j]*ph1[j]
                                               for j in range(u1))
                                   + gp.quicksum(lamb[j]*A[j, i]
                                                 for j in range(m1))
                                   == 0 for i in range(m2))

    if sub_loss:
        mdl.setObjective(sum_beta, gp.GRB.MINIMIZE)

        if Theta == 'nonnegative':
            mdl.addConstr(gp.quicksum(q) + gp.quicksum(Q) == 1)
            mdl.optimize()
            Q_opt = np.array([[Q[i, j].X for i in range(m2)]
                              for j in range(u1)])
            q_opt = np.array([q[i].X for i in range(u2)])
        else:
            # Search over facets of unit L-infinity sphere for the solution
            # with the lowest objective value.
            best_obj = np.inf
            for i in range(u1):
                for j in [-1, 1]:
                    cons = mdl.addConstr(q[i] == j)
                    mdl.optimize()
                    mdl.remove(cons)
                    # If an optimal solution was found
                    if mdl.status == 2:
                        obj_val = mdl.objVal
                        if obj_val < best_obj:
                            best_obj = obj_val
                            Q_opt = np.array([[Q[i, j].X for i in range(m2)]
                                              for j in range(u1)])
                            q_opt = np.array([q[i].X for i in range(u2)])
            for i in range(m2):
                for k in range(u1):
                    for j in [-1, 1]:
                        cons = mdl.addConstr(Q[i, k] == j)
                        mdl.optimize()
                        mdl.remove(cons)
                        # If an optimal solution was found
                        if mdl.status == 2:
                            obj_val = mdl.objVal
                            if obj_val < best_obj:
                                best_obj = obj_val
                                Q_opt = np.array([[Q[i, j].X
                                                   for i in range(m2)]
                                                  for j in range(u1)])
                                q_opt = np.array([q[i].X for i in range(u2)])
    else:
        if regularizer == 'L2_squared':
            Q_sum = gp.quicksum(Q[i, j]**2 for i in range(m2)
                                for j in range(u1))
            q_sum = gp.quicksum(q[i]**2 for i in range(u2))
            reg_term = (reg_param/2)*(Q_sum + q_sum)
        elif regularizer == 'L1':
            tQ = mdl.addVars(m2, u1, lb=-gp.GRB.INFINITY,
                             vtype=gp.GRB.CONTINUOUS)
            tq = mdl.addVars(u2, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            reg_term = reg_param*(gp.quicksum(tQ) + gp.quicksum(tq))
            mdl.addConstrs(Q[i, j] <= tQ[i, j]
                           for i in range(m2) for i in range(u1))
            mdl.addConstrs(-Q[i, j] <= tQ[i, j]
                           for i in range(m2) for i in range(u1))
            mdl.addConstrs(q[i] <= tq[i] for i in range(u2))
            mdl.addConstrs(-q[i] <= tq[i] for i in range(u2))

        mdl.setObjective(reg_term + sum_beta, gp.GRB.MINIMIZE)
        mdl.optimize()

        if mdl.status != 2:
            raise Exception('Optimal solution not found. Gurobi status code '
                            f'= {mdl.status}. Set the flag verbose=True for '
                            'more details.')

        Q_opt = np.array([[Q[i, j].X for i in range(m2)] for j in range(u1)])
        q_opt = np.array([q[i].X for i in range(u2)])

    theta_opt = np.concatenate((Q_opt.flatten('F'), q_opt))
    return theta_opt


def FOM(dataset, phi, theta_0, FOP, step_size, T,
        Theta=None,
        step='standard',
        regularizer='L2_squared',
        reg_param=0,
        theta_hat=None,
        batch_type=1,
        averaged=0,
        callback=None,
        normalize_grad=False,
        verbose=False):
    """
    Optimize (Augmented) Suboptimality loss using first-order methods.

    See an example usage at
    https://github.com/pedroszattoni/invopt/tree/main/examples

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    theta_0 : 1D ndarray
        Initial cost vector theta.
    FOP : callable
        (Augmented) forward optimization problem. Takes as input a cost vector
        theta, a signal s, and the respective response x. Returns the optimal
        (augmented) response. When using the Augmented Suboptimality loss,
        an augmented FOP should be used. Syntax: FOP(theta, s), or
        FOP(theta, s, x) for an augmented FOP.
    step_size : callable
        Step-size function. Takes as input the iteration counter t = 0,...,T-1
        and returns the step-size. Syntax: step_size(t).
    T : int
        Number of iterations the algorithm is run.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    step : {'standard', 'exponentiated'}, optional
        Type of update step used for the first-order algorithm. If 'standard',
        uses standard "subgradient method" update steps:
        theta_{t+1} = theta_t - step_size(t)*subgradient. If 'exponentiated',
        uses exponentiated steps:
        theta_{t+1} = theta_t * exp{-step_size(t)*subgradient}. The default is
        'standard'.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector. When not None,
        theta_hat is defined as the vector of zeros. The default is None.
    batch_type : {float, 'reshuffled'}, optional
        If float, it is the fraction of the dataset used to compute stochastic
        subgradients of the loss function, where batch_type=b means use 10*b %
        of the data to compute (stochastic) subgradients. If
        batch_type='reshuffled', T is the number of epochs instead of the
        number of iterations. For each epoch, run the algorithm for N
        iterations, where N is the size of the dataset. That is, perform one
        update step for each example in the dataset. At the beginning of each
        epoch, shuffle the order or the examples in the dataset. For more
        details, see Mishchenko et al. "Random Reshuffling: Simple Analysis
        with Vast Improvements". The default is 1.
    averaged : {0, 1, 2}, optional
        Add the option to return the average of the iterates of the algorithm
        instead of the final iterate. If averaged=0, it does not average the
        iterates of the algorithm. If averaged=1, uses a simple average of
        iterations: (1/T)*sum_{t=1}^{T} theta_t. If averaged=2, uses a weighted
        average of iterations: (2/(T*(T+1)))*sum_{t=1}^{T} t*theta_t. In
        theory, for strongly convex problems, the weighted average works better
        (see Lacoste-Julien et al. "A simpler approach to obtaining an O(1/t)
        convergence rate for the projected stochastic subgradient method"). The
        default is 0.
    callback : {callable, None}, optional
        If not None, called after each iteration of the algorithm. The default
        is None.
    normalize_grad : bool, optional
        If True, subgradient vectors are normalized before each iteration of
        the algorithm. If step='standard', the L2 norm of the subgradient is
        used. If step='standard', the L-infinity norm of the subgradient is
        used. The default is False.
    verbose : bool, optional
        If True, prints iteration counter. The default is False.

    Raises
    ------
    Exception
        If unsupported Theta or regularizer. If step = 'exponentiated',
        reg_param > 0, and regularizer is not 'L1'.

    Returns
    -------
    theta_T : {1D ndarray, list}
        If callback=None, returns the final (averaged) vector found after T
        iterations of the algorithm. Otherwise, returns a list of size T+1
        with elements callback(theta_t) for t=0,...,T, where theta_t is the
        (averaged) vector after t iterations of the algorithm.

    """
    # Check if inputs are valid
    check_Theta(Theta)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)

    if step == 'exponentiated':
        if (reg_param == 0) or (regularizer != 'L1'):
            raise Exception('To use step = \'exponentiated\', reg_param > 0,'
                            'and regularizer = \'L1\' are required.')

    # Get the number of examples and dimension of the problem
    N = len(dataset)
    p = len(theta_0)

    if theta_hat is None:
        theta_hat = np.zeros(p)

    theta_t = theta_0
    theta_avg = theta_0
    callback_list = []
    if callback is not None:
        # Evaluate theta_0
        callback_list.append(callback(theta_0))

    for t in range(T):
        if verbose:
            print(f'Iteration {t+1} out of {T}')
            print('')

        eta_t = step_size(t)
        if batch_type == 'reshuffled':
            arr = np.arange(N)
            np.random.shuffle(arr)
            for i in arr:
                samples = [dataset[i]]
                reg_grad, loss_grad = compute_grad(theta_t, samples, phi, FOP,
                                                   regularizer, reg_param,
                                                   theta_hat)
                theta_t = grad_step(theta_t, eta_t, reg_grad, loss_grad,
                                    reg_param, Theta, step, normalize_grad)
        else:
            batch_size = int(np.ceil(batch_type*N))
            sample_idxs = np.random.choice(N, batch_size, replace=False)
            samples = [dataset[i] for i in sample_idxs]
            reg_grad, loss_grad = compute_grad(theta_t, samples, phi, FOP,
                                               regularizer, reg_param,
                                               theta_hat)
            theta_t = grad_step(theta_t, eta_t, reg_grad, loss_grad, reg_param,
                                Theta, step, normalize_grad)

        if averaged == 0:
            theta_avg = theta_t
        elif averaged == 1:
            theta_avg = (1/(t+1))*theta_t + (t/(t+1))*theta_avg
        elif averaged == 2:
            theta_avg = (2/(t+2))*theta_t + (t/(t+2))*theta_avg

        if callback is not None:
            callback_list.append(callback(theta_avg))

    # Check if callback_list is empty
    if callback_list:
        theta_T = callback_list
    else:
        theta_T = theta_avg

    return theta_T


def gradient_regularizer(theta_t, regularizer, reg_param, theta_hat):
    """
    Compute (sub)gradient of regularizer.

    Parameters
    ----------
    theta_t : 1D ndarray
        Vector where the gradient will be evaluated at.
    regularizer : {'L2_squared', 'L1'}
        Type of regularization on cost vector theta.
    reg_param : float
        Nonnegative regularization parameter..
    theta_hat : 1D ndarray
        A priory belief or estimate of the true cost vector. When not None, the
        cost vector returned will be the feasible vector closest to theta_hat.

    Returns
    -------
    reg_grad : 1D ndarray
        (Sub)gradient of the regularizer evaluated at theta_t.

    """
    if regularizer == 'L2_squared':
        grad = reg_param*(theta_t - theta_hat)
    elif regularizer == 'L1':
        grad = reg_param*np.sign(theta_t - theta_hat)

    return grad


def compute_grad(theta_t, samples, phi, FOP, regularizer, reg_param,
                 theta_hat):
    """
    Compute a (sub)gradient of the regularizer and the sum of losses.

    Parameters
    ----------
    theta_t : 1D ndarray
        Vector where the (sub)gradient will be evaluated at.
    samples : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
        Each tuple defines a loss function.
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    FOP : callable
        (Augmented) forward optimization problem. Takes as input a cost vector
        theta, a signal s, and the respective response x. Returns the optimal
        (augmented) response. When using the Augmented Suboptimality loss,
        an augmented FOP should be used. Syntax: FOP(theta, s), or
        FOP(theta, s, x) for an augmented FOP.
    regularizer : {'L2_squared', 'L1'}
        Type of regularization on cost vector theta.
    reg_param : float
        Nonnegative regularization parameter.
    theta_hat : 1D ndarray
        A priory belief or estimate of the true cost vector.

    Returns
    -------
    reg_grad : 1D ndarray
        (Sub)gradient of the regularizer evaluated at theta_t.
    loss_grad : 1D ndarray
        (Sub)gradient of the sum of losses evaluated at theta_t.

    """
    aux = 0
    for s_hat, x_hat in samples:
        # Check if FOP is augmented
        try:
            x_opt = FOP(theta_t, s_hat, x_hat)
        except TypeError:
            x_opt = FOP(theta_t, s_hat)

        aux += phi(s_hat, x_hat) - phi(s_hat, x_opt)

    # Subgradient of the sum of losses
    loss_grad = (1/len(samples))*aux

    # (Sub)gradient of the regularizer
    reg_grad = gradient_regularizer(theta_t, regularizer, reg_param, theta_hat)

    return reg_grad, loss_grad


def grad_step(theta_t, eta_t, reg_grad, loss_grad, reg_param, Theta, step,
              normalize_grad):
    """
    Perform a subgradient step.

    Parameters
    ----------
    theta_t : 1D ndarray
        Initial vector.
    eta_t : float
        Step-size constant.
    reg_grad : 1D ndarray
        (Sub)gradient of the regularizer.
    loss_grad : 1D ndarray
        (Sub)gradient of the sum of loss functions.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    Theta : {None, 'nonnegative'}
        Constraints on cost vector theta.
    step : {'standard', 'exponentiated'}
        Type of update step used for the first-order algorithm. If 'standard',
        uses standard "subgradient method" update steps:
        theta_{t+1} = theta_t - step_size(t)*subgradient. If 'exponentiated',
        uses exponentiated steps:
        theta_{t+1} = theta_t * exp{-step_size(t)*subgradient}.
    normalize_grad : bool
        If True, subgradient vectors are normalized before each iteration of
        the algorithm. If step='standard', the L2 norm of the subgradient is
        used. If step='standard', the L-infinity norm of the subgradient is
        used.

    Returns
    -------
    theta_t1 : 1D ndarray
        Vector subgradient step.

    """
    if step == 'standard':
        grad = reg_grad + loss_grad
        if normalize_grad:
            grad = normalize(grad, 2)
        theta_t1 = theta_t - eta_t*grad
    elif step == 'exponentiated':
        grad = loss_grad
        if normalize_grad:
            grad = normalize(grad, np.inf)
        if Theta == 'nonnegative':
            theta_t1 = np.multiply(theta_t, np.exp(-eta_t*grad))
            norm_theta_t1 = np.sum(theta_t1)
        else:
            warnings.warn('The combination of  step = \'exponentiated\' and '
                          'Theta != \'nonnegative\' still needs to be tested.')
            theta_pos_t = np.clip(theta_t, 0, None)
            theta_neg_t = np.clip(theta_t, None, 0)
            theta_pos_t1 = np.multiply(theta_pos_t, np.exp(-eta_t*grad))
            theta_neg_t1 = np.multiply(theta_neg_t, np.exp(eta_t*grad))
            theta_t1 = theta_pos_t1 - theta_neg_t1
            norm_theta_t1 = np.sum(theta_pos_t1) + np.sum(theta_neg_t1)

        # If outside the simplex, projec onto it
        if reg_param*norm_theta_t1 > 1:
            theta_t1 = theta_t1/(reg_param*norm_theta_t1)

    # Projection onto Theta
    if Theta == 'nonnegative':
        theta_t = np.clip(theta_t, 0, None)

    return theta_t1


def MIP_quadratic(dataset, decision_space,
                  Z=None,
                  phi1=None,
                  phi2=None,
                  dist_func=None,
                  Theta=None,
                  regularizer='L2_squared',
                  reg_param=0,
                  sub_loss=False,
                  verbose=False,
                  solver='mosek'):
    """
    Inverse optimization for quadratic models with mixed-integer feasible sets.

    See an example usage at
    https://github.com/pedroszattoni/invopt/tree/main/examples

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    decision_space : {tuple('binary', n)}
        Tuple containing type and dimension of the decision space.
    Z : {callable, None}, optional
        Constraint set of the integer part of the decision vector. Given a
        signal s = (A, B, c, w) and response x = (y, z), returns True if z
        (i.e., the integer part of x) is a feasible response, and False
        otherwise. Syntax: Z(w, z). If None, it will be defined as
        "def Z(w, x): return True". The default is None.
    phi1 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi1(w, z): return np.array([0])". The default is None.
    phi2 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi2(w, z): return np.array([0])". The default is None.
    dist_func : {callable, None}, optional
        Distance function. Given two responses x1=(y1,z1) and x2=(y2,z2),
        returns the distance of their integer parts according to some distance
        metric. Not required when sub_loss=True. Syntax: dist_func(z1, z2).
        The default is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    sub_loss : bool, optional
        If True, solve the problem using the Suboptimality loss. Namely,
        searches over the facts of the unit L-infinity sphere for the cost
        vector with the smallest loss. If Theta='nonnegative', only searches
        over the nonnegative facet of the L-1 sphere. If False, solve the
        problem using the Augmented Suboptimality loss. The default is False.
    verbose : bool, optional
        If True, print solver's output. The default is False.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution. If sub_loss=False and dist_func is None.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector according to the chosen strategy.

    """
    import cvxpy as cp

    # Check if inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)

    # Warnings
    warning_large_decision_space(decision_space)
    warning_dist_func_reg_param_sub_loss(dist_func, reg_param, sub_loss)

    N = len(dataset)

    if Z is None:
        def Z(s, z): return True

    if phi1 is None:
        def phi1(w, z): return np.array([0])
    if phi2 is None:
        def phi2(w, z): return np.array([0])

    # Sample signal and response to get the the dimensions of the problem
    s_test, x_test = dataset[0]
    A_test, _, _, w_test = s_test
    y_test, z_test = x_test
    u1 = len(phi1(w_test, z_test))
    u2 = len(phi2(w_test, z_test))
    m1, m2 = A_test.shape

    Qyy = cp.Variable((m2, m2), symmetric=True)
    Q = cp.Variable((m2, u1))
    q = cp.Variable((u2, 1))
    beta = cp.Variable(N)

    constraints = []

    sum_beta = (1/N)*cp.sum(beta)

    if Theta == 'nonnegative':
        constraints += [Q >= 0, q >= 0]

    for i in range(N):
        s_hat, x_hat = dataset[i]
        y_hat, z_hat = x_hat
        A, B, c, w_hat = s_hat
        if decision_space[0] == 'binary':
            n = decision_space[1]
            for k in range(2**n):
                z = dec_to_bin(k, n)
                if Z(w_hat, z):
                    alpha = cp.Variable((1, 1))
                    lamb = cp.Variable((m1, 1))

                    if sub_loss:
                        dist = 0
                    else:
                        dist = dist_func(z_hat, z)

                    theta_phi_hat = (y_hat.T @ Qyy @ y_hat
                                     + y_hat.T @ Q @ phi1(w_hat, z_hat)
                                     + q.T @ phi2(w_hat, z_hat))

                    lambcBz = lamb.T @ (c - B @ z)

                    qphi2 = q.T @ phi2(w_hat, z)

                    constraints += [theta_phi_hat + alpha + lambcBz - qphi2
                                    <= beta[i] - dist]

                    off_diag = Q @ phi1(w_hat, z).reshape((u1, 1)) + A.T @ lamb
                    constraints += [cp.bmat([[Qyy, off_diag],
                                             [off_diag.T, 4*alpha]]) >> 0]
                    constraints += [lamb >= 0]

    if sub_loss:
        constraints += [cp.trace(Qyy) == 1]
        obj = cp.Minimize(sum_beta)
    else:
        if regularizer == 'L2_squared':
            Qyy_sum = cp.sum_squares(Qyy)
            Q_sum = cp.sum_squares(Q)
            q_sum = cp.sum_squares(q)
            reg_term = (reg_param/2)*(Qyy_sum + Q_sum + q_sum)
        elif regularizer == 'L1':
            tQyy = cp.Variable((m2, m2), symmetric=True)
            tQ = cp.Variable((m2, u1))
            tq = cp.Variable((u2, 1))
            reg_term = reg_param*(cp.sum(tQyy) + cp.sum(tQ) + cp.sum(tq))
            constraints += [Qyy <= tQyy, -Qyy <= tQyy]
            constraints += [Q <= tQ, -Q <= tQ]
            constraints += [q <= tq, -q <= tq]

        obj = cp.Minimize(reg_term + sum_beta)

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose)

    if prob.status != 'optimal':
        raise Exception('Optimal solution not found. CVXPY status code '
                        f'= {prob.status}. Set the flag verbose=True for more '
                        'details.')

    Qyy_opt = Qyy.value
    Q_opt = Q.value
    q_opt = q.value

    theta_opt = np.concatenate((Qyy_opt.flatten('F'),
                                Q_opt.flatten('F'),
                                q_opt.flatten('F')))
    return theta_opt
