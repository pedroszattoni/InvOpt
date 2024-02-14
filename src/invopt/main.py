"""
invopt: Inverse Optimization with Python.

Author: Pedro Zattoni Scroccaro
"""

import numpy as np
import warnings


def check_Theta(Theta):
    """Check if Theta is valid."""
    if Theta not in [None, 'nonnegative']:
        raise Exception(
            'Invalid Theta. Accepted values are: None (default)'
            'and \'nonnegative\'.'
        )


def check_decision_space(decision_space):
    """Check if decision_space is valid."""
    if decision_space not in ['binary', 'one_hot']:
        raise Exception(
            'Invalid decision space. Accepted values are:'
            '\'binary\' and \'one_hot\'.'
        )


def check_regularizer(regularizer):
    """Check if regularizer is valid."""
    if regularizer not in ['L2_squared', 'L1']:
        raise Exception(
            'Invalid regularizer. Accepted values are:'
            ' \'L2_squared\' (default) and \'L1\'.'
        )


def check_reg_parameter(reg_param):
    """Check if reg_param is valid."""
    if reg_param < 0:
        raise Exception('reg_param must be nonnegative.')


def warning_large_decision_space(decision_space, n):
    """Warn user if decision space is binary and high-dimensional."""
    if (decision_space == 'binary') and (n > 15):
        warnings.warn(
            'Attention! Using this function for FOPs with binary decision'
            'variables requires solving an optimization problem with'
            'potentially O(N*2^n) constraints, where N is the number of'
            'training examples and n is the dimension of the binary'
            'decision vector.'
        )


def warning_theta_hat_reg_param(theta_hat, reg_param):
    """Warn user theta_hat is not used when reg_param=0."""
    if (theta_hat is not None) and (reg_param == 0):
        warnings.warn('theta_hat is not used when reg_param=0.')


def warning_add_dist_func_y(add_dist_func_y):
    """Warn user that add_dist_func_y increases the size of the problem."""
    if add_dist_func_y:
        warnings.warn(
            'Setting add_dist_func_y=True increases the number of constraints'
            'of the IO problem by a factor of 2n, where n is the dimension of'
            'the continuous part of the decision vector.'
        )


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


def ASL(
    theta,
    dataset,
    FOP_aug,
    phi,
    dist_func,
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
):
    """
    Evaluate Augmented Suboptimality loss.

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
        Distance penalization function. Given two responses x1 and x2, returns
        the distance between them according to some distance metric. Syntax:
        dist_func(x1, x2). Alternatively, the function can also take the signal
        s_hat as a third argument. For instace, to use a distance function in
        feature space instead of action space, we can use
        dist_func(x1, x2, s_hat) = dist(phi(s_hat, x1), phi(s_hat, x2)). In
        some cases, this can improve the performance of the model.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector theta. The default
        is None.

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
    # Check if the inputs are valid
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
        try:
            dist = dist_func(x_hat, x)
        except TypeError:
            dist = dist_func(x_hat, x, s_hat)

        loss += theta @ phi_diff + dist

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


def candidate_action(k, decision_space, n):
    """
    Generate candidate action x.

    Parameters
    ----------
    k : int
        Index.
    decision_space : str
        Type of the decision space.
    n : int
        Dimention of the decision space.

    Returns
    -------
    x : 1D ndarray
        Candidate action.

    """
    if decision_space == 'binary':
        x = dec_to_bin(k, n)
    elif decision_space == 'one_hot':
        x = np.zeros(n)
        x[k] = 1

    return x


def evaluate(
    theta,
    dataset,
    FOP,
    dist_func,
    theta_true=None,
    phi=None,
    scale_obj_diff=True,
):
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
        Distance penalization function. Given two responses x1 and x2, returns
        the distance between them according to some distance metric. Syntax:
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
        If theta_true is given but the phi function is not.

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
    obj_diff = 0
    for s_hat, x_hat in dataset:
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


def discrete_consistent(
    dataset,
    X,
    phi,
    dist_func=None,
    Theta=None,
    regularizer='L2_squared',
    theta_hat=None,
    verbose=False,
    gurobi_params=None,
):
    """
    Inverse optimization for discrete FOPs with consistent data.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/discrete_consistent

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    X : tuple(str, int, {callable, None})
        Constraint set. Given a signal s and response x, X is a tuple
        containing the type of the decision space, the size of the decision
        space, and any extra constraints as an indicator function. For example,
        if the decision vector is binary, with size n, and respects the
        constraint Ax <= b., then X = ('binary', n, ind_func), where
        ind_func(s, x) equals True if Ax <= b, and False otherwise. Notice that
        A and b may be part of the signal s, which is why the indicator
        function takes both s and x as inputs. If ind_func=None, it will be
        defined as "def ind_func(s, x): return True".
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    dist_func : {callable, None}, optional
        Distance penalization function. Given two responses x1 and x2, returns
        the distance between them according to some distance metric. Syntax:
        dist_func(x1, x2). Alternatively, the function can also take the signal
        s_hat as a third argument. For instace, to use a distance function in
        feature space instead of action space, we can use
        dist_func(x1, x2, s_hat) = dist(phi(s_hat, x1), phi(s_hat, x2)). In
        some cases, this can improve the performance of the model. The default
        is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector theta. The default
        is None.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector.

    """
    import gurobipy as gp

    decision_space, n, ind_func = X
    if decision_space == 'binary':
        cardinality = 2**n
    elif decision_space == 'one_hot':
        cardinality = n

    # Check if the inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)

    warning_large_decision_space(decision_space, n)

    if ind_func is None:
        def ind_func(s, x): return True

    N = len(dataset)

    # Sample a signal and response to get the dimension of the cost vector
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
        for k in range(cardinality):
            x = candidate_action(k, decision_space, n)
            if ind_func(s_hat, x):
                phi_1 = phi(s_hat, x)
                phi_2 = phi(s_hat, x_hat)
                if dist_func is None:
                    dist = 0
                else:
                    try:
                        dist = dist_func(x_hat, x)
                    except TypeError:
                        dist = dist_func(x_hat, x, s_hat)

                mdl.addConstr(
                    gp.quicksum(
                        theta[j]*(phi_1[j] - phi_2[j]) for j in range(p)
                    ) >= dist
                )

    if (dist_func is None) and (theta_hat is None):
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
            obj = 0.5 * gp.quicksum(
                (theta[i] - theta_hat[i])**2 for i in range(p)
            )
        elif regularizer == 'L1':
            t = mdl.addVars(p, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            obj = gp.quicksum(t)
            mdl.addConstrs(theta[i] - theta_hat[i] <= t[i] for i in range(p))
            mdl.addConstrs(theta_hat[i] - theta[i] <= t[i] for i in range(p))

        mdl.setObjective(obj, gp.GRB.MINIMIZE)
        mdl.optimize()

    theta_opt = np.array([theta[i].X for i in range(p)])

    if mdl.status != 2:
        print(
            f'Optimal solution not found. Gurobi status code = {mdl.status}.'
            'Set the flag verbose=True for more details. Note: the IO'
            'optimization problem will always be infeasible if the data is not'
            'consistent.'
        )

    return theta_opt


def discrete(
    dataset,
    X,
    phi,
    dist_func=None,
    Theta=None,
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
    verbose=False,
    gurobi_params=None,
):
    """
    Inverse optimization for discrete FOPs.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/discrete

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response.
    X : tuple(str, int, {callable, None})
        Constraint set. Given a signal s and response x, X is a tuple
        containing the type of the decision space, the size of the decision
        space, and any extra constraints as an indicator function. For example,
        if the decision vector is binary, with size n, and respects the
        constraint Ax <= b., then X = ('binary', n, ind_func), where
        ind_func(s, x) equals True if Ax <= b, and False otherwise. Notice that
        A and b may be part of the signal s, which is why the indicator
        function takes both s and x as inputs. If ind_func=None, it will be
        defined as "def ind_func(s, x): return True".
    phi : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    dist_func : {callable, None}, optional
        Distance penalization function. Given two responses x1 and x2, returns
        the distance between them according to some distance metric. Syntax:
        dist_func(x1, x2). Alternatively, the function can also take the signal
        s_hat as a third argument. For instace, to use a distance function in
        feature space instead of action space, we can use
        dist_func(x1, x2, s_hat) = dist(phi(s_hat, x1), phi(s_hat, x2)). In
        some cases, this can improve the performance of the model. The default
        is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector theta. The default
        is None.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector.

    """
    _, n, _ = X

    if theta_hat is None:
        theta_hat_mod = None
    else:
        theta_hat_mod = np.concatenate(([0], theta_hat))

    dataset_mod = []
    for data in dataset:
        s_hat, x_hat = data
        s_hat_mod = (
            np.zeros((1, 1)), np.zeros((1, n)), np.array([0]), s_hat
        )
        x_hat_mod = (np.array([0]), x_hat)
        dataset_mod.append((s_hat_mod, x_hat_mod))

    theta_opt_mod = mixed_integer_linear(
        dataset_mod,
        X,
        phi2=phi,
        dist_func_z=dist_func,
        Theta=Theta,
        regularizer=regularizer,
        reg_param=reg_param,
        theta_hat=theta_hat_mod,
        verbose=verbose,
        gurobi_params=gurobi_params,
    )

    theta_opt = theta_opt_mod[1:]

    return theta_opt


def continuous_linear(
    dataset,
    phi1,
    add_dist_func_y=False,
    Theta=None,
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
    verbose=False,
    gurobi_params=None,
):
    """
    Inverse optimization for continuous linear FOPs.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/continuous_linear

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response. The
        signal s must be provided as a tuple (A,b,w). For more details, see the
        link above.
    phi1 : callable
        Feature function. Given a signal s and response x, returns a 1D
        ndarray feature vector. Syntax: phi(s, x).
    add_dist_func_y: bool, optional
        If True, adds l-infinity distance penalization to the continuous part
        or response vector. The default is 'None'.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector theta=vec(Q). The
        default is None.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta or regularizer. If Gurobi does not find an optimal
        solution.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector in the form theta = Q.flatten('F')

    """
    Z = ('binary', 0, None)

    if theta_hat is None:
        theta_hat_mod = None
    else:
        theta_hat_mod = np.concatenate((theta_hat, [0]))

    dataset_mod = []
    for data in dataset:
        s_hat, x_hat = data
        A, b, w = s_hat
        s_hat_mod = (A, np.zeros((len(b), 1)), b, w)
        x_hat_mod = (x_hat, np.array([0]))
        dataset_mod.append((s_hat_mod, x_hat_mod))

    def phi1_mod(w, z): return phi1(w)

    theta_opt_mod = mixed_integer_linear(
        dataset_mod,
        Z,
        add_dist_func_y=add_dist_func_y,
        phi1=phi1_mod,
        Theta=Theta,
        regularizer=regularizer,
        reg_param=reg_param,
        theta_hat=theta_hat_mod,
        verbose=verbose,
        gurobi_params=gurobi_params,
    )

    theta_opt = theta_opt_mod[:-1]

    return theta_opt


def continuous_quadratic(
    dataset,
    phi1=None,
    add_dist_func_y=False,
    Theta=None,
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
    verbose=False,
):
    """
    Inverse optimization for continuous quadratic FOPs.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/continuous_quadratic

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response. The
        signal s must be provided as a tuple (A,b,w). For more details, see the
        link above.
    phi1 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi1(w, z): return np.array([0])". The default is None.
    add_dist_func_y: bool, optional
        If True, adds l-infinity distance penalization to the continuous part
        of the response vector. The default is 'None'.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector
        theta=(vec(Qyy), vec(Q)). The default is None.
    verbose : bool, optional
        If True, print the solver's output. The default is False.

    Raises
    ------
    Exception
        If unsupported Theta or regularizer. If Gurobi does not find an optimal
        solution.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector in the form
        theta = np.concatenate((Qyy.flatten('F'), Q.flatten('F')))

    """
    Z = ('binary', 0, None)

    if theta_hat is None:
        theta_hat_mod = None
    else:
        theta_hat_mod = np.concatenate((theta_hat, [0]))

    dataset_mod = []
    for data in dataset:
        s_hat, x_hat = data
        A, b, w = s_hat
        s_hat_mod = (A, np.zeros((len(b), 1)), b, w)
        x_hat_mod = (x_hat, np.array([0]))
        dataset_mod.append((s_hat_mod, x_hat_mod))

    def phi1_mod(w, z): return phi1(w)

    theta_opt_mod = mixed_integer_quadratic(
        dataset_mod,
        Z,
        phi1=phi1_mod,
        add_dist_func_y=add_dist_func_y,
        Theta=Theta,
        regularizer=regularizer,
        reg_param=reg_param,
        theta_hat=theta_hat_mod,
        verbose=verbose,
    )

    theta_opt = theta_opt_mod[:-1]

    return theta_opt


def mixed_integer_linear(
    dataset,
    Z,
    phi1=None,
    phi2=None,
    add_dist_func_y=False,
    dist_func_z=None,
    Theta=None,
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
    verbose=False,
    gurobi_params=None,
):
    """
    Inverse optimization for mixed-integer FOPs with linear continuous part.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/mixed_integer_linear

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response. The
        signal s must be provided as a tuple (A,B,c,w), and the reponse x must
        be provided as a tuple (y,z). For more details, see the link above.
    Z : tuple(str, int, {callable, None})
        Constraint set for the integer part of the decision vector. Given a
        signal s = (A, B, c, w) and response x = (y, z), Z is a tuple
        containing the type of the decision space, the size of the decision
        space, and any extra constraints as an indicator function, all w.r.t.
        the integer part or the decision vector, i.e., z. For example,
        if z is binary, with size n, and respects the
        constraint Dx <= e., then Z = ('binary', n, ind_func), where
        ind_func(w, z) equals True if Dz <= e, and False otherwise. Notice that
        D and e may be part of w, which is why the indicator function
        takes both w and z as inputs. If ind_func=None, it will be defined as
        "def ind_func(w, z): return True".
    phi1 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi1(w, z): return np.array([0])". The default is None.
    phi2 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi2(w, z): return np.array([0])". The default is None.
    add_dist_func_y: bool, optional
        If True, adds l-infinity distance penalization to the continuous part
        of the response vector. The default is None.
    dist_func_z : {callable, None}, optional
        Distance penalization function. Given the integer parts of two
        responses x1=(y1,z1) and x2=(y2,z2), returns their distance according
        to some distance metric. Syntax: dist_func_z(z1, z2). Alternatively,
        the function can also take the signal w_hat as a third argument. For
        instace, to use a distance function in feature space instead of action
        space, we can use
        dist_func_z(z1, z2, w_hat) = dist(phi2(w_hat, z1), phi2(w_hat, z2)). In
        some cases, this can improve the performance of the model. The default
        is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector theta=(vec(Q), q).
        The default is None.
    verbose : bool, optional
        If True, print Gurobi's solver output. The default is False.
    gurobi_params : {list of tuple(str, value), None}, optional
        List of tuples with Gurobi's parameter name and value. For example,
        [('TimeLimit', 0.5), ('Method', 2)]. The default is None.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution. If neither phi1 nor phi2 are given.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector in the form
        theta = np.concatenate((Q.flatten('F'), q))

    """
    import gurobipy as gp

    decision_space, v, ind_func = Z
    if decision_space == 'binary':
        cardinality = 2**v
    elif decision_space == 'one_hot':
        cardinality = v

    # Check if the inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)

    # Warnings
    warning_large_decision_space(decision_space, v)
    warning_theta_hat_reg_param(theta_hat, reg_param)
    warning_add_dist_func_y(add_dist_func_y)

    if (phi1 is None) and (phi2 is None):
        raise Exception('Either phi1 or phi2 have to be given.')

    N = len(dataset)

    if ind_func is None:
        def ind_func(w, z): return True

    # Check if phi1 or phi2 were not provided, which means the problem is
    # purely discrete or continuous, respectively.
    discrete_flag = False
    continuous_flag = False
    if phi1 is None:
        def phi1(w, z): return np.array([0])
        discrete_flag = True
    if phi2 is None:
        def phi2(w, z): return np.array([0])
        continuous_flag = True

    # Sample a signal and response to get the dimensions of the problem
    s_test, x_test = dataset[0]
    A_test, _, _, w_test = s_test
    y_test, z_test = x_test
    m = len(phi1(w_test, z_test))
    r = len(phi2(w_test, z_test))
    t, u = A_test.shape

    # Initialize Gurobi model
    mdl = gp.Model()
    if not verbose:
        mdl.setParam('OutputFlag', 0)
    if gurobi_params is not None:
        for param, value in gurobi_params:
            mdl.setParam(param, value)

    Q = mdl.addVars(u, m, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    q = mdl.addVars(r, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
    beta = mdl.addVars(N, vtype=gp.GRB.CONTINUOUS)
    sum_beta = (1/N)*gp.quicksum(beta)

    if discrete_flag:
        mdl.addConstrs(Q[i, j] == 0 for i in range(u) for j in range(m))
    if continuous_flag:
        mdl.addConstrs(q[i] == 0 for i in range(r))

    if Theta == 'nonnegative':
        mdl.addConstrs(Q[i, j] >= 0 for i in range(u) for j in range(m))
        mdl.addConstrs(q[i] >= 0 for i in range(r))

    for i in range(N):
        s_hat, x_hat = dataset[i]
        y_hat, z_hat = x_hat
        A, B, c, w_hat = s_hat
        for k in range(cardinality):
            z = candidate_action(k, decision_space, v)
            if ind_func(w_hat, z):
                if dist_func_z is None:
                    dist_z = 0
                else:
                    try:
                        dist_z = dist_func_z(z_hat, z)
                    except TypeError:
                        dist_z = dist_func_z(z_hat, z, w_hat)

                if add_dist_func_y:
                    gamma_list = []
                    for index in range(u):
                        gamma = np.zeros(u)
                        gamma[index] = 1
                        gamma_list.append(gamma)
                    for index in range(u):
                        gamma = np.zeros(u)
                        gamma[index] = -1
                        gamma_list.append(gamma)
                else:
                    gamma_list = [np.zeros(u)]

                for gamma in gamma_list:
                    lamb = mdl.addVars(t, vtype=gp.GRB.CONTINUOUS)

                    ph1_hat = phi1(w_hat, z_hat)
                    ph2_hat = phi2(w_hat, z_hat)
                    Qph1_hat = [
                        gp.quicksum(Q[i, j]*ph1_hat[j] for j in range(m))
                        for i in range(u)
                    ]
                    yQph1_hat = gp.quicksum(
                        y_hat[j]*Qph1_hat[j] for j in range(u)
                    )
                    qphi2_hat = gp.quicksum(q[j]*ph2_hat[j] for j in range(r))
                    theta_phi = yQph1_hat + qphi2_hat

                    Bz = B @ z
                    lambcBz = gp.quicksum(
                        lamb[j]*(c[j] - Bz[j]) for j in range(t)
                    )
                    ph2 = phi2(w_hat, z)
                    qphi2 = gp.quicksum(q[j]*ph2[j] for j in range(r))

                    gammay_hat = gp.quicksum(
                        gamma[j]*y_hat[j] for j in range(u)
                    )
                    mdl.addConstr(
                        theta_phi + lambcBz - qphi2 + gammay_hat + dist_z
                        <= beta[i]
                    )

                    ph1 = phi1(w_hat, z)
                    mdl.addConstrs(
                        gp.quicksum(Q[i, j]*ph1[j] for j in range(m))
                        + gp.quicksum(lamb[j]*A[j, i] for j in range(t))
                        + gamma[i] == 0 for i in range(u)
                    )

    if reg_param > 0:
        if theta_hat is None:
            Q_hat = np.zeros((u, m))
            q_hat = np.zeros(r)
        else:
            Q_hat = theta_hat[:-r].reshape((u, m))
            q_hat = theta_hat[-r:]

        if regularizer == 'L2_squared':
            Q_sum = gp.quicksum(
                (Q[i, j] - Q_hat[i, j])**2 for i in range(u) for j in range(m)
            )
            q_sum = gp.quicksum((q[i] - q_hat[i])**2 for i in range(r))
            reg_term = (reg_param/2)*(Q_sum + q_sum)
        elif regularizer == 'L1':
            tQ = mdl.addVars(
                u, m, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS
            )
            tq = mdl.addVars(r, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)

            reg_term = reg_param*(gp.quicksum(tQ) + gp.quicksum(tq))

            mdl.addConstrs(
                Q[i, j] - Q_hat[i, j] <= tQ[i, j] for i in range(u)
                for j in range(m)
            )
            mdl.addConstrs(
                Q_hat[i, j] - Q[i, j] <= tQ[i, j] for i in range(u)
                for j in range(m)
            )
            mdl.addConstrs(q[i] - q_hat[i] <= tq[i] for i in range(r))
            mdl.addConstrs(q_hat[i] - q[i] <= tq[i] for i in range(r))
    else:
        reg_term = 0

    mdl.setObjective(reg_term + sum_beta, gp.GRB.MINIMIZE)

    # Check if the norm equality constraint needs to be added to avoid the
    # trivial solution theta=0
    if (((not add_dist_func_y) and (dist_func_z is None))
            and ((reg_param == 0) or (theta_hat is None))):
        if Theta == 'nonnegative':
            mdl.addConstr(gp.quicksum(q) + gp.quicksum(Q) == 1)
            mdl.optimize()

            if mdl.status != 2:
                print(
                    'Optimal solution not found. Gurobi status code ='
                    f'{mdl.status}. Set the flag verbose=True for more'
                    'details.'
                )

            Q_opt = np.array(
                [[Q[i, j].X for i in range(u)] for j in range(m)]
            )
            q_opt = np.array([q[i].X for i in range(r)])
        else:
            # Search over facets of unit L-infinity sphere for the solution
            # with the lowest objective value.
            best_obj = np.inf
            for i in range(r):
                for j in [-1, 1]:
                    cons = mdl.addConstr(q[i] == j)
                    mdl.optimize()
                    mdl.remove(cons)
                    # If an optimal solution was found
                    if mdl.status == 2:
                        obj_val = mdl.objVal
                        if obj_val < best_obj:
                            best_obj = obj_val
                            Q_opt = np.array(
                                [[Q[i, j].X for i in range(u)]
                                 for j in range(m)]
                            )
                            q_opt = np.array([q[i].X for i in range(r)])
            for i in range(u):
                for k in range(m):
                    for j in [-1, 1]:
                        cons = mdl.addConstr(Q[i, k] == j)
                        mdl.optimize()
                        mdl.remove(cons)
                        # If an optimal solution was found
                        if mdl.status == 2:
                            obj_val = mdl.objVal
                            if obj_val < best_obj:
                                best_obj = obj_val
                                Q_opt = np.array(
                                    [[Q[i, j].X for i in range(u)]
                                     for j in range(m)]
                                )
                                q_opt = np.array([q[i].X for i in range(r)])
    else:
        mdl.optimize()

        if mdl.status != 2:
            print(
                'Optimal solution not found. Gurobi status code ='
                f' {mdl.status}.Set the flag verbose=True for more details.'
            )

        Q_opt = np.array(
            [[Q[i, j].X for i in range(u)] for j in range(m)]
        )
        q_opt = np.array([q[i].X for i in range(r)])

    theta_opt = np.concatenate((Q_opt.flatten('F'), q_opt))
    return theta_opt


def mixed_integer_quadratic(
    dataset,
    Z,
    phi1=None,
    phi2=None,
    add_dist_func_y=False,
    dist_func_z=None,
    Theta=None,
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
    verbose=False
):
    """
    Inverse optimization for mixed-integer FOPs with quadratic continuous part.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/mixed_integer_quadratic

    Parameters
    ----------
    dataset : list of tuples
        List of tuples (s, x), where s is the signal and x is the response. The
        signal s must be provided as a tuple (A,B,c,w), and the reponse x must
        be provided as a tuple (y,z). For more details, see the link above.
    Z : tuple(str, int, {callable, None})
        Constraint set for the integer part of the decision vector. Given a
        signal s = (A, B, c, w) and response x = (y, z), Z is a tuple
        containing the type of the decision space, the size of the decision
        space, and any extra constraints as an indicator function, all w.r.t.
        the integer part or the decision vector, i.e., z. For example,
        if z is binary, with size n, and respects the
        constraint Dx <= e., then Z = ('binary', n, ind_func), where
        ind_func(w, z) equals True if Dz <= e, and False otherwise. Notice that
        D and e may be part of w, which is why the indicator function
        takes both w and z as inputs. If ind_func=None, it will be defined as
        "def ind_func(w, z): return True".
    phi1 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi1(w, z): return np.array([0])". The default is None.
    phi2 : {callable, None}, optional
        Feature function. Given w and response z, returns a 1D
        ndarray feature vector. Syntax: phi1(w, z). If None, it will be defined
        as "def phi2(w, z): return np.array([0])". The default is None.
    add_dist_func_y: bool, optional
        If True, adds l-infinity distance penalization to the continuous part
        of the response vector. The default is None.
    dist_func_z : {callable, None}, optional
        Distance penalization function. Given the integer parts of two
        responses x1=(y1,z1) and x2=(y2,z2), returns their distance according
        to some distance metric. Syntax: dist_func_z(z1, z2). Alternatively,
        the function can also take the signal w_hat as a third argument. For
        instace, to use a distance function in feature space instead of action
        space, we can use
        dist_func_z(z1, z2, w_hat) = dist(phi2(w_hat, z1), phi2(w_hat, z2)). In
        some cases, this can improve the performance of the model. The default
        is None.
    Theta : {None, 'nonnegative'}, optional
        Constraints on cost vector theta. The default is None.
    regularizer : {'L2_squared', 'L1'}, optional
        Type of regularization on cost vector theta. The default is
        'L2_squared'.
    reg_param : float, optional
        Nonnegative regularization parameter. The default is 0.
    theta_hat : {1D ndarray, None}, optional
        A priory belief or estimate of the true cost vector
        theta=(vec(Qyy), vec(Q), q). The default is None.
    verbose : bool, optional
        If True, print the solver's output. The default is False.

    Raises
    ------
    Exception
        If unsupported Theta, regularizer, or decision_space. If Gurobi does
        not find an optimal solution.

    Returns
    -------
    theta_opt : 1D ndarray
        An optimal cost vector in the form theta =
        np.concatenate((Qyy.flatten('F'), Q.flatten('F'), q.flatten('F')))

    """
    import cvxpy as cp

    decision_space, v, ind_func = Z
    if decision_space == 'binary':
        cardinality = 2**v
    elif decision_space == 'one_hot':
        cardinality = v

    # Check if the inputs are valid
    check_Theta(Theta)
    check_decision_space(decision_space)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)

    # Warnings
    warning_large_decision_space(decision_space, v)
    warning_theta_hat_reg_param(theta_hat, reg_param)
    warning_add_dist_func_y(add_dist_func_y)

    N = len(dataset)

    if ind_func is None:
        def ind_func(w, z): return True

    if phi1 is None:
        def phi1(w, z): return np.array([0])
    if phi2 is None:
        def phi2(w, z): return np.array([0])

    # Sample a signal and response to get the dimensions of the problem
    s_test, x_test = dataset[0]
    A_test, _, _, w_test = s_test
    y_test, z_test = x_test
    m = len(phi1(w_test, z_test))
    r = len(phi2(w_test, z_test))
    t, u = A_test.shape

    Qyy = cp.Variable((u, u), PSD=True)
    Q = cp.Variable((u, m))
    q = cp.Variable((r, 1))
    beta = cp.Variable(N)

    constraints = []

    sum_beta = (1/N)*cp.sum(beta)

    if Theta == 'nonnegative':
        constraints += [Q >= 0, q >= 0]

    for i in range(N):
        s_hat, x_hat = dataset[i]
        y_hat, z_hat = x_hat
        A, B, c, w_hat = s_hat
        for k in range(cardinality):
            z = candidate_action(k, decision_space, v)
            if ind_func(w_hat, z):
                if dist_func_z is None:
                    dist_z = 0
                else:
                    try:
                        dist_z = dist_func_z(z_hat, z)
                    except TypeError:
                        dist_z = dist_func_z(z_hat, z, w_hat)

                if add_dist_func_y:
                    gamma_list = []
                    for index in range(u):
                        gamma = np.zeros(u)
                        gamma[index] = 1
                        gamma_list.append(gamma)
                    for index in range(u):
                        gamma = np.zeros(u)
                        gamma[index] = -1
                        gamma_list.append(gamma)
                else:
                    gamma_list = [np.zeros(u)]

                for gamma in gamma_list:
                    alpha = cp.Variable((1, 1))
                    lamb = cp.Variable((t, 1))

                    theta_phi_hat = (
                        y_hat.T @ Qyy @ y_hat
                        + y_hat.T @ Q @ phi1(w_hat, z_hat)
                        + q.T @ phi2(w_hat, z_hat)
                    )

                    lambcBz = lamb.T @ (c - B @ z)

                    qphi2 = q.T @ phi2(w_hat, z)

                    gammay_hat = gamma.T @ y_hat

                    constraints += [
                        theta_phi_hat + alpha + lambcBz - qphi2 + gammay_hat
                        + dist_z <= beta[i]
                    ]

                    off_diag = (
                        Q @ phi1(w_hat, z).reshape((m, 1)) + A.T @ lamb
                        + gamma.reshape((u, 1))
                    )
                    constraints += [
                        cp.bmat([[Qyy, off_diag], [off_diag.T, 4*alpha]]) >> 0
                    ]
                    constraints += [lamb >= 0]

    if reg_param > 0:
        if theta_hat is None:
            Qyy_hat = np.zeros((u, u))
            Q_hat = np.zeros((u, m))
            q_hat = np.zeros((r, 1))
        else:
            Qyy_hat = theta_hat[:u**2].reshape((u, u))
            Q_hat = theta_hat[u**2:-r].reshape((u, m))
            q_hat = theta_hat[-r:].reshape((r, 1))

        if regularizer == 'L2_squared':
            Qyy_sum = cp.sum_squares(Qyy - Qyy_hat)
            Q_sum = cp.sum_squares(Q - Q_hat)
            q_sum = cp.sum_squares(q - q_hat)
            reg_term = (reg_param/2)*(Qyy_sum + Q_sum + q_sum)
        elif regularizer == 'L1':
            tQyy = cp.Variable((u, u), symmetric=True)
            tQ = cp.Variable((u, m))
            tq = cp.Variable((r, 1))
            reg_term = reg_param*(cp.sum(tQyy) + cp.sum(tQ) + cp.sum(tq))
            constraints += [Qyy - Qyy_hat <= tQyy, Qyy_hat - Qyy <= tQyy]
            constraints += [Q - Q_hat <= tQ, Q_hat - Q <= tQ]
            constraints += [q - q_hat <= tq, q_hat - q <= tq]
    else:
        reg_term = 0

    obj = cp.Minimize(reg_term + sum_beta)

    # Check if the trace equality constraint needs to be added to avoid the
    # trivial solution theta=0
    if (((not add_dist_func_y) and (dist_func_z is None))
            and ((reg_param == 0) or (theta_hat is None))):
        constraints += [cp.trace(Qyy) == 1]

    prob = cp.Problem(obj, constraints)
    prob.solve(verbose=verbose)

    if prob.status != 'optimal':
        raise Exception(
            f'Optimal solution not found. CVXPY status code = {prob.status}.'
            'Set the flag verbose=True for more details.'
        )

    Qyy_opt = Qyy.value
    Q_opt = Q.value
    q_opt = q.value

    theta_opt = np.concatenate(
        (Qyy_opt.flatten('F'), Q_opt.flatten('F'), q_opt.flatten('F'))
    )
    return theta_opt


def FOM(
    dataset,
    phi,
    theta_0,
    FOP,
    step_size,
    T,
    Theta=None,
    step='standard',
    regularizer='L2_squared',
    reg_param=0,
    theta_hat=None,
    batch_type=1,
    averaged=0,
    callback=None,
    callback_resolution=1,
    normalize_grad=False,
    verbose=False
):
    """
    Optimize (Augmented) Suboptimality loss using first-order methods.

    For more details, see
    https://github.com/pedroszattoni/invopt/tree/main/examples/FOM

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
        The number of iterations for the algorithm.
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
        A priory belief or estimate of the true cost vector. The default is
        None.
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
        If not None, callback(theta_t) is evaluated for
        t=0,callback_resolution,2*callback_resolution... . The default is None.
    callback_resolution : int, optional
        Callback function resolution. The default is 1.
    normalize_grad : bool, optional
        If True, subgradient vectors are normalized before each iteration of
        the algorithm. If step='standard', the L2 norm of the subgradient is
        used. If step='exponentiated', the L-infinity norm of the subgradient
        is used. The default is False.
    verbose : bool, optional
        If True, prints iteration counter. The default is False.

    Raises
    ------
    Exception
        If unsupported Theta or regularizer. If step = 'exponentiated' and
        the regularizer is not 'L1' or theta_hat is not None.

    Returns
    -------
    theta_T : {1D ndarray, list}
        If callback=None, returns the final (averaged) vector found after T
        iterations of the algorithm. Otherwise, returns a list of size T+1
        with elements callback(theta_t) for t=0,...,T, where theta_t is the
        (averaged) vector after t iterations of the algorithm.

    """
    # Check if the inputs are valid
    check_Theta(Theta)
    check_regularizer(regularizer)
    check_reg_parameter(reg_param)

    # Warnings
    warning_theta_hat_reg_param(theta_hat, reg_param)

    if (step == 'exponentiated') and (regularizer != 'L1') and (reg_param > 0):
        raise Exception(
            'To use step = \'exponentiated\' with reg_param > 0,'
            'regularizer = \'L1\' is required.'
        )

    if (step == 'exponentiated') and (theta_hat is not None):
        raise Exception(
            'When step=\'exponentiated\', theta_hat must be None.'
        )

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
                reg_grad, loss_grad = compute_grad(
                    theta_t, samples, phi, FOP, regularizer, reg_param,
                    theta_hat
                )
                theta_t = grad_step(
                    theta_t, eta_t, reg_grad, loss_grad, reg_param, Theta,
                    step, normalize_grad
                )
        else:
            batch_size = int(np.ceil(batch_type*N))
            sample_idxs = np.random.choice(N, batch_size, replace=False)
            samples = [dataset[i] for i in sample_idxs]
            reg_grad, loss_grad = compute_grad(
                theta_t, samples, phi, FOP, regularizer, reg_param, theta_hat
            )
            theta_t = grad_step(
                theta_t, eta_t, reg_grad, loss_grad, reg_param, Theta, step,
                normalize_grad
            )

        if averaged == 0:
            theta_avg = theta_t
        elif averaged == 1:
            theta_avg = (1/(t+1))*theta_t + (t/(t+1))*theta_avg
        elif averaged == 2:
            theta_avg = (2/(t+2))*theta_t + (t/(t+2))*theta_avg

        if (callback is not None) and (np.mod(t+1, callback_resolution) == 0):
            callback_list.append(callback(theta_avg))

    # Check if the list callback_list is empty
    if callback_list:
        theta_T = callback_list
    else:
        theta_T = theta_avg

    return theta_T


def gradient_regularizer(theta_t, regularizer, reg_param, theta_hat):
    """
    Compute (sub)gradient of the regularizer.

    Parameters
    ----------
    theta_t : 1D ndarray
        Vector where the gradient will be evaluated at.
    regularizer : {'L2_squared', 'L1'}
        Type of regularization on cost vector theta.
    reg_param : float
        Nonnegative regularization parameter..
    theta_hat : 1D ndarray
        A priory belief or estimate of the true cost vector.

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


def compute_grad(
    theta_t, samples, phi, FOP, regularizer, reg_param, theta_hat
):
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
        # Check if the FOP is augmented
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


def grad_step(
    theta_t, eta_t, reg_grad, loss_grad, reg_param, Theta, step,
    normalize_grad
):
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
                          'Theta != \'nonnegative\' still need to be tested.')
            theta_pos_t = np.clip(theta_t, 0, None)
            theta_neg_t = np.clip(theta_t, None, 0)
            theta_pos_t1 = np.multiply(theta_pos_t, np.exp(-eta_t*grad))
            theta_neg_t1 = np.multiply(theta_neg_t, np.exp(eta_t*grad))
            theta_t1 = theta_pos_t1 - theta_neg_t1
            norm_theta_t1 = np.sum(theta_pos_t1) + np.sum(theta_neg_t1)

        # If outside the simplex, project onto it
        if reg_param*norm_theta_t1 > 1:
            theta_t1 = theta_t1/(reg_param*norm_theta_t1)

    # Projection onto Theta
    if Theta == 'nonnegative':
        theta_t1 = np.clip(theta_t1, 0, None)

    return theta_t1
