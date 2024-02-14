"""
InvOpt package example: auxiliary function for VRPTW example.

Author: Pedro Zattoni Scroccaro
"""

import numpy as np
from typing import List, Optional
import pyvrp as pv

np.random.seed(0)

colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#808080',
          '#a65628', '#FFD700']


def replace(
    data: pv.ProblemData,
    clients: Optional[List[pv.Client]] = None,
    vehicle_types: Optional[List[pv.VehicleType]] = None,
    distance_matrix: Optional[List[List[int]]] = None,
    duration_matrix: Optional[List[List[int]]] = None,
) -> pv.ProblemData:
    """Return a customized version of the given ``pv.ProblemData`` object."""
    if clients is None:
        clients = [data.client(idx) for idx in range(data.num_clients + 1)]

    if vehicle_types is None:
        vehicle_types = [
            data.vehicle_type(idx) for idx in range(data.num_vehicle_types)
        ]

    if distance_matrix is None:
        distance_matrix = [
            [data.dist(i, j) for j in range(data.num_clients + 1)]
            for i in range(data.num_clients + 1)
        ]

    if duration_matrix is None:
        duration_matrix = [
            [data.duration(i, j) for j in range(data.num_clients + 1)]
            for i in range(data.num_clients + 1)
        ]

    return pv.ProblemData(
        clients=clients,
        vehicle_types=vehicle_types,
        distance_matrix=distance_matrix,
        duration_matrix=duration_matrix,
    )


def new_clients_tw(instance, tw, n_total):
    """."""
    clients_list = [
        pv.Client(
            x=instance.client(idx).x,
            y=instance.client(idx).y,
            demand=instance.client(idx).demand,
            service_duration=instance.client(idx).service_duration,
            tw_early=tw[idx][1],
            tw_late=tw[idx][2]
        )
        for idx in range(n_total)
    ]
    return clients_list


def create_new_instance(instance, theta, s, n_total, M):
    """."""
    theta_mat = (M/sum(theta))*theta.reshape(n_total, n_total)
    np.fill_diagonal(theta_mat, 0)

    new_clients = new_clients_tw(instance, s, n_total)

    new_distance_matrix = [
        [theta_mat[i, j] for j in range(n_total)]
        for i in range(n_total)
    ]

    new_instance = replace(
        instance,
        clients=new_clients,
        distance_matrix=new_distance_matrix,
    )
    return new_instance


def get_routes(result):
    """."""
    routes = result.best.get_routes()

    routes_list = []
    for route in routes:
        routes_list.append(route.visits())
    routes_list.sort()

    return routes_list


def solve_VRPTW(instance, iter_limit):
    """."""
    model = pv.Model.from_data(instance)
    result = model.solve(stop=pv.stop.MaxIterations(iter_limit), seed=0)
    # cost = result.cost()
    # print(cost)
    return result


def routes_to_vec(routes, n_total):
    """
    Represent list of routes as a vector.

    Given a list of routes, create an incidence matrix, where the element
    (i,j) of the matrix equals to 1 if client j is visited after client i
    (in some route), and 0 otherwise. All routes start and end at the depot
    (node 0), even though this is not explictly stated in the routes. Return
    the flattened version of the resulting matrix.

    Parameters
    ----------
    routes : list of lists
        List of routes.

    Returns
    -------
    x_vec : ndarray
        1D numpy array corresponding to the flattened matrix binary matrix
        encoding the zone sequence.

    """
    x_mat = np.zeros((n_total, n_total))
    for route in routes:
        n_route = len(route)
        x_mat[0, route[0]] = 1
        x_mat[route[-1], 0] = 1
        for i in range(n_route-1):
            x_mat[route[i], route[i+1]] = 1

    x_vec = x_mat.flatten()
    return x_vec


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
