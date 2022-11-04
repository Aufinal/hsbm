import numpy as np
from functools import reduce
from math import isclose


def signal_matrix(P, pi=None):
    """Generates the signal matrix Q from the affinity tensor P,
    such that the eigenvalues of Q and E[A] are identical."""

    edge_size = len(P.shape)
    n_clusters = P.shape[0]
    if pi is None:
        pi = np.ones(n_clusters) / n_clusters

    pi_tensor = reduce(np.outer, (pi,) * (edge_size - 2))
    D = np.tensordot(P, pi_tensor, axes=edge_size - 2)

    return D @ np.diag(pi)


def symmetric_hsbm(edge_size, n_clusters, d, mu_2):
    """Generates an affinity tensor P for the symmetric HSBM, such that
    the eigenvalues of Q are d and mu_2"""

    c_out = d - mu_2
    c_in = n_clusters ** (edge_size - 1) * mu_2 + c_out

    P = c_out * np.ones((n_clusters,) * edge_size, dtype=int)
    for i in range(n_clusters):
        P[(i,) * edge_size] = c_in

    return P


def hsbm_clusters(n_nodes, n_clusters=None, pi=None):
    """Partitions the nodes of an HSBM according to the probability vector pi.
    Returns the effective size of each class, as well as the true labels.
    If pi is not provided, it is taken as uniform over n_clusters."""

    if pi is not None:
        if not isclose(np.sum(pi), 1):
            raise ValueError("Pi must be a probability vector")
        n_clusters = len(pi)
    elif n_clusters is not None:
        pi = np.ones(n_clusters) / n_clusters
    else:
        raise ValueError(
            "Either the number of clusters or the cluster proportions must be provided"
        )

    # Cluster sizes are rounded below
    sizes = np.array(n_nodes * pi, dtype=int)
    offsets = np.cumsum(sizes) - sizes
    # Pad the last cluster to reach n nodes in total
    sizes[-1] += n_nodes - (offsets[-1] + sizes[-1])

    # True labels
    true_labels = np.repeat(np.arange(n_clusters), sizes)

    return sizes, offsets, true_labels
