import numpy as np
import itertools
import math
import random
from hypergraph import HyperGraph
from hsbm_utils import hsbm_clusters
from scipy.special import comb


def increasing(edge):
    for i, elt in enumerate(edge[1:]):
        if elt <= edge[i]:
            return False

    return True


def fast_hsbm(n_nodes, F, pi=None, n_clusters=None):
    sizes, offsets, _ = hsbm_clusters(n_nodes, pi=pi, n_clusters=n_clusters)
    n_clusters = len(sizes)
    edge_size = len(F.shape)
    edge_list = []

    # We adapt the networkx function to tensors
    block_iter = itertools.combinations_with_replacement(range(n_clusters), edge_size)
    for block in block_iter:
        p = F[block] / comb(n_nodes, edge_size - 1)
        edge_index = 0
        block_sizes = sizes[np.array(block)]
        block_offsets = offsets[np.array(block)]
        max_index = math.prod(block_sizes)

        while True:
            # The number of non-edges between two edges is ~Geom(1-p)
            logrand = math.log(random.random())
            skip = math.floor(logrand / math.log(1 - p))
            edge_index += skip

            if edge_index > max_index:
                break

            edge = block_offsets + np.unravel_index(edge_index, block_sizes)
            edge_index += 1

            # we only keep increasing edges for uniqueness
            if not increasing(edge):
                continue

            edge_list.append(edge)

    return HyperGraph(edge_size, n_nodes, edge_list)
