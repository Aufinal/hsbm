import numpy as np
import scipy.sparse as sp
from itertools import permutations


class HyperGraph:
    def __init__(self, edge_size, n_nodes, edge_list=None):
        self.edge_size = edge_size
        self.n_nodes = n_nodes
        self.edge_list = edge_list

    def adjacency_matrix(self, sparse=True):
        attr_name = "_sparse_adj" if sparse else "_adj"
        if not hasattr(self, attr_name):
            getattr(self, "_compute" + attr_name)()

        return getattr(self, attr_name)

    def _compute_adj(self):
        A = np.zeros((self.n_nodes, self.n_nodes))
        for edge in self.edge_list:
            for (i, j) in permutations(edge, 2):
                A[i, j] += 1

        self._adj = A

    def _compute_sparse_adj(self):
        edge_perms = self.edge_size * (self.edge_size - 1)
        n_edges = edge_perms * len(self.edge_list)
        row_ind = np.zeros(n_edges)
        col_ind = np.zeros(n_edges)

        for (edge_idx, edge) in enumerate(self.edge_list):
            for (pair_idx, (i, j)) in enumerate(permutations(edge, 2)):
                # The COO format allows for repeated entries, so we don't need to care about that
                row_ind[edge_perms * edge_idx + pair_idx] = i
                col_ind[edge_perms * edge_idx + pair_idx] = j

        self._sparse_adj = sp.csr_array(
            (np.ones(n_edges), (row_ind, col_ind)), shape=(self.n_nodes, self.n_nodes)
        )

    @property
    def degree_vector(self):
        if not hasattr(self, "_deg"):
            self._compute_deg()

        return self._deg

    def _compute_deg(self):
        d = np.zeros(self.n_nodes)
        for edge in self.edge_list:
            for i in edge:
                d[i] += 1

        self._deg = d

    def nonb_reduced(self, sparse=True):
        A = self.adjacency_matrix(sparse=sparse)
        if sparse:
            D = sp.diags(self.degree_vector)
            I = sp.eye(self.n_nodes)
            return sp.bmat(
                [
                    [None, D - I],
                    [-(self.edge_size - 1) * I, A - (self.edge_size - 2) * I],
                ]
            )
        else:
            D = np.diag(self.degree_vector)
            I = np.eye(self.n_nodes)
            return np.block(
                [
                    [np.zeros((self.n_nodes, self.n_nodes)), D - I],
                    [-(self.edge_size - 1) * I, A - (self.edge_size - 2) * I],
                ]
            )
