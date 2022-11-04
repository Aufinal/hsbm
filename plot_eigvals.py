import os

import numpy as np
from matplotlib import pyplot as plt

from generator import fast_hsbm
from hsbm_utils import symmetric_hsbm

n_nodes = 2000
edge_size = 4
n_clusters = 4
d = 4
mu_2 = 2

P = symmetric_hsbm(edge_size, n_clusters, d, mu_2)

if not os.path.exists("data/eigvals.txt"):

    G = fast_hsbm(n_nodes, P, n_clusters=n_clusters)
    eigvals = np.linalg.eigvals(G.sparse_nonb_reduced.toarray())

    np.savetxt("data/eigvals.txt", eigvals)


eigvals = np.loadtxt("data/eigvals.txt", dtype=complex)
eigvals.sort()

bulk = eigvals[:-n_clusters]
outliers = eigvals[-n_clusters:]
print(outliers)

fig, ax = plt.subplots()
ax.axis("equal")

ax.scatter(bulk.real, bulk.imag, c="tab:blue", marker=".")
ax.scatter(outliers.real, outliers.imag, c="tab:red", marker="*", s=100)
ax.add_patch(
    plt.Circle((0, 0), np.sqrt((edge_size - 1) * d), fill=False, color="tab:red", lw=2)
)
plt.show()
