import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs
from generator import fast_hsbm
from hsbm_utils import symmetric_hsbm, hsbm_clusters
import os

n_nodes = 4000
edge_size = 4
n_clusters = 3
d = 4
mu_2 = 2

P = symmetric_hsbm(edge_size, n_clusters, d, mu_2)

if not os.path.exists("data/eigvecs.txt"):
    G = fast_hsbm(n_nodes, P, n_clusters=n_clusters)

    _, eigvecs = eigs(
        G.nonb_reduced(sparse=True), n_clusters, which="LM", return_eigenvectors=True
    )
    np.savetxt("data/eigvecs.txt", eigvecs.real)

_, _, labels = hsbm_clusters(n_nodes, n_clusters=n_clusters)
eigvecs = np.loadtxt("data/eigvecs.txt")
colorlist = ["tab:orange", "tab:green", "tab:purple"]
colors = [colorlist[label] for label in labels]

plt.scatter(eigvecs[n_nodes:, 1], eigvecs[n_nodes:, 2], c=labels, vmax=10, cmap="tab10")
plt.show()
