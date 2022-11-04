# Hypergraph SBM : generation and spectral methods

This repository contains the code to generate the figures of [our article](https://arxiv.org/abs/2203.07346) on hypergraph stochastic block models.

The main features are:
- a `HyperGraph` class that represent an unweighted hypergraph, stored as a list of hyperedges. Its adjacency matrix, degree profile, and reduced-nonbacktracking matrix can be accessed via methods.
- a function `fast_hsbm` that generates HSBM-distributed graphs in an efficient fashion, for any choice of parameters.
