import numpy as np
import math
from scipy.stats import qmc

def collapse_contact_matrix(C, group_encoding, group_dist):
    """
    Collapse contact matrix based on group encoding and group distribution
    (C) original contact matrix; must be a numpy array of shape (nxn)
        let original groups be indexed by i
    (group_encoding) matrix encoding collapsed groups; must be a numpy array of shape (kxn)
        k: number of groups after collapsing original contact matrix
        let collapsed groups be indexed by j
        each row j corresponds to the jth collapsed group and contains a sequence of 0s and 1s
            0: original group i does not belong in collapsed group j
            1: original group i does belong in collapsed group j
            e.g., row j = [0, 1, 1] means the jth collapsed group contains groups 2 and 3 from the original grouping
    (group_dist): original group distribution; must be a numpy array of length (n)
        the array must sum to 1
    """

    n = C.shape[0] # number of groups in original grouping
    k = group_encoding.shape[0] # number of groups in collapsed grouping
    
    # create scaled group_encoding matrix
    group_dist_mx = np.tile(group_dist, (k, 1)) # stack k copies of group_dist
    group_encoding_scaled = group_encoding * group_dist_mx
    # normalize so rows sum to 1
    row_sums = np.sum(group_encoding_scaled, axis = 1).reshape(k, 1)
    group_encoding_scaled = group_encoding_scaled / row_sums

    # create collapsed contact matrix
    collapsed_mx = group_encoding_scaled @ C @ group_encoding.T

    return collapsed_mx

def find_group_dist(C):
    """
    Returns group distribution for a given contact matrix
    (C) contact matrix; must be a numpy array of shape (nxn)
    """

    n = C.shape[0] # number of groups
    n_eqs = math.comb(n, 2) # number of equality comparisons

    # initialize matrix to be solved
    eqs_mx = np.zeros((n_eqs, n))
    counter = 0
    for i in range(n):
        for j in range(i+1, n):
            eqs_mx[counter, i] = C[i, j]
            eqs_mx[counter, j] = -C[j, i]
            counter += 1
    
    # solve system
    e_vals, e_vecs = np.linalg.eig(eqs_mx.T @ eqs_mx)
    # extract eigenvector with minimum eigenvalue
    sol = e_vecs[:, np.argmin(e_vals)]
    sol = sol / np.sum(sol)
    return(sol)

def generate_lhs_samples(n, l_bounds, u_bounds, rng):
    """
    Generates k-dimensional Latin hypercube samples
    (n): number of samples to generate
    (lower): array of lower bounds; must have length k
    (upper): array of upper bounds; must have length k
    (rng): numpy rng generator
    """

    # generate [0,1] k-dimensional LHS samples
    k = len(l_bounds)
    sampler = qmc.LatinHypercube(d=k, seed=rng)
    samples = sampler.random(n=n)
    # rescale LHS samples to correct bounds
    samples_scaled = qmc.scale(samples, l_bounds, u_bounds)

    return np.array(samples_scaled)