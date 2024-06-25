import copy, random

import numpy as np
import math as m
from scipy.optimize import linprog
from scipy.optimize import minimize
from evaluation import normalize_features

def compute_k_grid_samples(planner, K):
    print('compute ',K, 'grid samples')
    np.random.seed(71) # fix arbitrary seed
    weights = []
    if planner.dim == 2:
        for k in range(K+1):
            w_1_val = (k)/(K)
            weights.append([w_1_val, 1-w_1_val])
    else:
        for i in range(planner.dim):
            w=[0]*planner.dim
            w[i]=1
            weights.append(list(w))
        print('generating ', K, 'random samples')
        for k in range(K-planner.dim):
            w = np.random.random(planner.dim)
            w = np.divide(w, sum(w))
            weights.append(list(w))
    samples = []
    for w in weights:
        traj = planner.find_optimum(w)
        samples.append(traj)
    samples += planner.get_basis()
    samples = sorted(samples, key=lambda d: d['w'][0])

    # print('raw samples', len(samples))
    # filtered_samples = filter_duplicates(samples)
    # print('unique samples', len(filtered_samples))
    return samples

def filter_duplicates(samples):
    filtered_samples = []
    for s in samples:
        # print('sample', s['w'], s['f'])
        is_new = True
        for other_s in filtered_samples:
            if s['states'] == other_s['states']:
                is_new = False
                break
        if is_new:
            filtered_samples.append(s)
    return filtered_samples



