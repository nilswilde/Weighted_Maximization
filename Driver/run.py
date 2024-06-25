import copy

import demos
import numpy as np
from simulation_utils import load_trajectories, create_env
from algos import compute_delta
from config import CFG

# Load sampled trajectories
sample_mode = 'Original'
# sample_mode = 'Uniform'
sample_mode = 'GreedyHeuristic'
sample_modes = ['Original', 'Uniform', 'GreedyHeuristic']
# sample_modes = [ 'Uniform', 'GreedyHeuristic']
sample_modes = [ 'Original']

simulation_object = create_env(CFG['task'])

# Generate different simulated users
generated_users = []
weights= []
for _ in range(1):
    w = np.random.rand(simulation_object.num_of_features)  # draw : uniformly random user
    # w = np.array([0.35430563, 0.33672912, 0.22400807, 0.08495718])  # draw : uniformly random user
    w = w / sum(w)
    weights.append(w)
    for sample_mode in sample_modes:
        trajectories = load_trajectories(CFG['task'], 200, sample_mode)
        delta =  compute_delta(trajectories, w)
        for sigma in CFG['sigma_values']:
            for alpha in CFG['alpha_values']:
                user = {'w': w, 'alpha': alpha, 'noise_std': sigma, 'delta':delta}
                generated_users.append(user)


        slider_step_size = CFG['slider_step_size']
        demos.run(CFG['task'],sample_mode, trajectories, slider_step_size, copy.deepcopy(generated_users), CFG['acquisitions'])
