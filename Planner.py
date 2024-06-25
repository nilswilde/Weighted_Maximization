import numpy as np
# from sampling import get_point_of_equal_cost, max_reg_in_neighbourhood_linprog#, compute_linear_combination
import copy, pickle
import os
from config import CFG

class Planner():
    '''
    Generic Planner Class
    '''
    def __init__(self, dim, scalarization, label='generic'):
        self.dim = dim # number of features
        self.label = label
        self.scalarization_mode = scalarization
        self.value_bounds = [{'lb': 0, 'ub': 1} for _ in range(self.dim)]# min and max values for each feature
        self.basis = None # basic solutions, e.g., for the [1 0 0], [0 1 0], [0 0 1] vectors

        self.sampled_solutions = []  # high number of sampled solutions, only for evaluation

    def __repr__(self):
        return self.label

    def get_cost_of_traj(self, traj, w):
        w_lift = [w_i+.0000001 for w_i in w]
        if self.scalarization_mode == 'linear':
            return np.dot(w_lift, traj['f'])
        if self.scalarization_mode == 'chebyshev':
            return np.max(np.multiply(w_lift, traj['f'])) + .0000001 * np.dot(w_lift, traj['f'])



    def get_value_bounds(self):
        '''
        :return:
        '''
        value_bounds = [{'lb': float('inf'), 'ub': -float('inf')} for _ in range(self.dim)]
        for i in range(self.dim):
            w = [0.0] * self.dim
            w[i] = 1.0
            sol = self.find_optimum(w)
            value_bounds[i]['lb'] = sol['f'][i]
            for j in range(self.dim):
                value_bounds[j]['ub'] = max(sol['f'][j], value_bounds[j]['ub'])
        print('value bounds', value_bounds)
        return value_bounds

    def get_basis(self):
        '''

        :return:
        '''
        print('compute basis', self.dim)
        basis = []
        value_bounds = [{'lb': float('inf'), 'ub': -float('inf')} for _ in range(self.dim)]
        for i in range(self.dim):
            w = [0.0] * self.dim
            w[i] = 1.0
            sol = self.find_optimum(w)
            basis += [sol]
            value_bounds[i]['lb'] = [sol['f'][i]]
        self.basis = basis
        return basis

    def find_optimum_with_LUT(self,w):
        return self.find_optimum(w)

    def find_optimum(self, w):
        '''
        Solve the planning problem for a given weight w
        :param w:
        :return:
        '''
        return {'w':w,
                'f': [1]*self.dim,
                'states': [(i,i)for i in range(10)]}

    def find_optima_for_set_of_weights(self, weights):
        trajects, opt_costs = [], []
        for w in weights:
            traj = self.find_optimum(w)
            cost = self.get_cost_of_traj(traj, w)
            traj['u'] = cost
            trajects.append(traj)
            opt_costs.append(cost)
        return trajects

    def compute_pair_regret(self, traj_P, traj_Q):
        c_QQ = self.get_cost_of_traj(traj_Q, traj_Q['w'])
        c_PQ = self.get_cost_of_traj(traj_P, traj_Q['w'])
        return c_PQ-c_QQ, c_PQ/c_QQ


    def compute_regrets(self, weights):
        '''

        :param weights:
        :param trajects:
        :return:
        '''
        regrets = []
        for w_P in weights:
            traj_P = self.find_optimum(w_P)
            regrets_row = []
            for w_Q in weights:
                traj_Q = self.find_optimum(w_Q)
                regrets_row.append(self.compute_pair_regret(traj_P, traj_Q))
            regrets.append(regrets_row)
        return regrets


    def compute_minmax_regret(self, samples):
        """

        :param samples:
        :return:
        """
        if len(self.sampled_solutions) == 0:
            self.generate_evaluation_samples()
        max_regret_abs, max_regret_rel = -float('inf'), 0
        int_regret_abs, int_regret_rel = 0, 0
        max_absreg_weight, max_relreg_weight = None, None
        for i in range(len(self.sampled_solutions)):
            abs_regrets_at_w, rel_regrets_at_w = [], []  # save the regrets for each trajectory at test point i
            for traj in samples:
                r_abs, r_rel = self.compute_pair_regret(traj, self.sampled_solutions[i])
                abs_regrets_at_w.append(r_abs)
                rel_regrets_at_w.append(r_rel)
            if min(abs_regrets_at_w) > max_regret_abs:  # check if the smallest regret at i is a new overall maximum
                max_regret_abs = min(abs_regrets_at_w)
                max_absreg_weight = self.sampled_solutions[i]['w']
            if min(rel_regrets_at_w) > max_regret_rel:
                max_regret_rel = min(rel_regrets_at_w)
                max_relreg_weight = self.sampled_solutions[i]['w']
            int_regret_abs += min(abs_regrets_at_w)
            int_regret_rel += min(rel_regrets_at_w)
        print('max reg at', max_absreg_weight)
        return {
            'max_regret': round(max_regret_abs,5),
            'max_relative_regret': round(max_regret_rel,5),
            'total_regret': round(int_regret_abs,5),
            'total_relative_regret': round(int_regret_rel,5)
        }

    def save_samples(self, samples, tag='uniform'):
        path = 'rewardLearning/ctrl_samples'
        if not os.path.isdir(path):
            os.mkdir(path)

        w_set = [sol['w'] for sol in samples]
        feature_set = [sol['f'] for sol in samples]
        input_set = [sol['states'] for sol in samples]
        np.savez(path+'/' + self.label + '_' + tag+'.npz', feature_set=feature_set, input_set=input_set,
                 w_set=w_set)

    def save_object(self, tag):
        print('save planner', self.label)
        folder = 'presamples/' + self.label +'/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder +self.label+'_'+tag+'.pickle', 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


def load_planner(label, folder):
    try:
        print('load', label)
        with open(folder+label+'.pickle', 'rb') as inp:
            loaded_object = pickle.load(inp)
    except:
        loaded_object = None
    return loaded_object

