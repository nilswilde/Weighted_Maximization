import numpy as np
from simulation_utils import create_env, get_feedback, run_algo, load_trajectories, best_id_out_of_dataset
import algos
import scipy.io

NOISE_STD = 0.2
DATASET_SIZE = 120
SAMPLE_SIZE = 200
NUM_QUERIES = 10

class Experiment:
    def __init__(self, acquisition, resolution):
        self.task = 'fetch'
        self.acquisition = acquisition
        self.resolution = resolution
        self.simulation_object = create_env(self.task)
        d = self.simulation_object.num_of_features
        
        self.query_id = 0
        self.validation_query_id = 0
        self.validation_query_response_id = 0
        self.overall_id = 0
        
        self.PQ = []
        self.Up = []
        
        self.w_samples = np.zeros((NUM_QUERIES+1, SAMPLE_SIZE, d))
        self.w_samples[0] = np.random.randn(SAMPLE_SIZE, d)
        self.w_samples[0] = self.w_samples[0] / np.linalg.norm(self.w_samples[0], axis=1).reshape(-1, 1)    
        self.alpha_samples = np.zeros((NUM_QUERIES+1, SAMPLE_SIZE))
        self.alpha_samples[0] = np.random.rand(SAMPLE_SIZE)
        self.sample_logprobs = np.zeros((NUM_QUERIES+1, SAMPLE_SIZE)) # because there is no initial data
        
        self.trajectories = load_trajectories(self.task, DATASET_SIZE)
        self.delta_samples = np.zeros((NUM_QUERIES+1, SAMPLE_SIZE))
        self.delta_samples[0] = algos.compute_delta(self.trajectories, self.w_samples[0].T) # simple trick to get delta's of multiple w's
        
        self.input_id1s = np.zeros((NUM_QUERIES, 1), dtype=np.int32)
        self.input_id2s = np.zeros((NUM_QUERIES, 1), dtype=np.int32)
        self.scores = np.zeros((NUM_QUERIES, 1))
        self.phi_As = np.zeros((NUM_QUERIES, d))
        self.phi_Bs = np.zeros((NUM_QUERIES, d))
        
    def receive_feedback(self, up_i):
        self.overall_id += 1
        if self.validation_query_id == 0:
            self.phi_As[self.query_id] = self.trajectories['feature_set'][self.input_id1s[self.query_id]]
            self.phi_Bs[self.query_id] = self.trajectories['feature_set'][self.input_id2s[self.query_id]]
            self.PQ.append(self.phi_As[self.query_id] - self.phi_Bs[self.query_id])
            self.Up.append(up_i)
            new_w_samples, new_alpha_samples, new_delta_samples, new_sample_logprobs = algos.estimate_w_and_delta(self.trajectories, self.resolution, self.PQ, self.Up,
                                                                                                                    NOISE_STD, self.w_samples[self.query_id,-1],
                                                                                                                    self.alpha_samples[self.query_id,-1], SAMPLE_SIZE)
            self.query_id += 1
            self.w_samples[self.query_id] = new_w_samples
            self.alpha_samples[self.query_id] = new_alpha_samples
            self.delta_samples[self.query_id] = new_delta_samples
            self.sample_logprobs[self.query_id] = new_sample_logprobs
        elif self.validation_query_id == 1:
            self.validation_comp_up = up_i
            self.validation_query_response_id += 1
        elif self.validation_query_id == 2:
            self.validation_ordi_up = up_i
            self.validation_query_response_id += 1
        else:
            self.overall_id -= 1 # we take that back
            assert False, 'There is something wrong. This should not have happened.'
        
        
    def optimize_query(self):
        if not self.training_done:
            new_input_id1, new_input_id2, new_score = run_algo(self.acquisition, self.simulation_object, self.trajectories, self.resolution, NOISE_STD, self.w_samples[self.query_id], self.alpha_samples[self.query_id], self.delta_samples[self.query_id], self.sample_logprobs[self.query_id], self.PQ, self.Up)
            self.input_id1s[self.query_id] = new_input_id1
            self.input_id2s[self.query_id] = new_input_id2
            self.scores[self.query_id] = new_score
            return new_input_id1, new_input_id2, self.resolution
        elif self.validation_query_response_id == 0:
            self.validation_query_id = 1
            self.validation_comp_input_id1 = best_id_out_of_dataset(self.trajectories, np.mean(self.w_samples[0], axis=0))
            self.validation_comp_input_id2 = best_id_out_of_dataset(self.trajectories, np.mean(self.w_samples[-1], axis=0))
            return self.validation_comp_input_id1, self.validation_comp_input_id2, 0.1
        elif self.validation_query_response_id == 1:
            self.validation_query_id = 2
            self.validation_ordi_input_id = best_id_out_of_dataset(self.trajectories, np.mean(self.w_samples[-1], axis=0))
            return self.validation_ordi_input_id, None, 0.1
        assert False, 'There is something wrong. This should not have happened.'
               
    @property
    def done(self):
        return self.training_done and self.validation_done
        
    @property
    def training_done(self):
        return len(self.PQ) >= NUM_QUERIES
       
    @property
    def validation_done(self):
        return self.validation_query_response_id >= 2
    
    def save(self, username, id):
        scipy.io.savemat('user_data/' + username + '_' + str(id) + '.mat', {'id': id, 'task': self.task, 'acquisition': self.acquisition, 'resolution': self.resolution,
                            'PQ': np.array(self.PQ), 'Up': np.array(self.Up), 'w_samples': self.w_samples, 'alpha_samples': self.alpha_samples,
                            'delta_samples': self.delta_samples, 'sample_logprobs': self.sample_logprobs, 'scores': self.scores, 'input_id1s': self.input_id1s,
                            'input_id2s': self.input_id2s, 'phi_As': self.phi_As, 'phi_Bs': self.phi_Bs, 'validation_comp_input_id1': self.validation_comp_input_id1,
                            'validation_comp_input_id2': self.validation_comp_input_id2, 'validation_comp_up': self.validation_comp_up,
                            'validation_ordi_input_id': self.validation_ordi_input_id, 'validation_ordi_up': self.validation_ordi_up})
        