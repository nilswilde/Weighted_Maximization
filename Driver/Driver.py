import copy, time
from Planner import Planner
import numpy as np
np.bool = np.bool_
try:
    np.distutils.__config__.blas_opt_info = np.distutils.__config__.blas_ilp64_opt_info
except Exception:
    pass
from Driver.simulation_utils import create_env, compute_best, play
from Driver.demos import get_trajectory_for_weight
import sys, random
import dill as pickle
import Driver.car as car
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.image as image
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)#The OffsetBox is a simple container artist.
#
import os

# a lookup table for different initial settings to generate test instances
settings_LUP = [
    {'r':[0., -0.3, np.pi/2., 0.4],'h':[0.17, 0., np.pi/2., 0.41]},
    {'r':[0., -0.3, np.pi/2., 0.4],'h':[0.0, 0., np.pi/2., 0.41]},
    {'r':[0., -0.3, np.pi/2., 0.4],'h':[-0.17, 0., np.pi/2., 0.41]},
    {'r':[0.17, -0.3, np.pi/2., 0.1],'h':[0.17, 0.1, np.pi/2., 0.2]},
    {'r':[0.17, -0.3, np.pi/2., 0.4],'h':[0.0, 0., np.pi/2., 0.41]},
    {'r':[0.17, -0.3, np.pi/2., 0.4],'h':[-0.17, 0., np.pi/2., 0.41]},
    {'r':[0.17, -0.3, np.pi/2., 0.4],'h':[0.17, 0., np.pi/2., 0.41]},
    {'r':[0.17, -0.3, np.pi/2., 0.4],'h':[0.17, 0.5, np.pi/2., 0.41]},
    {'r':[0.17, -0.3, np.pi/2., 0.4],'h':[0.17, 0.5, np.pi/2., 0.2]},
]


class Driver(Planner):
    def __init__(self, scalarization):
        print("init Driver")
        super().__init__(4, scalarization, 'Driver')
        self.tag = 'driver'
        self.save_tag = self.tag +'_samples'
        self.generated_trajects = None
        self.sim_object = create_env(self.tag)
        print(self.save_tag)



    def save_object(self, tag):
        print('save planner', self.label, self.scalarization_mode)
        file_name = self.label + '_' + self.scalarization_mode + '_'+str(len(self.sampled_solutions))+'_multi.pickle'
        # file_name = self.label + '_' + self.scalarization_mode + '.pickle'
        save_object = copy.deepcopy(self)
        save_object.sim_object = None
        folder = 'presamples/' + self.label + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(folder+'/'+file_name,'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def save_samples_raw(self, filename=''):
        filename = self.save_tag
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def load_trajects(self):
        try:
            with open(self.save_tag , 'rb') as in_strm:
                driver_loaded = pickle.load(in_strm)
            self.generated_trajects = copy.deepcopy(driver_loaded.generated_trajects)
            return True
        except:
            return False

    def generate_trajectories(self, num_samples=20):
        '''
        Pre-generate a large set of Dubins' paths for different turn radia
        :return:
        '''
        if self.load_trajects():
            # print(len(self.generated_trajects), 'samples loaded from file',self.generated_trajects)
            return
        print("no saved samples found, generating new...")
        trajects = []
        for num in range(num_samples):
            print("sample trajectory ", num+1, '/', num_samples)
            w = np.random.rand(self.dim)
            w = w / sum(w)
            traj = self.find_optimum(w)
            # print('w', w, 'phi', traj['f'])
            trajects.append(traj)
        print('Sampling finished', num_samples)
        self.generated_trajects = trajects
        self.save_samples_raw()

    def find_optimum(self, w, radius_sampled_trajects=None):
        """

        :param w:
        :return:
        """
        # if self.generated_trajects is not None:
        #     return self.get_best_sample(w)
        # w = [1,1,.5,.2]
        # w = [0,1,1,0]
        # w = [.0,1,1,.1]
        # self.scalarization_mode = 'chebyshev'
        w_adapted = w #+ [1]
        # np.random.seed(1)
        # ctrl_inputs = compute_best_MPC(self.sim_object, w_adapted,1, self.scalarization_mode)
        t = time.time()
        ctrl_inputs = compute_best(self.sim_object, w_adapted,1, self.scalarization_mode)
        print('COMP TIME', round(time.time()-t,4))
        self.sim_object.feed(list(ctrl_inputs))
        features = self.sim_object.get_features()[0:4]
        states = self.sim_object.get_states()
        print('solution, w=', w, ', f=', features, np.sum(np.multiply(w, features)))
        traj = {'w': w,'f': features, 'states': states}
        return traj


    def get_best_sample(self,w ):
        best_value, best_traject = float('inf'), None
        for traj in self.generated_trajects:
            if self.scalarization_mode == 'linear':
                reward = -np.dot(traj['f'], w)
            else:
                reward = np.max(-np.multiply(traj['f'], w))
            if reward < best_value:
                best_traject = copy.deepcopy(traj)
                best_value = reward
        best_traject['w'] = list(best_traject['w'])
        best_traject['f'] = list(np.multiply(best_traject['f'],1))
        print('retrieved best presampled traject', w, best_traject['f'] )
        return best_traject


    def compute_pair_regret(self, traj_P, traj_Q):
        c_QQ = self.get_cost_of_traj(traj_Q, traj_Q['w'])
        c_PQ = self.get_cost_of_traj(traj_P, traj_Q['w'])
        # return c_PQ-c_QQ, c_PQ/c_QQ
        # return c_PQ/c_QQ
        if c_QQ != 0:
            return c_PQ / c_QQ
        return 1

    def generate_evaluation_samples(self, nun_samples=1000):
        """

        :return:
        """
        weights = []
        for _ in range(nun_samples):
            w = np.random.random(self.dim)
            w = w / sum(w)
            weights.append(w)
        opt_trajs = self.find_optima_for_set_of_weights(weights)
        print("generated eval samples", len(opt_trajs))
        self.sampled_solutions = opt_trajs
        return opt_trajs

    def plot_trajects_and_features(self, trajects, title='', block=True):
        # fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 5))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))
        fig.suptitle('Main title')
        print("plotting", len(trajects), 'sampled solutions')

        # create background
        from matplotlib.patches import Rectangle
        ax.add_patch(Rectangle((-.7, -.5), 4.2, 1, facecolor='darkolivegreen'))
        ax.add_patch(Rectangle((-.7, -.27), 4.2, .54, facecolor = 'darkgrey'))
        ax.plot([-1,4],[.25,.25], color='darkorange', linewidth=1)
        ax.plot([-1,4],[-.25,-.25], color='darkorange', linewidth=1)
        ax.plot([-1,4],[-.09,-.09],'--', color='darkorange', linewidth=1)
        ax.plot([-1,4],[.09,.09],'--', color='darkorange', linewidth=1)

        ax.set_title(title, fontsize=18)

        for i in range(len(trajects)):
            traj = trajects[i]
            # print(traj)
            x_rob = [elem[0] for elem in traj['states']]
            x_human = [elem[1] for elem in traj['states']]
            if i ==0:
                ax.plot([x[1] for x in x_human], [x[0] for x in x_human], color='snow', linewidth=2)
            ax.plot([x[1] for x in x_rob], [x[0] for x in x_rob], color='firebrick',linewidth=1, zorder=1000)

            # ax.plot([x[1] for x in x_rob], [x[0] for x in x_rob], '.', markevery=10, color='firebrick')
            # ax.plot([x[1] for x in x_rob], [x[0] for x in x_human], '.', markevery=10, color='seagreen')
            # if i == 0:
            #     ax.plot(x_rob[0][1], x_rob[0][0], 'D', color='firebrick', linewidth=1)
            #     ax.plot(x_human[0][1],x_human[0][0],'D', color='white', linewidth=1)


        im = image.imread('Driver/imgs/car-orange.png')
        imagebox = OffsetImage(im, zoom=0.05)
        ab = AnnotationBbox(imagebox, (x_rob[0][1], x_rob[0][0]), frameon=False)
        ax.add_artist(ab)
        im = image.imread('Driver/imgs/car-white.png')
        imagebox2 = OffsetImage(im, zoom=0.05)
        ab = AnnotationBbox(imagebox2, (x_human[0][1],x_human[0][0]), frameon=False)
        ax.add_artist(ab)


        ax.set_xlabel('x pos', fontsize=12)
        ax.set_xlabel('', fontsize=12)
        ax.set_ylabel('y pos', fontsize=12)
        ax.set_ylabel('', fontsize=12)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylim([-.35,.35])
        ax.set_xlim([-.7,3.5])

        # ax = axes[1]
        #
        # phi_1, phi_2 = [traj['f'][0] for traj in trajects], [traj['f'][1] for traj in trajects]
        # for idx in range(len(phi_1)):
        #     ax.plot(phi_1[idx], phi_2[idx], 'D', color=cmap(idx / len(trajects)), label='optimal trajectories')
        #
        # asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect('equal')
        # ax.set_xlabel('Trajectory Length', fontsize=16)
        # ax.set_ylabel('Closeness', fontsize=16)
        #
        # fig.tight_layout()
        if block:
            plt.show()


class DriverExtended(Driver):
    def __init__(self, num_feat=10):
        print("init DriverExtended")
        super().__init__(num_feat, 'driverextended')
        print("back to DriverExtended")
