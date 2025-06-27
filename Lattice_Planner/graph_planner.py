import time

from Planner import Planner
from Lattice_Planner.graph import Graph, GraphMinMax, get_distance
import numpy as np
import random, copy, pickle
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
np.set_printoptions(legacy='1.25')

class GraphPlanner(Planner):
    def __init__(self, scalarization):
        super().__init__(2, scalarization, 'Graph')

        self.g = Graph()
        # start and goal for simple map
        # self.s = self.g.get_closest_vertex((0, 1100))[0]
        # self.t = self.g.get_closest_vertex((1100, 400))[0]

        self.s = self.g.get_closest_vertex((250, 10))[0]
        self.t = self.g.get_closest_vertex((750, 700))[0]
        #
        self.s = self.g.get_closest_vertex((20, 600))[0]
        self.t = self.g.get_closest_vertex((1140, 600))[0]

    def set_planner_param(self, budget):
        self.g.planning_budget = budget
    def randomize_goals(self):
        min_dist = .6*self.g.x_range
        while True:
            s = random.choice(self.g.vertices)
            t = random.choice(self.g.vertices)
            if get_distance(s,t)>min_dist:
                self.s = s
                self.t = t
                break

    # def save_object(self, tag):
    #     print('save planner', self.label, self.scalarization_mode)
    #     file_name = 'graph_samples/'+self.label + '_' + tag + '_' + str(round(time.time(),8))+ '.pickle'
    #     print(file_name)
    #     save_object = copy.deepcopy(self)
    #     save_object.sim_object = None
    #     with open(file_name,
    #               'wb') as outp:  # Overwrites any existing file.
    #         pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)


    def find_optimum(self, w, heuristic=True):
        """

        :param w:
        :return:
        """

        print('find optimum via graph search', w, self.scalarization_mode, ', k=', self.g.planning_budget)
        # w = [w_i+.001 for w_i in w]
        self.g.set_edge_costs(w)

        path = self.g.compute_shortest_path(self.s, self.t, scalarization=self.scalarization_mode,
                                            heuristic=heuristic)
        path_pos = self.g.get_path_positions(path)
        _, f = self.g.compute_path_features(path)
        print('new path, w=', w, ', f= ', f, np.dot(w, f))
        sol = {'w': w,
                'f': f,
                'states': path_pos
                }
        # self.plot_trajects_and_features([sol])
        return sol

    def generate_evaluation_samples(self):
        """

        :return:
        """
        m = 1
        step_size = 1 / 10 ** m
        step_size = .5
        weights = [[round(w, m), round(1 - w, m)] for w in np.arange(0, 1 + step_size, step_size)]
        print("generating", len(weights), 'evaluation samples')
        self.sampled_solutions = self.find_optima_for_set_of_weights(weights)
        return self.sampled_solutions

    def plot_trajects_and_features(self, trajects, title='', block=True):
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 5))

        print("plotting", 'sampled solutions')

        ax = axes[0]
        # ax.imshow(self.g.map_img)
        fig, ax = self.g.plot(fig, ax, block=False)
        if trajects is not None:
            pal = plt.cm.viridis(np.linspace(0, 1, len(trajects)))
            ax.set_title(self.label+' - ' + title, fontsize=18)
            ax.set_title(title, fontsize=18)
            # plot start and goal
            # ax.plot(self.s[0], self.s[1], 'D', 'green', zorder=10000)
            # ax.plot(self.t[0], self.t[1], 'X', 'purple', zorder=10000)

            for i in range(len(trajects)):
                traj = trajects[i]
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color=pal[i],linewidth=5)

        # ax.set_xlabel('x pos', fontsize=12)
        # ax.set_ylabel('y pos', fontsize=12)
        ax.set_yticklabels([])
        ax.set_xticklabels([])

        ax = axes[1]

        # phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
        # ax.plot(phi_1, phi_2, 'x', color='grey', label='all trajects')

        phi_1, phi_2 = [traj['f'][0] for traj in trajects], [traj['f'][1] for traj in trajects]
        for idx in range(len(phi_1)):
            ax.plot(phi_1[idx], phi_2[idx], 'D', color=pal[idx], label='optimal trajectories')

        # if highlight is not None:
        #     ax.plot(highlight['f'][0], highlight['f'][1], 'D', color='green', label='optimal trajectories',
        #             markersize=6)
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)
        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)

        fig.tight_layout()
        if block:
            plt.show()


    def plot_animation(self, samples, title='', block=False):
        fig, ax = plt.subplots(figsize=(10, 10))
        # plot map and graph
        fig, ax = self.g.plot(fig, ax, block=False)
        ax.set_xlim([0, 1200])
        ax.set_ylim([0, 800])
        ax.set_title(self.label + ' - ' + title, fontsize=18)
        ax.set_title(title, fontsize=18)

        # ax.set_aspect('equal')
        num_lines = len(samples)
        x = [[s[0] for s in samples[i]['states']] for i in range(num_lines)]
        y = [[s[1] for s in samples[i]['states']] for i in range(num_lines)]
        k = 10
        eps = 10
        for i in range(num_lines):
            x_offset = (np.random.random()-.5)*eps
            y_offset = (np.random.random()-.5)*eps
            x[i] = x[i][::k]+[x[i][-1]]
            y[i] = y[i][::k]+[y[i][-1]]
            for j in range(1, len(x[i])-1):
                x[i][j] += x_offset
                y[i][j] += y_offset

        print('animating', num_lines, 'lines, ', [len(x_i) for x_i in x])
        empty_values = np.empty((1, num_lines))
        empty_values[:] = np.nan
        lines = ax.plot(empty_values, empty_values, linewidth=4)
        def animate(n):
            # for i, line in enumerate(lines):
            for i in range(num_lines):
                x_i = x[i]
                y_i = y[i]
                n_max = min(n, len(x_i)-1)
                lines[i].set_data(x_i[:n_max], y_i[:n_max])
            return lines
        anim = animation.FuncAnimation(fig, animate, frames=1000, interval=1, blit=True)
        fig.tight_layout()
        plt.show()

class GraphPlannerMinMax(GraphPlanner):
    def __init__(self, scalarization):
        super().__init__(scalarization)

        self.g = GraphMinMax()
        self.s = self.g.get_closest_vertex((250, 10))[0]
        self.t = self.g.get_closest_vertex((750, 700))[0]
        # self.s = self.g.get_closest_vertex((0, 625))[0]
        # self.t = self.g.get_closest_vertex((1140, 600))[0]
        self.label = 'GraphMinMax'