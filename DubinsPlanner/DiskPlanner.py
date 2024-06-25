import random
import matplotlib.pyplot as plt
from Planner import *
import math as m
import numpy as np
from matplotlib import animation


def get_distance(x_1, x_2):
    return m.sqrt((x_1[0] - x_2[0]) ** 2 + (x_1[1] - x_2[1]) ** 2)


def in_collision(traj, bounds, obstacles=[]):
    # return False
    for pos in traj:
        for o in obstacles:
            if get_distance(pos, o['pos']) < o['r']:
                return True
        if not bounds[0][0] <= pos[0] <= bounds[0][1] or not bounds[1][0] <= pos[1] <= bounds[1][1]:
            return True
    return False


def filter_dominated_samples(samples):
    new_samples = []
    for s in samples:
        dominated = False
        for s_other in samples:
            f = np.array(s['f'])

            f_other = np.array(s_other['f'])
            if np.any(f_other < f) and np.all(f_other <= f):
                dominated = True
                break
        if not dominated:
            new_samples += [s]
    new_samples.sort(key=lambda tup: tup['f'][0])
    return new_samples


def closeness_to_obstacles(traj, bounds, obstacles):
    """

    """
    closeness_measure = 0
    if in_collision(traj, bounds, obstacles):
        return None
    for pos in traj:
        min_dist = float('inf')
        for obst in obstacles:
            dist = get_distance(pos, obst['pos']) - obst['r']
            if dist < .0:
                return None
            min_dist = min(min_dist, dist)
            if min_dist < 0.00:
                return None
        closeness = np.exp(-min_dist)
        closeness_measure = max(closeness_measure, closeness)
    return closeness_measure


def generate_random_goal():
    while True:
        x = random.randint(0, 6) - 1
        y = random.randint(0, 6) - 1
        theta = random.randint(1, 8) * m.pi / 4
        if x == y == 0 or y == theta == 0:
            continue
        return (x, y, theta)


class DiskPlanner(Planner):
    def __init__(self, scalarization):
        super().__init__(2, scalarization, 'Dubins2DAdvanced')
        self.pregenerated_trajectories = None
        self.min_radius = .2
        self.max_radius = 1.0
        self.label = 'DiskPlanner2D'

        # self.start = (.5, .8)
        # self.goal = (3, 2.8, -m.pi / 4)
        self.start = (.8, 1.5)
        self.goal = (3, 2.3, -m.pi / 4)

        self.obstacles = [ # disk shaped obstacles
            {'pos': (1.8, 2.8), 'r': .2, },
            {'pos': (2, 1.6), 'r': .2, },
        ]
        # self.obstacles = [  # disk shaped obstacles
        #     {'pos': (1.8, 2.3), 'r': .25, },
        #     {'pos': (2, 1.1), 'r': .3, },
        # ]
        self.bounds = [(.15, 3.3), (.19, 3.1)]
        self.res = 1000
        self.generate_trajectories()

    def compute_dubins(self, clearance):
        """
        find a Dubin's path between any two fixed trajectories
        :param turning_radius: a given minimal turning radius
        :return: a trajectory, i.e., list of triplets (x,y,theta), and the features for that trajectory
        """

        def comp_tangents_points(point, circle):
            a = circle['r']
            C = circle['pos']
            P = point
            from math import sqrt, acos, atan2, sin, cos
            b = sqrt((P[0] - C[0]) ** 2 + (P[1] - C[1]) ** 2)  # hypot() also works here
            if not -1<=a/b<=1:
                return None
            th = acos(a / b)  # angle theta
            d = atan2(P[1] - C[1], P[0] - C[0])  # direction angle of point P from C
            d1 = d + th  # direction angle of point T1 from C
            d2 = d - th  # direction angle of point T2 from C
            T1 = (C[0] + a * cos(d1), C[1] + a * sin(d1))
            T2 = (C[0] + a * cos(d2), C[1] + a * sin(d2))
            if T1[1] < T2[1]:
                return T1, T2
            return T2, T1

        def comp_arcs(p1, p2, circle):
            c = circle['pos']
            arc = []
            theta1 = m.atan2(p1[1] - c[1], p1[0] - c[0]);
            theta2 = m.atan2(p2[1] - c[1], p2[0] - c[0]);
            angle = theta2 - theta1 if theta2 > theta1 else theta1 - theta2
            arc_length = angle * circle['r']
            for theta in list(np.linspace(theta1, theta2, int(arc_length * res))):
                p = [c[0] + circle['r'] * m.cos(theta), c[1] + circle['r'] * m.sin(theta)]
                arc.append(p)
            return arc

        def gradient_test(traj):
            for i in range(len(traj) - 2):
                grad1 = (traj[i + 1][0] - traj[i][0],traj[i + 1][1] - traj[i][1])
                grad2 = (traj[i + 2][0] - traj[i + 1][0],traj[i + 2][1] - traj[i + 1][1])
                angle = np.dot(grad1,grad2) / (np.linalg.norm(grad1)*np.linalg.norm(grad2))
                if angle < .7:
                    return False
            return True

        def plot_trajects(trajects_plot):
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
            for o in self.obstacles:
                circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
                ax.add_patch(circle1)
            for o in circles:
                circle1 = plt.Circle(o['pos'], o['r'], color='blue', alpha=0.3)
                ax.add_patch(circle1)
            for traj in trajects_plot:
                ax.plot([x[0] for x in traj], [x[1] for x in traj], '--')
            plt.show()

        q0 = self.start
        q2 = self.goal[0:2]
        res = self.res
        circles = copy.deepcopy(self.obstacles)
        for c in circles:
            c['r'] = c['r'] * clearance + .000001
        traj = np.linspace(q0, q2, int(get_distance(q0, q2) * res))
        traj = [list(elem) for elem in traj]
        trajects = [] if in_collision(traj, self.bounds, self.obstacles) else [traj]
        for circle in circles:
            via_points_0 = comp_tangents_points(q0, circle)
            via_points_2 = comp_tangents_points(q2, circle)
            if via_points_0 is None or via_points_2 is None:
                continue
            traj0_a = list(np.linspace(q0, via_points_0[0], int(get_distance(q0, via_points_0[0]) * res)))
            traj0_b = list(np.linspace(q0, via_points_0[1], int(get_distance(q0, via_points_0[1]) * res)))

            trajc_2 = list(np.linspace(via_points_2[0], q2, int(get_distance(q2, via_points_2[0]) * res)))
            trajd_2 = list(np.linspace(via_points_2[1], q2, int(get_distance(q2, via_points_2[1]) * res)))

            # arc_segments
            arc_ac = comp_arcs(via_points_0[0], via_points_2[0], circle)
            arc_ad = comp_arcs(via_points_0[0], via_points_2[1], circle)
            arc_bc = comp_arcs(via_points_0[1], via_points_2[0], circle)
            arc_bd = comp_arcs(via_points_0[1], via_points_2[1], circle)

            traj1 = traj0_a + arc_ac[1::] + trajc_2
            traj2 = traj0_b + arc_bc[1:-1] + trajc_2
            traj3 = traj0_a + arc_ad[1:-1] + trajd_2
            traj4 = traj0_b + arc_bd[1:-1] + trajd_2

            for traj in [traj1, traj2, traj3, traj4]:
                if not in_collision(traj, self.bounds, self.obstacles):
                    if gradient_test(traj):
                        traj_list = [list(elem) for elem in traj]
                        trajects.append(traj_list)

        sols = []
        for traj in trajects:
            sols.append({'f': self.get_features(traj), 'states': traj})
        return sols

    def generate_trajectories(self, force_new=False):
        '''
        Pre-generate a large set of Dubins' paths for different turn radia
        :return:
        '''
        if self.pregenerated_trajectories is not None and not force_new:
            return self.pregenerated_trajectories
        print('generate base set of trajectories')
        clearances = np.linspace(1, 7, 20)

        all_sols = []
        for clearance in clearances:
            sols = self.compute_dubins(clearance)
            for sol in sols:
                if sol not in all_sols:
                    all_sols.append(sol)
        all_sols.reverse()
        self.pregenerated_trajectories = all_sols
        return all_sols

    def find_optimum(self, w, sample_mode=False):
        """

        :param w:
        :return:
        """
        min_cost = float('inf')
        best_traject = None
        radius_sampled_trajects = self.generate_trajectories()
        for traj in radius_sampled_trajects:
            cost = self.get_cost_of_traj(traj, w)
            if cost < min_cost:
                min_cost = cost
                best_traject = traj
        return {'w': w,
                'f': best_traject['f'],
                'states': best_traject['states']
                }

    def get_features(self, traj):
        """
        custom designed features to evaluate a trajectory
        :param path:
        :param radius:
        :return:
        """

        L = 0
        for i in range(len(traj) - 1):
            L += get_distance(traj[i], traj[i + 1])
        obst_distances = closeness_to_obstacles(traj, self.bounds, self.obstacles)
        if obst_distances is None:
            return None
        features = [L, 10 * obst_distances]
        return features

    def compute_minmax_regret(self, samples):
        """

        :param samples:
        :return:
        """
        if len(self.sampled_solutions) == 0:
            self.generate_evaluation_samples()
        max_regret_abs, max_regret_rel = 0, 0
        int_regret_abs, int_regret_rel = 0, 0
        for i in range(len(self.sampled_solutions)):
            abs_regrets_at_w, rel_regrets_at_w = [], []
            for traj in samples:
                r_abs, r_rel = self.compute_pair_regret(traj, self.sampled_solutions[i])
                abs_regrets_at_w.append(r_abs)
                rel_regrets_at_w.append(r_rel)
            if min(abs_regrets_at_w) > max_regret_abs:
                max_regret_abs = min(abs_regrets_at_w)
            if min(rel_regrets_at_w) > max_regret_rel:
                max_regret_rel = min(rel_regrets_at_w)
            int_regret_abs += min(abs_regrets_at_w)
            int_regret_rel += min(rel_regrets_at_w)

        return {
            'max_regret': max_regret_abs,
            'max_relative_regret': max_regret_rel,
            'total_regret': int_regret_abs,
            'total_relative_regret': int_regret_rel
        }

    def generate_evaluation_samples(self):
        """

        :return:
        """
        m = 3
        step_size = 1 / 10 ** m
        weights = [[round(w, m), round(1 - w, m)] for w in np.arange(0, 1 + step_size, step_size)]
        print("generating", len(weights), 'evaluation samples')
        self.sampled_solutions = self.find_optima_for_set_of_weights(weights, sample_mode=True)

    def plot_trajects_and_features(self, samples, title='', block=False):

        def compute_matching_to_pareto_front(samples, pareto_samples):
            matching = {}
            for i in range(len(samples)):
                for j in range(len(pareto_samples)):
                    if samples[i]['f'] == pareto_samples[j]['f']:
                        matching[i] = j
                        break
            return matching

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        all_sols = self.generate_trajectories()  # [0::10]

        pareto_optimal_solutions = filter_dominated_samples(all_sols)
        pareto_optimal_solutions.reverse()
        print("plotting", len(self.sampled_solutions), 'sampled solutions')
        for ax in axes:
            ax.set_yticks([])
            ax.set_xticks([])
            ax.axis('off')

        pal = plt.cm.plasma(np.linspace(.0, .9, len(pareto_optimal_solutions)))
        ax = axes[0]
        # ax.set_title('' + title, fontsize=18)
        for o in self.obstacles:
            circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
            ax.add_patch(circle1)
        if samples is None: # if no samples are given, we plot the ground set of solutions
            for traj in all_sols:
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='lightgrey')
            for i in range(len(pareto_optimal_solutions)):
                traj = pareto_optimal_solutions[i]
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=2, color=pal[i])
        else:
            matching = compute_matching_to_pareto_front(samples, pareto_optimal_solutions)
            for i in range(len(samples)):
                traj = samples[i]
                if i in matching.keys():
                    col = pal[matching[i]]
                else:
                    print('dominated solution', traj['f'])
                    col = 'black'
                ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=2, color=col)
        # plot bounds
        ax.plot([self.bounds[0][0], self.bounds[0][1], self.bounds[0][1], self.bounds[0][0], self.bounds[0][0]],
                [self.bounds[1][0], self.bounds[1][0], self.bounds[1][1], self.bounds[1][1], self.bounds[1][0]], '--',
                color='darkgrey')
        ax.set_aspect('equal')

        # plot Pareto front
        ax = axes[1]
        if samples is None:
            # plot ground set of trajectories
            all_sols_plot = all_sols  # [0::10]
            phi_1, phi_2 = [traj['f'][0] for traj in all_sols_plot], [traj['f'][1] for traj in all_sols_plot]
            ax.plot(phi_1, phi_2, 'D', color='lightgrey', label='all trajects')

            for i in range(len(pareto_optimal_solutions)):
                s = pareto_optimal_solutions[i]
                # phi_1, phi_2 = [traj['f'][0] for traj in pareto_optimal_solutions], [traj['f'][1] for traj in pareto_optimal_solutions]
                ax.plot(s['f'][0], s['f'][1], 'D', color=pal[i], label='all trajects')
        else:
            # plot samples
            phi_1, phi_2 = [traj['f'][0] for traj in samples], [traj['f'][1] for traj in samples]
            for i in range(len(phi_1)):
                if i in matching.keys():
                    col = pal[matching[i]]
                else:
                    col = 'black'
                ax.plot(phi_1[i], phi_2[i], 'D', color=col, label='optimal trajectories')
        ax.set_xlim([3.1, 5.7])
        ax.set_ylim([3.4, 10.5])
        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)
        ax.set_title(title, fontsize=20)
        fig.tight_layout()
        if block:
            plt.show()

    def plot_animation(self, samples, title='', block=False):
        fig, ax = plt.subplots(figsize=(8, 8))
        # plot obstsacles
        for o in self.obstacles:
            circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
            ax.add_patch(circle1)
        # plot bounds
        ax.set_yticks([])
        ax.set_xticks([])
        ax.axis('off')
        print("BOUNDS", self.bounds)
        ax.plot([self.bounds[0][0], self.bounds[0][1], self.bounds[0][1], self.bounds[0][0], self.bounds[0][0]],
                [self.bounds[1][0], self.bounds[1][0], self.bounds[1][1], self.bounds[1][1], self.bounds[1][0]], '--',
                color='darkgrey')
        ax.set_aspect('equal')
        num_lines = len(samples)
        x = [[s[0] for s in samples[i]['states']] for i in range(num_lines)]
        y = [[s[1] for s in samples[i]['states']] for i in range(num_lines)]
        eps=.0
        k = 50
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
        lines = ax.plot(empty_values, empty_values)
        def animate(n):
            # lines = []
            for i, line in enumerate(lines):
                x_i = x[i]
                y_i = y[i]
                n_max = min(n, len(x_i)-1)
                line.set_data(x_i[:n_max], y_i[:n_max])
            return lines
        anim = animation.FuncAnimation(fig, animate, frames=1000, interval=10)
        plt.show()

    def plot_linear_convexification(self, samples, title='', block=False):

        def compute_matching(lin_samples, base_samples):
            print('computed lin cost matching', len(base_samples))
            matching = {}
            for i in range(len(base_samples)):
                s_base = base_samples[i]
                best_cost, best_j = 1000000, s_base
                for j in range(len(lin_samples)):
                    s_lin = lin_samples[j]
                    regret = np.dot(s_lin['w'], s_base['f']) - np.dot(s_lin['w'], s_lin['f'])

                    if regret < best_cost:
                        best_cost = regret
                        best_j = j
                if best_cost < 1000000:
                    matching[i] = best_j
            print('MATCHING', matching)
            return matching

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
        all_sols = self.generate_trajectories()
        samples = samples  # [0::5]
        all_sols = filter_dominated_samples(all_sols)  # [0::5]
        matching = compute_matching(samples, all_sols)
        print("plotting", len(self.sampled_solutions), 'sampled solutions')
        # for ax in axes:
        #     ax.set_yticks([])
        #     ax.set_xticks([])
        # ax.axis('off')

        ax = axes[0]
        ax.set_title('' + title, fontsize=18)
        pal = plt.cm.hsv(np.linspace(0, 1, len(samples)))
        for o in self.obstacles:
            circle1 = plt.Circle(o['pos'], o['r'], color='dimgrey', alpha=0.5)
            ax.add_patch(circle1)
            # ax.add_patch(Rectangle((o['x_0'], o['y_0']), o['x_1']-o['x_0'], o['y_1']-o['y_0']))

        for i in range(len(all_sols)):
            traj = all_sols[i]
            col = 'lightgrey'
            col = pal[matching[i]]
            ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color=col, alpha=.15)
        # for traj in pareto_optimal_solutions[0::4]:
        #     ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], color='darkseagreen')

        for i in range(len(samples)):
            traj = samples[i]
            ax.plot([x[0] for x in traj['states']], [x[1] for x in traj['states']], linewidth=4, color=pal[i])

        # ax.set_xlabel('x pos', fontsize=12)
        # ax.set_ylabel('y pos', fontsize=12)
        # axes[0].set_title('Sampled Trajectories'+title)
        ax.set_aspect('equal')

        # plot Pareto front
        ax = axes[1]

        # plot ground set of trajectories
        phi_1, phi_2 = [traj['f'][0] for traj in all_sols], [traj['f'][1] for traj in all_sols]
        for i in range(len(phi_1)):
            col = pal[matching[i]]
            ax.plot(phi_1[i], phi_2[i], 'D', color='lightgrey', alpha=.8)
            lin_phi = samples[matching[i]]['f']
            ax.plot([lin_phi[0], phi_1[i]], [lin_phi[1], phi_2[i]], '--', color=col, alpha=.3)

        # plot samples
        phi_1, phi_2 = [traj['f'][0] for traj in samples], [traj['f'][1] for traj in samples]
        for idx in range(len(phi_1)):
            ax.plot(phi_1[idx], phi_2[idx], 'D', color=pal[idx % len(pal)], label='optimal trajectories')

        asp = np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0]
        ax.set_aspect(asp)

        ax.set_xlabel('Trajectory Length', fontsize=16)
        ax.set_ylabel('Closeness', fontsize=16)

        fig.tight_layout()
        if block:
            plt.show()
