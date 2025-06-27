import numpy as np
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import math as m
import copy, time, random

from scipy.optimize import linprog
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle

# import gurobipy as gp
# from gurobipy import GRB


def get_distance(vertex, other):
    return m.sqrt((vertex[0] - other[0]) ** 2 + (vertex[1] - other[1]) ** 2)


class Graph:

    def __init__(self, img_file='Lattice_Planner/map.png'):
        """

        :param img_file: a black and white image. white is free space black are static obstacles
        """
        self.img_file = img_file
        self.map_img = mpimg.imread(img_file)
        self.vertices = []  # list with robot configurations / vertices
        self.vertex_obst_dists = {}
        self.edges = {}  # dictionary with (v,u) keys, values are the distance
        self.edge_list = []
        self.neighbours = {}
        self.costs = {}  # dictionary with (v,u) keys, values are the distance
        self.edge_features = {}
        self.dim = 2  # number of features
        self.w = [1, 1]
        self.weighted_edge_features = {}

        self.incidence_matrix = None

        self.x_range = len(self.map_img[0])
        self.y_range = len(self.map_img)
        self.generate_prm(3000, 4)

        self.planning_budget = 10

    def generate_prm(self, number_vertices=10, k=4, symmetric=True):
        """

        :param number_vertices: number of vertices to be sampled
        :param k: connection factor
        :param symmetric: should the PRM graph be symmetric? Default: True
        :return:
        """
        np.random.seed(10)
        vertices_reached = []

        min_dist = .03 * self.x_range
        while True:
            print("create PRM graph, n = ", number_vertices, ', k = ', k)
            self.vertices = []

            for _ in range(number_vertices * 10):
                if len(self.vertices) >= number_vertices:
                    break
                v = (int(np.random.random() * self.x_range), int(np.random.random() * self.y_range))
                if self.vertex_in_free_space(v):
                    _, dist = self.get_closest_vertex(v)
                    if dist > min_dist:
                        self.vertices.append(v)
                        dist_to_obst = self.get_dist_to_closest_obst(v)
                        self.vertex_obst_dists[v] = dist_to_obst
            for v in self.vertices:
                self.neighbours[v] = []

            for v in self.vertices:
                dists_to_neighbours = []
                for u in self.vertices:
                    if v != u:
                        dists_to_neighbours.append({'u': u,
                                                    'dist': get_distance(u, v)})
                dists_to_neighbours = sorted(dists_to_neighbours, key=lambda i: i['dist'])
                num_connections = 0
                for i in range(k):
                    u = dists_to_neighbours[i]['u']
                    d = dists_to_neighbours[i]['dist']
                    collision_free = True
                    for step in np.linspace(0, 1, 11):
                        intermediate_point = np.add(np.multiply(step, u), np.multiply((1 - step), v))
                        if not self.vertex_in_free_space(intermediate_point):
                            collision_free = False
                            break
                    if collision_free:
                        if u not in self.neighbours[v]:
                            self.neighbours[v] += [u]

                        if (v, u) not in self.edges.keys():
                            vertices_reached += [u]
                            num_connections += 1
                            self.edges[(v, u)] = d
                            self.costs[(v, u)] = d
                            # compute features
                            self.edge_features[(v, u)] = self.compute_edge_features((v, u))
                            if symmetric and (u, v) not in self.edges.keys():
                                if v not in self.neighbours[u]:
                                    self.neighbours[u] += [v]

                                self.edges[(u, v)] = d
                                self.costs[(u, v)] = d
                                self.edge_features[(u, v)] = self.compute_edge_features((u, v))
                if num_connections == 0 and v not in vertices_reached:  # if a vertex cannot be connected to any other vertex, remove it from the graph
                    self.vertices.remove(v)

            if max(self.edges.values()) < float('inf'):  # check if we were able to construct a connected graph
                break
        print("PRM finished, n=", len(self.vertices), 'm=', len(list(self.edges.values())))
    def set_edge_costs(self, w):
        self.w = copy.deepcopy(w)
        eps = .0000
        for e in self.edges.keys():
            self.costs[e] = np.dot(w, self.edge_features[e]) + eps
            self.weighted_edge_features[e] = list(np.multiply(w, self.edge_features[e]) + eps)

    def compute_edge_features(self, e):
        obst_dist = (self.vertex_obst_dists[e[0]] + self.vertex_obst_dists[e[1]]) / 2
        closeness = np.exp(-.05 * obst_dist)
        # closeness = np.exp(-.005 * obst_dist)
        return [self.edges[e] / 100, closeness * 1000]


    def compute_path_features(self, path):
        path_features = [0, 0]
        weighted_path_features = [0, 0]
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            weighted_path_features[0] += self.weighted_edge_features[e][0]
            weighted_path_features[1] += self.weighted_edge_features[e][1]
            path_features[0] += self.edge_features[e][0]
            path_features[1] += self.edge_features[e][1]
        return weighted_path_features, path_features


    def setup_flow_constraints(self, s, t):
        A_eq = copy.deepcopy(self.incidence_matrix)
        s_idx, t_idx = self.vertices.index(s), self.vertices.index(t)
        b_eq = [0] * len(self.vertices)
        b_eq[s_idx] = 1
        b_eq[t_idx] = -1
        return A_eq, b_eq

    def check_connectivity(self):
        for v in self.vertices:
            for u in self.vertices:
                if (v, u) not in self.path_pointers.keys():
                    return False
        return True

    def vertex_in_free_space(self, vertex):
        """
        collision check for png black and white images representing obstacles
        :param vertex:
        :return:
        """
        pixel = self.map_img[int(round(vertex[1]))][int(round(vertex[0]))]
        return pixel[0] != 0

    def get_closest_vertex(self, pos):
        min_dist = float('inf')
        u = None
        for v in self.vertices:
            d = get_distance(v, pos)
            if d < min_dist:
                min_dist = d
                u = v
        return u, min_dist

    def get_dist_to_closest_obst(self, pos):
        t = time.time()
        res = 20
        min_dist = float('inf')
        x_points = np.linspace(0, self.x_range - 1, num=int(self.x_range / res))
        y_points = np.linspace(0, self.y_range - 1, num=int(self.x_range / res))
        for x in x_points:
            for y in y_points:
                if not self.vertex_in_free_space((x, y)):
                    dist = get_distance(pos, (x, y))
                    min_dist = min(min_dist, dist)
        return min_dist

    def plot(self, fig=None, ax=None, paths=[], title='', show_graph=True, block=True):
        """
        simple plot of the environment only
        :param block:
        :return:
        """
        print("plot graph", paths)
        if fig is None or ax is None:
            print('gen subplots')
            fig, ax = plt.subplots()
        # fig.suptitle(title)
        x, y = [], []
        ax.imshow(self.map_img)

        if show_graph:
            # plot graph edges
            for e in self.edges:
                c = 'lightgrey'
                x_start, y_start = e[0][0], e[0][1]
                x_goal, y_goal = e[1][0], e[1][1]
                ax.plot([x_start, x_goal], [y_start, y_goal], color=c, zorder=1)

            # plot vertices
            for i in range(len(self.vertices)):
                v = self.vertices[i]
                x.append(v[0])
                y.append(v[1])
                ax.scatter(v[0], v[1], color='lightgrey', s=50, zorder=2)

        cols = ['b', 'g', 'r', 'purple']
        for path_idx in range(len(paths)):
            path = paths[path_idx]
            if path is not None:
                for idx in range(len(path)):
                    v = path[idx]
                    x.append(v[0] + random.random()), y.append(v[1] + random.random())
                    ax.scatter(v[0], v[1], color=cols[path_idx], s=60, zorder=3)
                    if idx < len(path) - 1:
                        u = path[idx + 1]
                        ax.plot([v[0], u[0]], [v[1], u[1]], color=cols[path_idx], zorder=1, linewidth=5)

        plt.show(block=block)
        return fig, ax

    def get_path_positions(self, path, speed=1):

        pos_log = []
        u = path[0]
        for idx in range(len(path) - 1):
            v, u = path[idx], path[idx + 1]
            pos_log += [v]
            steps = int(self.edges[(v, u)] / speed) - 1
            for i in range(steps):
                w = (i / steps * u[0] + (1 - i / steps) * v[0], i / steps * u[1] + (1 - i / steps) * v[1])
                pos_log.append(w)
        pos_log.append(u)
        return pos_log

    def animate_path(self, path):
        """

        :param path:
        :return:
        """
        dt = 0.01
        log = self.get_path_positions(path)

        # Initialize the plot
        fig, ax = self.plot(paths=[path], show_graph=False, block=False)

        # Create and add a circle patch to the axis
        patch = Circle(log[0], radius=15)
        ax.add_patch(patch)

        # Animation function to update and return the patch at each frame
        def animate(i):
            patch.center = log[i]
            return patch,

        # Specify the animation parameters and call animate
        ani = FuncAnimation(fig,
                            animate,
                            frames=len(log),  # Total number of frames in the animation
                            interval=int(1000 * dt),  # Set the length of each frame (milliseconds)
                            blit=True,  # Only update patches that have changed (more efficient)
                            repeat=False)  # Only play the animation once

        plt.show()

    def compute_shortest_path(self, s, g, scalarization='linear', heuristic=False, ):

        """

        """
        import heapq
        def path_dominated(path_to_v, other_paths_to_v, v):
            _, features_j = self.compute_path_features(list(path_to_v) + [v])
            for other_path in other_paths_to_v:
                _, features_i = self.compute_path_features(list(other_path) + [v])
                if np.all(np.array(features_i) <= np.array(features_j)) \
                        and np.any(np.array(features_i) < np.array(features_j)):
                    return True
            return False

        print('Run shortest path search', scalarization, s, g,'use heuristic?', heuristic, self.planning_budget)
        t_s = time.time()

        paths_to_v = {}
        for v in self.vertices:
            paths_to_v[v] = []

        paths_to_v[s] = [[]]
        open_set = [(0, s, [])]
        heapq.heapify(open_set)
        max_path_records = self.planning_budget
        iter=0
        while len(open_set) > 0:
            iter += 1
            f, curr, path_to_prev = heapq.heappop(open_set)

            if curr == g:  # terminate when the goal is found
                print('shortest path in ', (round(time.time() - t_s, 4)), self.compute_path_features(list(list(path_to_prev) + [curr])))
                return list(path_to_prev) + [curr]

            path_to_curr = tuple(list(path_to_prev) + [curr])
            neighbours = self.neighbours[curr]
            for neigh in neighbours:
                if neigh in path_to_curr:  # avoid cycles
                    continue
                if path_to_curr in paths_to_v[neigh]:  # avoid duplicates
                    continue
                if len(paths_to_v[neigh]) >= max_path_records:  # budget cuttoff for paths leading to a vertex
                    continue

                if path_dominated(path_to_curr, paths_to_v[neigh], neigh):
                    continue
                if len(path_to_curr) != len(set(path_to_curr)):
                    raise
                paths_to_v[neigh] += [copy.deepcopy(path_to_curr)]
                weighted_cost_vec, _ = self.compute_path_features(list(path_to_curr) + [neigh])
                if heuristic:
                    weighted_cost_vec[0] += self.w[0] * get_distance(neigh, g) / 100
                if scalarization == 'linear':
                    tent_score = sum(weighted_cost_vec)
                else:
                    tent_score = max(weighted_cost_vec) + .00001 *sum(weighted_cost_vec)
                heapq.heappush(open_set, (tent_score, neigh, path_to_curr))


class GraphMinMax(Graph):
    def compute_path_features(self, path):
        path_features = [0, 0]
        weighted_path_features = [0, 0]
        for i in range(len(path) - 1):
            e = (path[i], path[i + 1])
            weighted_path_features[0] += self.weighted_edge_features[e][0]
            weighted_path_features[1] = max(weighted_path_features[1], self.weighted_edge_features[e][1])
            path_features[0] += self.edge_features[e][0]
            path_features[1] = max(path_features[1], self.edge_features[e][1])
        return weighted_path_features, path_features