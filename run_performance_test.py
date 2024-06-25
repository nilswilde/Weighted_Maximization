from Lattice_Planner.graph_planner import GraphPlanner, GraphPlannerMinMax
from main import get_planner_class
import copy, time, os, errno
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm

def save_metrics(metrics_dict):

    identifier = int(time.time() * 100)
    folder = "performanceMetrics/" + '/'
    filename = 'METRICSFILE_planner:' +   '_ID:' + str(identifier) \
               + '.csv'
    try:
        os.makedirs(folder+'/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    metrics_df = pd.DataFrame(metrics_dict)
    metrics_df.to_csv(folder + filename)

def generate_weights(K):
    weights = []
    for k in range(K):
        w = np.random.random(2)
        w = np.divide(w, sum(w))
        weights.append(list(w))
    return weights

def run_performance_test(planner_type):
    planner_original, _ = get_planner_class(planner_type)

    weights = generate_weights(10)
    planning_budgets = [1, 5, 10, 20, 50,100]
    data_rec = []
    for _ in range(1):
        planner = copy.deepcopy(planner_original)
        planner.scalarization_mode = 'chebyshev'
        # planner.randomize_goals()
        for w in weights:
            # comp opt cost
            t = time.time()
            planner.set_planner_param(1)
            opt_time = time.time() - t
            planner.set_planner_param(1000)
            traj_opt = planner.find_optimum(w)

            opt_cost = planner.get_cost_of_traj(traj_opt, w)
            for budget in planning_budgets:
                for mode in ["Standard","Heuristic"]:
                    t = time.time()
                    planner.set_planner_param(budget)

                    traj = planner.find_optimum(w, heuristic=mode == "Heuristic")
                    comp_time = time.time() - t
                    cost = planner.get_cost_of_traj(traj, w)
                    data_rec += [{'Mode':mode,
                                    'Budget': budget,
                                  'Cost Gap': cost-opt_cost,
                                  'Cost Ratio': cost/opt_cost,
                                  'Computation Time': comp_time,
                                  'Comp. Time Ratio': comp_time/opt_time}]
    save_metrics(data_rec)

def visualize():
    path = "performanceMetrics/"
    print(path)
    files = [f for f in listdir(path) if isfile(join(path, f))]
    print(files)
    df = pd.DataFrame()
    for file in files:
        if 'csv' in file:
            print('file', file)
            df = df.append(pd.read_csv(path + "/" + file))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))

    ax = axes[0]
    sns.boxplot(ax=ax, data=df, x='Budget', y='Cost Ratio',hue='Mode', showfliers=False)
    ax = axes[1]
    sns.boxplot(ax=ax, data=df, x='Budget', y='Computation Time', hue='Mode',showfliers=False, showmeans=True)
    y_labels = ['Cost Ratio', 'Comp. Time Ratio']
    for i in range(len(axes)):
        ax = axes[i]
        ax.set_xlabel('Budget', fontsize=18)
        ax.set_ylabel(y_labels[i], fontsize=18)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(14)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(14)
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_performance_test('Graph')
    # visualize()