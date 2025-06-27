from config import CFG
from os import listdir
from os.path import isfile, join
from auxillary import *
from evaluation import compute_metrics
from DubinsPlanner.DiskPlanner import *
# from Driver.Driver import Driver
from Lattice_Planner.graph_planner import GraphPlanner, GraphPlannerMinMax

def get_planner_class(planner_type, load_planner=False):
    scalarization = 'linear'
    if load_planner:
        planner = load_planner(planner_type+'_'+scalarization, 'presamples/'+planner_type+'/')
        print('Planner loaded', planner)
        if planner:
            return planner, True
    print('generate new planner')
    if planner_type == 'DiskPlanner2D':
        return DiskPlanner(scalarization), False
    elif planner_type == 'Graph':
        return GraphPlanner(scalarization), False
    elif planner_type == 'GraphMinMax':
        return GraphPlannerMinMax(scalarization), False
    elif planner_type == 'Driver':
        return Driver(scalarization), False

def presample(planner_type, K=20, loaded_from_file=False):
    print("Run Presampling")
    planner_original, loaded_from_file = get_planner_class(planner_type, loaded_from_file)

    planners = {}
    for scalarization in ['linear','chebyshev']:
        if loaded_from_file:
            planner = load_planner(planner_type + '_'+scalarization, 'presamples/' + planner_type + '/')
            print('planner', scalarization, 'loaded')
        else:
            planner = copy.deepcopy(planner_original)
            n = 1
            random.seed(n)
            np.random.seed(n)
            planner.scalarization_mode = scalarization
            planner.sampled_solutions = compute_k_grid_samples(planner, K)
            planner.save_object(tag=scalarization)
        planners[scalarization] = planner
    return planners, planner_original

#

if __name__ == '__main__':

    n = 7 # fix a random seed
    random.seed(n)
    np.random.seed(n)
    for _ in range(1):
        print("Run experiment for planner: ", CFG['planner_type'])
        K = 100 # number of samples
        planners, planner_orig = presample(CFG['planner_type'], K)
        samples = {s:planners[s].sampled_solutions for s in planners.keys()}
        metric = compute_metrics(planner_orig, samples, K, save=False)
        planner_orig.plot_trajects_and_features([], title='Ground Set', block=False)
        for scalarization in planners.keys():
            planner = planners[scalarization]
            samples = filter_duplicates(planner.sampled_solutions)
            planner.plot_trajects_and_features(samples, title=scalarization, block=False)
            # planner.plot_animation(samples, title=scalarization, block=False)
        plt.show()




