import numpy as np
import scipy.optimize as opt
from Driver.models import LDS, Driver, DriverExtended, Tosser, Fetch

import warnings


def get_feedback(simulation_object, resolution, user, input_A, input_B):
    simulation_object.feed(input_A)
    phi_A = np.array(simulation_object.get_features())
    simulation_object.feed(input_B)
    phi_B = np.array(simulation_object.get_features())
    pq = - phi_A + phi_B
    u = None
    while u == None:
        if not (user is None):
            thres = user['delta']*user['alpha']
            psi = np.clip(np.dot(user['w'], pq) / thres, -1, 1)
            u = np.clip(psi + np.random.randn() * user['noise_std'], -1, 1)
            break
        selection = input('A/B to watch, [-1,1] to vote: ').lower()
        if selection == 'a':
            simulation_object.feed(input_A)
            simulation_object.watch(1)
        elif selection == 'b':
            simulation_object.feed(input_B)
            simulation_object.watch(1)
        else:
            try:
                u = float(selection)
                if not -1 <= u <= 1:
                    u = None
            except ValueError:
                continue
                
    if np.isclose(resolution, 2):
        up = np.sign(u)
    else:
        up = np.round(u / resolution) * resolution
        up = np.round(up, 5)
    return phi_A, phi_B, up
    
def get_feedback_no_sim(resolution, user, phi_A, phi_B):
    pq = - phi_A + phi_B
    u = None
    while u == None:
        if not (user is None):
            thres = user['delta']*user['alpha']
            psi = np.clip(np.dot(user['w'], pq) / thres, -1, 1)
            u = np.clip(psi + np.random.randn() * user['noise_std'], -1, 1)
            break
        print('A: ' + str(phi_A))
        print('B: ' + str(phi_B))
        selection = input('[-1,1] to vote: ').lower()
        try:
            u = float(selection)
            if not -1 <= u <= 1:
                u = None
        except ValueError:
            continue
            
    if np.isclose(resolution, 2):
        up = np.sign(u)
    else:
        up = np.round(u / resolution) * resolution
        up = np.round(up, 5)
    return up


def load_trajectories(task, num_trajectories, tag='Uniform'):
    """
    load pre sampled trajectories from file
    :param task:
    :param num_trajectories:
    :return:
    """
    path = 'ctrl_samples/' + task + '_'+tag +'.npz'
    if tag =='Original':
        path = 'ctrl_samples/' + task + '.npz'
    A = np.load(path)
    print("LOADED FEATURE SETS",tag,  A['feature_set'])
    if A['input_set'].shape[0] < num_trajectories:
        warnings.warn(str(num_trajectories) + ' trajectories were requested, but the dataset contains only '
                      + str(A['input_set'].shape[0]) + ' trajectories. Returning the dataset.')
        return A
    B = {}
    B['input_set'] = A['input_set'][:num_trajectories]
    B['feature_set'] = A['feature_set'][:num_trajectories]
    B['w_set'] = A['w_set'][:num_trajectories]

    return B
    

def create_env(task):
    if task == 'lds':
        return LDS()
    elif task == 'driver':
        return Driver()
    elif task == 'driverextended':
        return DriverExtended()
    elif task == 'tosser':
        return Tosser()
    elif task == 'fetch':
        return Fetch()
    else:
        print('There is no task called ' + task)
        exit(0)


def run_algo(acquisition, simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples, sample_logprobs, PQ, Up):
    if acquisition == 'random':
        return algos.random(trajectories)
    elif acquisition == 'information':
        return algos.infogain(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples)
    elif acquisition == 'regret':
        return algos.maxregret(simulation_object, trajectories, resolution, noise_std, w_samples, alpha_samples, delta_samples, sample_logprobs, PQ, Up)
    else:
        assert False, 'There is no acquisition called ' + acquisition


def func(ctrl_array, *args):
    simulation_object = args[0]
    w = np.array(args[1])
    scalarization_mode = np.array(args[2])
    # print('linear sets control', len(ctrl_array))
    simulation_object.set_ctrl(ctrl_array)
    features = simulation_object.get_features()
    if scalarization_mode =='linear':
        return np.dot(features,w)
    else:
        return np.max(np.multiply(features, w))

def objective(ctrl_array, *args):
    simulation_object = args[0]
    simulation_object.set_ctrl(ctrl_array)
    features = simulation_object.get_features()
    return ctrl_array[-1] + .0000001*np.sum(features)

def const_general(ctrl_array, idx, *args):
    simulation_object = args[0]
    w = np.array(args[1])
    x = ctrl_array[0:-1]
    simulation_object.set_ctrl(x)
    # print('cheby sets control', len(x))
    features = simulation_object.get_features()
    return ctrl_array[-1] - w[idx] * features[idx]

def constr1(ctrl_array, *args):
    return const_general(ctrl_array,0, *args)
def constr2(ctrl_array, *args):
    return const_general(ctrl_array,1, *args)
def constr3(ctrl_array, *args):
    return const_general(ctrl_array,2, *args)
def constr4(ctrl_array, *args):
    return const_general(ctrl_array,3, *args)

def bounds(ctrl_array, *args):
    simulation_object = args[0]
    x = ctrl_array[0:10]

    # lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
    # upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
    if np.min(x) < -1 or np.max(x)>1:
        return -1
    return 1

def best_id_out_of_dataset(trajectories, w):
    return np.argmax(trajectories['feature_set'] @ w, axis=0)

def compute_best(simulation_object, w, iter_count=5, scalarization_mode='linear'):
    u = simulation_object.ctrl_size
    lower_ctrl_bound = [x[0] for x in simulation_object.ctrl_bounds]
    upper_ctrl_bound = [x[1] for x in simulation_object.ctrl_bounds]
    opt_val = np.inf
    opt_ctrl = None
    for _ in range(iter_count):
        x0 = np.random.uniform(low=lower_ctrl_bound, high=upper_ctrl_bound, size=(u))
        if scalarization_mode =='linear':
            ctrl = opt.fmin_cobyla(func, x0=x0, cons=[bounds],
                                       consargs=(simulation_object, w, scalarization_mode), args=(simulation_object, w, scalarization_mode))
            simulation_object.set_ctrl(ctrl)
            features = simulation_object.get_features()
            cost = np.dot(features, w)
        elif scalarization_mode == 'chebyshev':
            x0_aug= list(x0) + [10]
            # print('x0', x0_aug)
            temp_res = opt.fmin_cobyla(objective, x0=x0_aug, cons=[bounds, constr1, constr2, constr3, constr4],
                                       consargs=(simulation_object, w, scalarization_mode), args=(simulation_object, w, scalarization_mode))
            # print('sol', temp_res, len(temp_res), len(temp_res[0:10]))
            ctrl = temp_res[0:10]
            simulation_object.set_ctrl(ctrl)
            features = simulation_object.get_features()
            cost = np.max(np.multiply(features, w))


        if cost < opt_val:
            opt_val = cost
            opt_ctrl = ctrl

    return opt_ctrl




def play(simulation_object, optimal_ctrl):
    simulation_object.set_ctrl(optimal_ctrl)
    keep_playing = 'y'
    while keep_playing == 'y':
        keep_playing = 'u'
        simulation_object.watch(1)
        while keep_playing != 'n' and keep_playing != 'y':
            keep_playing = input('Again? [y/n]: ').lower()
    return optimal_ctrl
