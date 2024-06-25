import copy
import time
import csv
import os, errno
import numpy as np
import math as m
import pandas as pd

def compute_metrics(planner, samples, K, save=True):
    """

    :param planner:
    :param samples:
    :return:
    """

    metrics = []
    for label in samples.keys():
        stats = {'sampler': label, 'planner': planner.label, 'K':K}
        norm_samples = normalize_features(planner, samples[label])
        stats = compute_dispersion(stats, norm_samples)
        print(stats)
        stats['samples'] = samples[label]
        metrics.append(stats)

    df = pd.DataFrame(metrics)
    if save:
        save_metrics(planner, df, K)
    return metrics


def compute_dispersion(stats, samples):
    dispersion = 0
    summed_diff = 0
    unique_features = []
    for i in range(len(samples)-1):
        f_i = [round(samples[i]['f'][idx],4)for idx in range(len(samples[i]['f']))]
        if f_i not in unique_features:
            unique_features.append(f_i)
        smallest_radius = float('inf')
        for j in range(i+1,len(samples)):
            f_j = samples[j]['f']
            dist = get_distance(f_i, f_j)
            smallest_radius = min(smallest_radius, dist/2)
        summed_diff += smallest_radius
        dispersion = max(dispersion, smallest_radius)
    stats['dispersion'] = round(dispersion,5)
    stats['mean_distance'] = round(summed_diff/len(samples),5)
    stats['num_unique_samples'] = len(unique_features)
    return stats

def compute_hypervolume(stats, planner, samples):
    def parteto_dominated(point, samples):
        for sample in samples:
            if np.all(np.array(sample) <= np.array(point)):
                return 0
        return 1

    norm_samples = normalize_features(planner, samples)
    if planner.dim == 2:
        from scipy.interpolate import InterpolatedUnivariateSpline

        sorted_samples = sorted(norm_samples, key=lambda d: d['f'][0])
        f_1_vals, f_2_vals = [], []
        for sample in sorted_samples:
            if sample['f'][0] in f_1_vals: # skip duplicates
                continue
            f_1_vals.append(sample['f'][0])
            f_2_vals.append(sample['f'][1])

        f = InterpolatedUnivariateSpline(f_1_vals, f_2_vals, k=1)
        volume = f.integral(min(f_1_vals), max(f_1_vals))
        print('volume', volume)
        stats['hypervolume'] = volume
        return stats
    else: # use approximation for higher order
        num_vol_samples, volume = 500000,0
        sample_fs = [s['f'] for s in norm_samples]
        for _ in range(num_vol_samples):
            volume += parteto_dominated(np.random.random(planner.dim), sample_fs)
        print('approx volume', volume/num_vol_samples)
        stats['hypervolume'] = volume/num_vol_samples
        return stats



def get_distance(x, y, mode='L2'):
    if mode == 'L2':
        return np.linalg.norm(np.subtract(x, y))
    if mode == 'L1':
        return np.linalg.norm(np.subtract(x, y), ord=1)


def normalize_features(planner, samples):
    return samples
    normalized_samples = copy.deepcopy(samples)
    value_bounds = planner.get_value_bounds()
    for idx in range(len(normalized_samples)):
        f = copy.deepcopy(normalized_samples[idx]['f'])
        for i in range(planner.dim):
            f[i] = (f[i]-value_bounds[i]['lb'])/(value_bounds[i]['ub']-value_bounds[i]['lb'])
        normalized_samples[idx]['f'] = f
    return normalized_samples

def save_samples(planner, samples, K):
    """

    :param error_log:
    :param number_iterations:
    :param solver: The solver used for the problem
    :return:
    """
    identifier = int(time.time() * 100)
    folder = "simulation_data/"+str(planner)+'/'

    print("save metrics")
    for label in samples.keys():
        print('label', label)
        data = samples[label]
        filename = 'SAMPLEFILE_planner:' + planner.label +'_sampler:' + label +'_K:' + str(K) +'_ID:' + str(identifier) \
                   + '.csv'
        try:
            os.makedirs(folder)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        keys = data[0].keys()

        with open(folder + filename, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

def save_metrics(planner, metrics_df, K):

    identifier = int(time.time() * 100)
    folder = "simulation_data/" + str(planner) + '/'
    filename = 'METRICSFILE_planner:' + planner.label + '_K:' + str(K) + '_ID:' + str(identifier) \
               + '.csv'
    try:
        os.makedirs(folder+'/')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


    metrics_df.to_csv(folder + filename)