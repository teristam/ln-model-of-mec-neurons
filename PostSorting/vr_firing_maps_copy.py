from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from numba import jit
import numpy as np
import math
import time


def get_dwell(spatial_data, prm):
    min_dwell_distance_cm = 5  # from point to determine min dwell time

    dt_position_ms = spatial_data.time_seconds.diff().mean()*1000  # average sampling interval in position data (ms)
    min_dwell_time_ms = 3 * dt_position_ms  # this is about 100 ms
    min_dwell = round(min_dwell_time_ms/dt_position_ms)
    return min_dwell, min_dwell_distance_cm


def get_bin_size(prm):
    bin_size_cm = 2
    return bin_size_cm


def get_number_of_bins(spatial_data, prm):
    bin_size_cm = get_bin_size(prm)
    length_of_arena_x = spatial_data.x_position_cm[~np.isnan(spatial_data.x_position_cm)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_cm)
    length_of_arena_y = spatial_data.trial_number[~np.isnan(spatial_data.trial_number)].max()
    number_of_bins_y = length_of_arena_y
    return number_of_bins_x, number_of_bins_y


"""

def get_number_of_bins(spatial_data, prm):
    bin_size_cm = get_bin_size(prm)
    length_of_arena_x = spatial_data.x_position_cm[~np.isnan(spatial_data.x_position_cm)].max()
    number_of_bins_x = math.ceil(length_of_arena_x / bin_size_cm)
    length_of_arena_y = spatial_data.trial_number[~np.isnan(spatial_data.trial_number)].max()
    number_of_bins_y = length_of_arena_y
    return number_of_bins_x, number_of_bins_y
"""

@jit
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny


# not trial by trial calculation
def calculate_firing_rate_for_cluster_parallel(cluster, smooth, firing_data_spatial, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('Started another cluster')
    cluster_index = firing_data_spatial.cluster_id.values[cluster] - 1
    cluster_firings = pd.DataFrame({'x_position_cm': firing_data_spatial.x_position_cm[cluster_index], 'y_position_cm': firing_data_spatial.trial_number[cluster_index]})
    spike_positions_x = cluster_firings.x_position_cm.values
    firing_rate_map = np.zeros((number_of_bins_x))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                firing_rate_map[x] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))
            else:
                firing_rate_map[x] = 0

    return firing_rate_map

'''
def calculate_firing_rate_for_cluster_parallel_trial_by_trial(cluster, smooth, firing_data_spatial, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('Started another cluster')
    print(cluster)
    cluster_index = firing_data_spatial.cluster_id.values[cluster] - 1
    cluster_firings = pd.DataFrame({'x_position_cm': firing_data_spatial.beaconed_position_cm[cluster_index], 'y_position_cm': firing_data_spatial.beaconed_trial_number[cluster_index]})
    spike_positions_x = cluster_firings.x_position_cm.values
    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                firing_rate_map[x,y] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))
            else:
                firing_rate_map[x,y] = 0

    return firing_rate_map
'''

def calculate_firing_rate_for_cluster_parallel_trial_by_trial(cluster, smooth, firing_data_spatial, positions_x, positions_y, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms):
    print('Started another cluster')
    print(cluster)
    cluster_index = firing_data_spatial.cluster_id.values[cluster] - 1
    cluster_firings = pd.DataFrame({'x_position_cm': firing_data_spatial.x_position_cm[cluster_index], 'y_position_cm': firing_data_spatial.trial_number[cluster_index], 'trial_type': firing_data_spatial.trial_type[cluster_index]})
    spike_positions_x = cluster_firings.x_position_cm.values
    spike_trial_types = cluster_firings.trial_type.values
    spike_trials = cluster_firings.y_position_cm.values
    firing_rate_map = np.zeros((number_of_bins_x, number_of_bins_y, 3))
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            trial_type = np.unique(np.take(spike_trial_types, np.where(spike_trials == y)))
            px = x * bin_size_pixels + (bin_size_pixels / 2)
            py = y * bin_size_pixels + (bin_size_pixels / 2)
            spike_distances = np.sqrt(np.power(px - spike_positions_x, 2))
            spike_distances = spike_distances[~np.isnan(spike_distances)]
            occupancy_distances = np.sqrt(np.power((px - positions_x), 2))
            occupancy_distances = occupancy_distances[~np.isnan(occupancy_distances)]
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_pixels)[0])

            if bin_occupancy >= min_dwell:
                firing_rate_map[x,y,trial_type] = sum(gaussian_kernel(spike_distances/smooth)) / (sum(gaussian_kernel(occupancy_distances/smooth)) * (dt_position_ms/1000))
            else:
                firing_rate_map[x,y,trial_type] = 0

    return firing_rate_map


def get_spike_heatmap_parallel(spatial_data, firing_data_spatial, prm):
    print('I will calculate firing rate maps now.')
    dt_position_ms = spatial_data.time_seconds.diff().mean()*1000
    min_dwell, min_dwell_distance_pixels = get_dwell(spatial_data, prm)
    smooth = 2
    bin_size_pixels = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)
    clusters = range(len(firing_data_spatial))
    num_cores = multiprocessing.cpu_count()
    time_start = time.time()
    firing_rate_maps = Parallel(n_jobs=num_cores)(delayed(calculate_firing_rate_for_cluster_parallel_trial_by_trial)(cluster, smooth, firing_data_spatial, spatial_data.x_position_cm.values,  spatial_data.trial_number.values, number_of_bins_x, number_of_bins_y, bin_size_pixels, min_dwell, min_dwell_distance_pixels, dt_position_ms) for cluster in clusters)
    time_end = time.time()
    print('Making the rate maps took:')
    time_diff = time_end - time_start
    print(time_diff)
    firing_data_spatial['firing_maps'] = firing_rate_maps

    return firing_data_spatial



def get_position_heatmap(spatial_data, prm):
    min_dwell, min_dwell_distance_cm = get_dwell(spatial_data, prm)
    bin_size_cm = get_bin_size(prm)
    number_of_bins_x, number_of_bins_y = get_number_of_bins(spatial_data, prm)
    position_heat_map = np.zeros((number_of_bins_x, number_of_bins_y))

    # find value for each bin for heatmap
    for x in range(number_of_bins_x):
        for y in range(number_of_bins_y):
            px = x * bin_size_cm + (bin_size_cm / 2)
            occupancy_distances = np.sqrt(np.power((px - spatial_data.x_position_cm.values), 2))
            bin_occupancy = len(np.where(occupancy_distances < min_dwell_distance_cm)[0])

            if bin_occupancy >= min_dwell:
                position_heat_map[x, y] = bin_occupancy
            else:
                position_heat_map[x, y] = None
    return position_heat_map


# this is the firing rate in the bin with the highest rate
def find_maximum_firing_rate(spatial_firing):
    max_firing_rates = []
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1
        firing_rate_map = spatial_firing.firing_maps[cluster]
        max_firing_rate = np.max(firing_rate_map.flatten())
        max_firing_rates.append(max_firing_rate)
    spatial_firing['max_firing_rate'] = max_firing_rates
    return spatial_firing


def make_firing_field_maps(spatial_data, firing_data_spatial, prm):
    #position_heat_map = get_position_heatmap(spatial_data, prm)
    firing_data_spatial = get_spike_heatmap_parallel(spatial_data, firing_data_spatial, prm)
    #firing_data_spatial = find_maximum_firing_rate(firing_data_spatial)
    return firing_data_spatial
