import numpy as np
import pandas as pd
import PostSorting.parameters
import gc
import PostSorting.vr_stop_analysis
import PostSorting.vr_make_plots
from scipy import stats
from tqdm import tqdm


def calculate_total_trial_numbers(raw_position_data,processed_position_data):
    print('calculating total trial numbers for trial types')
    trial_numbers = np.array(raw_position_data['trial_number'])
    trial_type = np.array(raw_position_data['trial_type'])
    trial_data=np.transpose(np.vstack((trial_numbers, trial_type)))
    beaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]>0),0)
    unique_beaconed_trials = np.unique(beaconed_trials[:,0])
    nonbeaconed_trials = np.delete(trial_data, np.where(trial_data[:,1]!=1),0)
    unique_nonbeaconed_trials = np.unique(nonbeaconed_trials[1:,0])
    probe_trials = np.delete(trial_data, np.where(trial_data[:,1]!=2),0)
    unique_probe_trials = np.unique(probe_trials[1:,0])

    processed_position_data.at[0,'beaconed_total_trial_number'] = len(unique_beaconed_trials)
    processed_position_data.at[0,'nonbeaconed_total_trial_number'] = len(unique_nonbeaconed_trials)
    processed_position_data.at[0,'probe_total_trial_number'] = len(unique_probe_trials)
    return processed_position_data


def find_dwell_time_in_bin(dwell_time_per_sample, speed_ms, locations, loc):
    time_in_bin = dwell_time_per_sample[np.where(np.logical_and(locations > loc, locations <= (loc+1)))]
    speed_in_bin = speed_ms[np.where(np.logical_and(locations > loc, locations <= (loc+1)))]
    time_in_bin_moving = sum(time_in_bin[np.where(speed_in_bin >= 2.5)])
    time_in_bin_stationary = sum(time_in_bin[np.where(speed_in_bin < 2.5)])
    return time_in_bin,time_in_bin_moving, time_in_bin_stationary

def find_dwell_time_in_bin_fast(dwell_time_per_sample, speed_ms, locations, loc):
    # assuming the sampling rate is constant
    idx = np.where((locations > loc) & (locations <= (loc+1)))
    time_in_bin = dwell_time_per_sample[idx]
    speed_in_bin = speed_ms[idx]
    time_in_bin_moving = sum(time_in_bin[np.where(speed_in_bin >= 2.5)])
    time_in_bin_stationary = sum(time_in_bin[np.where(speed_in_bin < 2.5)])
    mean_speed_in_bin = np.nanmean(speed_in_bin)
    return time_in_bin,time_in_bin_moving, time_in_bin_stationary, mean_speed_in_bin,idx


def find_time_in_bin(time_per_sample, locations, loc):
    time_in_bin = time_per_sample[np.where(np.logical_and(locations > loc, locations <= (loc+1)))]
    return time_in_bin

def find_speed_in_bin(speed_ms, locations, loc):
    speed_in_bin = (np.nanmean(speed_ms[np.where(np.logical_and(locations > loc, locations <= (loc+1)))]))
    return speed_in_bin


def find_trial_type_in_bin(trial_types, locations, loc):
    trial_type_in_bin = stats.mode(trial_types[np.where(np.logical_and(locations > loc, locations <= (loc+1)))])[0]
    return trial_type_in_bin


"""
calculates speed for each location bin (0-200cm) across all trials

inputs:
    position_data : pandas dataframe containing position information for mouse during that session
    
outputs:
    position_data : with additional column added for processed data
"""

def bin_data_trial_by_trial(raw_position_data,processed_position_data):
    print('calculate binned data per trial...')
    binned_data = pd.DataFrame(columns=['trial_number_in_bin','bin_count', 'trial_type_in_bin', 'binned_speed_ms_per_trial', 'binned_time_ms_per_trial', 'dwell_time_ms_moving', 'dwell_time_ms_stationary', 'binned_apsolute_elapsed_time'])
    bin_size_cm,number_of_bins, bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    trials = np.array(raw_position_data['trial_number'])
    trial_types = np.array(raw_position_data['trial_type'])
    speed_ms = np.array(raw_position_data['speed_per200ms'])
    locations = np.array(raw_position_data['x_position_cm'])
    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'])
    time_per_sample = np.array(raw_position_data['time_seconds'])

    for t in tqdm(range(1,int(number_of_trials)+1)):
        trial_locations = np.take(locations, np.where(trials == t)[0])
        trial_type = int(stats.mode(np.take(trial_types, np.where(trials == t)[0]))[0])
        for loc in range(int(number_of_bins)):
            time_in_bin,time_in_bin_moving, time_in_bin_stationary,speed_in_bin,locIdx = find_dwell_time_in_bin_fast(dwell_time_per_sample, speed_ms, trial_locations, loc)
            # speed_in_bin = find_speed_in_bin(speed_ms, trial_locations, loc)
            # time_in_bin,time_in_bin_moving, time_in_bin_stationary = find_dwell_time_in_bin(dwell_time_per_sample, speed_ms, trial_locations, loc)
            # apsolute_elapsed_time_in_bin = find_time_in_bin(time_per_sample, trial_locations, loc)
            apsolute_elapsed_time_in_bin=time_per_sample[locIdx]
            binned_data = binned_data.append({"trial_number_in_bin": int(t), "bin_count": int(loc), "trial_type_in_bin": int(trial_type), "binned_speed_ms_per_trial":  np.float16(speed_in_bin), "binned_time_ms_per_trial":  np.float16(sum(time_in_bin)), "binned_time_ms_per_trial_moving":  np.float16(time_in_bin_moving), "dwell_time_ms_stationary":  np.float16(time_in_bin_stationary), "binned_apsolute_elapsed_time" : np.float16(apsolute_elapsed_time_in_bin),}, ignore_index=True)

    processed_position_data = pd.concat([processed_position_data, binned_data['binned_speed_ms_per_trial']], axis=1)
    processed_position_data = pd.concat([processed_position_data, binned_data['binned_time_ms_per_trial_moving']], axis=1)
    processed_position_data = pd.concat([processed_position_data, binned_data['binned_time_ms_per_trial']], axis=1)
    processed_position_data = pd.concat([processed_position_data, binned_data['trial_type_in_bin']], axis=1)
    processed_position_data = pd.concat([processed_position_data, binned_data['trial_number_in_bin']], axis=1)
    processed_position_data = pd.concat([processed_position_data, binned_data['binned_apsolute_elapsed_time']], axis=1)
    return processed_position_data


def bin_data_over_trials(raw_position_data,processed_position_data):
    print('Calculating binned data over trials...')
    binned_data = pd.DataFrame(columns=['dwell_time_ms', 'dwell_time_ms_moving', 'dwell_time_ms_stationary', 'speed_in_bin'])
    bin_size_cm,number_of_bins,bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    locations = np.array(raw_position_data['x_position_cm'])
    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'])
    speed_ms = np.array(raw_position_data['speed_per200ms'])

    for loc in tqdm(range(int(number_of_bins))):
        # time_in_bin,time_in_bin_moving, time_in_bin_stationary = find_dwell_time_in_bin(dwell_time_per_sample, speed_ms, locations, loc)
        # speed_in_bin = find_speed_in_bin(speed_ms, locations, loc)
        time_in_bin,time_in_bin_moving, time_in_bin_stationary,speed_in_bin,locIdx = find_dwell_time_in_bin_fast(dwell_time_per_sample, speed_ms, locations, loc)
        binned_data = binned_data.append({"dwell_time_ms":  np.float16(sum(time_in_bin))/number_of_trials, "dwell_time_ms_moving":  np.float16(time_in_bin_moving)/number_of_trials, "dwell_time_ms_stationary":  np.float16(time_in_bin_stationary)/number_of_trials, "speed_in_bin": np.float16(speed_in_bin)}, ignore_index=True)

    processed_position_data['binned_time_ms'] = binned_data['dwell_time_ms']
    processed_position_data['binned_time_moving_ms'] = binned_data['dwell_time_ms_moving']
    processed_position_data['binned_time_stationary_ms'] = binned_data['dwell_time_ms_stationary']
    processed_position_data['binned_speed_ms'] = binned_data['speed_in_bin']
    return processed_position_data

def bin_data_over_trials_fast(raw_position_data,processed_position_data):
    print('Calculating binned data over trials...')
    binned_data = pd.DataFrame(columns=['dwell_time_ms', 'dwell_time_ms_moving', 'dwell_time_ms_stationary', 'speed_in_bin'])
    bin_size_cm,number_of_bins,bins = PostSorting.vr_stop_analysis.get_bin_size(raw_position_data)
    number_of_trials = raw_position_data.trial_number.max() # total number of trials
    locations = np.array(raw_position_data['x_position_cm'])
    dwell_time_per_sample = np.array(raw_position_data['dwell_time_ms'])
    speed_ms = np.array(raw_position_data['speed_per200ms'])

    for loc in tqdm(range(int(number_of_bins))):
        # time_in_bin,time_in_bin_moving, time_in_bin_stationary = find_dwell_time_in_bin(dwell_time_per_sample, speed_ms, locations, loc)
        # speed_in_bin = find_speed_in_bin(speed_ms, locations, loc)
        time_in_bin,time_in_bin_moving, time_in_bin_stationary,speed_in_bin,locIdx = find_dwell_time_in_bin_fast(dwell_time_per_sample, speed_ms, locations, loc)
        binned_data = binned_data.append({"dwell_time_ms":  np.float16(sum(time_in_bin))/number_of_trials, "dwell_time_ms_moving":  np.float16(time_in_bin_moving)/number_of_trials, "dwell_time_ms_stationary":  np.float16(time_in_bin_stationary)/number_of_trials, "speed_in_bin": np.float16(speed_in_bin)}, ignore_index=True)

    processed_position_data['binned_time_ms'] = binned_data['dwell_time_ms']
    processed_position_data['binned_time_moving_ms'] = binned_data['dwell_time_ms_moving']
    processed_position_data['binned_time_stationary_ms'] = binned_data['dwell_time_ms_stationary']
    processed_position_data['binned_speed_ms'] = binned_data['speed_in_bin']
    return processed_position_data

def drop_columns_from_dataframe(raw_position_data):
    raw_position_data.drop(['dwell_time_seconds'], axis='columns', inplace=True, errors='ignore')
    #raw_position_data.drop(['velocity'], axis='columns', inplace=True, errors='ignore')
    return raw_position_data


def process_position_data(raw_position_data, prm):
    # processed_position_data = pd.DataFrame() # make dataframe for processed position data
    # processed_position_data = bin_data_over_trials(raw_position_data,processed_position_data)
    # processed_position_data = bin_data_trial_by_trial(raw_position_data,processed_position_data)
    # processed_position_data = calculate_total_trial_numbers(raw_position_data, processed_position_data)
    # processed_position_data = PostSorting.vr_stop_analysis.generate_stop_lists(raw_position_data, processed_position_data)
    gc.collect()
    processed_position_data = pd.read_pickle(prm.get_output_path() + '/DataFrames/processed_position_data.pkl')
    prm.set_total_length_sampling_points(raw_position_data.time_seconds.values[-1])  # seconds
    processed_position_data["new_trial_indices"] = raw_position_data["new_trial_indices"]
    raw_position_data = drop_columns_from_dataframe(raw_position_data)


    print('-------------------------------------------------------------')
    print('position data processed')
    print('-------------------------------------------------------------')
    return raw_position_data, processed_position_data


#  for testing
def main():
    print('-------------------------------------------------------------')

    params = PostSorting.parameters.Parameters()

    recording_folder = 'C:/Users/s1466507/Documents/Ephys/test_overall_analysis/M5_2018-03-06_15-34-44_of'

    vr_spatial_data = process_position_data(recording_folder)


if __name__ == '__main__':
    main()
