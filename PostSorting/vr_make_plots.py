import os
import matplotlib.pylab as plt
import plot_utility
import numpy as np
import PostSorting.vr_stop_analysis
import PostSorting.vr_extract_data
import matplotlib.image as mpimg
import pandas as pd
from scipy import stats

'''

# Plot basic info to check recording is good:
> movement channel
> trial channels (one and two)

'''

# plot the raw movement channel to check all is good
def plot_movement_channel(location, prm):
    plt.plot(location)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/movement' + '.png')
    plt.close()

# plot the trials to check all is good
def plot_trials(trials, prm):
    plt.plot(trials)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trials' + '.png')
    plt.close()

# plot the raw trial channels to check all is good
def plot_trial_channels(trial1, trial2, prm):
    plt.plot(trial1[0,:])
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trial_type1' + '.png')
    plt.close()
    plt.plot(trial2[0,:])
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/trial_type2' + '.png')
    plt.close()


'''

# Plot behavioural info:
> stops on trials 
> avg stop histogram
> avg speed histogram
> combined plot

'''

def split_stop_data_by_trial_type(spatial_data):
    locations,trials,trial_type = PostSorting.vr_stop_analysis.load_stop_data(spatial_data)
    stop_data=np.transpose(np.vstack((locations, trials, trial_type)))
    beaconed = np.delete(stop_data, np.where(stop_data[:,2]>0),0)
    nonbeaconed = np.delete(stop_data, np.where(stop_data[:,2]!=1),0)
    probe = np.delete(stop_data, np.where(stop_data[:,2]!=2),0)
    return beaconed, nonbeaconed, probe


def plot_stops_on_track(raw_position_data, processed_position_data, prm):
    print('I am plotting stop rasta...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stops_on_track = plt.figure(figsize=(6,6))
    ax = stops_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(processed_position_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    ax.plot(processed_position_data.rewarded_stop_locations, processed_position_data.rewarded_trials, '>', color='Red', markersize=3)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    #plt.xlim(min(spatial_data.position_bins),max(spatial_data.position_bins))
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(raw_position_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_raster' + '.png', dpi=200)
    plt.close()


def plot_stop_histogram(raw_position_data, processed_position_data, prm):
    print('plotting stop histogram...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    stop_histogram = plt.figure(figsize=(6,4))
    ax = stop_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"])
    average_stops = np.array(processed_position_data["average_stops"])
    ax.plot(position_bins,average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.average_stops)+0.1
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/stop_histogram' + '.png', dpi=200)
    plt.close()


def plot_speed_histogram(raw_position_data, processed_position_data, prm):
    print('plotting speed histogram...')
    save_path = prm.get_output_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    speed_histogram = plt.figure(figsize=(6,4))
    ax = speed_histogram.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"].dropna(axis=0))
    average_speed = np.array(processed_position_data["binned_speed_ms"].dropna(axis=0))
    ax.plot(position_bins,average_speed, '-', color='Black')
    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.binned_speed_ms)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .35, wspace = .35,  bottom = 0.2, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_output_path() + '/Figures/behaviour/speed_histogram' + '.png', dpi=200)
    plt.close()


def plot_combined_behaviour(raw_position_data,processed_position_data, prm):
    print('making combined behaviour plot...')
    save_path = prm.get_local_recording_folder_path() + '/Figures/behaviour'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    combined = plt.figure(figsize=(6,9))
    ax = combined.add_subplot(3, 1, 1)  # specify (nrows, ncols, axnum)

    beaconed,nonbeaconed,probe = split_stop_data_by_trial_type(processed_position_data)

    ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='0.5', markersize=2)
    ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='red', markersize=2)
    ax.plot(probe[:,0], probe[:,1], 'o', color='blue', markersize=2)
    plt.ylabel('Stops on trials', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(raw_position_data.trial_number)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = combined.add_subplot(3, 1, 2)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"].dropna(axis=0))
    average_stops = np.array(processed_position_data["average_stops"].dropna(axis=0))
    ax.plot(position_bins,average_stops, '-', color='Black')
    plt.ylabel('Stops (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.average_stops)+0.5
    plot_utility.style_vr_plot(ax, x_max)

    ax = combined.add_subplot(3, 1, 3)  # specify (nrows, ncols, axnum)
    position_bins = np.array(processed_position_data["position_bins"].dropna(axis=0))
    average_speed = np.array(processed_position_data["binned_speed_ms"].dropna(axis=0))
    ax.plot(position_bins,average_speed, '-', color='Black')
    plt.ylabel('Speed (cm/s)', fontsize=12, labelpad = 10)
    plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
    plt.xlim(0,200)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    plot_utility.style_track_plot(ax, 200)
    x_max = max(processed_position_data.binned_speed_ms)+0.5
    plot_utility.style_vr_plot(ax, x_max)
    plt.subplots_adjust(hspace = .3, wspace = .3,  bottom = 0.06, left = 0.12, right = 0.87, top = 0.92)
    plt.savefig(prm.get_local_recording_folder_path() + '/Figures/behaviour/combined_behaviour' + '.png', dpi=200)
    plt.close()



'''

# Plot spatial firing info:
> spikes per trial
> firing rate

'''

def plot_spikes_on_track(spike_data,raw_position_data,processed_position_data, prm, prefix):
    print('plotting spike rastas...')
    save_path = prm.get_output_path() + '/Figures/spike_trajectories'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    rewarded_locations = np.array(processed_position_data['rewarded_stop_locations'].dropna(axis=0)) #
    rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        x_max = max(np.array(spike_data.at[cluster_index, 'beaconed_trial_number'])) + 1
        spikes_on_track = plt.figure(figsize=(4,(x_max/32)))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)

        #uncomment if you want to plot stops
        #ax.plot(beaconed[:,0], beaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        #ax.plot(nonbeaconed[:,0], nonbeaconed[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)
        #ax.plot(probe[:,0], probe[:,1], 'o', color='LimeGreen', markersize=2, alpha=0.5)

        ax.plot(spike_data.loc[cluster_index].beaconed_position_cm, spike_data.loc[cluster_index].beaconed_trial_number, '|', color='Black', markersize=4)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=4)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=4)
        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=3)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

        plot_utility.style_track_plot(ax, 200)
        plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_track_firing_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


def plot_firing_rate_maps(spike_data, prm, prefix):
    print('I am plotting firing rate maps...')
    save_path = prm.get_output_path() + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure(figsize=(6,4))

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = PostSorting.vr_extract_data.extract_smoothed_average_firing_rate_data(spike_data, cluster_index)

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        nb_x_max = np.nanmax(avg_beaconed_spike_rate)
        b_x_max = np.nanmax(avg_nonbeaconed_spike_rate)
        p_x_max = np.nanmax(avg_probe_spike_rate)
        if b_x_max > nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        elif b_x_max > nb_x_max and b_x_max < p_x_max:
            plot_utility.style_vr_plot(ax, p_x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.12, right=0.87, top=0.92)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


def plot_gc_firing_rate_maps(spike_data, prm, prefix):
    print('I am plotting firing rate maps...')
    save_path = prm.get_output_path() + '/Figures/spike_rate'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        avg_spikes_on_track = plt.figure(figsize=(6,4))

        avg_beaconed_spike_rate, avg_nonbeaconed_spike_rate, avg_probe_spike_rate = PostSorting.vr_extract_data.extract_gc_firing_rate_data(spike_data, cluster_index)

        ax = avg_spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        ax.plot(avg_beaconed_spike_rate, '-', color='Black')
        ax.plot(avg_nonbeaconed_spike_rate, '-', color='Red')
        ax.plot(avg_probe_spike_rate, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        nb_x_max = np.nanmax(avg_beaconed_spike_rate)
        b_x_max = np.nanmax(avg_nonbeaconed_spike_rate)
        p_x_max = np.nanmax(avg_probe_spike_rate)
        if b_x_max > nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        elif b_x_max > nb_x_max and b_x_max < p_x_max:
            plot_utility.style_vr_plot(ax, p_x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace=.35, wspace=.35, bottom=0.15, left=0.12, right=0.87, top=0.92)

        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_map_Cluster_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()



'''
plot gaussian convolved firing rate in time against similarly convolved speed and location. 
'''

def plot_convolved_rates_in_time(spike_data, prm):
    print('plotting spike rastas...')
    save_path = prm.get_output_path() + '/Figures/ConvolvedRates_InTime'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(4,5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        firing_rate = spike_data.loc[cluster_index].spike_rate_in_time
        speed = spike_data.loc[cluster_index].speed_rate_in_time
        x_max= np.max(firing_rate)
        ax.plot(firing_rate, speed, '|', color='Black', markersize=4)
        plt.ylabel('Firing rate (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Speed (cm/sec)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, 200)
        plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_versus_SPEED_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()

        spikes_on_track = plt.figure(figsize=(4,5))
        ax = spikes_on_track.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        position = spike_data.loc[cluster_index].location_rate_in_time
        ax.plot(firing_rate, position, '|', color='Black', markersize=4)
        plt.ylabel('Firing rate (Hz)', fontsize=12, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, 200)
        plot_utility.style_vr_plot(ax, x_max)
        plt.locator_params(axis = 'y', nbins  = 4)
        try:
            plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        except ValueError:
            continue
        plt.savefig(save_path + '/' + spike_data.session_id[cluster_index] + '_rate_versus_POSITION_' + str(cluster_index +1) + '.png', dpi=200)
        plt.close()


# unused code but might use in future

'''
def plot_combined_spike_raster_and_rate(spike_data,raw_position_data,processed_position_data, prm, prefix):
    print('plotting combined spike rastas and spike rate...')
    save_path = prm.get_output_path() + '/Figures/combined_spike_plots'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    rewarded_locations = np.array(processed_position_data['rewarded_stop_locations'].dropna(axis=0))
    rewarded_trials = np.array(processed_position_data['rewarded_trials'].dropna(axis=0))

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        spikes_on_track = plt.figure(figsize=(6,10))

        ax = spikes_on_track.add_subplot(2, 1, 1)  # specify (nrows, ncols, axnum)
        cluster_firing_indices = spike_data.firing_times[cluster_index]
        ax.plot(spike_data.loc[cluster_index].beaconed_position_cm, spike_data.loc[cluster_index].beaconed_trial_number, '|', color='Black', markersize=4)
        ax.plot(spike_data.loc[cluster_index].nonbeaconed_position_cm, spike_data.loc[cluster_index].nonbeaconed_trial_number, '|', color='Red', markersize=4)
        ax.plot(spike_data.loc[cluster_index].probe_position_cm, spike_data.loc[cluster_index].probe_trial_number, '|', color='Blue', markersize=4)
        ax.plot(rewarded_locations, rewarded_trials, '>', color='Red', markersize=2)

        plt.ylabel('Spikes on trials', fontsize=12, labelpad = 10)
        plt.xlim(0,200)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        plot_utility.style_track_plot(ax, 200)
        x_max = max(raw_position_data.trial_number[cluster_firing_indices])+0.5
        plot_utility.style_vr_plot(ax, x_max)

        ax = spikes_on_track.add_subplot(2, 1, 2)  # specify (nrows, ncols, axnum)
        unsmooth_b = np.array(spike_data.at[cluster_index, 'avg_b_spike_rate'])
        unsmooth_nb = np.array(spike_data.at[cluster_index, 'avg_nb_spike_rate'])
        unsmooth_p = np.array(spike_data.at[cluster_index, 'avg_p_spike_rate'])
        ax.plot(unsmooth_b, '-', color='Black')
        ax.plot(unsmooth_nb, '-', color='Red')
        ax.plot(unsmooth_p, '-', color='Blue')
        ax.locator_params(axis = 'x', nbins=3)
        ax.set_xticklabels(['0', '100', '200'])
        plt.ylabel('Spike rate (hz)', fontsize=14, labelpad = 10)
        plt.xlabel('Location (cm)', fontsize=14, labelpad = 10)
        plt.xlim(0,200)
        nb_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_b_spike_rate']))
        b_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_nb_spike_rate']))
        p_x_max = np.nanmax(np.array(spike_data.at[cluster_index, 'avg_p_spike_rate']))
        if b_x_max > nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, b_x_max)
        elif b_x_max < nb_x_max and b_x_max > p_x_max:
            plot_utility.style_vr_plot(ax, nb_x_max)
        elif b_x_max > nb_x_max and b_x_max < p_x_max:
            plot_utility.style_vr_plot(ax, p_x_max)
        plot_utility.style_track_plot(ax, 200)
        plt.subplots_adjust(hspace = .2, wspace = .2,  bottom = 0.06, left = 0.12, right = 0.87, top = 0.92)
        plt.savefig(prm.get_output_path() + '/Figures/combined_spike_plots/' + spike_data.session_id[cluster_index] + '_track_firing_Cluster_' + str(cluster_index +1) + str(prefix) + '.png', dpi=200)
        plt.close()


def make_combined_figure(prm, spatial_firing, prefix):
    print('I will make the combined images now.')
    save_path = prm.get_output_path() + '/Figures/combined'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    plt.close('all')
    figures_path = prm.get_output_path() + '/Figures/'
    for cluster in range(len(spatial_firing)):
        cluster = spatial_firing.cluster_id.values[cluster] - 1

        spike_raster_path = figures_path + 'combined_spike_plots/' + spatial_firing.session_id[cluster] + '_track_firing_Cluster_' + str(cluster +1) + str(prefix) + '.png'
        spike_histogram_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_spike_histogram.png'
        autocorrelogram_10_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_10ms.png'
        autocorrelogram_250_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_autocorrelogram_250ms.png'
        waveforms_path = figures_path + 'firing_properties/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + '_waveforms.png'
        combined_behaviour_path = figures_path + 'behaviour/combined_behaviour.png'
        grid = plt.GridSpec(6, 3, wspace=0.003, hspace=0.01)
        if os.path.exists(waveforms_path):
            waveforms = mpimg.imread(waveforms_path)
            waveforms_plot = plt.subplot(grid[0, 0])
            waveforms_plot.axis('off')
            waveforms_plot.imshow(waveforms)
        if os.path.exists(spike_histogram_path):
            spike_hist = mpimg.imread(spike_histogram_path)
            spike_hist_plot = plt.subplot(grid[1, 0])
            spike_hist_plot.axis('off')
            spike_hist_plot.imshow(spike_hist)
        if os.path.exists(autocorrelogram_10_path):
            autocorrelogram_10 = mpimg.imread(autocorrelogram_10_path)
            autocorrelogram_10_plot = plt.subplot(grid[2, 0])
            autocorrelogram_10_plot.axis('off')
            autocorrelogram_10_plot.imshow(autocorrelogram_10)
        if os.path.exists(autocorrelogram_250_path):
            autocorrelogram_250 = mpimg.imread(autocorrelogram_250_path)
            autocorrelogram_250_plot = plt.subplot(grid[3, 0])
            autocorrelogram_250_plot.axis('off')
            autocorrelogram_250_plot.imshow(autocorrelogram_250)
        if os.path.exists(spike_raster_path):
            spike_raster = mpimg.imread(spike_raster_path)
            spike_raster_plot = plt.subplot(grid[:, 1])
            spike_raster_plot.axis('off')
            spike_raster_plot.imshow(spike_raster)
        if os.path.exists(combined_behaviour_path):
            combined_behaviour = mpimg.imread(combined_behaviour_path)
            combined_behaviour_plot = plt.subplot(grid[:, 2])
            combined_behaviour_plot.axis('off')
            combined_behaviour_plot.imshow(combined_behaviour)
        plt.subplots_adjust(hspace = .0, wspace = .0,  bottom = 0.06, left = 0.06, right = 0.94, top = 0.94)
        plt.savefig(save_path + '/' + spatial_firing.session_id[cluster] + '_' + str(cluster + 1) + str(prefix) + '.png', dpi=1000)
        plt.close()

'''