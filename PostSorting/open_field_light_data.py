import open_ephys_IO
import os
import numpy as np
import pandas as pd
from scipy import stats
import PostSorting.parameters

import PostSorting.open_field_make_plots


def load_opto_data(recording_to_process, prm):
    is_found = False
    opto_data = None
    print('loading opto channel...')
    file_path = recording_to_process + '/' + prm.get_opto_channel()
    if os.path.exists(file_path):
        opto_data = open_ephys_IO.get_data_continuous(prm, file_path)
        is_found = True
    else:
        print('Opto data was not found.')
    return opto_data, is_found


def get_ons_and_offs(opto_data):
    # opto_on = np.where(opto_data > np.min(opto_data) + 10 * np.std(opto_data))
    # opto_off = np.where(opto_data <= np.min(opto_data) + 10 * np.std(opto_data))
    mode = stats.mode(opto_data[::30000])[0][0]
    opto_on = np.where(opto_data > 0.2 + mode)
    opto_off = np.where(opto_data <= 0.2 + mode)
    return opto_on, opto_off


def process_opto_data(recording_to_process, prm):
    opto_on = opto_off = None
    opto_data, is_found = load_opto_data(recording_to_process, prm)
    if is_found:
        opto_on, opto_off = get_ons_and_offs(opto_data)
        if not np.asarray(opto_on).size:
            prm.set_opto_tagging_start_index(None)
            is_found = None
        else:
            first_opto_pulse_index = min(opto_on[0])
            prm.set_opto_tagging_start_index(first_opto_pulse_index)

    else:
        prm.set_opto_tagging_start_index(None)

    return opto_on, opto_off, is_found


def make_opto_data_frame(opto_on: tuple) -> pd.DataFrame:
    opto_data_frame = pd.DataFrame()
    opto_end_times = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1))
    opto_start_times_from_second = np.take(opto_on, np.where(np.diff(opto_on)[0] > 1)[0] + 1)
    opto_start_times = np.append(opto_on[0][0], opto_start_times_from_second)
    opto_data_frame['opto_start_times'] = opto_start_times
    opto_end_times = np.append(opto_end_times, opto_on[0][-1])
    opto_data_frame['opto_end_times'] = opto_end_times
    return opto_data_frame


def check_parity_of_window_size(window_size_ms):
    if window_size_ms % 2 != 0:
        print("Window size must be divisible by 2")
        assert window_size_ms % 2 == 0


def get_on_pulse_times(prm):
    path_to_pulses = prm.get_output_path() + '/DataFrames/opto_pulses.pkl'
    pulses = pd.read_pickle(path_to_pulses)
    on_pulses = pulses.opto_start_times
    return on_pulses


def get_firing_times(cell):
    if 'firing_times_opto' in cell:
        firing_times = np.append(cell.firing_times, cell.firing_times_opto)
    else:
        firing_times = cell.fiting_times
    return firing_times


def find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate):
    spikes_in_window_binary = np.zeros(window_size_sampling_rate)
    window_start = int(pulse - window_size_sampling_rate / 2)
    window_end = int(pulse + window_size_sampling_rate / 2)
    spikes_in_window_indices = np.where((firing_times > window_start) & (firing_times < window_end))
    spike_times = np.take(firing_times, spikes_in_window_indices)[0]
    position_of_spikes = spike_times.astype(int) - window_start
    spikes_in_window_binary[position_of_spikes] = 1
    return spikes_in_window_binary


def make_df_to_append_for_pulse(session_id, cluster_id, spikes_in_window_binary, window_size_sampling_rate):
    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    df_row = np.append([session_id, cluster_id], spikes_in_window_binary.astype(int))
    df_to_append = pd.DataFrame([(df_row)], columns=columns)
    return df_to_append


def process_spikes_around_light(spatial_firing, prm, window_size_ms=40):
    check_parity_of_window_size(window_size_ms)
    on_pulses = get_on_pulse_times(prm)
    sampling_rate = prm.get_sampling_rate()
    window_size_sampling_rate = int(sampling_rate/1000 * window_size_ms)

    columns = np.append(['session_id', 'cluster_id'], range(window_size_sampling_rate))
    peristimulus_spikes = pd.DataFrame(columns=columns)

    for index, cell in spatial_firing.iterrows():
        session_id = cell.session_id
        cluster_id = cell.cluster_id
        for pulse in on_pulses:
            firing_times = get_firing_times(cell)
            spikes_in_window_binary = find_spike_positions_in_window(pulse, firing_times, window_size_sampling_rate)
            df_to_append = make_df_to_append_for_pulse(session_id, cluster_id, spikes_in_window_binary, window_size_sampling_rate)
            peristimulus_spikes = peristimulus_spikes.append(df_to_append)
    peristimulus_spikes.to_pickle(prm.get_output_path() + '/DataFrames/peristimulus_spikes.pkl')
    # plt.plot((peristimulus_spikes.iloc[:, 2:].astype(int)).sum().rolling(50).sum())


def main():
    # recording_folder = '/Users/briannavandrey/Documents/recordings'
    recording_folder = 'C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of'
    prm = PostSorting.parameters.Parameters()
    prm.set_output_path(recording_folder + '/MountainSort')
    prm.set_sampling_rate(30000)
    spikes_path = prm.get_output_path() + '/DataFrames/spatial_firing.pkl'
    spikes = pd.read_pickle(spikes_path)
    process_spikes_around_light(spikes, prm)


if __name__ == '__main__':
    main()



