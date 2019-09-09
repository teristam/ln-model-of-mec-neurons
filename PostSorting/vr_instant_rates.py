

import numpy as np


def add_columns_to_dataframe(spike_data):
    spike_data["instant_rates"] = ""
    return spike_data


def round_down(num, divisor):
    return num-(num%divisor)


def calculate_window():
    sampling_rate=30000
    window_sec = 0.25
    window = sampling_rate/(1/window_sec)
    return window


def calculate_array_out(array, window):
    array_out_size = round_down((array.shape[0]/window), 1)
    return array_out_size


def calculate_instant_speed(speed_ms, window, array_out_size):
    instant_speed=np.zeros((int(array_out_size)))
    for row in range(int(array_out_size)):
        i = row
        start_index=int(i*window)
        end_index=int((i+1)*window)
        index_speed=np.mean(speed_ms[start_index:end_index])
        instant_speed[row] = gaussian_kernel(index_speed)
    return instant_speed


def calculate_instant_location(locations, window, array_out_size):
    instant_location=np.zeros((int(array_out_size)))
    for row in range(int(array_out_size)):
        i = row
        start_index=int(i*window)
        end_index=int((i+1)*window)
        index_location=np.mean(locations[start_index:end_index])
        instant_location[row] = index_location
    return instant_location


def calculate_instant_firingrate(firing_times, window, array_out_size):
    instant_firing_rate=np.zeros((int(array_out_size)))
    for row in range(int(array_out_size)):
        i = row
        start_index=int(i*window)
        end_index=int((i+1)*window)
        firing_events=firing_times[np.where(np.logical_and(firing_times > start_index, firing_times <= (end_index + 1)))]
        instant_firing_rate[row] = gaussian_kernel(firing_events.shape[0])
    return instant_firing_rate


def calculate_instant_rates(spike_data, raw_position_data):
    spike_data=add_columns_to_dataframe(spike_data)
    speed_ms = np.array(raw_position_data['velocity'])
    locations = np.array(raw_position_data['x_position_cm'])
    window = calculate_window()
    array_out_size = calculate_array_out(speed_ms, window)

    for cluster_index in range(len(spike_data)):
        cluster_index = spike_data.cluster_id.values[cluster_index] - 1
        firing_times = np.array(spike_data.at[cluster_index, 'firing_times'])
        instant_speed=calculate_instant_speed(speed_ms, window, array_out_size)
        instant_location=calculate_instant_location(locations, window, array_out_size)
        instant_firing_rate=calculate_instant_firingrate(firing_times, window, array_out_size)
        spike_data = add_data_to_frame(spike_data, cluster_index, instant_speed, instant_firing_rate, instant_location)
    return spike_data


def add_data_to_frame(spike_data, cluster_index, instant_speed, instant_firing_rate, instant_location):
    sn=[]
    sn.append(np.array(instant_speed))
    sn.append(np.array(instant_location))
    sn.append(np.array(instant_firing_rate))
    spike_data.at[cluster_index, 'instant_rates'] = list(sn)
    return spike_data



@jit
def gaussian_kernel(kernx):
    kerny = np.exp(np.power(kernx, 2)/2 * (-1))
    return kerny
