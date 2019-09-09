import os
import mdaio
import numpy as np
import PreClustering.dead_channels
import matplotlib.pylab as plt


def extract_random_snippets(filtered_data, firing_times, tetrode, number_of_snippets, prm):
    dead_channels = prm.get_dead_channels()
    if len(dead_channels) != 0:
        for dead_ch in range(len(dead_channels[0])):
            to_insert = np.zeros(len(filtered_data[0]))
            filtered_data = np.insert(filtered_data, int(dead_channels[0][dead_ch]) - 1, to_insert, 0)
    random_indices = np.ceil(np.random.uniform(16, len(firing_times)-16, number_of_snippets)).astype(int)
    snippets = np.zeros((4, 30, number_of_snippets))

    channels = [(tetrode-1)*4, (tetrode-1)*4 + 1, (tetrode-1)*4 + 2, (tetrode-1)*4 + 3]

    for index, event in enumerate(random_indices):
        snippets_indices = np.arange(firing_times[event]-10, firing_times[event]+20, 1).astype(int)
        snippets[:, :, index] = filtered_data[channels[0]:channels[3]+1, snippets_indices]
    # plt.plot(snippets[3,:,:]) # example ch plot
    return snippets


def get_snippets(firing_data, prm):
    print('I will get some random snippets now for each cluster.')
    file_path = prm.get_local_recording_folder_path()
    filtered_data_path = []

    filtered_data_path = file_path + '/Electrophysiology' + prm.get_sorter_name() + '/filt.mda'

    snippets_all_clusters = []
    if os.path.exists(filtered_data_path):
        filtered_data = mdaio.readmda(filtered_data_path)
        for cluster in range(len(firing_data)):
            #TODO: cluster ID problems
            # cluster = firing_data.cluster_id.values[cluster] - 1
            firing_times = firing_data.firing_times[cluster]

            snippets = extract_random_snippets(filtered_data, firing_times, firing_data.tetrode[cluster], 50, prm)
            snippets_all_clusters.append(snippets)
            
        firing_data['random_snippets'] = snippets_all_clusters
    #plt.plot(firing_data.random_snippets[4][3,:,:])
    return firing_data