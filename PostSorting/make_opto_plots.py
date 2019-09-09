import array_utility
import os
import matplotlib.pylab as plt
import math
import numpy as np
import pandas as pd
import plot_utility
import PostSorting.parameters
import scipy.ndimage


# do not use this on data from more than one session
def plot_peristimulus_raster(peristimulus_spikes, prm):
    assert len(peristimulus_spikes.groupby('session_id')['session_id'].nunique()) == 1
    save_path = prm.get_output_path() + '/Figures/opto_stimulation'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    cluster_ids = np.unique(peristimulus_spikes.cluster_id)
    for cluster in cluster_ids:
        cluster_rows_boolean = peristimulus_spikes.cluster_id.astype(int) == int(cluster)
        cluster_rows_annotated = peristimulus_spikes[cluster_rows_boolean]
        cluster_rows = cluster_rows_annotated.iloc[:, 2:]
        print(cluster_rows.head())
        plt.cla()
        peristimulus_figure = plt.figure()
        peristimulus_figure.set_size_inches(5, 5, forward=True)
        ax = peristimulus_figure.add_subplot(1, 1, 1)  # specify (nrows, ncols, axnum)
        sample_times = np.argwhere(cluster_rows.to_numpy().astype(int) == 1)[:, 1]
        trial_numbers = np.argwhere(cluster_rows.to_numpy().astype(int) == 1)[:, 0]
        stimulation_start = cluster_rows.shape[1] / 2 - 45  # todo remove magic number
        stimulation_end = cluster_rows.shape[1] / 2 + 45
        ax.axvspan(stimulation_start, stimulation_end, 0, cluster_rows.shape[0], alpha=0.5, color='lightblue')
        ax.vlines(x=sample_times, ymin=trial_numbers, ymax=(trial_numbers + 1), color='black', zorder=2, linewidth=3)
        plt.xlabel('Time (sampling points)')
        plt.ylabel('Trial (sampling points)')
        plt.ylim(0, cluster_rows.shape[0])
        plt.savefig(save_path + '/' + cluster + '_peristimulus_raster.png', dpi=300)
        plt.close()

        # plt.plot((cluster_rows.astype(int)).sum().rolling(100).sum())


def main():
    prm = PostSorting.parameters.Parameters()
    path = 'C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of/MountainSort/DataFrames/peristimulus_spikes.pkl'
    peristimulus_spikes = pd.read_pickle(path)
    prm.set_output_path('C:/Users/s1466507/Documents/Ephys/recordings/M0_2017-12-14_15-00-13_of/MountainSort/')
    plot_peristimulus_raster(peristimulus_spikes, prm)


if __name__ == '__main__':
    main()
