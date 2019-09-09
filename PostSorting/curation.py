import os
import json


def load_curation_metrics(spike_data_frame, prm):
    isolations = []
    noise_overlaps = []
    signal_to_noise_ratios = []
    peak_amplitudes = []
    sorter_name = prm.get_sorter_name()
    path_to_metrics = prm.get_local_recording_folder_path() + '/Electrophysiology' + sorter_name + '/cluster_metrics.json'
    if os.path.exists(path_to_metrics):
        with open(path_to_metrics) as metrics_file:
            cluster_metrics = json.load(metrics_file)
            metrics_file.close()
        for cluster in range(len(spike_data_frame)):
            isolation = cluster_metrics["clusters"][cluster]["metrics"]["isolation"]
            noise_overlap = cluster_metrics["clusters"][cluster]["metrics"]["noise_overlap"]
            peak_snr = cluster_metrics["clusters"][cluster]["metrics"]["peak_snr"]
            peak_amp = cluster_metrics["clusters"][cluster]["metrics"]["peak_amp"]

            isolations.append(isolation)
            noise_overlaps.append(noise_overlap)
            signal_to_noise_ratios.append(peak_snr)
            peak_amplitudes.append(peak_amp)

        spike_data_frame['isolation'] = isolations
        spike_data_frame['noise_overlap'] = noise_overlaps
        spike_data_frame['peak_snr'] = signal_to_noise_ratios
        spike_data_frame['peak_amp'] = peak_amplitudes
    return spike_data_frame


def curate_data(spike_data_frame, prm):
    spike_data_frame = load_curation_metrics(spike_data_frame, prm)
    isolation_threshold = 0.9
    noise_overlap_threshold = 0.05
    peak_snr_threshold = 1
    firing_rate_threshold = 0.5

    isolated_cluster = spike_data_frame['isolation'] > isolation_threshold
    low_noise_cluster = spike_data_frame['noise_overlap'] < noise_overlap_threshold
    high_peak_snr = spike_data_frame['peak_snr'] > peak_snr_threshold
    high_mean_firing_rate = spike_data_frame['mean_firing_rate'] > firing_rate_threshold

    good_cluster = spike_data_frame[isolated_cluster & low_noise_cluster & high_peak_snr & high_mean_firing_rate].copy()
    noisy_cluster = spike_data_frame.loc[~spike_data_frame.index.isin(list(good_cluster.index))]

    return good_cluster, noisy_cluster


