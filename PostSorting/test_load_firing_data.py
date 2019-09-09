import PostSorting.load_firing_data
import numpy as np


def test_correct_detected_ch_for_dead_channels():
    dead_channels = [1, 2]
    primary_channels = [1, 1, 3, 6, 7, 9, 11, 3]

    desired_result = [2, 2, 5, 7, 9, 11, 13, 5]
    result = PostSorting.load_firing_data.correct_detected_ch_for_dead_channels(dead_channels, primary_channels)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)