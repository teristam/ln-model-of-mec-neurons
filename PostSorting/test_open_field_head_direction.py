import numpy as np
import open_field_head_direction


def test_get_rolling_sum():

    array_in = [1, 2, 3, 4, 5, 6]
    window = 3

    desired_result = [9, 6, 9, 12, 15, 12]
    result = open_field_head_direction.get_rolling_sum(array_in, window)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    array_in = [3, 4, 5, 8, 11, 1, 3, 5]
    window = 3

    desired_result = [12, 12, 17, 24, 20, 15, 9, 11]
    result = open_field_head_direction.get_rolling_sum(array_in, window)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)

    array_in = [3, 4, 5, 8, 11, 1, 3, 5, 4]
    window = 3

    desired_result = [11, 12, 17, 24, 20, 15, 9, 12, 12]
    result = open_field_head_direction.get_rolling_sum(array_in, window)

    assert np.allclose(result, desired_result, rtol=1e-05, atol=1e-08)