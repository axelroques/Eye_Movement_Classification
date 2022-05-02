
""""
Basic common functions
"""

import numpy as np


def compute_velocity(t, x, y):

    dt, dx, dy = np.diff(t), np.diff(x), np.diff(y)
    v_x, v_y = dx/dt, dy/dt

    return v_x, v_y


def dispersion(x_array, y_array):
    return max(x_array)-min(x_array) + max(y_array)-min(y_array)


def merge_function(fixations, saccades,
                   temporal_threshold=75,
                   spatial_threshold=0.5):
    """
    Fixations and saccades points are merged into fixation and saccade segments.
    """

    return
