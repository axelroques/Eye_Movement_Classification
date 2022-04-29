
""""
Basic common functions
"""


def compute_velocity(t, x, y):

    dt, dx, dy = t.diff(), x.diff(), y.diff()
    v_x, v_y = dx/dt, dy/dt

    return v_x.dropna().reset_index(drop=True), \
        v_y.dropna().reset_index(drop=True)


def dispersion(x_array, y_array):
    return max(x_array)-min(x_array) + max(y_array)-min(y_array)


def merge_function(fixations, saccades,
                   temporal_threshold=75,
                   spatial_threshold=0.5):
    """
    Fixations and saccades points are merged into fixation and saccade segments.
    """

    return
