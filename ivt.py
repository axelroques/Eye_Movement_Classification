
from common import compute_velocity
import numpy as np


def IVT(t, x, y, threshold):
    """
    First described in Salvucci  and  Goldberg  (2000). 
    Each sampled is classified as either a fixation or a saccade using a 
    threshold value on the signal's velocity (in °/s).
    """

    v_x, v_y = compute_velocity(t, x, y)
    v = np.sqrt(v_x**2 + v_y**2)

    fixations = np.where(v < threshold, 1, 0)
    saccades = np.where(v >= threshold, 1, 0)

    return fixations, saccades
