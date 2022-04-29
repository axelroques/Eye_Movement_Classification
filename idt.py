
from common import dispersion
import numpy as np


def IDT(t, x, y, threshold, window_size):
    """
    First described in Salvucci  and  Goldberg  (2000). The algorithm defines
    a temporal window which moves one point at a time. The spacial dispersion
    created by the points within this window is compared against the threshold.
    If  such  dispersion falls below the threshold, the points within the
    temporal window are classified as a part of fixation. Otherwise, the window
    is moved by one sample, and the first sample of the previous window is 
    classified as a saccade.
    """

    # Initialize arrays
    fixations = np.zeros(len(x))
    saccades = np.zeros(len(x))

    # Iterate over samples
    i = 0
    while (i+window_size) < len(x):

        current_window = window_size
        D = dispersion(x[i:i+current_window],
                       y[i:i+current_window])

        if D < threshold:
            while (D < threshold) and (i+current_window < len(x)):
                D = dispersion(x[i:i+current_window],
                               y[i:i+current_window])
                current_window += 1

            fixations[i:i+current_window] = 1
            i += current_window

        else:
            saccades[i] = 1

        i += 1

    return fixations, saccades
