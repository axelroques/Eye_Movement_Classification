
from .sigmoid_fit import initial_guess, sigmoid, fit
from .RANSAC import remove_outliers
from ihmm import IHMM

import numpy as np
import warnings


class Saccade():
    """
    Simple class for saccades.
    Each saccade is characterized by:
        - A start and end timestamp
        - A duration
        - A peak velocity 
    """

    def __init__(self, t, x, y):

        self.t = t
        self.x = x
        self.y = y

    def compute_parameters(self, ransac, y_bar):
        """
        Update/compute saccade parameters using the results from the 
        fit.
        """

        # Local Sigmoid function and its derivative
        def S(t, params): return params[0]+(params[1]-params[0]) * \
            np.power(t, params[3]) / \
            (np.power(params[2], params[3])+np.power(t, params[3]))

        def S_p(t, params): return (params[1]-params[0]) * \
            params[3] * np.power(params[2], params[3]) * \
            np.power(t, params[3]-1) / \
            (np.power(params[2], params[3])+np.power(t, params[3]))**2

        # Basic parameters
        self.t = ransac.t
        self.x = ransac.x
        self.y = ransac.y

        # Amplitude
        min_amplitude = np.min(S(self.t, y_bar))
        max_amplitude = np.max(S(self.t, y_bar))
        self.amplitude = max_amplitude - min_amplitude

        # Start and end timestamps
        self.t_start = self.t[np.nonzero(y_bar > 0.02*max_amplitude)[0][0]]
        self.t_end = self.t[np.nonzero(y_bar > 0.98*max_amplitude)[0][0]]

        # Duration
        self.duration = self.t_end - self.t_start

        # Peak velocity
        self.peak_velocity = np.max(S_p(self.t, y_bar))

        return


def get_saccades(sacc, t, x, y, n_min):
    """
    Get the start and end timestamps estimates of each saccade. 
    """

    # Get all saccades positions
    indices = []
    for i, (val1, val2) in enumerate(zip(sacc[0:-1], sacc[1:])):

        # Initialization
        if (i == 0) and (val1 == 1):
            indices.append(i)

        # Tests to append
        if (val2 - val1) == 0:
            continue
        else:
            indices.append(i+1)

    # Termination
    indices += [len(sacc)]

    return [Saccade(t[i_start:i_end], x[i_start:i_end], y[i_start:i_end])
            for i_start, i_end in zip(indices[::2], indices[1::2])
            if (i_end-i_start) >= n_min]


def ISF(t, x, y, threshold, n_iter=10, outlier_proba=0.2, n_min=5):
    """
    Identification by Sigmoid Fitting (ISF) algorithm.
    Pretty cool name right?

    Inputs:
        - t = numpy array
        - x, y = numpy arrays containing the eye trace
        - threshold = velocity threshold for the IVT algorithm
        - n_iter = number of iterations to estimate the probabilistic parameters
        in the IHMM algorithm
        - outlier_proba = probability for a sample to be an outlier
        - n_min = minimal number of points in a saccade event to process
        the saccade
    """

    # Consider warnings as errors
    warnings.filterwarnings('error')

    # Rough estimate of saccade position
    # Shoudld we use IVT like in the article or IHMM?
    _, sacc = IHMM(t, x, y, threshold, n_iter)

    # Get a list of saccades using the previous rough estimates of
    # saccade position
    saccades = get_saccades(sacc, t, x, y, n_min)

    # Process each saccade
    for saccade in saccades:

        print('hi', len(saccade.t))

        # RANSAC method to remove outliers
        ransac = remove_outliers(saccade.t, saccade.x,
                                 saccade.y, outlier_proba)

        # Curve fitting
        try:
            p0 = initial_guess(t, ransac.x)
            x_bar = fit(ransac.t, ransac.x, p0, fun=sigmoid)
            p0 = initial_guess(t, ransac.y)
            y_bar = fit(ransac.t, ransac.y, p0, fun=sigmoid)

        # If model does not converge
        except (RuntimeError, RuntimeWarning):
            print('No parameters for you')
            continue

        # Update saccade parameters
        saccade.compute_parameters(ransac, y_bar)

    return saccades
