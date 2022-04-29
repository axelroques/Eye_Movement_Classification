
from .sigmoid_fit import initial_guess, sigmoid
from .RANSAC import remove_outliers
from ..ihmm import IHMM
from .fit import fit

import numpy as np


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

    def compute_parameters(self, x_bar, y_bar):
        """
        Update/compute saccade parameters using the results from the 
        fit.
        """

        # TO DO
        return


def get_saccades(sacc, t, x, y):
    """
    Get the start and end timestamps estimates of each saccade. 
    """

    indices = np.nonzero(sacc)[0]

    return [Saccade(t[i_start:i_end], x[i_start:i_end], y[i_start:i_end])
            for i_start, i_end in zip(indices[:-1], indices[1:])]


def ISF(t, x, y, threshold, outlier_proba):
    """
    Identification by Sigmoid Fitting (ISF) algorithm.
    Pretty cool name right?

    Inputs:
        - t = numpy array
        - x = numpy array
        - y = numpy array
    """

    # Rough estimate of saccade position
    # Shoudld we use IVT like in the article or IHMM?
    fix, sacc = IHMM(t, x, y, threshold, n_iter=10)

    # Get saccades list using the previous rough estimate of
    # saccade position
    saccades = get_saccades(sacc, t, x, y)

    # Process each saccade
    for saccade in saccades:

        # RANSAC method to remove outliers
        ransac = remove_outliers(saccade.t, saccade.x,
                                 saccade.y, outlier_proba)

        # Curve fitting
        x0 = initial_guess(ransac.t)
        x_bar = fit(ransac.t, ransac.x, x0, fun=sigmoid)
        y_bar = fit(ransac.t, ransac.y, x0, fun=sigmoid)

        # Update saccade parameters
        saccade.compute_parameters(x_bar, y_bar)

    return
