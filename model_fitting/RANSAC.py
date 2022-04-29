
from .sigmoid_fit import initial_guess, sigmoid
from .fit import fit

import numpy as np


class Sample():
    """
    Each sample is characterized by 3 parameters:
        - Its timestamp t
        - Its value y
        - Its confidence level (CL), a statistical measure of the affinity of 
        the sample to the fitted model. In short, a lower CL means a higher
        chance of being an outlier.
    """

    def __init__(self, t, x, y):
        self.t = t
        self.x = x
        self.y = y
        self.CL = 0


class RANSAC():
    """
    Random Sample Consensus class.
    Contains every sample of interest.
    """

    def __init__(self, timestamps, x_samples, y_samples, outlier_proba):

        self.n = len(timestamps)
        self.t = timestamps
        self.x = x_samples
        self.y = y_samples

        # Assign each sample to a Sample class
        self.samples = [Sample(t, x, y)
                        for t, x, y in zip(timestamps, x_samples, y_samples)]

        # Outlier parameters
        self.outlier_proba = outlier_proba
        self.n_outliers = self.estimate_outliers()

        # Curve fitting parameters
        self.GOF = {
            'RMSE_x': [],
            'RMSE_x': [],
            'CD_x': [],
            'CD_y': []
        }
        self.x_fit = []
        self.y_fit = []

    def estimate_outliers(self):
        """
        Estimates the number of outliers using the outlier_proba parameter.
        """
        return int(self.outlier_proba*len(self.n)) + 1

    def generate_subsets(self):
        """
        Generates a list of every possible subset of n-n_outliers samples.
        """

        self.subsets = [[self.samples[:i]+self.samples[i+2:]]
                        for i in range(self.n-1)] + [[self.samples[1:-1]]]

        return

    def compute_GOF(self, x_bar, y_bar):
        """
        Goodness of fit computation:
            - RMSE
            - Pearson correlation distance (CD)

        Also stores the results of the fit for each subset.
        """

        RMSE_x = np.sqrt(np.sum((self.x-x_bar)**2)/self.n)
        RMSE_y = np.sqrt(np.sum((self.y-y_bar)**2)/self.n)

        CD_x = (1-(np.cov(self.x, x_bar)/np.std(self.x)/np.std(x_bar)))/2
        CD_y = (1-(np.cov(self.y, y_bar)/np.std(self.y)/np.std(y_bar)))/2

        self.GOF['RMSE_x'].append((RMSE_x))
        self.GOF['RMSE_y'].append((RMSE_y))
        self.GOF['CD_x'].append(CD_x)
        self.GOF['CD_y'].append(CD_y)

        self.x_fit.append(y_bar)
        self.y_fit.append(y_bar)

        return

    def update_CL(self):
        """
        Updates each sample's CL.
        """
        for sample, x_fit, y_fit in zip(self.samples, self.x_fit, self.y_fit):

            # Confidence interval on the y_fit distribution
            # TO DO

            # CL computation
            # TO DO
            sample.CL += y_fit-sample.y

        return

    def prune(self):
        """
        Remove the outliers: the n_outliers outliers are removed from t and y.
        """

        # Get indices of the n_outliers samples with the lowest CL
        CL_list = [sample.CL for sample in self.samples]
        order = np.argsort(CL_list)[:self.n_outliers]

        # Creates a mask set to False for all indices in order
        mask = np.ones_like(self.t, dtype=bool)
        mask[order] = False

        # Update t and y
        self.t = self.t[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]

        return


def remove_outliers(t, x, y, outlier_proba):
    """
    Runs the Random Sample Consensus (RANSAC) method. 
    Smoothes data containing outliers. 

    It first estimates the number of outliers n_outliers in the dataset using the 
    input outlier_probability parameter.
    Then, a curve is fitted over all subsets of size n - n_outliers. A metric called 
    the confidence level (CL) is accumulated for each original sample. This metric is 
    statistical in nature and reflects the affinity of that sample with the fitted
    model. Thus, the lower the CL, the higher the chance that this sample is an outlier.    

    Returns a ransac object, containing the pruned version of t and y where the n_outliers 
    outliers were removed.
    """

    # Initializing RANSAC instance
    ransac = RANSAC(t, y, outlier_proba)

    # If we potentially have outliers, start the process
    if ransac.n_outliers > 0:

        # Generate subsets
        ransac.generate_subsets()

        # Iterate over all subsets
        for subset in ransac.subsets:

            # Retrieve values from subset
            t_sub = np.array([sample.t for sample in subset])
            x_sub = np.array([sample.x for sample in subset])
            y_sub = np.array([sample.x for sample in subset])

            # Curve fitting
            x0 = initial_guess(t_sub)
            x_bar = fit(t_sub, x_sub, x0, fun=sigmoid)
            y_bar = fit(t_sub, y_sub, x0, fun=sigmoid)

            # GOF computation
            ransac.compute_GOF(ransac, x_bar, y_bar)

            # Update sample confidence level
            ransac.update_CL()

        ransac.prune()

    return ransac
