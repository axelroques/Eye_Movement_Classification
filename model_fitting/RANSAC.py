
from .sigmoid_fit import initial_guess, sigmoid, fit

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


class Subset():
    """
    Simple class for a subset.
    Stores the t, x and y information for each sample in the subset.
    """

    def __init__(self, subset):

        # Sample parameters
        self.subset = subset
        self.n = len(subset)
        self.t = np.array([sample.t for sample in subset])
        self.x = np.array([sample.x for sample in subset])
        self.y = np.array([sample.y for sample in subset])

        # Curve fitting parameters
        self.GOF = {
            'RMSE_x': [],
            'RMSE_y': [],
            'CD_x': [],
            'CD_y': []
        }

    def get_fit(self, popt_x, popt_y):
        """
        Computes the fit of the subset using the resulting fit paramers.
        """

        self.popt_x = popt_x
        self.popt_y = popt_y
        self.x_fit = sigmoid(self.t, *popt_x)
        self.y_fit = sigmoid(self.t, *popt_y)

        return

    def compute_GOF(self):
        """
        Goodness of fit computation:
            - RMSE
            - Pearson correlation distance (CD)

        Also stores the results of the fit for each subset.
        """

        RMSE_x = np.sqrt(np.sum((self.x-self.x_fit)**2)/self.n)
        RMSE_y = np.sqrt(np.sum((self.y-self.y_fit)**2)/self.n)

        CD_x = (1-(np.cov(self.x, self.x_fit) /
                   np.std(self.x)/np.std(self.x_fit)))/2
        CD_y = (1-(np.cov(self.y, self.y_fit) /
                   np.std(self.y)/np.std(self.y_fit)))/2

        self.GOF['RMSE_x'].append((RMSE_x))
        self.GOF['RMSE_y'].append((RMSE_y))
        self.GOF['CD_x'].append(CD_x)
        self.GOF['CD_y'].append(CD_y)

        return

    def update_CL(self):
        """
        Updates each sample's CL.

        For each sample, compute the difference between its value and the value
        of the fit. Then, order the samples in increasing order of difference.
        Finally, each sample is assigned a CL based on their order, ranging from 0  
        for the sample with a value closest to its fit to 1 for the sample with a 
        value the further away to its fit.   
        """

        # Compute the difference between the value and the fit
        differences = np.zeros(len(self.subset))
        for i, sample in enumerate(self.subset):

            diff_x = abs(self.x_fit[i]-sample.x)
            diff_y = abs(self.y_fit[i]-sample.y)
            differences[i] = diff_x + diff_y

        # Normalize the differences and order the samples
        differences /= np.max(differences)
        i_sorted = np.argsort(differences)
        CL_values = np.linspace(0, 1, 11, endpoint=True)
        binning = np.digitize(differences, CL_values, right=True)

        # Assign CL
        for i in i_sorted:
            self.subset[i].CL += CL_values[binning[i]-1]

        return

    def plot_fit(self, t):
        """
        Plots the eye trace vs. the Sigmoid fit.
        """

        import matplotlib.pyplot as plt

        _, ax = plt.subplots(figsize=(15, 8))

        ax.scatter(self.t+t[0], self.x, c='royalblue', alpha=0.8, s=40,
                   facecolors='none', edgecolors='orchid', label='x')
        ax.plot(self.t+t[0], self.x_fit, c='royalblue',
                alpha=0.8, label='x fit')
        ax.scatter(self.t+t[0], self.y, c='crimson', alpha=0.8, s=40,
                   facecolors='none', edgecolors='orchid', label='y')
        ax.plot(self.t+t[0], self.y_fit, c='crimson',
                alpha=0.8, label='y fit')

        ax.set_title('Eye position', fontsize=19)
        ax.set_xlabel('Time (s)', fontsize=15)
        ax.set_ylabel('Angle (°)', fontsize=15)

        ax.legend()
        plt.show()

        return


class RANSAC():
    """
    Random Sample Consensus class.
    Contains every sample of interest.
    """

    def __init__(self, timestamps, x_samples, y_samples, outlier_proba):

        self.n = len(timestamps)
        self.t = timestamps - timestamps[0]
        self.t[0] += 0.00001  # Prevents warnings when fitting the sigmoid
        self.x = x_samples
        self.y = y_samples

        # Assign each sample to a Sample class
        self.samples = [Sample(t, x, y)
                        for t, x, y in zip(self.t, x_samples, y_samples)]

        # Outlier parameters
        self.outlier_proba = outlier_proba
        self.n_outliers = self.estimate_outliers()

    def estimate_outliers(self):
        """
        Estimates the number of outliers using the outlier_proba parameter.
        """
        return int(self.outlier_proba*self.n) + 1

    def generate_subsets(self):
        """
        RANSAC method of selecting samples.

        Here we impose that all subsets generated did not have two successive samples 
        removed.
        The number of subsets created is only a portion of the combinations of
        n-n_outliers subsets possible. We make sure that this selection is sufficiently 
        large to be statistically sure (with 90% confidence) that it contains a subset 
        free of outliers (see Fischler & Bolles, 1981).
        """

        def prune_subset(indices):
            """
            Prunes the subsets: checks whether the input indices are successive or not.
            If they are successive, the function returns False and the subset
            is not kept. If they are not, the function returns True and the 
            subset is stored.
            """

            # Difference in indices
            diff = np.diff(np.sort(choice))

            # If there is a 1 in diff, at least two deletion were successive samples
            if np.any(diff == 1):
                return False

            else:
                return True

        # Max selections (see Fischler & Bolles, 1981)
        k = int(np.log(0.1)/np.log(1-np.power(1-self.outlier_proba,
                                              self.n-self.n_outliers))) + 1

        # Subset generation process
        i = 0
        subsets = []
        while i < k:

            # Select n_outliers samples to be removed
            choice = np.random.choice(np.arange(self.n),
                                      size=self.n_outliers,
                                      replace=False)

            # Check if subset is OK
            if prune_subset(choice):

                # Create subset without these two samples
                subsets.append(Subset([self.samples[i]
                                       for i in range(self.n) if i not in choice]))
                i += 1

        self.subsets = subsets

        return

    def prune(self):
        """
        Remove the outliers: the n_outliers outliers are removed from t and y.
        """

        # Get indices of the n_outliers samples with the lowest CL
        CL_list = [sample.CL for sample in self.samples]
        order = np.argsort(CL_list)[-self.n_outliers:]

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
    ransac = RANSAC(t, x, y, outlier_proba)

    # If we potentially have outliers, start the process
    if ransac.n_outliers > 0:

        # Generate subsets
        ransac.generate_subsets()

        # Iterate over all subsets
        for subset in ransac.subsets:

            # Curve fitting
            try:
                p0 = initial_guess(t, subset.x)
                popt_x = fit(subset.t, subset.x, p0, fun=sigmoid)
                p0 = initial_guess(t, subset.y)
                popt_y = fit(subset.t, subset.y, p0, fun=sigmoid)

                # Store fit results
                subset.get_fit(popt_x, popt_y)

                # Optional plot
                # subset.plot_fit(t)

                # GOF computation
                subset.compute_GOF()

                # Update sample confidence level
                subset.update_CL()

            # If model does not converge
            except (RuntimeError, RuntimeWarning):
                print('\tOptimal parameters not found, skipping this subset.')
                continue

        ransac.prune()

    return ransac
