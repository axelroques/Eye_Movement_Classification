
from .sigmoid_fit import initial_guess, sigmoid, fit
from .RANSAC import remove_outliers
from ..ihmm.ihmm import IHMM
from .saccade import Saccade

from scipy.optimize import OptimizeWarning
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings


class ISF:
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

    def __init__(self, t, x, y, threshold, n_iter=10,
                 outlier_proba=0.2, n_min=5):

        # Raw parameters
        self.t, self.x, self.y = self._get_raw_parameters(t, x, y)

        # Classification parameter
        self.threshold = threshold
        self.n_iter = n_iter
        self.outlier_proba = outlier_proba
        self.n_min = n_min

        # Saccades initialization
        self.saccades = None
        self.n_real_sacc = 0

    def process(self):
        """
        Run ISF process.
        """

        # Consider warnings as errors
        warnings.simplefilter("error")

        # Rough estimate of saccade position
        # Shoudld we use IVT like in the article or IHMM?
        self.ihmm = IHMM(self.t, self.x, self.y, self.threshold, self.n_iter)
        self.ihmm.process()
        sacc = self.ihmm.saccades

        # Clen data by removing NaNs
        # self._clean_data()

        # Get a list of saccades using the previous rough estimates of
        # saccade position
        self.saccades = self._get_saccades(sacc)

        # Process each saccade
        for i, saccade in enumerate(self.saccades):

            # Progress bar
            print('\rFitting the saccades: [{0:<50s}] {1:5.1f}%'
                  .format('#' * int((i+1)/len(self.saccades)*50),
                          (i+1)/len(self.saccades)*100), end="")

            # RANSAC method to remove outliers
            ransac = remove_outliers(saccade.t, saccade.x,
                                     saccade.y, self.outlier_proba)

            # Curve fitting
            try:
                p0 = initial_guess(ransac.t, ransac.x)
                popt_x = fit(ransac.t, ransac.x, p0, fun=sigmoid)
                p0 = initial_guess(ransac.t, ransac.y)
                popt_y = fit(ransac.t, ransac.y, p0, fun=sigmoid)

            # If model does not converge
            except (RuntimeError, OptimizeWarning):
                # print('Model failed to converge.')

                # This saccade could not be fitted
                saccade.is_real = False

                continue

            # Update saccade parameters
            saccade.compute_parameters(ransac, saccade.t, popt_x, popt_y)
            self.n_real_sacc += 1

        # Exit statement
        print(
            f'\nFinished! {self.n_real_sacc}/{len(self.saccades)} saccades were fitted.'
        )

        return

    def _clean_data(self):
        """
        Remove unwanted NaNs in the data.

        This step has to be done after the rough saccade position
        estimate not to mess with the velocity computation.
        """

        # NaN positions
        mask_x = ~np.isnan(self.x)
        mask_y = ~np.isnan(self.y)
        mask = mask_x * mask_y

        # Update arrays
        self.t = self.t[mask]
        self.x = self.x[mask]
        self.y = self.y[mask]

        return

    def _get_saccades(self, sacc):
        """
        Get the start and end timestamps estimates of each saccade.
        """

        # Get all saccades positions
        self.sacc_pos = []
        for i, (val1, val2) in enumerate(zip(sacc[0:-1], sacc[1:])):

            # Initialization
            if (i == 0) and (val1 == 1):
                self.sacc_pos.append(i)

            # Tests to append
            if (val2 - val1) == 0:
                continue
            else:
                self.sacc_pos.append(i+1)

        # Termination
        self.sacc_pos += [len(sacc)]

        return [Saccade(self.t[i_start:i_end],
                        self.x[i_start:i_end],
                        self.y[i_start:i_end])
                for i_start, i_end in zip(self.sacc_pos[::2],
                                          self.sacc_pos[1::2])
                if (i_end-i_start) >= self.n_min]

    def get_saccade_parameters(self):
        """
        Returns a list of dictionaries with the saccade parameters. 
        """

        return [saccade.__dict__ for saccade in self.saccades]

    def plot(self):
        """
        Roughly localizes the saccade position on the whole experiment.
        """
        _, axes = plt.subplots(2, 1, figsize=(15, 8))

        # All saccades as found by the IHMM algorithm
        for i_start, i_end in zip(self.sacc_pos[::2], self.sacc_pos[1::2]):
            if (i_end-i_start) >= self.n_min:

                axes[0].axvspan(self.t[i_start:i_end][0], self.t[i_start:i_end][-1],
                                color='lightgrey', ec=None, alpha=0.1)
                axes[1].axvspan(self.t[i_start:i_end][0], self.t[i_start:i_end][-1],
                                color='lightgrey', ec=None, alpha=0.1)

        # Saccades that could be fitted
        for saccade in self.saccades:
            if saccade.is_real:

                axes[0].axvspan(saccade.t_start_x, saccade.t_end_x,
                                color='crimson', ec=None, alpha=0.3)
                axes[1].axvspan(saccade.t_start_y, saccade.t_end_y,
                                color='crimson', ec=None, alpha=0.3)

        # Eye traces
        axes[0].plot(self.t, self.x, c='silver', alpha=0.9)
        axes[0].set_ylabel('x', fontsize=15)

        axes[1].plot(self.t, self.y, c='silver', alpha=0.9)
        axes[1].set_ylabel('y', fontsize=15)

        # Plot parameters
        legend_elements = [
            Patch(facecolor='lightgrey', alpha=0.1, label='Saccades (IHMM)'),
            Patch(facecolor='crimson', alpha=0.3, label='Saccades (ISF)')
        ]
        for ax in axes:
            ax.set_xlabel('Time', fontsize=15)
            ax.set_xlim((self.t[0], self.t[-1]))
            ax.legend(handles=legend_elements)

        plt.tight_layout()
        plt.show()

        return

    def plot_saccades(self, n=None):
        """
        Plot n of the identified saccades with their sigmoid fit.
        """

        if n:
            saccades = self.saccades[:n]
        else:
            saccades = self.saccades

        for saccade in saccades:

            # Only focus on the saccades that could be fitted
            if saccade.is_real:

                print(
                    '------------------------------ * ------------------------------\n')
                print('Saccade parameters:')
                print('\t* x component')
                print(
                    f'\t\t- Duration = {saccade.duration_x*1000:.1f} ms ; [{saccade.t_start_x:.3f} - {saccade.t_end_x:.3f}] s')
                print(f'\t\t- Amplitude = {saccade.amplitude_x:.2f} °')
                print(
                    f'\t\t- Peak velocity = {saccade.peak_velocity_x:.2f} °/s')
                print('\t* y component')
                print(
                    f'\t\t- Duration = {saccade.duration_y*1000:.1f} ms ; [{saccade.t_start_y:.3f} - {saccade.t_end_y:.3f}] s')
                print(f'\t\t- Amplitude = {saccade.amplitude_y:.2f} °')
                print(
                    f'\t\t- Peak velocity = {saccade.peak_velocity_y:.2f} °/s\n')

                saccade.plot_saccade()

        return

    @staticmethod
    def _get_raw_parameters(t, x, y):
        """
        Type check.
        """

        # t
        if isinstance(t, pd.core.series.Series):
            t = t.to_numpy()
        elif isinstance(t, np.ndarray):
            pass
        elif isinstance(t, list):
            t = np.array(t)
        else:
            raise RuntimeError('Unacceptable data type for parameter t')

        # x
        if isinstance(x, pd.core.series.Series):
            x = x.to_numpy()
        elif isinstance(x, np.ndarray):
            pass
        elif isinstance(x, list):
            x = np.array(x)
        else:
            raise RuntimeError('Unacceptable data type for parameter x')

        # y
        if isinstance(y, pd.core.series.Series):
            y = y.to_numpy()
        elif isinstance(y, np.ndarray):
            pass
        elif isinstance(y, list):
            y = np.array(y)
        else:
            raise RuntimeError('Unacceptable data type for parameter y')

        return t, x, y
