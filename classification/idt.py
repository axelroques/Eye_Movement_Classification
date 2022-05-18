
from .common import dispersion

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class IDT:
    """
    First described in Salvucci  and  Goldberg  (2000). The algorithm defines
    a temporal window which moves one point at a time. The spacial dispersion
    created by the points within this window is compared against the threshold.
    If  such  dispersion falls below the threshold, the points within the
    temporal window are classified as a part of fixation. Otherwise, the window
    is moved by one sample, and the first sample of the previous window is 
    classified as a saccade.
    """

    def __init__(self, t, x, y, threshold, window_size):

        # Raw parameters
        self.t, self.x, self.y = self._get_raw_parameters(t, x, y)
        self.n = len(self.x)

        # Classification parameter
        self.threshold = threshold
        self.window_size = window_size

        # Fixations and saccades initialization
        self.fixations = np.zeros(self.n)
        self.saccades = np.zeros(self.n)

    def process(self):
        """
        Run IDT.
        """

        # Iterate over samples
        i = 0
        while (i+self.window_size) < self.n:

            current_window = self.window_size
            D = dispersion(self.x[i:i+current_window],
                           self.y[i:i+current_window])

            if D < self.threshold:
                while (D < self.threshold) and (i+current_window < self.n):
                    D = dispersion(self.x[i:i+current_window],
                                   self.y[i:i+current_window])
                    current_window += 1

                self.fixations[i:i+current_window] = 1
                i += current_window

            else:
                self.saccades[i] = 1

            i += 1

        return

    def plot(self):
        """
        Simple plot of the fixations and the saccades that
        were found during the process step.

        Fixations are shown in blue, saccades in red.
        """

        if self.fixations.any():

            _, axes = plt.subplots(2, 1, figsize=(15, 8))

            # Plot eye trace
            axes[0].plot(self.t, self.x, c='k', alpha=0.8)
            axes[0].set_ylabel('x', fontsize=15)

            axes[1].plot(self.t, self.y, c='k', alpha=0.8)
            axes[1].set_ylabel('y', fontsize=15)

            # Plot fixations and saccades
            for i in range(len(self.fixations)-1):
                axes[0].axvspan(self.t[i], self.t[i+1],
                                color='royalblue' if self.fixations[i] == 1 else 'crimson',
                                ec=None, alpha=0.2)
                axes[1].axvspan(self.t[i], self.t[i+1],
                                color='royalblue' if self.fixations[i] == 1 else 'crimson',
                                ec=None, alpha=0.2)

            for ax in axes:
                ax.set_xlabel('Time', fontsize=15)
                ax.set_xlim((self.t[0], self.t[-1]))

            plt.tight_layout()
            plt.show()

        else:
            raise RuntimeError('Run process method first')

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
