
from .common import compute_velocity

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class IVT:
    """
    First described in Salvucci  and  Goldberg  (2000).
    Each sampled is classified as either a fixation or a saccade
    using a threshold value on the signal's velocity (in °/s).
    """

    def __init__(self, t, x, y, threshold):

        # Raw parameters
        self.t, self.x, self.y = self._get_raw_parameters(t, x, y)

        # Classification parameter
        self.threshold = threshold

        # Velocity parameters
        self.v_x, self.v_y = compute_velocity(self.t,
                                              self.x,
                                              self.y)
        self.v = np.sqrt(self.v_x**2 + self.v_y**2)

        # Fixations and saccades initialization
        self.fixations = None
        self.saccades = None

    def process(self):
        """
        Run IVT.
        """

        # Compute fixations and saccades
        # We allow warnings because there may be NaNs
        with np.errstate(invalid='ignore'):
            self.fixations = np.where(self.v < self.threshold, 1, 0)
            self.saccades = np.where(self.v >= self.threshold, 1, 0)

        return

    def plot(self):
        """
        Simple plot of the fixations and the saccades that
        were found during the process step.

        Fixations are shown in blue, saccades in red.
        """

        if isinstance(self.fixations, np.ndarray):

            _, axes = plt.subplots(2, 1, figsize=(15, 8))

            # Plot eye trace
            axes[0].plot(self.t, self.x, c='silver', alpha=0.9)
            axes[0].set_ylabel('x', fontsize=15)

            axes[1].plot(self.t, self.y, c='silver', alpha=0.9)
            axes[1].set_ylabel('y', fontsize=15)

            # Plot fixations and saccades
            for i in range(len(self.fixations)-1):
                axes[0].axvspan(self.t[i], self.t[i+1],
                                color='royalblue' if self.fixations[i] == 1 else 'crimson',
                                ec=None, alpha=0.4)
                axes[1].axvspan(self.t[i], self.t[i+1],
                                color='royalblue' if self.fixations[i] == 1 else 'crimson',
                                ec=None, alpha=0.4)

            for ax in axes:
                ax.set_xlabel('Time', fontsize=15)
                ax.set_xlim((self.t[0], self.t[-1]))

            plt.tight_layout()
            plt.show()

        else:
            raise RuntimeError('Run process method first')

        return

    @ staticmethod
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
