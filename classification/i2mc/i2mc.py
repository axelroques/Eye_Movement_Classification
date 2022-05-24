

from .interp import interpolation_pipeline
from .cluster import clustering_pipeline
from .toolbox import average

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class I2MC:

    def __init__(self, t, type='binocular',
                 x_l=None, x_r=None, y_l=None, y_r=None,
                 x=None, y=None,
                 max_gap_duration=100, verbose=True,
                 window_length=200, step=3, n_iter=10, plot=True):

        # Initialization
        if type == 'binocular':
            # Construct a pandas DataFrame with the eye data
            self.df = pd.DataFrame(data={
                't': t,
                'x_l': x_l,
                'y_l': y_l,
                'x_r': x_r,
                'y_r': y_r})

            # Compute the average x and y positions of the eyes (when possible)
            self.df = average(self.df)

        elif type == 'monocular':
            self.df = pd.DataFrame(data={
                't': t,
                'x_avg': x,
                'y_avg': y})

        else:
            raise RuntimeError(
                'Unrecognized type: should be binocular or monocular'
            )

        # Classification parameters
        self.type = type
        self.max_gap_duration = max_gap_duration
        self.window_length = window_length
        self.step = step
        self.n_iter = n_iter

        # Interpolation
        self.df_interp = interpolation_pipeline(
            self.df, self.max_gap_duration, type, verbose)

        # Plot parameters
        self.plot_clustering_weights = plot
        self.fixations = None
        self.saccades = None

    def process(self):

        # 2-Means Clustering
        self.df_data, self.df_processing = clustering_pipeline(
            self.df_interp, self.window_length,
            self.step, self.n_iter, self.type, self.plot_clustering_weights
        )

        # Fixation and saccades initialization
        self.fixations = np.zeros(len(self.df_processing))
        self.saccades = np.zeros(len(self.df_processing))

        return

    def get_results(self, threshold=None):
        """
        Detect saccades and fixations using a threshold.
        """

        # If a manual threshold is given, use this threshold otherwise
        # by default, threshold is set to the mean clustering weight + 2 stds
        if not threshold:
            threshold = self.df_processing.mean() + 2*self.df_processing.std()

        self.fixations = np.where(self.df_processing < threshold, 1, 0)
        self.saccades = np.where(self.df_processing >= threshold, 1, 0)

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
            axes[0].plot(self.df['t'], self.df['x_avg'], c='silver', alpha=0.9)
            axes[0].set_ylabel('x', fontsize=15)

            axes[1].plot(self.df['t'], self.df['y_avg'], c='silver', alpha=0.9)
            axes[1].set_ylabel('y', fontsize=15)

            # Plot fixations and saccades
            for i in range(len(self.fixations)-1):
                axes[0].axvspan(self.df['t'].iloc[i], self.df['t'].iloc[i+1],
                                color='royalblue' if self.fixations[i] == 1 else 'crimson',
                                ec=None, alpha=0.4)
                axes[1].axvspan(self.df['t'].iloc[i], self.df['t'].iloc[i+1],
                                color='royalblue' if self.fixations[i] == 1 else 'crimson',
                                ec=None, alpha=0.4)

            for ax in axes:
                ax.set_xlabel('Time', fontsize=15)
                ax.set_xlim((self.df['t'].iloc[0], self.df['t'].iloc[-1]))

            plt.tight_layout()
            plt.show()

        else:
            raise RuntimeError('Run process method first')

        return
