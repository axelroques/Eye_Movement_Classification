
from .baum_welch import Baum_Welch
from .viterbi import Viterbi
from ..ivt import IVT

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class IHMM:
    """
    First described in Salvucci and Goldberg (2000). The first 
    stage of the I-HMM is identical to I-VT, where each eye 
    position sample is classified either as a fixation or a saccade 
    depending on the velocity threshold. Second stage is defined by 
    the Viterbi Sampler (Forney (1973)), where each eye position can 
    be re-classified, depending on the probabilistic parameters (initial 
    state, state transition and observation probability distributions) 
    of the model. The goal of the Viterbi Sampler is to maximize the 
    probability of the state assignment given probabilistic parameters 
    of the model. The initial probabilistic parameters given to I-HMM 
    are not optimal and can be improved. The third stage of the I-HMM 
    is defined by Baum-Welch re-estimation algorithm (Baum et al. (1970)). 
    This algorithm re-estimates initial probabilistic parameters and 
    attempts to minimize errors in the state assignments. Parameter 
    re-estimation performed by Baum-Welch can be conducted multiple 
    times. Here, the number of such re-estimations is defined with
    the n_iter parameter.
    """

    def __init__(self, t, x, y, threshold, n_iter=10):

        # Raw parameters
        self.t, self.x, self.y = self._get_raw_parameters(t, x, y)

        # Classification parameter
        self.threshold = threshold
        self.n_iter = n_iter

        # Hidden Markov Model parameters estimation
        # State transition matrix
        # State 0 = Fixation, State 1 = Saccade
        self.A = np.array([[0.95, 0.05], [0.05, 0.95]])
        # Emission matrix
        self.B = np.array([[0.9, 0.1], [0.1, 0.9]])
        # Initial state distribution
        self.P = np.array([0.5, 0.5])

        # Fixations and saccades initialization
        self.fixations = None
        self.saccades = None

    def process(self):
        """
        Run IHMM.

        Initial estimation of A was defined like
        in Salvucci & Goldberg (2000).
        """

        # IVT classification
        classifier = IVT(self.t, self.x, self.y, self.threshold)
        classifier.process()

        # Observations
        O = classifier.saccades

        # Parameter reestimation
        self.a, self.b = Baum_Welch(
            self.A, self.B, self.P, O, n_iter=self.n_iter)

        # Catches potential issue in the Baum Welch algorithm
        if (np.count_nonzero(self.a) == 0) and (np.count_nonzero(self.b) == 0):
            self.a, self.b = self.A, self.B
            print(
                'IHMM parameters re-estimation failed! Returning to original parameters.\n'
            )

        # Decoding
        _, S, _ = Viterbi(self.a, self.b, self.P, O)

        self.fixations = np.where(S == 0, 1, 0)
        self.saccades = np.where(S == 1, 1, 0)

        return

    def stats(self):
        """
        Some simple prints for the Markov model probabilistic parameters.
        """

        print(
            f'Original probabilistic parameters: \nA = \n{self.A}\nB = \n{self.B}\n'
        )

        print(
            f'Re-estimated probabilistic parameters: \nA = \n{self.a}\nB = \n{self.b}\n'
        )

        print(f'Initial state distribution: \nP = {self.P}')

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
