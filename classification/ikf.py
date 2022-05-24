
from .common import compute_velocity

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class IKF:
    """
    Here, we employ a Two State Kalman Filter (TSKF), first described 
    in Komogortsev and Khan (2009). The TSKF models an eye as a system 
    with two states: position and velocity. The acceleration of the eye 
    is modeled as white noise with fixed maximum acceleration. When 
    applied to the recorded eye position signal the TSKF generates 
    predicted eye velocity signal. The values of the measured and 
    predicted eye velocity allow employing Chi-square test to detect
    the onset and the offset of a saccade (Sauter ,1991).
    """

    def __init__(self, t, x, y, threshold=150):

        # Raw parameters
        self.t, self.x, self.y = self._get_raw_parameters(t, x, y)

        # Classification parameter
        self.threshold = threshold
        self.delta = 1

        # Kalman Filter variables
        self.P_0 = np.zeros((2, 2))
        self.dt = np.diff(self.t)
        self.x_KF = self.x[1:]
        self.y_KF = self.y[1:]

        # Velocity parameters
        self.v_x, self.v_y = compute_velocity(self.t,
                                              self.x,
                                              self.y)

        # Fixations and saccades initialization
        self.fixations = None
        self.saccades = None

    def process(self):
        """
        Run IKF.
        """

        # Estimated velocity computation
        v_x_estimates, v_y_estimates = self._kalman_filter(self.dt, self.x_KF,
                                                           self.y_KF, self.v_x,
                                                           self.v_y, self.P_0)

        # plt.plot(v_x_estimates, alpha=0.5)
        # plt.plot(self.v_x, alpha=0.5)
        # plt.xlim((360, 400))
        # plt.show()

        # Chi-square test between the actual and predicted point to point velocities
        Xi_sq = ((v_x_estimates-self.v_x)**2 +
                 (v_y_estimates-self.v_y)**2)/self.delta**2

        # print(Xi_sq)
        self.fixations = np.where(Xi_sq < self.threshold, 1, 0)
        self.saccades = np.where(Xi_sq >= self.threshold, 1, 0)

        return

    @staticmethod
    def _kalman_filter(dt, x, y, v_x, v_y, P_k):
        """
        Following Komogortsev et. al. (2007), the control input U is null.
        """

        ####################
        # Model Parameters #
        ####################

        # Measurement noise
        R = np.eye(1)
        # Variance of the physiological noise
        delta_w = 1
        # Observation matrix
        H = np.array([[1, 0]])
        # Covariance matrix for system's noise
        Q = np.array([[delta_w**2, 0], [0, delta_w**2]])

        ##################
        # Initialization #
        ##################
        x_estimates = []
        y_estimates = []

        ##########
        # Filter #
        ##########

        for k in range(len(x)):

            # State vectors
            x_k = np.array([[x[k]], [v_x[k]]])
            y_k = np.array([[y[k]], [v_y[k]]])
            # State transition matrix
            A_k = np.array([[1, dt[k]], [0, 1]])

            x_k_plus_1, y_k_plus_1, P_k_plus_1 = IKF._predict(
                x_k, y_k, A_k, P_k, Q)
            x_k_plus_1, y_k_plus_1, P_k = IKF._update(x_k_plus_1, y_k_plus_1,
                                                      P_k_plus_1, H, R)

            x_estimates.append(x_k_plus_1)
            y_estimates.append(y_k_plus_1)

        v_x_estimates = np.array([i[1] for i in x_estimates]).flatten()
        v_y_estimates = np.array([i[1] for i in y_estimates]).flatten()

        return v_x_estimates, v_y_estimates

    @staticmethod
    def _predict(x_k, y_k, A_k, P_k, Q, B_k=0, U_k=0):
        """
        This step estimates the values x_k_plus_1 and y_k_plus_1 and the covariance P of the
        system state at timestep k given the previous measurements.

        Inputs:
            - x_k, y_k = Measurements of the system at timestep k
            - A_k = Transition matrix at timestep k
            - P_k = State covariance matrix at timestep k
            - Q = Process noise covariance matrix
            - B_k = Input effect matrix at timestep k
            - U_k = Control input at timestep k

        Outputs:
            - x_k_plus_1, y_k_plus_1 = Mean state prediction at timestep k+1
            - P_k_plus_one = State covariance matrix prediction at timestep k+1
        """

        x_k_plus_1 = A_k @ x_k + np.dot(B_k, U_k)
        y_k_plus_1 = A_k @ y_k + np.dot(B_k, U_k)
        P_k_plus_1 = A_k @ P_k @ A_k.T + Q

        return x_k_plus_1, y_k_plus_1, P_k_plus_1

    @staticmethod
    def _update(x_k_plus_1, y_k_plus_1, P_k_plus_1, H, R):
        """
        This steps updates the estimations of x_k and y_k using the measurement
        at the current time step.

        Inputs:
            - x_k_plus_one, y_k_plus_one = Mean state prediction at timestep k+1
            - P_k_plus_one = State covariance matrix prediction at timestep k+1
            - H = Observation model matrix
            - R = v_k = Covariance of observation noise

        Outputs:
            - x_k_plus_1, y_k_plus_1 = Mean state estimate of step k+1
            - P_k_plus_one = State covariance matrix of step k+1
        """

        # Compute Kalman gain matrix
        K = P_k_plus_1 @ H.T @ np.linalg.inv(H @ P_k_plus_1 @ H.T + R)
        # Correct the prediction using the measurement
        x_k_plus_1 = x_k_plus_1 + \
            K @ (IKF._measurement(x_k_plus_1, H, R) - H @ x_k_plus_1)
        y_k_plus_1 = y_k_plus_1 + \
            K @ (IKF._measurement(y_k_plus_1, H, R) - H @ y_k_plus_1)
        # Update the covariance matrix
        P_k_plus_1 = P_k_plus_1 - (K @ H @ P_k_plus_1)

        return x_k_plus_1, y_k_plus_1, P_k_plus_1

    @staticmethod
    def _measurement(x_k, H, v_k):
        """
        Computes the measurement vector of the system at timestep k.

        Inputs:
            - x_k = Observation at timestep k
            - H = Observation model matrix
            - v_k = Observation noise with covariance R_k at timestep k

        Outputs:
            - z_k = Measurement vector at timestep k

        Note that in this implementation, v_k = R
        """
        return H @ x_k + v_k

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

                # Color selection
                if self.fixations[i] == 1:
                    color = 'royalblue'
                elif self.saccades[i] == 1:
                    color = 'crimson'
                else:
                    color = 'w'

                # Draw rectangle
                axes[0].axvspan(self.t[i], self.t[i+1],
                                color=color, ec=None, alpha=0.4)
                axes[1].axvspan(self.t[i], self.t[i+1],
                                color=color, ec=None, alpha=0.4)

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
