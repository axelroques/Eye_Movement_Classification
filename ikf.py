
from common import compute_velocity
import numpy as np


def predict(x_k, y_k, A_k, P_k, Q, B_k=0, U_k=0):
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


def measurement(x_k, H, v_k):
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


def update(x_k_plus_1, y_k_plus_1, P_k_plus_1, H, R):
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
        K @ (measurement(x_k_plus_1, H, R) - H @ x_k_plus_1)
    y_k_plus_1 = y_k_plus_1 + \
        K @ (measurement(y_k_plus_1, H, R) - H @ y_k_plus_1)
    # Update the covariance matrix
    P_k_plus_1 = P_k_plus_1 - (K @ H @ P_k_plus_1)

    return x_k_plus_1, y_k_plus_1, P_k_plus_1


def kalman_filter(dt, x, y, v_x, v_y, P_k):
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
        x_k = np.array([[x.iloc[k]], [v_x.iloc[k]]])
        y_k = np.array([[y.iloc[k]], [v_y.iloc[k]]])
        # State transition matrix
        A_k = np.array([[1, dt.iloc[k]], [0, 1]])

        x_k_plus_1, y_k_plus_1, P_k_plus_1 = predict(x_k, y_k, A_k, P_k, Q)
        x_k_plus_1, y_k_plus_1, P_k = update(x_k_plus_1, y_k_plus_1,
                                             P_k_plus_1, H, R)

        x_estimates.append(x_k_plus_1)
        y_estimates.append(y_k_plus_1)

    v_x_estimates = np.array([i[1] for i in x_estimates]).flatten()
    v_y_estimates = np.array([i[1] for i in y_estimates]).flatten()

    return v_x_estimates, v_y_estimates


def IKF(t, x, y, threshold=150):

    # Parameters
    delta = 1

    # True velocity computation
    v_x, v_y = compute_velocity(t, x, y)

    # Estimated velocity computation
    P_0 = np.zeros((2, 2))
    dt = t.diff().dropna()
    x_sub = x.iloc[1:]
    y_sub = y.iloc[1:]
    v_x_estimates, v_y_estimates = kalman_filter(dt, x_sub, y_sub,
                                                 v_x, v_y, P_0)

    import matplotlib.pyplot as plt
    plt.plot(v_x_estimates, alpha=0.5)
    plt.plot(v_x, alpha=0.5)
    plt.xlim((360, 400))
    plt.show()

    # Chi-square test between the actual and predicted point to point velocities
    Xi_sq = ((v_x_estimates-v_x)**2 + (v_y_estimates-v_y)**2)/delta**2
    print(Xi_sq)
    fixations = np.where(Xi_sq < threshold, 1, 0)
    saccades = np.where(Xi_sq >= threshold, 1, 0)

    return fixations, saccades
