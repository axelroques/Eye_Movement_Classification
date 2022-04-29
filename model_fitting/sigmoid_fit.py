
import numpy as np


def sigmoid(x, t, y):
    """
    Sigmoid function using Hill's equation.
    Small difference in the denominator compared to Gibaldi et.al. (2020) to
    reflect the true form of Hill's equation.
    """

    t_start = x[0]
    t_end = x[1]
    t_50 = x[2]
    alpha = x[3]

    return t_start + (t_end-t_start)*np.pow(t, alpha)/(np.pow(t_50, alpha)+np.pow(t, alpha)) - y


def initial_guess(t):
    """
    Plausible initial estimate of sigmoid parameters:
        x0[0] = t_start
        x0[1] = t_end
        x0[2] = t_50
        x0[3] = alpha
    """

    x0 = np.zeros(4)
    x0[0] = t[0]
    x0[1] = t[-1]
    x0[2] = t[len(t)//2]
    x0[3] = 2

    return x0
