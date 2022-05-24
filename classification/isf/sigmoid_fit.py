
from scipy.optimize import curve_fit
import numpy as np


def sigmoid(t, E_0, E_max, t_50, alpha):
    """
    Sigmoid function using Hill's equation.

    Here we make a small deviation from Gibaldi et. al. (2020) in the denominator to use
    the traditional form of the Hill's equation.
    """

    # Prevents RuntimeWarning as numpy does not like to take
    # a non-integer power of a negative number. We put 0.001
    # rather than 0 to avoid dividing by zero
    t_50 = max(0.001, t_50)

    # print('E_0 =', E_0, 'E_max =', E_max, 't_50 =', t_50, 'alpha =', alpha)

    return E_0 + (E_max-E_0)*np.power(t, alpha)/(np.power(t_50, alpha) + np.power(t, alpha))


def initial_guess(t, s):
    """
    Plausible initial estimate of sigmoid parameters:
        p0[0] = E_0
        p0[1] = E_max
        p0[2] = t_50
        p0[3] = alpha
    """

    p0 = np.zeros(4)
    p0[0] = s[0]
    p0[1] = s[-1]
    p0[2] = t[len(t)//2]-t[0]
    p0[3] = 2

    return p0


def fit(t, y, p0, fun):
    """
    Fitting performed using a nonlinear minimization algorithm.

    Bounds are set to restrict the parameters search space:
        - x[0] = E_0 +- 1°
        - x[1] = E_max +- 1°
        - x[2] = t_50 +- 30 ms
        - x[3] = alpha in [-20, 20]
    """

    # If I really want to try and use the Levemberg-Marquadt algorithm:
    # try:
    #     popt, _ = curve_fit(f=fun, xdata=t, ydata=y,
    #                         p0=p0, method='lm')

    # except RuntimeError:
    #     print('Levemberg-Marquadt minimization failed')
    #     print('Trying Trust Region Reflective(TRF) algorithm with predefined bounds')

    #     bounds = ([p0[0]-1, p0[1]-1, p0[2]-0.03, -20],
    #               [p0[0]+1, p0[1]+1, p0[2]+0.03, 20])

    #     popt, _ = curve_fit(f=fun, xdata=t, ydata=y,
    #                         p0=p0, bounds=bounds)

    bounds = ([p0[0]-1, p0[1]-1, max(p0[2]-0.03, 0), -20],
              [p0[0]+1, p0[1]+1, p0[2]+0.03, 20])

    popt, _ = curve_fit(f=fun, xdata=t, ydata=y,
                        p0=p0, bounds=bounds)

    return popt
