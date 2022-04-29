
from scipy.optimize import least_squares

def fit(t, y, x0, fun):
    """
    Fitting performed using a Levemberg-Marquadt nonlinear minimization algorithm.
    """

    residuals = least_squares(fun=fun, x0=x0,
                              args=(t, y), method='lm')

    return residuals.x
