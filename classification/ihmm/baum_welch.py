
import numpy as np


def Baum_Welch(A, B, P, O, n_iter=4):
    """
    Implementation of the Baum_Welch algorithm.
    Based on:
    https://medium.com/mlearning-ai/baum-welch-algorithm-4d4514cf9dbe and
    http://www.adeveloperdiary.com/data-science/machine-learning/derivation-and-implementation-of-baum-welch-algorithm-for-hidden-markov-model/

    Inputs:
        - A = State transition matrix (I x I)
        - B = Emission matrix (I X K with K the number of discrete
            observations)
        - P = Initial state distribution (1 X I)
        - O = Observations (1 x N)

    Outputs:
        - a = Estimation of the state transition matrix
        - b = Estimation of the emission matrix
    """

    # Parameters & variables
    I = A.shape[0]
    N = len(O)
    a = A.copy()
    b = B.copy()

    # A, B estimation is repeated n_iter times
    for _ in range(n_iter):

        ###################
        # Estimation step #
        ###################

        # Initialization
        alpha = forward(a, b, P, O)
        beta = backward(a, b, O)
        Xi = np.zeros((I, I, N-1))

        # Computing ξ: Xi hold values of ξᵢⱼ(t)
        for t in range(N-1):
            denominator = (alpha[t, :].T @ a * b[:, O[t+1]].T) @ beta[t+1, :]
            for i in range(I):
                numerator = alpha[t, i] * a[i, :] * \
                    b[:, O[t+1]].T * beta[t+1, :].T
                Xi[i, :, t] = np.divide(numerator, denominator,
                                        out=np.zeros_like(numerator),
                                        where=denominator != 0)

        # Computing γᵢ(t) values
        gamma = np.sum(Xi, axis=1)

        ###################
        # Estimation step #
        ###################

        # Estimation of a
        a = np.divide(np.sum(Xi, 2), np.sum(gamma, axis=1).reshape((-1, 1)),
                      out=np.zeros_like(np.sum(Xi, 2)),
                      where=np.sum(gamma, axis=1).reshape((-1, 1)) != 0)

        # Add additional T'th element in gamma
        gamma = np.hstack(
            (gamma, np.sum(Xi[:, :, N-2], axis=0).reshape((-1, 1))))

        # Estimation of b
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, O == l], axis=1)
        b = np.divide(b, denominator.reshape((-1, 1)),
                      out=np.zeros_like(b),
                      where=denominator.reshape((-1, 1)) != 0)

    return a, b


def forward(A, B, P, O):

    # Parameters & variables
    I = A.shape[0]
    N = len(O)
    alpha = np.zeros((N, I))

    # Initialization
    alpha[0, :] = P * B[:, O[0]]

    # Computation
    for t in range(1, N):
        for j in range(I):
            alpha[t, j] = alpha[t-1] @ A[:, j] * B[j, O[t]]

    return alpha


def backward(A, B, O):

    # Parameters & variables
    I = A.shape[0]
    N = len(O)
    beta = np.zeros((N, I))

    # Initialization
    # Setting beta(N) = 1
    beta[N-1] = np.ones(I)

    # Loop in backward way from T-1 to
    # Due to python indexing the actual loop will be T-2 to 0
    for t in range(N-2, -1, -1):
        for j in range(I):
            beta[t, j] = (beta[t+1] * B[:, O[t+1]]) @ A[j, :]

    return beta
