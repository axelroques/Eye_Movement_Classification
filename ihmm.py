
from ivt import IVT
import numpy as np


def Viterbi(A, B, P, O):
    """
    Log implementation of the Viterbi algorithm. In the log implementation,
    all probability multiplications are replaced by sums.Based on: 
    https://www.audiolabs-erlangen.de/resources/MIR/FMP/C5/C5S3_Viterbi.html

    Inputs:
        - A = State transition matrix (I x I)
        - B = Emission matrix (I X K with K the number of discrete
            observations)
        - P = Initial state distribution (1 X I)
        - O = Observations (1 x N)

    Outputs:
        - D = Accumulated probability matrix (I x N). D[i, n] is the 
            highest probability along a single state sequence (s_1,…,s_n) 
            that accounts for the first n observations and ends in state
            s_n=alpha_i
        - S = Optimal state sequence (1 x N)
        - backtracking = Backtracking matrix

    """

    # Log matrices
    epsilon = np.finfo(0.).tiny  # Tiny number to prevent log(0)
    A = np.log(A + epsilon)
    B = np.log(B + epsilon)
    P = np.log(P + epsilon)

    # Parameters & variables
    I = A.shape[0]  # Number of states
    N = len(O)  # Length of observation sequence
    D = np.zeros((I, N))
    backtracking = np.zeros((I, N-1)).astype(np.int32)
    S = np.zeros(N).astype(np.int32)

    # Initialization
    # First column is initialized using the initial state distribution
    # and the first observation:
    D[:, 0] = P + B[:, O[0]]

    # Computation
    for n in range(1, N):  # Loop over observations
        for i in range(I):  # Loop over states
            sum = D[:, n-1] + A[:, i]
            D[i, n] = np.max(sum) + B[i, O[n]]
            backtracking[i, n-1] = np.argmax(sum)

    # Backtracking
    S[-1] = np.argmax(D[:, -1])
    for n in range(N-2, -1, -1):
        S[n] = backtracking[int(S[n+1]), n]

    return D, S, backtracking


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
                Xi[i, :, t] = numerator / denominator

        # Computing γᵢ(t) values
        gamma = np.sum(Xi, axis=1)

        ###################
        # Estimation step #
        ###################

        # Estimation of a
        a = np.sum(Xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        # Add additional T'th element in gamma
        gamma = np.hstack(
            (gamma, np.sum(Xi[:, :, N-2], axis=0).reshape((-1, 1))))

        # Estimation of b
        K = b.shape[1]
        denominator = np.sum(gamma, axis=1)
        for l in range(K):
            b[:, l] = np.sum(gamma[:, O == l], axis=1)
        b = np.divide(b, denominator.reshape((-1, 1)))

    return a, b


def IHMM(t, x, y, threshold, n_iter=10):
    """
    A was defined like Salvucci & Goldberg (2000).
    """

    ######################
    # IVT classification #
    ######################
    _, saccades = IVT(t, x, y, threshold)

    #############################################
    # Hidden Markov Model parameters estimation #
    #############################################
    # State transition matrix
    # State 0 = Fixation, State 1 = Saccade
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    # Emission matrix
    B = np.array([[0.9, 0.1], [0.1, 0.9]])
    # Initial state distribution
    P = np.array([0.5, 0.5])
    # Observations
    O = saccades

    a, b = Baum_Welch(A, B, P, O, n_iter=n_iter)

    print(f'Original probabilistic parameters: \nA = \n{A}\nB = \n{B}')
    print(f'Re-estimated probabilistic parameters: \nA = \n{a}\nB = \n{b}')

    # Decoding
    _, S, _ = Viterbi(a, b, P, O)

    fixations = np.where(S == 0, 1, 0)
    saccades = np.where(S == 1, 1, 0)

    return fixations, saccades
