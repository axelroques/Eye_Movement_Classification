
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
