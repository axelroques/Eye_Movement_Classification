
import numpy as np


def distance_matrix(x_window, y_window):
    """
    Efficient distance matrix calculation using broadcasting
    """
    coords = np.stack([x_window, y_window], axis=1)

    return np.linalg.norm(coords[:, :, None]-coords[:, :, None].T, axis=1)


def Prim(A):
    """
    Inspired by 
    https://stackabuse.com/graphs-in-python-minimum-spanning-trees-prims-algorithm/
    """

    # Initialization
    MST_matrix = np.zeros_like(A)
    n_vertices = A.shape[0]

    # Arbitrary big number as a temporary minimum
    temp = float('inf')

    # Track which vertices were visited
    visited_vertices = [False for vertex in range(n_vertices)]

    # Construct the tree
    while False in visited_vertices:

        # Initial possible minimum weight
        minimum = temp

        # Starting and ending vertices
        start, end = 0, 0

        # Loop over vertices
        for i in range(n_vertices):

            # If the vertex is part of the MST, look for its relationships
            if visited_vertices[i]:

                # Index starts at i, because the matrix is symmetric
                for j in range(i, n_vertices):

                    # If both vertices are connected and the end vertex is not already visited (no loop)
                    if (not visited_vertices[j]) and (A[i][j] > 0):

                        # If this path is better than the current path
                        if A[i][j] < minimum:
                            minimum = A[i][j]
                            start, end = i, j

        # Update visited vertices array
        visited_vertices[end] = True

        # Update MST
        if minimum == temp:
            MST_matrix[start][end] = 0
        else:
            MST_matrix[start][end] = minimum
        MST_matrix[end][start] = MST_matrix[start][end]  # Symmetry

    return MST_matrix


def IMST(x, y, threshold, window_size):
    """
    First described in Salvucci  and  Goldberg  (2000). 
    The I-MST is a dispersion-based identification algorithm that builds a 
    minimum spanning tree taking a predefined number of eye position points
    using Prim's algorithm. The I-MST traverses the MST and separates the points 
    into  fixations and saccades based on the predefined distance thresholds. 
    The I-MST requires a sampling window to build a sequence of MST trees allowing 
    it to parse a long eye movement recording. Here, the window selected is 200 
    points.
    """

    # Initialization
    fixations = np.zeros(len(x))
    saccades = np.zeros(len(x))

    # Iterate over samples
    i = 0
    i_window = 0
    while i < len(x):
        x_window = x[i:i+window_size]
        y_window = y[i:i+window_size]

        # Adjacency matrix = distance matrix
        A = distance_matrix(x_window, y_window)

        # MST
        MST_matrix = Prim(A)

        # Classify
        for i_mst in range(MST_matrix.shape[0]):
            for j_mst in range(i_mst+1, MST_matrix.shape[1]):
                # Edges have to be connected
                if MST_matrix[i_mst, j_mst] > 0:
                    # Fixation/Saccade distinction
                    if MST_matrix[i_mst, j_mst] < threshold:
                        fixations[i_window*window_size+i_mst] = 1
                        fixations[i_window*window_size+j_mst] = 1
                    else:
                        saccades[i_window*window_size+i_mst] = 1
                        saccades[i_window*window_size+j_mst] = 1

        i += window_size
        i_window += 1

    return fixations, saccades
