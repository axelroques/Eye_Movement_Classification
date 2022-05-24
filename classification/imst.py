
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class IMST:
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

    def __init__(self, t, x, y, threshold, window_size):

        # Raw parameters
        self.t, self.x, self.y = self._get_raw_parameters(t, x, y)
        self.n = len(self.x)

        # Classification parameter
        self.threshold = threshold
        self.window_size = window_size

        # Fixations and saccades initialization
        self.fixations = np.zeros(len(x))
        self.saccades = np.zeros(len(x))

    def process(self):
        """
        Run IMST.
        """

        # Iterate over samples
        i = 0
        i_window = 0
        while i < self.n:
            x_window = self.x[i:i+self.window_size]
            y_window = self.y[i:i+self.window_size]

            # Adjacency matrix = distance matrix
            A = self._distance_matrix(x_window, y_window)

            # MST
            MST_matrix = self._Prim(A)

            # Classify
            for i_mst in range(MST_matrix.shape[0]):
                for j_mst in range(i_mst+1, MST_matrix.shape[1]):
                    # Edges have to be connected
                    if MST_matrix[i_mst, j_mst] > 0:
                        # Fixation/Saccade distinction
                        if MST_matrix[i_mst, j_mst] < self.threshold:
                            self.fixations[i_window*self.window_size+i_mst] = 1
                            self.fixations[i_window*self.window_size+j_mst] = 1
                        else:
                            self.saccades[i_window*self.window_size+i_mst] = 1
                            self.saccades[i_window*self.window_size+j_mst] = 1

            i += self.window_size
            i_window += 1

        return

    @staticmethod
    def _Prim(A):
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

    @staticmethod
    def _distance_matrix(x_window, y_window):
        """
        Efficient distance matrix calculation using broadcasting
        """
        coords = np.stack([x_window, y_window], axis=1)

        return np.linalg.norm(coords[:, :, None]-coords[:, :, None].T, axis=1)

    def plot(self):
        """
        Simple plot of the fixations and the saccades that
        were found during the process step.

        Fixations are shown in blue, saccades in red.
        """

        if self.fixations.any():

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
