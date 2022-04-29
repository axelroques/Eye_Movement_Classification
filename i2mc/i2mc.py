
from .interp import interpolation_pipeline
from .cluster import clustering_pipeline
from .toolbox import average
import pandas as pd
import numpy as np


def I2MC(t,
         type='binocular', x_l=None, x_r=None, y_l=None, y_r=None,
         x=None, y=None,
         max_gap_duration=100, verbose=True,
         window_length=200, step=3, n_iter=10, plot=True):

    ##################
    # Initialization #
    ##################

    if type == 'binocular':
        # Construct a pandas DataFrame with the eye data
        df = pd.DataFrame(data={
            't': t,
            'x_l': x_l,
            'y_l': y_l,
            'x_r': x_r,
            'y_r': y_r})

        # Compute the average x and y positions of the eyes (when possible)
        df = average(df)

    elif type == 'monocular':
        df = pd.DataFrame(data={
            't': t,
            'x_avg': x,
            'y_avg': y})

    else:
        print('Unrecognized type: should be binocular or monocular')
        return None

    #################
    # Interpolation #
    #################

    df_interp = interpolation_pipeline(df, max_gap_duration, type, verbose)

    ######################
    # 2-Means Clustering #
    ######################

    df_data, df_processing = clustering_pipeline(
        df_interp, window_length, step, n_iter, type, plot)

    ######################
    # Fixation Detection #
    ######################
    fixations = np.zeros(len(df_processing))
    saccades = np.zeros(len(df_processing))

    # Threshold is the mean clustering weight + 2 stds
    threshold = df_processing.mean() + 2*df_processing.std()

    fixations = np.where(df_processing < threshold, 1, 0)
    saccades = np.where(df_processing >= threshold, 1, 0)

    return df_data, df_processing, fixations, saccades


def IM2C_detection(df_processing, threshold):
    """
    Separate function to detect saccades and fixations with a manual threshold
    """
    fixations = np.zeros(len(df_processing))
    saccades = np.zeros(len(df_processing))

    fixations = np.where(df_processing < threshold, 1, 0)
    saccades = np.where(df_processing >= threshold, 1, 0)

    return fixations, saccades
