
import numpy as np


def average(df):
    """
    Computes the average x and y eye position.
    By default, pandas .mean() ignores missing values.
    """

    df['x_avg'] = df.loc[:, ['x_l', 'x_r']].mean(axis=1)
    df['y_avg'] = df.loc[:, ['y_l', 'y_r']].mean(axis=1)

    return df


def downsample(t, x, y, i_ds):
    """
    Downsamples the signal by a factor 1/i_ds
    """

    i = np.arange(0, len(t), step=i_ds)

    return t.iloc[i], x.iloc[i], y.iloc[i]
