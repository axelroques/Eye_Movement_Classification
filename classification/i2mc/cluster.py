
from .toolbox import downsample

from scipy.cluster.vq import kmeans2
import pandas as pd
import numpy as np


def find_clusters(x_ds, y_ds, n_iter, cluster_plot):
    """
    Uses scipy's kmeans2 algorithm to classify a set of observations into 2 clusters using the k-means algorithm with k=2.
    The algorithm attempts to minimize the Euclidean distance between observations and centroids. 
    Cluster centroid initialization is done using the kmeans++ method (careful seeding) from  Arthur & Vassilvitskii (2007).
    """

    # Reshape the data as a M X 2 matrix of M 'observations' in 2 dimensions
    data = np.stack([x_ds, y_ds], axis=1)

    # 2-means algorithm
    centroids, labels = kmeans2(data, k=2, minit='++', iter=n_iter)

    # Plot clustered data
    if cluster_plot:
        import matplotlib.pyplot as plt
        w0 = data[labels == 0]
        w1 = data[labels == 1]
        _, ax = plt.subplots(2, 1, figsize=(12, 8))
        ax[0].plot(x_ds, c='royalblue', alpha=0.8, label='x')
        ax[0].plot(y_ds, c='crimson', alpha=0.8, label='y')
        ax[0].set_xlabel('timestep')
        ax[0].set_ylabel('x/y (°)')
        ax[1].plot(w0[:, 0], w0[:, 1], 'o', alpha=0.5, label='cluster 0')
        ax[1].plot(w1[:, 0], w1[:, 1], 'd', alpha=0.5, label='cluster 1')
        ax[1].plot(centroids[:, 0], centroids[:, 1], 'k*', label='centroids')
        ax[1].set_xlabel('x (°)')
        ax[1].set_ylabel('y (°)')
        for a in ax:
            a.legend()
        plt.show()

    return centroids, labels


def on_off_switch(weights, t_window, common_indices):
    """
    Converts discrete weight assignment into piecewise constant linear function.
    The output new_weights = weights if i_ds = 1.
    """

    new_weights = np.zeros_like(t_window)
    slices = [slice(i, j) for i, j in zip(common_indices[:-1],
                                          common_indices[1:])]

    for k, s in enumerate(slices):
        new_weights[s] = weights[k]

    return new_weights


def clustering_weight(labels):
    """
    Computes the clustering weight of each sample. 
    The weight is defined as 1 over the total number of transitions.
    """

    # Find the "timestamp" of transitions from one cluster to another
    # We need to append a 0 at the end to avoid shape mismatch
    transitions = np.abs(np.diff(labels))
    transitions = np.append(transitions, 0)

    # Compute weight
    n_transitions = np.sum(transitions)
    if n_transitions == 0:
        transition_weights = transitions
    else:
        transition_weights = 1/n_transitions*transitions

    return transition_weights


def clustering_procedure(t, x, y, window_length,
                         step, n_iter):
    """
    Clustering procedure. 
    While running a window over the samples iif the window does not 
    contain any NaN:
        - For each downsampling factor in ds_list (i_ds):
            - Cluster the data in two groups using 2-means algorithm.
            - Compute a clustering weight for each sample.

    The array ds_list contains the values i_ds that will determine the 
    frequency of the downsampled signals (Fs/i_ds). Note that highly 
    downsampled signals will contain very few points and therefore 
    will have high clustering weights (could create a bias).
    """

    # Initialization
    # Divide the sampling frequency by these numbers
    ds_list = [1, 2, 5, 10]
    data = {
        f'i_ds={i_ds}': {'t': [], 'weights': []} for i_ds in ds_list
    }

    # Iterate over samples
    i = 0
    while i < len(t):

        # Moving window
        n_window = len(t.loc[(t >= t.iloc[i]) &
                             (t < t.iloc[i]+window_length/1000)])
        t_window = t.iloc[i:i+n_window]
        x_window = x.iloc[i:i+n_window]
        y_window = y.iloc[i:i+n_window]

        # If NaNs in the window, skip to the next window
        if (np.isnan(x_window).any()) or (np.isnan(y_window).any()) or (n_window <= 20):
            pass

        # Otherwise run the clustering procedure
        else:

            cluster_plot = False
            # Uncomment this to plot the clustering results
            # if (t_window.iloc[0] > 48.3) and (t_window.iloc[0] < 48.5):
            #     print(f'Window = {t_window.iloc[0]}-{t_window.iloc[1]}')
            #     cluster_plot = True

            for i_ds in ds_list:

                # Downsample the signal
                _, x_ds, y_ds = downsample(t_window, x_window,
                                           y_window, i_ds)

                # Cluster data points

                _, labels = find_clusters(x_ds, y_ds, n_iter, cluster_plot)

                # Assign weights
                weights = clustering_weight(labels)

                if cluster_plot:
                    print(f'i_ds = {i_ds}')
                    print('labels =', labels)
                    print('weights = ', weights)
                    print('-------------')

                    # Transform discrete weight assignment in a piece constant signal
                    # using t_window. E.g. for i_ds > 1, the weight should remain
                    # the same until the next time stamp
                if i_ds > 1:
                    weights = on_off_switch(weights, t_window,
                                            np.arange(0, len(t_window), step=i_ds))

                # Store weights & additional data
                data[f'i_ds={i_ds}']['t'] += t_window.tolist()
                data[f'i_ds={i_ds}']['weights'] += weights.tolist()

        i = i + step

    return pd.concat({k: pd.DataFrame(v).T for k, v in data.items()}, axis=0)


def plot_clustering_weight(df, df_data, df_processing):

    import matplotlib.pyplot as plt
    _, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Plot raw eye signal
    axes[0].plot(df['t'], df['x_l'], alpha=0.5, label=f'Left eye')
    axes[0].plot(df['t'], df['x_r'], alpha=0.5, label=f'Right eye')
    axes[0].plot(df['t'], df['x_avg'], alpha=0.5, label=f'Average')
    axes[0].set_ylabel('x', fontsize=15)
    axes[1].plot(df['t'], df['y_l'], alpha=0.5, label=f'Left eye')
    axes[1].plot(df['t'], df['y_r'], alpha=0.5, label=f'Right eye')
    axes[1].plot(df['t'], df['y_avg'], alpha=0.5, label=f'Average')
    axes[1].set_ylabel('y', fontsize=15)

    # Plot cluster weights wrt. downsampling index
    # ds_plot = df_data.loc['avg', :]
    # for i, (t, s) in enumerate(zip(ds_plot.columns.values.tolist()[:-1:2],
    #                                ds_plot.columns.values.tolist()[1::2])):
    #     if i < 5:  # To restrict number of curves
    #         axes[2].scatter(ds_plot[t], ds_plot[s], s=2,
    #                         alpha=0.2, label=f'{t[0]}')
    #     axes[2].set_ylabel('Clustering weight')
    # axes[2].set_xlim((47, 47.2))

    # Plot processed clustering weight
    axes[2].plot(df_processing.index.values, df_processing,
                 alpha=0.8, c='k', ls='--', label=f'Average weight')
    axes[2].set_ylabel('Clustering weight', fontsize=15)
    axes[2].set_xlabel('Time', fontsize=15)

    for ax in axes.ravel():
        ax.legend(fontsize=15, loc='best')

    plt.tight_layout()
    plt.show()

    return


def clustering_pipeline(df, window_length, step, n_iter, type, plot):
    """
    Run the 2-means clustering algorithm.
    By default, the clustering is done for frequencies: fs/1, fs/2, fs/5, fs/10.

    Inputs:
        - df: result from the interpolation_pipeline function
        - window_length: length of the window in ms.
        - step: size of the step between two windows in samples.
    """

    if type == 'binocular':
        eye_list = ['l', 'r', 'avg']
    else:
        eye_list = ['avg']
    data = {
        f'{eye}': 0 for eye in eye_list
    }

    # Apply the clustering procedure separately for each eye, and one more
    # time for the average of both eyes
    for eye in eye_list:

        # Clustering procedure
        t = df['t']
        x = df[f'x_{eye}']
        y = df[f'y_{eye}']
        data_cluster = clustering_procedure(t, x, y, window_length,
                                            step, n_iter)

        # Saving results
        data[f'{eye}'] = data_cluster

    # Create a unique Dataframe with the previous results
    df_data = pd.concat(
        {k: pd.DataFrame(v).T for k, v in data.items()}, axis=0)

    # Sum weights for the different downsampling values
    df_processing = df_data.loc[:, [col for col in df_data.columns.values.tolist()[
        1::2]]].sum(axis=1)

    # Reorganize the dataframe and add the 't' column
    df_processing = df_processing.unstack(level=0)
    t_series = pd.Series(
        df_data.loc['avg', df_data.columns.values.tolist()[0]].tolist(), name='t')
    df_processing['t'] = t_series

    # Average weight values when a sample was present in multiple windows
    df_processing = df_processing.groupby(['t']).mean()

    # Average weights from left, right and both eyes
    df_processing = df_processing.mean(axis=1)

    # Plot
    if plot:
        plot_clustering_weight(df, df_data, df_processing)

    return df_data, df_processing
