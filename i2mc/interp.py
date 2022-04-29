
import pandas as pd
import numpy as np


def cubic_interpolation_function(t, t_i, a_i, b_i, c_i, d_i):
    return a_i*(t-t_i)**3 + b_i*(t-t_i)**2 + c_i*(t-t_i) + d_i


def h(t_i, t_i_plus_1):
    return t_i_plus_1 - t_i


def s(t_i, t_i_plus_1, y_i, y_i_plus_1):
    return (y_i_plus_1-y_i) / (t_i_plus_1-t_i)


def slope(s_i_minus_1, s_i, h_i_minus_1, h_i):

    if s_i_minus_1*s_i <= 0:
        return 0

    else:
        p_i = (s_i_minus_1*h_i + s_i*h_i_minus_1) / (h_i+h_i_minus_1)

        if (abs(p_i) > 2*abs(s_i_minus_1)) or \
                (abs(p_i) > 2*abs(s_i)):
            return 2*np.sign(s_i)*abs(min([s_i_minus_1, s_i], key=abs))

        else:
            return p_i


def a(y_p_i, y_p_i_plus_1, s_i, h_i):
    return (y_p_i + y_p_i_plus_1 - 2*s_i) / h_i**2


def b(y_p_i, y_p_i_plus_1, s_i, h_i):
    return (3*s_i - 2*y_p_i - y_p_i_plus_1) / h_i


def c(y_p_i):
    return y_p_i


def d(y_i):
    return y_i


def interpolate(t, t_i_minus_1, t_i, t_i_plus_1, t_i_plus_2,
                y_i_minus_1, y_i, y_i_plus_1, y_i_plus_2):

    # Basic computations
    h_i_minus_1 = h(t_i_minus_1, t_i)
    h_i = h(t_i, t_i_plus_1)
    h_i_plus_1 = h(t_i_plus_1, t_i_plus_2)
    s_i_minus_1 = s(t_i_minus_1, t_i, y_i_minus_1, y_i)
    s_i = s(t_i, t_i_plus_1, y_i, y_i_plus_1)
    s_i_plus_1 = s(t_i_plus_1, t_i_plus_2, y_i_plus_1, y_i_plus_2)

    # Derivatives
    y_p_i = slope(s_i_minus_1, s_i, h_i_minus_1, h_i)
    y_p_i_plus_1 = slope(s_i, s_i_plus_1, h_i, h_i_plus_1)

    # Coefficients
    a_i = a(y_p_i, y_p_i_plus_1, s_i, h_i)
    b_i = b(y_p_i, y_p_i_plus_1, s_i, h_i)
    c_i = c(y_p_i)
    d_i = d(y_i)

    return cubic_interpolation_function(t, t_i, a_i, b_i, c_i, d_i)


def fill_missing_data(t_array, signal, max_gap_duration, verbose):
    """
    Use Steffen's interpolation method to fill gaps in the data.
    The function uses parameter max_gap_duration (in ms) to limit the maximum 
    number of NaNs it tries to fill. 
    """

    interpolated = []

    # Find NaNs in the column
    NaNs = signal.copy()
    NaNs = (NaNs.isnull().astype(int)
            .groupby(NaNs.notnull().astype(int).cumsum())
            .cumsum().to_frame('Consecutive NaNs'))

    # Iterate over values
    i = 0
    while i < len(NaNs):

        nan = NaNs['Consecutive NaNs'].iloc[i]

        # If start of a NaN
        if nan != 0:

            # Count number of successive NaNs
            n_nan = 1
            while (i+n_nan < len(NaNs)) and (NaNs['Consecutive NaNs'].iloc[i+n_nan] != 0):
                n_nan += 1

            dt = (t_array.iloc[i+n_nan] - t_array.iloc[i-1])*1000

            if verbose:
                print(f'Gap of size {n_nan} ({dt:.1f} ms) at i = {i}')

            # If gap is not too long, try to interpolate
            if dt <= max_gap_duration:

                # If NaN at the beginning or end, might not be able to do i+2 or i-1
                try:

                    # Grid point variables
                    # Here i is offset by 1 (since i is on a NaN)
                    # and i+1 represents the end of the NaN fragment
                    # so i+n_nan
                    start = i-1  # Article notation = i
                    end = i+n_nan  # Article notation = i+1

                    y_i_minus_1 = signal.iloc[start-1]
                    y_i = signal.iloc[start]
                    y_i_plus_1 = signal.iloc[end]
                    y_i_plus_2 = signal.iloc[end+1]

                    # If we don't have at least 2 non-NaNs before and
                    # after the window, we do not interpolate
                    if np.isnan(y_i_minus_1) or np.isnan(y_i_plus_1) or np.isnan(y_i_plus_2):
                        print(
                            '\tValid data is not available for at least two samples on each side of the fragment\n')
                        interpolated += [np.nan for r in range(n_nan)]

                    # Otherwise we interpolate
                    else:
                        t_i_minus_1 = t_array.iloc[start-1]
                        t_i = t_array.iloc[start]
                        t_i_plus_1 = t_array.iloc[end]
                        t_i_plus_2 = t_array.iloc[end+1]

                        n_step = (t_i_plus_1 - t_i)//0.005
                        t = np.linspace(t_i, t_i_plus_1, int(
                            n_step), endpoint=False)

                        # if verbose:
                        #     print('Interpolation parameters:')
                        #     print(f'\t[*] y_i_minus_1 = {y_i_minus_1}, y_i = {y_i}')
                        #     print(f'\t    y_i_plus_1 = {y_i_plus_1}, y_i_plus_2 = {y_i_plus_2}')
                        #     print(f'\t[*] t_i_minus_1 = {t_i_minus_1}, t_i = {t_i}')
                        #     print(f'\t    t_i_plus_1 = {t_i_plus_1}, t_i_plus_2 = {t_i_plus_2}')
                        #     print(f'\t[*] n_step = {n_step}, t_start = {t[0]}, t_end = {t[-1]}\n')

                        interp = interpolate(t, t_i_minus_1, t_i, t_i_plus_1, t_i_plus_2,
                                             y_i_minus_1, y_i, y_i_plus_1, y_i_plus_2)

                        interpolated += interp.tolist()

                        if verbose:
                            print('\t[*] Interpolation done\n')

                except Exception as e:
                    print(f'Error: {e}\n')
                    interpolated += [np.nan for r in range(n_nan)]

            # If the gap is too big, simply report the NaNs
            else:
                interpolated += [np.nan for r in range(n_nan)]

                if verbose:
                    print('\tGap is too long\n')

            i += n_nan

        else:
            interpolated.append(signal.iloc[i])

            i += 1

    return interpolated


def interpolation_pipeline(df, max_gap_duration, type, verbose):
    """
    Run Steffan's interpolation for all pairs of (x, y) signals in the DataFrame.

    The input DataFrame should contain:
        - 't'; timestamps
        - 'x_l': x position of the left eye
        - 'y_l': y position of the left eye
        - 'x_r': x position of the right eye
        - 'y_l': y position of the right eye
        - 'x_avg': averaged x position of both eyes
        - 'y_avg': averaged y position of both eyes
    """

    if type == 'binocular':
        data = {
            't': df['t'],
            'x_l': 0,
            'y_l': 0,
            'x_r': 0,
            'y_r': 0,
            'x_avg': 0,
            'y_avg': 0
        }

    else:
        data = {
            't': df['t'],
            'x_l': 0,
            'y_l': 0,
            'x_r': 0,
            'y_r': 0
        }

    for label, signal in df.iloc[:, 1:].iteritems():

        interpolated = fill_missing_data(df['t'], signal,
                                         max_gap_duration,
                                         verbose)
        data[label] = interpolated

    return pd.DataFrame(data=data)
