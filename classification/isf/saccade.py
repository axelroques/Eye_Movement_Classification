
import matplotlib.pyplot as plt
import numpy as np


class Saccade():
    """
    Simple class for saccades.
    Each saccade is characterized by:
        - A start and end timestamp
        - A duration
        - A peak velocity
    """

    def __init__(self, t, x, y):

        self.t = t
        self.x = x
        self.y = y

    def compute_parameters(self, ransac, t, popt_x, popt_y):
        """
        Update/compute saccade parameters using the results from the
        fit.
        """

        # Local Sigmoid function and its derivative
        def S(t, E_0, E_max, t_50, alpha): return E_0+(E_max-E_0) * \
            np.power(t, alpha) / \
            (np.power(t_50, alpha)+np.power(t, alpha))

        def S_p(t, E_0, E_max, t_50, alpha): return (E_max-E_0) * \
            alpha * np.power(t_50, alpha) * \
            np.power(t, alpha-1) / \
            (np.power(t_50, alpha)+np.power(t, alpha))**2

        # Basic parameters
        self.t = ransac.t
        self.x = ransac.x
        self.y = ransac.y
        self.popt_x = popt_x
        self.popt_y = popt_y

        # 'High definition' parameters
        self.t_HD = np.linspace(self.t[0], self.t[-1], 10*len(self.t))
        self.x_HD = S(self.t_HD, *self.popt_x)
        self.y_HD = S(self.t_HD, *self.popt_y)

        # Fit computation
        self.x_fit = S(self.t_HD, *popt_x)
        self.y_fit = S(self.t_HD, *popt_y)

        # Amplitude
        self.min_amplitude_x = self.x_fit[0]
        self.max_amplitude_x = self.x_fit[-1]
        self.amplitude_x = self.max_amplitude_x - self.min_amplitude_x
        self.min_amplitude_y = self.y_fit[0]
        self.max_amplitude_y = self.y_fit[-1]
        self.amplitude_y = self.max_amplitude_y - self.min_amplitude_y

        # Start and end timestamps
        if self.amplitude_x >= 0:
            self.t_start_x = self.t_HD[np.nonzero(
                self.x_fit > self.x_fit[0] + 0.02*self.amplitude_x)[0][0]]
            self.t_end_x = self.t_HD[np.nonzero(
                self.x_fit > self.x_fit[0] + 0.98*self.amplitude_x)[0][0]]
        else:
            self.t_start_x = self.t_HD[np.nonzero(
                self.x_fit < self.x_fit[0] + 0.02*self.amplitude_x)[0][0]]
            self.t_end_x = self.t_HD[np.nonzero(
                self.x_fit < self.x_fit[0] + 0.98*self.amplitude_x)[0][0]]

        if self.amplitude_y >= 0:
            self.t_start_y = self.t_HD[np.nonzero(
                self.y_fit > self.y_fit[0] + 0.02*self.amplitude_y)[0][0]]
            self.t_end_y = self.t_HD[np.nonzero(
                self.y_fit > self.y_fit[0] + 0.98*self.amplitude_y)[0][0]]
        else:
            self.t_start_y = self.t_HD[np.nonzero(
                self.y_fit < self.y_fit[0] + 0.02*self.amplitude_y)[0][0]]
            self.t_end_y = self.t_HD[np.nonzero(
                self.y_fit < self.y_fit[0] + 0.98*self.amplitude_y)[0][0]]

        # Duration
        self.duration_x = self.t_end_x - self.t_start_x
        self.duration_y = self.t_end_y - self.t_start_y

        # Peak velocity
        self.x_p_fit = S_p(self.t_HD, *popt_x)
        self.y_p_fit = S_p(self.t_HD, *popt_y)
        self.i_max_vel_x = np.argmax(np.abs(self.x_p_fit))
        self.peak_velocity_x = np.sign(
            self.amplitude_x)*self.x_p_fit[self.i_max_vel_x]
        self.i_max_vel_y = np.argmax(np.abs(self.y_p_fit))
        self.peak_velocity_y = np.sign(
            self.amplitude_y)*self.y_p_fit[self.i_max_vel_y]

        # Add temporal offset to reflect the true t values
        # (this offset was removed to make the fitting easier)
        self.t += t[0]
        self.t_HD += t[0]
        self.t_start_x += t[0]
        self.t_start_y += t[0]
        self.t_end_x += t[0]
        self.t_end_y += t[0]

        return

    def plot_saccade(self):
        """
        Plots the eye trace vs. the Sigmoid fit.
        """

        _, ax = plt.subplots(figsize=(15, 8))

        # Simple plots
        ax.scatter(self.t, self.x, alpha=0.8, s=40,
                   facecolors='none', edgecolors='royalblue', label='x')
        ax.plot(self.t_HD, self.x_fit, c='royalblue',
                alpha=0.8, label='x fit')
        ax.scatter(self.t, self.y, alpha=0.8, s=40,
                   facecolors='none', edgecolors='crimson', label='y')
        ax.plot(self.t_HD, self.y_fit, c='crimson',
                alpha=0.8, label='y fit')

        # Durations
        ax.axvline(self.t_start_x, ls='--', color='royalblue', alpha=0.5)
        ax.axvline(self.t_end_x, ls='--', color='royalblue', alpha=0.5)
        ax.axvline(self.t_start_y, ls='--', color='crimson', alpha=0.5)
        ax.axvline(self.t_end_y, ls='--', color='crimson', alpha=0.5)

        # Max velocity
        ax.scatter(self.t_HD[self.i_max_vel_x], self.x_fit[self.i_max_vel_x],
                   c='royalblue', alpha=0.8, s=100, marker='D')
        ax.scatter(self.t_HD[self.i_max_vel_y], self.y_fit[self.i_max_vel_y],
                   c='crimson', alpha=0.8, s=100, marker='D')

        ax.set_title('Eye position', fontsize=19)
        ax.set_xlabel('Time', fontsize=15)

        ax.legend(fontsize=15)
        plt.show()

        return
