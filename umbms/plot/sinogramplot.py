"""
Tyson Reimer
University of Manitoba
June 27th, 2019
"""

import matplotlib.pyplot as plt
import numpy as np

###############################################################################

# Conversion factor from [GHz] to [Hz]
__GHz_to_Hz = 1e9

###############################################################################


def plot_sino(td_data, ini_t, fin_t, title='',
              save_fig=False, save_str='', normalize=False,
              normalize_values=(0, 1), cbar_fmt='%.3f', cmap='inferno',
              transparent=True, dpi=600, close_save=True):
    """Plots a time-domain sinogram

    Displays a sinogram in the time domain.

    Parameters
    ----------
    td_data : array_like
        Raw data in the time-domain.
    ini_t : float
        The initial time-point in the time-domain, in seconds
    fin_t : float
        The final time-point in the time-domain, in seconds
    title : str
        The title used for displaying the raw data
    save_fig : bool
        If true, will save the figure to a .png file - default is False
    save_str : str
        The title of the .png file to be saved if save_image is true -
        default is empty str, triggering the save_string to be set to
        the title
    normalize : bool
        If set to True, will normalize the colorbar intensity to the
        specified value (default is to the maximum value in the image)
    normalize_values : tuple
        Default set to (0, 1) will cause the colorbar to be normalized
        to the maximum value in the image, that is, the image values
        will be scaled to have maximum 1. If set to any other value,
        the image values will not change, but the colorbar will be
        scaled to have (min, max) values of the two values in the tuple
    cbar_fmt : str
        The format specifier to be used for the colorbar, default
        is '%.3f'
    cmap : str
        The colormap that will be used to display the sinogram
    transparent : bool
        If True, will save the figure when True, else will save
        with default white background
    dpi : int
        The DPI to use if saving the figure
    close_save : bool
        If True, will close the figure after saving it.
    """

    td_data = np.abs(td_data)

    n_time_pts = np.size(td_data, axis=1)

    # Find the vector of temporal points in the time domain used in the
    # scan
    scan_times = np.linspace(ini_t, fin_t, n_time_pts)

    # Declare the extent for the plot, along x-axis from antenna
    # angle from 0 deg to 355 deg, along y-axis from user specified
    # points in time-domain
    plot_extent = [0, 355, scan_times[-1] * 1e9, scan_times[0] * 1e9]

    # Determine the aspect ratio for the plot to make it have equal axes
    plot_aspect_ratio = 355 / (scan_times[-1] * 1e9)

    # Make the figure for displaying the reconstruction
    plt.figure()

    # Declare the default font for our figure to be Times New Roman
    plt.rc('font', family='Times New Roman')

    # If the user wanted to normalize the data, make our imshow() and
    # set the colorbar() bounds using vmin, vmax
    if normalize:

        # Assert that the normalize values are of the form
        # (cbar_min, cbar_max)
        assert len(normalize_values) == 2, \
            'Normalize values must be 2-element tuple of form ' \
            '(cbar_min, cbar_max)'
        assert normalize_values[1] > normalize_values[0], \
            'Normalize values must be 2-element tuple of the form ' \
            '(cbar_min, cbar_max)'

        # If the normalize value is default, indicating to normalize to
        # the maximum value in the reconstruction
        if normalize_values == (0, 1):

            # Map values in reconstructed image to be over range (0, 1)
            td_data -= np.min(td_data)
            td_data /= np.max(td_data)

        # Plot the data, using the using the normalize_values as
        # the colorbar bounds
        plt.imshow(td_data, cmap=cmap, extent=plot_extent,
                   vmin=normalize_values[0], vmax=normalize_values[1],
                   aspect=plot_aspect_ratio)

    # If user did not want to normalize the data, do not set
    # the colorbar() bounds
    else:
        plt.imshow(td_data, cmap=cmap, extent=plot_extent,
                   aspect=plot_aspect_ratio)

    # Set the size for the x,y ticks and set which ticks to display
    plt.tick_params(labelsize=14)
    scan_times *= 1e9
    plt.gca().set_yticks([round(ii, 2)
                          for ii in scan_times[::n_time_pts // 8]])
    plt.gca().set_xticks([round(ii)
                          for ii in np.linspace(0, 355, 355)[::75]])

    # Create the colorbar and set the colorbar tick size, also set
    # the format specifier to be as entered by user - default is '%.3f'
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Label the plot axes and assign the plot a title
    plt.title(title, fontsize=20)
    plt.xlabel('Rotational Position (' + r'$^\circ$' + ')', fontsize=16)
    plt.ylabel('Time of Response (ns)', fontsize=16)
    plt.tight_layout()

    # If the user set save_data to True, and therefore wanted to save
    # the figure as a png file
    if save_fig:

        # If the user did not specify a save string, then set the
        # save string to be the title string, without spaces
        if not save_str:
            # Define a string for saving the figure, replace any spaces
            # in the title with underscores
            save_str = title.replace(' ', '_') + '.png'

        # If the user did specify a save string, then add '.png' file
        # extension to it for saving purposes
        else:
            save_str += '.png'

        # Save the figure to a png file
        plt.savefig(save_str, transparent=transparent, dpi=dpi)
        if close_save:
            plt.close()

    else:  # If not saving the figure, then show the figure
        plt.show()
