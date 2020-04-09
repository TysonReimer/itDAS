"""
Tyson Reimer
University of Manitoba
June 18, 2019
"""

import numpy as np
import matplotlib.pyplot as plt

import umbms.beamform.breastmodels as breastmodels

###############################################################################

# Conversion factor from [m] to [cm]
__M_to_CM = 100

# Conversion factor from [GHz] to [Hz]
__GHz_to_Hz = 1e9

###############################################################################


def plot_img(img, tum_x=0.0, tum_y=0.0, tum_rad=0.0, adi_rad=0.0, ant_rad=0.0,
             save_str='', save_fig=False, cmap='inferno', max_val=1.0,
             title='', normalize=True, crop_img=True, cbar_fmt='%.1f',
             norm_cbar=False, cbar_max=1.0, transparent=True, dpi=300,
             save_close=True):
    """Displays a reconstruction, making a publication-ready figure

    Parameters
    ----------
    img : array_like
        The 2D reconstruction that will be displayed
    tum_x : float
        The x-position of the tumor in the scan, in meters
    tum_y : float
        The y-position of the tumor in the scan, in meters
    tum_rad : float
        The radius of the tumor in the scan, in meters
    adi_rad : float
        The approximate radius of the breast in the scan, in meters
    ant_rad : float
        The corrected antenna radius during the scan (corrected for
        antenna time-delay)
    save_str : str, optional
        The complete path for saving the figure as a .png - only used if
        save_fig is True
    save_fig : bool
        If True, will save the displayed image to the location specified
        in save_str
    cmap : str
        Specifies the colormap that will be used when displaying the
        image
    max_val : float
        The maximum intensity to display in the image
    title : str
        The title for the plot
    normalize : bool
        If True, will normalize the image to have the maximum value
        max_val
    crop_img : bool
        If True, will set the values in the image to NaN outside of the
        inner antenna trajectory
    cbar_fmt : str
        The format for the tick labels on the colorbar
    norm_cbar : bool
        If True, will normalize the colorbar to have min value 0,
        max value of cbar_max
    cbar_max : float
        If norm_cbar, this will be the maximum value of the cbar
    transparent : bool
        If True, will save the image with a transparent background
        (i.e., whitespace will be transparent)
    dpi : int
        The DPI to be used if saving the image
    save_close : bool
        If True, closes the figure after saving
    """

    ant_rad *= 100  # Convert from m to cm to facilitate plot

    # If cropping the image at the antenna-trajectory boundary
    if crop_img:

        temp_val = (ant_rad - 14.8) / 0.97 + 10.6

        # Find the region inside the antenna trajectory
        in_ant_trajectory = breastmodels.get_roi(temp_val - 10,
                                                 np.size(img, axis=0), ant_rad)

        # Set the pixels outside of the antenna trajectory to NaN
        img[np.logical_not(in_ant_trajectory)] = np.NaN

    # Define angles for plot the tissue geometry
    draw_angs = np.linspace(0, 2 * np.pi, 1000)

    if normalize:  # If normalizing the image

        # If the max value isn't set to unity, then scale the image so
        # that intensity is between 0 and max_val
        if max_val != 1.0:
            img = img - np.min(img[np.logical_not(np.isnan(img))])
            img = img / max_val

        # If the max value is 1.0, scale so that the intensity is
        # between 0 and 1.0
        else:
            img = img - np.min(img[np.logical_not(np.isnan(img))])
            img = img / np.max(img[np.logical_not(np.isnan(img))])

    # Rotate and flip img to match proper x/y axes labels
    img_to_plot = (img**2).T

    temp_val = (ant_rad - 14.8) / 0.97 + 10.6
    ant_xs, ant_ys = ((temp_val - 10) * np.cos(draw_angs),
                      (temp_val - 10) * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate breast outline
    breast_xs, breast_ys = (adi_rad * 100 * np.cos(draw_angs),
                            adi_rad * 100 * np.sin(draw_angs))

    # Define the x/y coordinates of the approximate tumor outline
    tum_xs, tum_ys = (tum_rad * 100 * np.cos(draw_angs) + tum_x * 100,
                      tum_rad * 100 * np.sin(draw_angs) + tum_y * 100)

    tick_bounds = [-ant_rad, ant_rad, -ant_rad, ant_rad]

    # Set the font to times new roman
    plt.rc('font', family='Times New Roman')
    plt.figure()  # Make the figure window

    # If normalizing the image, ensure the cbar axis extends from 0 to 1
    if normalize:
        plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds,
                   aspect='equal', vmin=0, vmax=1.0)
    elif norm_cbar:
        plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds,
                   aspect='equal', vmin=0, vmax=cbar_max)
    else:  # If nor normalizing the image
        plt.imshow(img_to_plot, cmap=cmap, extent=tick_bounds, aspect='equal')

    # Set the size of the axis tick labels
    plt.tick_params(labelsize=14)

    # Set the x/y-ticks at multiples of 5 cm
    plt.gca().set_xticks([-10, -5, 0, 5, 10])
    plt.gca().set_yticks([-10, -5, 0, 5, 10])

    # Specify the colorbar tick format and size
    plt.colorbar(format=cbar_fmt).ax.tick_params(labelsize=14)

    # Set the x/y axes limits
    plt.xlim([-temp_val + 10, temp_val - 10])
    plt.ylim([-temp_val + 10, temp_val - 10])

    plt.plot(ant_xs, ant_ys, 'k--', linewidth=2.5)
    plt.plot(breast_xs, breast_ys, 'w--', linewidth=2)

    # Plot the approximate tumor boundary
    plt.plot(tum_xs, tum_ys, 'g', linewidth=1.5)

    plt.title(title, fontsize=20)  # Make the plot title
    plt.xlabel('x-axis (cm)', fontsize=16)  # Make the x-axis label
    plt.ylabel('y-axis (cm)', fontsize=16)  # Make the y-axis label
    plt.tight_layout()  # Remove excess whitespace in the figure

    # If saving the image, save it to the save_str path and close it
    if save_fig:
        plt.savefig(save_str, transparent=transparent, dpi=dpi,
                    bbox_inches='tight')

        if save_close:  # If wanting to close the fig after saving
            plt.close()
