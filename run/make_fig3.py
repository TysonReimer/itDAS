"""
Tyson Reimer
University of Manitoba
December 21, 2018
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from umbms import get_proj_path, verify_path

from umbms.loadsave import load_pickle
from umbms.beamform.extras import apply_ant_t_delay
from umbms.beamform.breastmodels import get_roi

###############################################################################

# Directory where geom_params are stored
__GEOM_PARAM_DIR = os.path.join(get_proj_path(), 'data/')

# Directory where .png will be output
output_dir = os.path.join(get_proj_path(), 'output/paper-figs/')
verify_path(output_dir)

__M_SIZE = 500  # Model size

# Load the geometry parameters
geom_params = load_pickle(os.path.join(__GEOM_PARAM_DIR, 'geom_params.pickle'))

###############################################################################


def import_recon_imgs():
    """Imports the recon images and makes dict to facilitate fig

    Returns
    -------
    all_fig_imgs : dict
        Dict containing the reconstructions of each phantom experiment
        to be included in Fig. 2 in the paper, by each recon method
    """

    # Directory where reconstructed images are stored
    img_data_dir = os.path.join(get_proj_path(), 'output/recon-imgs/')

    # Load reconstructed images by each beamformer
    das_imgs = load_pickle(os.path.join(img_data_dir, 'das_imgs.pickle'))
    dmas_imgs = load_pickle(os.path.join(img_data_dir, 'dmas_imgs.pickle'))
    itdas_imgs = load_pickle(os.path.join(img_data_dir, 'itdas_imgs.pickle'))
    itdmas_imgs = load_pickle(os.path.join(img_data_dir, 'itdmas_imgs.pickle'))

    all_fig_imgs = dict()  # Init dict to return

    # Get the DAS images of interest
    all_fig_imgs['DAS_C1'] = das_imgs['c1mf3cm']
    all_fig_imgs['DAS_C2'] = das_imgs['c2lf2cm']
    all_fig_imgs['DAS_C3'] = das_imgs['c3mf3cm']
    all_fig_imgs['DAS_C4'] = das_imgs['c4sf3cm']

    # Get the DMAS images of interest
    all_fig_imgs['DMAS_C1'] = dmas_imgs['c1mf3cm']
    all_fig_imgs['DMAS_C2'] = dmas_imgs['c2lf2cm']
    all_fig_imgs['DMAS_C3'] = dmas_imgs['c3mf3cm']
    all_fig_imgs['DMAS_C4'] = dmas_imgs['c4sf3cm']

    # Get the itDAS images of interest
    all_fig_imgs['itDAS_C1'] = itdas_imgs['c1mf3cm']
    all_fig_imgs['itDAS_C2'] = itdas_imgs['c2lf2cm']
    all_fig_imgs['itDAS_C3'] = itdas_imgs['c3mf3cm']
    all_fig_imgs['itDAS_C4'] = itdas_imgs['c4sf3cm']

    # Get the itDMAS images of interest
    all_fig_imgs['itDMAS_C1'] = itdmas_imgs['c1mf3cm']
    all_fig_imgs['itDMAS_C2'] = itdmas_imgs['c2lf2cm']
    all_fig_imgs['itDMAS_C3'] = itdmas_imgs['c3mf3cm']
    all_fig_imgs['itDMAS_C4'] = itdmas_imgs['c4sf3cm']

    return all_fig_imgs


def get_big_fig_imgs():
    """Gets the images and normalizes them to facilitate making the fig

    Returns
    -------
    algo_images : dict
        Dict containing the images to be used in the figure
    tar_geom_params : dict
        Dict containing the geometry parameters of the images
        used in the figure
    """

    tar_geom_params = dict()  # Init dict

    # Get the target geometry parameters
    tar_geom_params['C1'] = geom_params['c1mf3cm']
    tar_geom_params['C2'] = geom_params['c2lf2cm']
    tar_geom_params['C3'] = geom_params['c3mf3cm']
    tar_geom_params['C4'] = geom_params['c4sf3cm']

    algo_images = import_recon_imgs()  # Load the reconstructions

    for img_key in algo_images.keys():  # For each target image

        # Get the scan radius
        _, _, _, _, scan_rad = tar_geom_params[img_key[-2:]]
        true_ant_rad = apply_ant_t_delay(scan_rad)  # Apply time delay

        # Get ROI to remove responses outside of antenna region
        roi = get_roi(roi_rad=scan_rad-0.10,
                      ant_rad=true_ant_rad, m_size=__M_SIZE)

        img_here = algo_images[img_key]  # Get this image
        img_here[~roi] = np.NaN  # Apply window

        # Normalize the image to have a min of 0 and a max of unity
        img_here -= np.min(img_here[~np.isnan(img_here)])
        img_here /= np.max(img_here[~np.isnan(img_here)])

        algo_images[img_key] = img_here  # Store the image

    return algo_images, tar_geom_params


def plot_one_img_for_big_fig(ax, imgs, dists, expt_id):
    """Plots one image, for use in making the big fig (Fig. 2)

    Plots one of the 16 images used in Fig. 2.

    Parameters
    ----------
    ax : axes object
        Axes on which the image will be plotted
    imgs : dict
        Dict containing all the images to be plotted
    dists : dict
        Dict containing the geometry parameters of the images to be
        plotted
    expt_id : str
        The expt ID of the image to be plotted

    Returns
    -------
    this_img : imshow object
        The plotted image on the axes
    """

    # Define angles for drawing tissue components and antennas
    draw_angles = np.linspace(0, 2 * np.pi, 1000)

    # Get geometry parameters and image for this expt_id
    tum_x, tum_y, tum_rad, breast_rad, scan_rad = dists[expt_id[-2:]]
    img = imgs[expt_id]

    # Account for the time-delay introduced by the antenna and
    # convert from m to cm
    true_ant_rad = 100 * apply_ant_t_delay(scan_rad)

    scan_rad *= 100  # Store un-corrected ant_rad

    # Get the x/y positions of the trajectory of the inner-edge of the
    # antenna during the scan
    antenna_xs, antenna_ys = ((scan_rad - 10) * np.cos(draw_angles),
                              (scan_rad - 10) * np.sin(draw_angles))

    # Get the x/y positions of the breast outline
    breast_xs, breast_ys = (breast_rad * 100 * np.cos(draw_angles),
                            breast_rad * 100 * np.sin(draw_angles))

    # Get the x/y positions of the tumour outline
    tum_xs, tum_ys = (tum_rad * 100 * np.cos(draw_angles) + tum_x * 100,
                      tum_rad * 100 * np.sin(draw_angles) + tum_y * 100)

    # Get the tick bounds for plotting the image
    tick_bounds = [-true_ant_rad, true_ant_rad, -true_ant_rad, true_ant_rad]

    ax.set_adjustable('box')  # Make plot square

    # Plot the image
    this_img = ax.imshow(img.T**2, cmap='inferno', extent=tick_bounds,
                         aspect='equal', vmin=0, vmax=1.0)

    # Set the axes limits
    ax.set_xlim([-scan_rad + 10, scan_rad - 10])
    ax.set_ylim([-scan_rad + 10, scan_rad - 10])

    # Plot antenna trajectory, breast outline, tumour outline
    ax.plot(antenna_xs, antenna_ys, 'k--', linewidth=1)
    ax.plot(breast_xs, breast_ys, 'w--', linewidth=1)
    ax.plot(tum_xs, tum_ys, 'g', linewidth=0.5)

    return this_img


def make_big_fig(save_fig=False, save_str=''):
    """Makes Fig. 2 from the itDAS paper

    Parameters
    ----------
    save_fig : bool
        If True, will save the fig as a .png
    save_str : str
        The full path of the .png file that will be saved, if save_fig
    """

    # Get the images and geometry parameters for each image to be
    # ploted
    imgs, dists = get_big_fig_imgs()

    # Init the plot
    plt.rc('font', family='Times New Roman')
    font = FontProperties()
    font.set_family('serif')
    font.set_name('Times New Roman')

    # Get the figure and axes objects
    fig, axes = plt.subplots(4, 4, figsize=(7, 8))

    # Define alphabet for labeling each plot as if it was a sub-fig
    alphabet = 'abcdefghijklmnopqrstuvwxyz'

    # Label each column in the fig
    axes[0, 0].text(0.5, 1.1, 'DAS', transform=axes[0, 0].transAxes,
                    horizontalalignment='center', fontsize=12)
    axes[0, 1].text(0.5, 1.1, 'DMAS', transform=axes[0, 1].transAxes,
                    horizontalalignment='center', fontsize=12)
    axes[0, 2].text(0.5, 1.1, 'itDAS', transform=axes[0, 2].transAxes,
                    horizontalalignment='center', fontsize=12)
    axes[0, 3].text(0.5, 1.1, 'itDMAS', transform=axes[0, 3].transAxes,
                    horizontalalignment='center', fontsize=12)

    # Label each row in the fig
    axes[0, 0].text(-0.25, 0.5, 'Class I\nRecontsructions',
                    transform=axes[0, 0].transAxes,
                    horizontalalignment='center',
                    fontsize=12, verticalalignment='center',
                    rotation='vertical')
    axes[1, 0].text(-0.25, 0.5, 'Class II\nRecontsructions',
                    transform=axes[1, 0].transAxes,
                    horizontalalignment='center',
                    fontsize=12, verticalalignment='center',
                    rotation='vertical')
    axes[2, 0].text(-0.25, 0.5, 'Class III\nRecontsructions',
                    transform=axes[2, 0].transAxes,
                    horizontalalignment='center',
                    fontsize=12, verticalalignment='center',
                    rotation='vertical')
    axes[3, 0].text(-0.25, 0.5, 'Class IV\nRecontsructions',
                    transform=axes[3, 0].transAxes,
                    horizontalalignment='center',
                    fontsize=12, verticalalignment='center',
                    rotation='vertical')

    # Adjust to make space for the sub-fig labels
    fig.subplots_adjust(hspace=0.75, wspace=0.1)

    # Set all plots to have the same x/y-tick label locations,
    # but set all to have no tick labels
    for ii in range(4):
        for jj in range(3):
            axes[ii, jj].set_xticks([-10, -5, 0, 5, 10])
            axes[ii, jj].set_yticks([-10, -5, 0, 5, 10])
            axes[ii, jj].set_xticklabels(['', '', '', '', ''])
            axes[ii, jj].set_yticklabels(['', '', '', '', ''])
            axes[ii, jj].yaxis.tick_right()

    # Set the right-most plot in each row to have tick labels
    for ii in range(4):
        axes[ii, 3].yaxis.set_label_position('right')
        axes[ii, 3].set_yticks([-10, -5, 0, 5, 10])
        axes[ii, 3].set_xticks([-10, -5, 0, 5, 10])
        axes[ii, 3].set_yticklabels(['-10', '-5', '0', '5', '10'])
        axes[ii, 3].yaxis.tick_right()
        axes[ii, 3].set_xlabel('x-axis (cm)', fontproperties=font, labelpad=-2)
        axes[ii, 3].set_ylabel('y-axis (cm)', fontproperties=font, labelpad=-1)

    # Label each plot as if it was a sub-fig
    alphabet_counter = 0
    for ii in range(4):
        for jj in range(4):
            axes[ii, jj].text(0.5, -0.5, '(%s)' % alphabet[alphabet_counter],
                              transform=axes[ii, jj].transAxes,
                              horizontalalignment='center')
            alphabet_counter += 1

    # Plot all of the images
    plot_one_img_for_big_fig(axes[0, 0], imgs, dists, 'DAS_C1')
    plot_one_img_for_big_fig(axes[0, 1], imgs, dists, 'DMAS_C1')
    plot_one_img_for_big_fig(axes[0, 2], imgs, dists, 'itDAS_C1')
    plot_one_img_for_big_fig(axes[0, 3], imgs, dists, 'itDMAS_C1')

    plot_one_img_for_big_fig(axes[1, 0], imgs, dists, 'DAS_C2')
    plot_one_img_for_big_fig(axes[1, 1], imgs, dists, 'DMAS_C2')
    plot_one_img_for_big_fig(axes[1, 2], imgs, dists, 'itDAS_C2')
    plot_one_img_for_big_fig(axes[1, 3], imgs, dists, 'itDMAS_C2')

    plot_one_img_for_big_fig(axes[2, 0], imgs, dists, 'DAS_C3')
    plot_one_img_for_big_fig(axes[2, 1], imgs, dists, 'DMAS_C3')
    plot_one_img_for_big_fig(axes[2, 2], imgs, dists, 'itDAS_C3')
    plot_one_img_for_big_fig(axes[2, 3], imgs, dists, 'itDMAS_C3')

    plot_one_img_for_big_fig(axes[3, 0], imgs, dists, 'DAS_C4')
    plot_one_img_for_big_fig(axes[3, 1], imgs, dists, 'DMAS_C4')
    plot_one_img_for_big_fig(axes[3, 2], imgs, dists, 'itDAS_C4')

    # Return the last image for making the colorbar
    my_img = plot_one_img_for_big_fig(axes[3, 3], imgs, dists, 'itDMAS_C4')

    # Make room for the cbar
    fig.subplots_adjust(right=0.85)
    cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])  # Add cbar axes

    # Make the colorbar
    fig.colorbar(my_img, cax=cax, cmap='inferno')
    fig.show()

    if save_fig:  # If saving the fig
        fig.savefig(save_str, transparent=False, dpi=300,
                    bbox_inches='tight')


###############################################################################

if __name__ == '__main__':

    # Make the big figure
    make_big_fig(save_fig=False,
                 save_str=os.path.join(output_dir, 'four_class_recons.png'))
