"""
Tyson Reimer
University of Manitoba
January 24th, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from umbms import get_proj_path, verify_path

from umbms.loadsave import load_pickle

from umbms.beamform.backproj import back_proj
from umbms.beamform.fwdproj import fwd_proj
from umbms.beamform.extras import get_pix_dists_angs, apply_ant_t_delay
from umbms.beamform.propspeed import estimate_speed

###############################################################################

# Directory time-domain sinograms are located
__DATA_DIR = os.path.join(get_proj_path(), 'data/')

# Directory to which the figure will be saved
__FIG_OUT_DIR = os.path.join(get_proj_path(), 'output/paper-figs/')
verify_path(__FIG_OUT_DIR)

###############################################################################


# Duplicate the itDAS function, but cause it to return the forward
# projections at each iteration to facilitate the plot


def itdas(td_data, ini_t, fin_t, ant_rad, speed, m_size, ini_ant_ang=-130.0,
          use_dmas=False, n_iters=6):
    """Reconstructs an image of the return_td using the itDAS algorithm

    Uses the itDAS reconstruction algorithm to reconstruct an image,
    using return_td that was obtained via the inverse chirp z-transform
    (ICZT).

    Parameters
    ----------
    td_data : array_like
        The time-domain data obtained from an experimental scan that
        will be
        used to reconstruct an image of the scanned object
    ini_t : float
        The initial time-point used in the sinogram (typically 0 s),
        in seconds
    fin_t : float
        The final time-point used in the sinogram (typically 6e-9 s),
        in seconds
    ant_rad : float
        The radius of the antenna trajectory during the scan (as
        measured from the black-line on the antenna stand), in meters
    speed : float
        The estimated propagation speed of the signal in the scan, in
        meters/second
    m_size : int
        The number of pixels along one dimension to be used when
        reconstructing an image
    ini_ant_ang : float
        The angle off of the positive x-axis of the *initial* antenna
        position during the scan
    use_dmas : bool
        If True, uses the DMAS beamformer as the back-projection
        operation
    n_iters : int
        The number of iterations that will be used to reconstruct the
        image

    Returns
    -------
    img : array_like
        A 2D reconstructed image of the object space
    """

    n_time_pts, n_ant_pos = td_data.shape

    # Calculate these arrays once to save computation time
    pix_dists, pix_angs = get_pix_dists_angs(m_size, n_ant_pos, ant_rad,
                                             ini_ant_ang=ini_ant_ang)

    # Initialize image estimate to be a homogeneous map
    img = 0.001 * np.ones([m_size, m_size])

    # Get the back-projection of a unity matrix in the data domain, for
    # normalization purposes
    unity_back_proj = back_proj(np.ones_like(td_data), pix_angs,
                                pix_dists, speed, ini_t, fin_t,
                                use_dmas=use_dmas)

    # Get the forward-projection of a unity matrix in the image-speace,
    # for normalization purposes
    unity_fwd_proj = fwd_proj(np.ones_like(img), pix_angs, pix_dists,
                              speed, ini_t, fin_t, n_time_pts, ant_rad)

    fwd_projs_to_plot = np.zeros([n_iters,
                                  np.size(unity_fwd_proj, axis=0),
                                  np.size(unity_fwd_proj, axis=1)],
                                 dtype=complex)

    # Iterate until the set number of iterations has elapsed
    for iter_idx in range(n_iters):

        # Compute the forward projection of the current image estimate
        fwd_proj_img = fwd_proj(img, pix_angs, pix_dists, speed,
                                ini_t, fin_t, n_time_pts, ant_rad)

        # Normalize the forward projection
        fwd_proj_img /= unity_fwd_proj
        fwd_proj_img[np.isnan(fwd_proj_img)] = \
            np.min(fwd_proj_img[~np.isnan(fwd_proj_img)])

        fwd_proj_to_plot = fwd_proj_img * np.ones_like(fwd_proj_img)
        fwd_projs_to_plot[iter_idx, :, :] = fwd_proj_to_plot

        # Compute the ratio between the experimental data and this
        # forward projection
        fwd_proj_ratio = td_data / fwd_proj_img

        # Back project this ratio into the object space
        back_proj_ratio = back_proj(fwd_proj_ratio, pix_angs, pix_dists,
                                    speed, ini_t, fin_t, use_dmas=use_dmas)

        # Normalize the back projection
        back_proj_ratio /= unity_back_proj

        # Update the image estimate by multiplying it with this
        # back-projected ratio
        img *= back_proj_ratio

    return img, fwd_projs_to_plot


def normalize_sino(sino):
    """Normalizes a sinogram to have a max of unity

    Parameters
    ----------
    sino : array_like
        The time-domain sinogram

    Returns
    -------
    norm_sino : array_like
        The normalized sinogram
    """

    norm_sino = np.abs(sino)

    norm_sino /= np.max(norm_sino)

    return norm_sino


###############################################################################


def make_fwd_proj_fig(save_fig=False):
    """Makes the fwd-projection figure from the paper

    Parameters
    ----------
    save_fig : bool
        If True, saves the figure as a .png
    """

    # Load the sinogram data and geometry parameters of each scan
    sinograms = load_pickle(os.path.join(__DATA_DIR, 'td_cal_data.pickle'))
    geom_params = load_pickle(os.path.join(__DATA_DIR, 'geom_params.pickle'))

    # The ID of the experiment for which the figure will be made
    tar_id = 'c1mf3cm'

    # Get the geometry parameters for the target experiment
    tum_x, tum_y, tum_rad, adi_rad, ant_rad = geom_params[tar_id]
    ant_rad = apply_ant_t_delay(ant_rad)  # Apply the antenna time delay

    # Estimate the average propagation speed
    speed = estimate_speed(adi_rad=adi_rad, ant_rad=ant_rad)

    # Get the target sinogram
    tar_sino = sinograms[tar_id]

    # Get the forward projections at each of the first six iterations
    _, fwd_projs = itdas(tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
                         m_size=500, speed=speed, ini_ant_ang=-102.5,
                         n_iters=6)

    # Init the minimum observed value out of any of the sinograms
    min_val = np.min(normalize_sino(tar_sino))

    # For the forward projection at each iteration
    for ii in range(6):

        # Normalize the forward projection
        fwd_projs[ii, :, :] = normalize_sino(fwd_projs[ii, :, :])

        # If this sinogram has the smallest observed value
        if np.min(fwd_projs[ii, :, :]) < min_val:
            min_val = np.min(fwd_projs[ii, :, :])

    # The time-points used to represent the sinograms
    scan_ts = np.linspace(0, 6, 700)

    # Define the plot extent and aspect ratio
    plot_extent = [0, 355, scan_ts[-1], scan_ts[0]]
    plot_aspect_ratio = 355 / (scan_ts[-1])

    fig = plt.figure(figsize=(12, 6))  # Init fig window

    plt.rc('font', family='Times New Roman')  # Set font

    # Find x/y-tick locations
    ytick_locs = [round(ii, 2) for ii in scan_ts[::700 // 8]]
    xtick_locs = [round(ii) for ii in np.linspace(0, 355, 355)[::75]]

    gridspec.GridSpec(3, 4)  # Make the subplot grid

    # Plot the measured sinogram
    plt.subplot2grid((3, 4), (0, 0), colspan=2, rowspan=3)
    plt.tick_params(labelsize=16)
    sino_to_plt = normalize_sino(tar_sino)
    plt.imshow(sino_to_plt, cmap='inferno', extent=plot_extent,
               aspect=plot_aspect_ratio, vmax=1)
    plt.gca().set_yticks(ytick_locs)
    plt.gca().set_xticks(xtick_locs)
    plt.xlabel('Rotational Position (' + r'$^\circ$' + ')', fontsize=20)
    plt.ylabel('Time of Response (ns)', fontsize=20)
    plt.title('Measured Sinogram', fontsize=24)
    plt.tight_layout()

    # Plot the forward projection at the 1st iteration
    plt.subplot2grid((3, 4), (0, 2))
    sino_to_plt = normalize_sino(fwd_projs[0, :])
    plt.imshow(sino_to_plt, cmap='inferno',
               extent=plot_extent, aspect=plot_aspect_ratio, vmin=min_val,
               vmax=1)
    plt.xticks(xtick_locs, ['' for ii in range(len(xtick_locs))])
    plt.yticks(ytick_locs, ['' for ii in range(len(ytick_locs))])
    plt.tight_layout()
    plt.title('Iteration %d' % 1)

    # Plot the forward projection at the 2nd iteration
    plt.subplot2grid((3, 4), (0, 3))
    sino_to_plt = normalize_sino(fwd_projs[1, :])
    plt.imshow(sino_to_plt, cmap='inferno',
               extent=plot_extent, aspect=plot_aspect_ratio, vmin=min_val,
               vmax=1)
    plt.xticks(xtick_locs, ['' for ii in range(len(xtick_locs))])
    plt.yticks(ytick_locs, ['' for ii in range(len(ytick_locs))])
    plt.tight_layout()
    plt.title('Iteration %d' % 2)

    # Plot the forward projection at the 3rd iteration
    plt.subplot2grid((3, 4), (1, 2))
    sino_to_plt = normalize_sino(fwd_projs[2, :])
    plt.imshow(sino_to_plt, cmap='inferno',
               extent=plot_extent, aspect=plot_aspect_ratio, vmin=min_val,
               vmax=1)
    plt.xticks(xtick_locs, ['' for ii in range(len(xtick_locs))])
    plt.yticks(ytick_locs, ['' for ii in range(len(ytick_locs))])
    plt.tight_layout()
    plt.title('Iteration %d' % 3)

    # Plot the forward projection at the 4th iteration
    plt.subplot2grid((3, 4), (1, 3))
    sino_to_plt = normalize_sino(fwd_projs[3, :])
    plt.imshow(sino_to_plt, cmap='inferno',
               extent=plot_extent, aspect=plot_aspect_ratio, vmin=min_val,
               vmax=1)
    plt.xticks(xtick_locs, ['' for ii in range(len(xtick_locs))])
    plt.yticks(ytick_locs, ['' for ii in range(len(ytick_locs))])
    plt.tight_layout()
    plt.title('Iteration %d' % 4)

    # Plot the forward projection at the 5th iteration
    plt.subplot2grid((3, 4), (2, 2))
    sino_to_plt = normalize_sino(fwd_projs[4, :])
    plt.imshow(sino_to_plt, cmap='inferno',
               extent=plot_extent, aspect=plot_aspect_ratio, vmin=min_val,
               vmax=1)
    plt.xticks(xtick_locs, ['' for ii in range(len(xtick_locs))])
    plt.yticks(ytick_locs, ['' for ii in range(len(ytick_locs))])
    plt.tight_layout()
    plt.title('Iteration %d' % 5)

    # Plot the forward projection at the 6th iteration
    plt.subplot2grid((3, 4), (2, 3))
    sino_to_plt = normalize_sino(fwd_projs[5, :])
    last_img = plt.imshow(sino_to_plt, cmap='inferno', extent=plot_extent,
                          aspect=plot_aspect_ratio, vmin=min_val,
                          vmax=1)
    plt.xticks(xtick_locs, ['' for ii in range(len(xtick_locs))])
    plt.yticks(ytick_locs, ['' for ii in range(len(ytick_locs))])
    plt.tight_layout()
    plt.title('Iteration %d' % 6)

    # Adjust the fig to make room for the cbar
    fig.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8)
    cax = fig.add_axes([0.83, 0.1, 0.02, 0.8])  # Add cbar axes
    cbar = fig.colorbar(last_img, cax=cax, cmap='inferno')  # Add cbar
    cbar.ax.tick_params(labelsize=18)  # Edit cbar tick label size

    if save_fig:
        plt.savefig(os.path.join(__FIG_OUT_DIR, 'fwd_proj_fig.png'),
                    transparent=False, dpi=300, bbox_inches='tight')


###############################################################################


if __name__ == "__main__":

    make_fwd_proj_fig(save_fig=True)
