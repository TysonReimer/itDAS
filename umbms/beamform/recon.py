"""
Tyson Reimer
University of Manitoba
June 4th, 2019
"""

import numpy as np

from umbms.beamform.backproj import back_proj
from umbms.beamform.fwdproj import fwd_proj
from umbms.beamform.extras import get_pix_dists_angs

###############################################################################


def itdas(td_data, ini_t, fin_t, ant_rad, speed, m_size, ini_ant_ang=-130.0,
          use_dmas=False, n_iters=6):
    """Reconstructs an image of the td_data using the itDAS algorithm

    Uses the itDAS reconstruction algorithm to reconstruct an image.

    Parameters
    ----------
    td_data : array_like
        The time-domain data obtained from an experimental scan that
        will be used to reconstruct an image of the scanned object
    ini_t : float
        The initial time-point used in the sinogram (typically 0 s),
        in seconds
    fin_t : float
        The final time-point used in the sinogram (typically 6e-9 s),
        in seconds
    ant_rad : float
        The radius of the antennas trajectory during the scan,
        after accounting for phase-delay using apply_ant_t_delay() func
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
        operator
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

    # Iterate until the set number of iterations has elapsed
    for iter_idx in range(n_iters):

        # Compute the forward projection of the current image estimate
        fwd_proj_img = fwd_proj(img, pix_angs, pix_dists, speed,
                                ini_t, fin_t, n_time_pts, ant_rad)

        # Normalize the forward projection
        fwd_proj_img /= unity_fwd_proj

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

    return img


###############################################################################


def das(td_data, ini_t, fin_t, ant_rad, speed, m_size, ini_ant_ang=-130.0):
    """Reconstructs an image using the delay-and-sum (DAS) beamformer

    Uses the delay-and-sum beamformer to reconstruct an image.

    Parameters
    ----------
    td_data : array_lke
        The time-domain data (sinogram) that will be back-projected into
        the image space
    ini_t : float
        The initial time (s) of the return_td (i.e., the first element
        in each radar signal corresponds to the ini_t of response)
    fin_t : float
        The final time (s) of the return_td (i.e., the last element in
        each radar signal corresponds to the fin_t of response)
    ant_rad : float
                The radius of the antennas trajectory during the scan,
        after accounting for phase-delay using apply_ant_t_delay() func
    speed : float
        The estimated propagation speed of the microwave signal in m/s
    m_size : int
        The number of pixels along each dimension used when forming the
        reconstruction
    ini_ant_ang : float
        The angle off of the positive x-axis of the *initial* antenna
        position during the scan

    Returns
    -------
    img : array_like
        The back-projection, in the image-space, of the return_td
    """

    n_ant_pos = td_data.shape[1]  # Get number of antenna positions

    # Get the distances and angles of all pixels from each antenna
    # position
    pix_dists, pix_angs = get_pix_dists_angs(m_size, n_ant_pos, ant_rad,
                                             ini_ant_ang=ini_ant_ang)

    # Back-project using the DAS beamformer
    img = back_proj(td_data, pix_angs, pix_dists, speed, ini_t, fin_t)

    return img


def dmas(td_data, ini_t, fin_t, ant_rad, speed, m_size, ini_ant_ang=-130.0):
    """Reconstructs an image using the delay-multiply-and-sum beamformer

    Uses the delay-multiply-and-sum beamformer to reconstruct an image.

    Parameters
    ----------
    td_data : array_lke
        The time-domain data (sinogram) that will be back-projected into
        the image space
    ini_t : float
        The initial time (s) of the return_td (i.e., the first element
        in each radar signal corresponds to the ini_t of response)
    fin_t : float
        The final time (s) of the return_td (i.e., the last element in
        each radar signal corresponds to the fin_t of response)
    ant_rad : float
        The radius of the antennas trajectory during the scan,
        after accounting for phase-delay using apply_ant_t_delay() func
    speed : float
        The estimated propagation speed of the microwave signal in m/s
    m_size : int
        The number of pixels along each dimension used when forming the
        reconstruction
    ini_ant_ang : float
        The angle off of the positive x-axis of the *initial* antenna
        position during the scan

    Returns
    -------
    img : array_like
        The back-projection, in the image-space, of the return_td
    """

    n_ant_pos = td_data.shape[1]  # Get the number of antenna positions

    # Get the distances and angles of all pixels from each antenna
    # position
    pix_dists, pix_angs = get_pix_dists_angs(m_size, n_ant_pos, ant_rad,
                                             ini_ant_ang=ini_ant_ang)

    # Back-project using the DMAS beamformer
    img = back_proj(td_data, pix_angs, pix_dists, speed, ini_t, fin_t,
                    use_dmas=True)

    return img
