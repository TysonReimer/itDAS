"""
Tyson Reimer
University of Manitoba
January 31st, 2020
"""

import os
import time
import numpy as np

from umbms import get_proj_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.beamform.recon import das, dmas
from umbms.beamform.propspeed import estimate_speed
from umbms.beamform.extras import apply_ant_t_delay, get_pix_dists_angs
from umbms.beamform.backproj import back_proj
from umbms.beamform.fwdproj import fwd_proj

###############################################################################

# Directory where time-domain sinograms are located
__DATA_DIR = os.path.join(get_proj_path(), 'data/')

__M_SIZE = 500

###############################################################################


# Replicate the itDAS algorithm but modify to measure compute times
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

    setup_start_t = time.time()

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

    # Convert total reconstruction time to minutes and seconds
    total_t = time.time() - setup_start_t
    tot_t_min = total_t // 60
    tot_t_sec = total_t - tot_t_min * 60

    logger.info("\t\t\titDAS Set-up complete in:\t%d min, %.1f sec"
                % (tot_t_min, tot_t_sec))

    # Iterate until the set number of iterations has elapsed
    for iter_idx in range(n_iters):

        fwd_proj_start_t = time.time()

        # Compute the forward projection of the current image estimate
        fwd_proj_img = fwd_proj(img, pix_angs, pix_dists, speed,
                                ini_t, fin_t, n_time_pts, ant_rad)

        # Convert total reconstruction time to minutes and seconds
        total_t = time.time() - fwd_proj_start_t
        tot_t_min = total_t // 60
        tot_t_sec = total_t - tot_t_min * 60

        logger.info("\t\t\titDAS Fwd-proj (iter %d) complete"
                    " in:\t%d min, %.1f sec"
                    % (iter_idx, tot_t_min, tot_t_sec))

        # Normalize the forward projection
        fwd_proj_img /= unity_fwd_proj

        # Compute the ratio between the experimental data and this
        # forward projection
        fwd_proj_ratio = td_data / fwd_proj_img

        back_proj_start_t = time.time()

        # Back project this ratio into the object space
        back_proj_ratio = back_proj(fwd_proj_ratio, pix_angs, pix_dists,
                                    speed, ini_t, fin_t, use_dmas=use_dmas)

        # Convert total reconstruction time to minutes and seconds
        total_t = time.time() - back_proj_start_t
        tot_t_min = total_t // 60
        tot_t_sec = total_t - tot_t_min * 60

        logger.info("\t\t\titDAS Back-proj (iter %d) complete"
                    " in:\t%d min, %.1f sec"
                    % (iter_idx, tot_t_min, tot_t_sec))

        # Normalize the back projection
        back_proj_ratio /= unity_back_proj

        # Update the image estimate by multiplying it with this
        # back-projected ratio
        img *= back_proj_ratio

    return img


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    logger.info("Beginning...Computation Time Comparison...")

    # Load time-domain data and geometry parameters
    td_data = load_pickle(os.path.join(__DATA_DIR, 'td_cal_data.pickle'))
    geom_params = load_pickle(os.path.join(__DATA_DIR, 'geom_params.pickle'))

    tar_expt_id = 'c1mf3cm'  # Target expt ID

    # Get the target sinogram and geometry parameters
    tar_sino = td_data[tar_expt_id]
    tum_x, tum_y, tum_rad, adi_rad, ant_rad = geom_params[tar_expt_id]
    ant_rad = apply_ant_t_delay(ant_rad)  # Correct for time delay

    logger.info("\tUsing Sinogram of %d Time Points"
                % (np.size(tar_sino, axis=0)))
    logger.info("\tReconstructing %d x %d Images"
                % (__M_SIZE, __M_SIZE))

    # Estimate propagation speed
    speed = estimate_speed(adi_rad=adi_rad, ant_rad=ant_rad)

    n_runs = 10

    das_ts = np.zeros([10, ])
    dmas_ts = np.zeros([10, ])
    itdas_ts = np.zeros([10, ])

    for run_idx in range(n_runs):

        logger.info("\tBeginning: DAS Reconstruction...")
        start_t = time.time()

        # Reconstruct image with DAS
        _ = das(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
                speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5)

        # Convert total reconstruction time to minutes and seconds
        total_t = time.time() - start_t
        das_ts[run_idx] = total_t
        tot_t_min = total_t // 60
        tot_t_sec = total_t - tot_t_min * 60

        logger.info("\t\tDAS Reconstruction complete in:\t%d min, %.1f sec"
                    % (tot_t_min, tot_t_sec))

        logger.info("\tBeginning: DMAS Reconstruction...")
        start_t = time.time()

        # Reconstruct with DMAS
        _ = dmas(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
                 speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5)

        # Convert total reconstruction time to minutes and seconds
        total_t = time.time() - start_t
        dmas_ts[run_idx] = total_t
        tot_t_min = total_t // 60
        tot_t_sec = total_t - tot_t_min * 60

        logger.info("\t\tDMAS Reconstruction complete in:\t%d min, %.1f sec"
                    % (tot_t_min, tot_t_sec))

        logger.info("\tBeginning: itDAS Reconstruction...")
        start_t = time.time()

        # Reconstruct with itDAS
        _ = itdas(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
                  speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5,
                  n_iters=6, use_dmas=False)

        # Convert total reconstruction time to minutes and seconds
        total_t = time.time() - start_t
        itdas_ts[run_idx] = total_t
        tot_t_min = total_t // 60
        tot_t_sec = total_t - tot_t_min * 60

        logger.info("\t\titDAS Reconstruction complete in:\t%d min, %.1f sec"
                    % (tot_t_min, tot_t_sec))

        # logger.info("\tBeginning: itDMAS Reconstruction...")
        # start_t = time.time()
        #
        # # Reconstruct with itDMAS
        # _ = itdas(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
        #           speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5,
        #           n_iters=6, use_dmas=True)
        #
        # # Convert total reconstruction time to minutes and seconds
        # total_t = time.time() - start_t
        # tot_t_min = total_t // 60
        # tot_t_sec = total_t - tot_t_min * 60
        #
        # logger.info("\t\titDMAS Reconstruction complete in:\t%d min, %.1f sec"
        #             % (tot_t_min, tot_t_sec))

    logger.info('\tAverage Results over %d runs:' % n_runs)
    logger.info('\t\tDAS:\t%.1f +/- %.1f sec'
                % (np.mean(das_ts), np.std(das_ts)))
    logger.info('\t\tDMAS:\t%.1f +/- %.1f sec'
                % (np.mean(dmas_ts), np.std(dmas_ts)))
    logger.info('\t\titDAS:\t%.1f +/- %.1f sec'
                % (np.mean(itdas_ts), np.std(itdas_ts)))
