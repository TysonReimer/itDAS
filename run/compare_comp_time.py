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

from umbms.beamform.recon import das, dmas, itdas
from umbms.beamform.propspeed import estimate_speed
from umbms.beamform.extras import apply_ant_t_delay

###############################################################################

# Directory where time-domain sinograms are located
__DATA_DIR = os.path.join(get_proj_path(), 'data/')

__M_SIZE = 500

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

    logger.info("\tBeginning: DAS Reconstruction...")
    start_t = time.time()

    # Reconstruct image with DAS
    _ = das(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
            speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5)

    # Convert total reconstruction time to minutes and seconds
    total_t = time.time() - start_t
    tot_t_min = total_t // 60
    tot_t_sec = total_t - tot_t_min * 60
    
    logger.info("\t\tDAS Reconstruction complete in:\t%d min, %d sec"
                % (tot_t_min, tot_t_sec))

    logger.info("\tBeginning: DMAS Reconstruction...")
    start_t = time.time()

    # Reconstruct with DMAS
    _ = dmas(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
             speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5)

    # Convert total reconstruction time to minutes and seconds
    total_t = time.time() - start_t
    tot_t_min = total_t // 60
    tot_t_sec = total_t - tot_t_min * 60

    logger.info("\t\tDMAS Reconstruction complete in:\t%d min, %d sec"
                % (tot_t_min, tot_t_sec))

    logger.info("\tBeginning: itDAS Reconstruction...")
    start_t = time.time()

    # Reconstruct with itDAS
    _ = itdas(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
              speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5,
              n_iters=6, use_dmas=False)

    # Convert total reconstruction time to minutes and seconds
    total_t = time.time() - start_t
    tot_t_min = total_t // 60
    tot_t_sec = total_t - tot_t_min * 60

    logger.info("\t\titDAS Reconstruction complete in:\t%d min, %d sec"
                % (tot_t_min, tot_t_sec))

    logger.info("\tBeginning: itDMAS Reconstruction...")
    start_t = time.time()

    # Reconstruct with itDMAS
    _ = itdas(td_data=tar_sino, ini_t=0, fin_t=6e-9, ant_rad=ant_rad,
              speed=speed, m_size=__M_SIZE, ini_ant_ang=-102.5,
              n_iters=6, use_dmas=True)

    # Convert total reconstruction time to minutes and seconds
    total_t = time.time() - start_t
    tot_t_min = total_t // 60
    tot_t_sec = total_t - tot_t_min * 60

    logger.info("\t\titDMAS Reconstruction complete in:\t%d min, %d sec"
                % (tot_t_min, tot_t_sec))
