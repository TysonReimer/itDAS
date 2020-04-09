"""
Tyson Reimer
University of Manitoba
January 28th, 2020
"""

import os

from umbms import get_proj_path, get_script_logger

from umbms.loadsave import load_pickle, save_pickle
from umbms.beamform.sigproc import iczt

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/')

# Define the parameters for the ICZT, used to convert from the
# frequency domain to the time domain
__INI_T = 0
__FIN_T = 6e-9
__N_TIME_PTS = 700

# Define the initial and final frequencies used in the phantom scans
__INI_F = 1e9
__FIN_F = 8e9

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)  # Init logger

    logger.info("Beginning...Making Calibrated Sinograms...")

    # Import the frequency-domain, uncalibrated data
    fd_data = load_pickle(os.path.join(__DATA_DIR, 'fd_data.pickle'))

    cal_td_data = dict()  # Init dict to save

    for expt_id in fd_data:  # For each expt

        # If the experiment was not an adipose-only reference scan
        if 'adi' not in expt_id:

            logger.info('\tCalibrating expt:\t%s...' % expt_id)

            # Get the target (tumour-containing) and reference
            # (adipose-only) scan data
            tar_fd_data = fd_data[expt_id]
            ref_fd_data = fd_data['%sadi' % expt_id[:-3]]

            # Perform ideal air-tissue reflection subtraction
            cal_fd_data = tar_fd_data - ref_fd_data

            # Convert to the time-domain via ICZT
            td_cal_data = iczt(fd_data=cal_fd_data,
                               ini_t=__INI_T, fin_t=__FIN_T,
                               n_time_pts=__N_TIME_PTS,
                               ini_f=__INI_F, fin_f=__FIN_F)

            cal_td_data[expt_id] = td_cal_data  # Store this

    save_pickle(cal_td_data,
                os.path.join(__DATA_DIR, 'td_cal_data.pickle'))
