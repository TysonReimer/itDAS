"""
Tyson Reimer
University of Manitoba
January 21st, 2020
"""

import os
import numpy as np

from umbms import get_proj_path, verify_path, get_script_logger, null_logger
from umbms.loadsave import load_pickle, save_pickle

from umbms.beamform.iqms import (get_smr, get_scr, get_loc_err,
                                 round_uncertainty)
from umbms.beamform.extras import apply_ant_t_delay

###############################################################################

# Directory where geometry parameters are located
__GEOM_PARAM_DIR = os.path.join(get_proj_path(), 'data/')

# Directory where reconstructed images .pickle files are located
__IMG_DIR = os.path.join(get_proj_path(), 'output/recon-imgs/')

# Directory where IQMs will be stored as .pickle files
__OUT_DIR = os.path.join(get_proj_path(), 'output/all_iqms/')
verify_path(__OUT_DIR)

###############################################################################


def compute_iqms(logger=null_logger):
    """Computes the IQMs for each image by each beamformer

    Parameters
    ----------
    logger : logging object
        Logger for reporting results and progress
    """

    # Load reconstructed images
    das_imgs = load_pickle(os.path.join(__IMG_DIR, 'das_imgs.pickle'))
    dmas_imgs = load_pickle(os.path.join(__IMG_DIR, 'dmas_imgs.pickle'))
    itdas_imgs = load_pickle(os.path.join(__IMG_DIR, 'itdas_imgs.pickle'))
    itdmas_imgs = load_pickle(os.path.join(__IMG_DIR, 'itdmas_imgs.pickle'))

    # Load the geometry parameters
    geom_params = load_pickle(os.path.join(__GEOM_PARAM_DIR,
                                           'geom_params.pickle'))

    # Init dicts for storing the IQMs for each beamformer
    iqms = dict()
    iqms['das'] = dict()
    iqms['dmas'] = dict()
    iqms['itdas'] = dict()
    iqms['itdmas'] = dict()

    # Init lists for storing the localization errors of each beamformer
    loc_errs = dict()
    loc_errs['das'] = []
    loc_errs['dmas'] = []
    loc_errs['itdas'] = []
    loc_errs['itdmas'] = []

    for expt_id in das_imgs.keys():  # For each image

        # Get the geometry parameters for this specific image
        tum_x, tum_y, tum_rad, adi_rad, ant_rad = geom_params[expt_id]
        ant_rad = apply_ant_t_delay(ant_rad)  # Apply time delay

        logger.info('\tImage ID:\t%s' % expt_id)

        # Get the images of this expt_id
        das_img = das_imgs[expt_id]
        dmas_img = dmas_imgs[expt_id]
        itdas_img = itdas_imgs[expt_id]
        itdmas_img = itdmas_imgs[expt_id]

        # Compute the SCR and its uncertainty for each image
        das_scr, das_scr_uncty = get_scr(das_img, ant_rad=ant_rad,
                                         adi_rad=adi_rad, tum_rad=tum_rad,
                                         tum_x=tum_x, tum_y=tum_y)
        dmas_scr, dmas_scr_uncty = get_scr(dmas_img, ant_rad=ant_rad,
                                           adi_rad=adi_rad, tum_rad=tum_rad,
                                           tum_x=tum_x, tum_y=tum_y)
        itdas_scr, itdas_scr_uncty = get_scr(itdas_img, ant_rad=ant_rad,
                                             adi_rad=adi_rad,
                                             tum_rad=tum_rad,
                                             tum_x=tum_x, tum_y=tum_y)
        itdmas_scr, itdmas_scr_uncty = get_scr(itdmas_img, ant_rad=ant_rad,
                                               adi_rad=adi_rad,
                                               tum_rad=tum_rad, tum_x=tum_x,
                                               tum_y=tum_y)

        # Compute the SMR and its uncertainty for each image
        das_smr, das_smr_uncty = get_smr(das_img, ant_rad=ant_rad,
                                         adi_rad=adi_rad, tum_rad=tum_rad,
                                         tum_x=tum_x, tum_y=tum_y)
        dmas_smr, dmas_smr_uncty = get_smr(dmas_img, ant_rad=ant_rad,
                                           adi_rad=adi_rad, tum_rad=tum_rad,
                                           tum_x=tum_x, tum_y=tum_y)
        itdas_smr, itdas_smr_uncty = get_smr(itdas_img, ant_rad=ant_rad,
                                             adi_rad=adi_rad,
                                             tum_rad=tum_rad,
                                             tum_x=tum_x, tum_y=tum_y)
        itdmas_smr, itdmas_smr_uncty = get_smr(itdmas_img, ant_rad=ant_rad,
                                               adi_rad=adi_rad,
                                               tum_rad=tum_rad, tum_x=tum_x,
                                               tum_y=tum_y)

        # If the tumor was detected in the image, compute and store the
        # localization error
        if das_scr >= 0:
            das_locerr = get_loc_err(img=das_img, ant_rad=ant_rad,
                                     tum_x=tum_x, tum_y=tum_y)
            loc_errs['das'].append(das_locerr)
        if dmas_scr >= 0:
            dmas_locerr = get_loc_err(img=dmas_img, ant_rad=ant_rad,
                                      tum_x=tum_x, tum_y=tum_y)
            loc_errs['dmas'].append(dmas_locerr)
        if itdas_scr >= 0:
            itdas_locerr = get_loc_err(img=itdas_img, ant_rad=ant_rad,
                                       tum_x=tum_x, tum_y=tum_y)
            loc_errs['itdas'].append(itdas_locerr)
        if itdmas_scr >= 0:
            itdmas_locerr = get_loc_err(img=itdmas_img, ant_rad=ant_rad,
                                        tum_x=tum_x, tum_y=tum_y)
            loc_errs['itdmas'].append(itdmas_locerr)

        # Round the SMRs to match their uncertainties
        das_smr, das_smr_uncty = round_uncertainty(das_smr, das_smr_uncty)
        dmas_smr, dmas_smr_uncty = round_uncertainty(dmas_smr, dmas_smr_uncty)
        itdas_smr, itdas_smr_uncty = round_uncertainty(itdas_smr,
                                                       itdas_smr_uncty)
        itdmas_smr, itdmas_smr_uncty = round_uncertainty(itdmas_smr,
                                                         itdmas_smr_uncty)

        # Round the SCRs to match their uncertainties
        das_scr, das_scr_uncty = round_uncertainty(das_scr, das_scr_uncty)
        dmas_scr, dmas_scr_uncty = round_uncertainty(dmas_scr, dmas_scr_uncty)
        itdas_scr, itdas_scr_uncty = round_uncertainty(itdas_scr,
                                                       itdas_scr_uncty)
        itdmas_scr, itdmas_scr_uncty = round_uncertainty(itdmas_scr,
                                                         itdmas_scr_uncty)

        # Store the IQM results for each beamformer for this image
        iqms['das']['%s' % expt_id] = (das_smr, das_smr_uncty,
                                       das_scr, das_scr_uncty)
        iqms['dmas']['%s' % expt_id] = (dmas_smr, dmas_smr_uncty,
                                        dmas_scr, dmas_scr_uncty)
        iqms['itdas']['%s' % expt_id] = (itdas_smr, itdas_smr_uncty,
                                         itdas_scr, itdas_scr_uncty)
        iqms['itdmas']['%s' % expt_id] = (itdmas_smr, itdmas_smr_uncty,
                                          itdmas_scr, itdmas_scr_uncty)

        # Report the results to the logger
        logger.info('\t\tDAS')
        logger.info('\t\t---')
        logger.info('\t\t\tSMR: (%s +/- %s)\t\tSCR: (%s +/- %s)'
                    % (das_smr, das_smr_uncty, das_scr, das_scr_uncty))

        logger.info('\t\tDMAS')
        logger.info('\t\t---')
        logger.info('\t\t\tSMR: (%s +/- %s)\t\tSCR: (%s +/- %s)'
                    % (dmas_smr, dmas_smr_uncty, dmas_scr, dmas_scr_uncty))

        logger.info('\t\titDAS')
        logger.info('\t\t---')
        logger.info('\t\t\tSMR: (%s +/- %s)\t\tSCR: (%s +/- %s)'
                    % (itdas_smr, itdas_smr_uncty, itdas_scr, itdas_scr_uncty))

        logger.info('\t\titDMAS')
        logger.info('\t\t---')
        logger.info('\t\t\tSMR: (%s +/- %s)\t\tSCR: (%s +/- %s)'
                    % (itdmas_smr, itdmas_smr_uncty, itdmas_scr,
                       itdmas_scr_uncty))

    logger.info('\tDAS Loc Err:\t%.3f +/- %.3f cm'
                % (100 * np.mean(np.array(loc_errs['das'])),
                   100 * np.std(np.array(loc_errs['das']))))

    logger.info('\tdmas Loc Err:\t%.3f +/- %.3f cm'
                % (100 * np.mean(np.array(loc_errs['dmas'])),
                   100 * np.std(np.array(loc_errs['dmas']))))

    logger.info('\titdas Loc Err:\t%.3f +/- %.3f cm'
                % (100 * np.mean(np.array(loc_errs['itdas'])),
                   100 * np.std(np.array(loc_errs['itdas']))))

    logger.info('\titdmas Loc Err:\t%.3f +/- %.3f cm'
                % (100 * np.mean(np.array(loc_errs['itdmas'])),
                   100 * np.std(np.array(loc_errs['itdmas']))))

    # Save the IQM dict to a .pickle file
    save_pickle(iqms, os.path.join(__OUT_DIR, 'all_iqms.pickle'))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    compute_iqms(logger=logger)
