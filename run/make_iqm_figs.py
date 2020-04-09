"""
Tyson Reimer
University of Manitoba
January 23rd, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path
from umbms.loadsave import load_pickle

###############################################################################

__IQM_DIR = os.path.join(get_proj_path(), 'output/all_iqms/all_iqms.pickle')

__FIG_OUT_DIR = os.path.join(get_proj_path(), 'output/paper-figs/')
verify_path(__FIG_OUT_DIR)

###############################################################################

dark_red = '#A72323'  # Define code for dark red colour

# Load the IQM dict
all_iqms = load_pickle(__IQM_DIR)

###############################################################################


def iqms_dict_to_arrs(iqms):
    """Converts IQM dict to np arrays

    Parameters
    ----------
    iqms : dict
        Dict of IQMS for one beamformer

    Returns
    -------
    smrs : array_like
        SMRs of each expt
    smr_unctys : array_like
        Uncertainty in the SMRs of each expt
    scrs : array_like
        SCRs of each expt
    scr_unctys : array_like
        Uncertainty in the SCRs of each expt
    """

    # Init lists
    smrs, smr_unctys = [], []
    scrs, scr_unctys = [], []

    for expt_id in iqms.keys():  # For each expt

        # Store the SMR, SCR and their uncertainties
        smrs.append(iqms[expt_id][0])
        smr_unctys.append(iqms[expt_id][1])
        scrs.append(iqms[expt_id][2])
        scr_unctys.append(iqms[expt_id][3])

    # Convert the lists to arrays
    smrs = np.array(smrs)
    smr_unctys = np.array(smr_unctys)
    scrs = np.array(scrs)
    scr_unctys = np.array(scr_unctys)

    return smrs, smr_unctys, scrs, scr_unctys


def prune_by_birads(iqms, birads='c1'):
    """Gets the IQMs for only one BI-RADS class for a beamformer

    Parameters
    ----------
    iqms : dict
        The dict of IQMs for one beamformer
    birads : str
        The BI-RADS class of the IQMs to be returned

    Returns
    -------
    pruned_iqms : dict
        The IQMs of the images of the specified BI-RADS class
    """

    # Define valid birads options, assert user choice is valid
    valid_birads = ['c1', 'c2', 'c3', 'c4']
    error_msg = "Error: birads must be in %s" % valid_birads
    assert birads in valid_birads, error_msg

    pruned_iqms = dict()  # Init dict to return

    for old_key in iqms.keys():  # For each expt ID

        # If it belongs to specified BI-RADS class
        if birads in old_key:
            pruned_iqms[old_key] = iqms[old_key]

    return pruned_iqms


def prune_by_tum_detect(iqms):
    """Prunes to only contain IQMs for imgs with identifiable tumour

    Parameters
    ----------
    iqms : dict
        The dict of IQMs for one beamformer

    Returns
    -------
    pruned_iqms : dict
        The dict of the IQMs for all images which had an identifiable
        tumour response
    """

    pruned_iqms = dict()  # Init dict to return

    for old_key in iqms.keys():  # For each expt

        # If the tumour for this image was detected, using the SCR
        # criterion
        if iqms[old_key][2] >= 0:
            pruned_iqms[old_key] = iqms[old_key]

    return pruned_iqms


###############################################################################


def get_class_iqm_avgs(iqms):
    """Get the avg IQMs for each BI-RADS class

    Parameters
    ----------
    iqms : dict
        Dict of IQMs for specific beamformer

    Returns
    -------
    smr_avgs : array_like
        Avg SMR for reconstructions of each BI-RADS class
    smr_unctys : array_like
        Uncertainty in avg SMR for reconstructions of each BI-RADS class
    scr_avgs : array_like
        Avg SCR for reconstructions of each BI-RADS class
    scr_unctys : array_like
        Uncertainty in avg SCR for reconstructions of each BI-RADS class
    """

    # Init arrs to return
    smr_avgs = np.ones([4, ])
    smr_unctys = np.ones([4, ])
    scr_avgs = np.ones([4, ])
    scr_unctys = np.ones([4, ])

    cc = 0  # Init BI-RADS counter
    for birads in ['c1', 'c2', 'c3', 'c4']:  # For each density class

        # Prune by BI-RADS and tumour-detection
        iqms_here = prune_by_birads(iqms, birads=birads)
        iqms_here = prune_by_tum_detect(iqms_here)

        # Convert the dicts to arrays
        (iqm_smr, iqm_smr_unctys,
         iqm_scr, iqm_scr_unctys) = iqms_dict_to_arrs(iqms_here)

        # Get the average and uncertainty of each metric
        smr_avgs[cc] = np.mean(iqm_smr)
        smr_unctys[cc] = np.mean(iqm_smr_unctys)
        scr_avgs[cc] = np.mean(iqm_scr)
        scr_unctys[cc] = np.mean(iqm_scr_unctys)

        cc += 1  # Increment BI-RADS counter

    return smr_avgs, smr_unctys, scr_avgs, scr_unctys


def plot_smr_by_birads(save_fig=False):
    """Plots the average SMR for each BI-RADs class of beamformers

    Parameters
    ----------
    save_fig : bool
        If True, saves the figure as a .png
    """

    # Load the IQMs
    das_iqms = all_iqms['das']
    dmas_iqms = all_iqms['dmas']
    itdas_iqms = all_iqms['itdas']
    itdmas_iqms = all_iqms['itdmas']

    # Get the BI-RADS class avg IQMs
    das_iqms = get_class_iqm_avgs(das_iqms)
    dmas_iqms = get_class_iqm_avgs(dmas_iqms)
    itdas_iqms = get_class_iqm_avgs(itdas_iqms)
    itdmas_iqms = get_class_iqm_avgs(itdmas_iqms)

    print('DMAS to DAS:')
    for ii in range(4):
        print('\t%s SMR:\t%.3f' % (ii + 1, 100 * dmas_iqms[0][ii] / das_iqms[0][ii]))
        print('\t%s SCR:\t%.3f' % (ii + 1, 100 * dmas_iqms[2][ii] / das_iqms[2][ii]))

    print('itDAS to DAS:')
    for ii in range(4):
        print('\t%s SMR:\t%.3f' % (ii + 1, 100 * itdas_iqms[0][ii] / das_iqms[0][ii]))
        print('\t%s SCR:\t%.3f' % (ii + 1, 100 * itdas_iqms[2][ii] / das_iqms[2][ii]))

    print('itDMAS to DAS:')
    for ii in range(4):
        print(
            '\t%s SMR:\t%.3f' % (ii + 1, 100 * itdmas_iqms[0][ii] / das_iqms[0][ii]))
        print(
            '\t%s SCR:\t%.3f' % (ii + 1, 100 * itdmas_iqms[2][ii] / das_iqms[2][ii]))

    # Define xs for making the plot
    plot_xs = np.arange(1, 5)

    plt.figure(figsize=(10, 6))  # Make fig
    plt.rc('font', family='Times New Roman')  # Set font

    plt.tick_params(labelsize=18)

    plt.errorbar(x=plot_xs, y=itdmas_iqms[0], yerr=itdmas_iqms[1],
                 label='itDMAS', color=dark_red, marker='^', capsize=10,
                 linestyle='--')
    plt.errorbar(x=plot_xs, y=itdas_iqms[0], yerr=itdas_iqms[1], label='itDAS',
                 color=dark_red, marker='D', capsize=10)
    plt.errorbar(x=plot_xs, y=dmas_iqms[0], yerr=dmas_iqms[1], label='DMAS',
                 color='k', marker='s', capsize=10, linestyle='--')
    plt.errorbar(x=plot_xs, y=das_iqms[0], yerr=das_iqms[1], label='DAS',
                 color='k', marker='o', capsize=10)

    plt.legend(fontsize=18)
    plt.xticks(plot_xs, ['I', 'II', 'III', 'IV'])
    plt.ylabel('SMR Magnitude (dB)', fontsize=20)
    plt.xlabel('BI-RADS Density Class', fontsize=20)

    plt.tight_layout()
    plt.show()

    if save_fig:
        plt.savefig(os.path.join(__FIG_OUT_DIR, 'smr_by_class.png'),
                    transparent=True, dpi=450)


def plot_scr_by_birads(save_fig=False):
    """Plots the average SCR for each BI-RADs class of beamformers

    Parameters
    ----------
    save_fig : bool
        If True, saves the figure as a .png
    """

    # Load the IQMs
    das_iqms = all_iqms['das']
    dmas_iqms = all_iqms['dmas']
    itdas_iqms = all_iqms['itdas']
    itdmas_iqms = all_iqms['itdmas']

    # Get the BI-RADS class avg IQMs
    das_iqms = get_class_iqm_avgs(das_iqms)
    dmas_iqms = get_class_iqm_avgs(dmas_iqms)
    itdas_iqms = get_class_iqm_avgs(itdas_iqms)
    itdmas_iqms = get_class_iqm_avgs(itdmas_iqms)

    # Define xs for making the plot
    plot_xs = np.arange(1, 5)

    plt.figure(figsize=(10, 6))  # Make fig
    plt.rc('font', family='Times New Roman')  # Set font

    plt.tick_params(labelsize=18)

    plt.errorbar(x=plot_xs, y=itdmas_iqms[2], yerr=itdmas_iqms[3],
                 label='itDMAS', color=dark_red, marker='^', capsize=10,
                 linestyle='--')
    plt.errorbar(x=plot_xs, y=itdas_iqms[2], yerr=itdas_iqms[3], label='itDAS',
                 color=dark_red, marker='D', capsize=10)
    plt.errorbar(x=plot_xs, y=dmas_iqms[2], yerr=dmas_iqms[3], label='DMAS',
                 color='k', marker='s', capsize=10, linestyle='--')
    plt.errorbar(x=plot_xs, y=das_iqms[2], yerr=das_iqms[3], label='DAS',
                 color='k', marker='o', capsize=10)

    plt.legend(fontsize=18)
    plt.xticks(plot_xs, ['I', 'II', 'III', 'IV'])
    plt.ylabel('SCR Magnitude (dB)', fontsize=20)
    plt.xlabel('BI-RADS Density Class', fontsize=20)

    plt.tight_layout()
    plt.show()

    if save_fig:
        plt.savefig(os.path.join(__FIG_OUT_DIR, 'scr_by_class.png'),
                    transparent=True, dpi=450)


###############################################################################


if __name__ == "__main__":

    plot_smr_by_birads(save_fig=True)
    plot_scr_by_birads(save_fig=True)
