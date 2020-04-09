"""
Tyson Reimer
University of Manitoba
December 14, 2018
"""

import numpy as np
from umbms.beamform.breastmodels import get_breast, get_roi

###############################################################################

# The top percentile of tumor pixels used for evaluating the tumor
# response
_top_tum_percent = 25

# The top percentile of clutter pixels used for evaluating the clutter
# response
_top_clutter_percent = 5

# The distance, in  meters, used to increase the "tumor" region of the
# image
_tum_rad_increase = 0.005

###############################################################################


def get_smr(img, ant_rad, adi_rad, tum_rad, tum_x, tum_y, normalize=True):
    """Returns the SMR of an image

    Returns the signal-to-mean ratio (SMR) of a reconstructed image.

    Parameters
    ----------
    img : array_like
        The reconstructed image
    ant_rad : array_like
        The radius of the antenna trajectory during the scan
    adi_rad : float
        The radius used to approximate the breast region as a circle,
        in meters
    tum_rad : float
        The radius used to define the tumor region, in meters
    tum_x : float
        The known x-position of the tumor, in meters
    tum_y : float
        The known y-position of the tumor, in meters
    normalize : bool
        If True, normalizes the image before computing the SMR

    Returns
    -------
    smr : float
        The SMR of the reconstructed image
    smr_uncty : float
        The uncertainty in the SMR value
    """

    # Create a breast model to index for the different tissue regions
    # (air, adipose, fibroglandular, and tumor)
    indexing_breast = get_breast(np.size(img, 0), ant_rad=ant_rad,
                                 adi_rad=adi_rad, adi_x=0, adi_y=0,
                                 fib_rad=0, fib_x=0, fib_y=0,
                                 tum_rad=tum_rad + _tum_rad_increase,
                                 tum_x=tum_x, tum_y=tum_y, skin_thickness=0,
                                 adi_perm=2, fib_perm=3, tum_perm=4,
                                 skin_perm=1, air_perm=1)

    # Square the image, as would be done when it is displayed
    img = img**2

    # If we want to normalize the reconstructed image prior to
    # computing the SMR
    if normalize:

        # Find the black-line radius of the antenna in the scan
        temp_val = (ant_rad - 0.148) / 0.97 + 0.106

        # Remove any NaN values
        img[np.isnan(img)] = np.min(img[~np.isnan(img)])

        # Make ROI defined by within the antenna trajectory, measured
        # from the most central edge of the antenna
        in_ant_trajectory = get_roi(temp_val - 0.1,
                                    np.size(img, axis=0),
                                    ant_rad)

        # Apply the ROI
        img[np.logical_not(in_ant_trajectory)] = np.min(img)

        # Normalize to have max of unity and min of 0
        img = img - np.min(img)
        img = img / np.max(img)

    tum_pix = img[indexing_breast == 4]  # Find tumour pixels

    # Find the maximum value in the known tumor region
    sig_val = np.max(tum_pix)

    # Find the mean clutter value in the breast region (excluding the
    # tumor region)
    mean_clut_val = np.mean(img[np.logical_and(indexing_breast != 4,
                                               indexing_breast != 1)])

    # Find the estimated uncertainty in the tumor response
    _, tum_uncty = _get_top_percent_tum(img, indexing_breast)

    # Find the SMR value and its estimated uncertainty
    smr = 20 * np.log10(sig_val / mean_clut_val)
    smr_uncty = 20 * tum_uncty / (sig_val * np.log(10))

    return smr, smr_uncty


def get_scr(img, ant_rad, adi_rad, tum_rad, tum_x, tum_y, normalize=True):
    """Returns the SCR of an image

    Returns the signal-to-clutter ratio (SCR) of a reconstructed image.

    Parameters
    ----------
    img : array_like
        The reconstructed image
    ant_rad : array_like
        The radius of the antenna trajectory during the scan
    adi_rad : float
        The radius used to approximate the breast region as a circle,
        in meters
    tum_rad : float
        The radius used to define the tumor region, in meters
    tum_x : float
        The known x-position of the tumor, in meters
    tum_y : float
        The known y-position of the tumor, in meters
    normalize : bool
        If True, normalizes the image before computing the SMR

    Returns
    -------
    scr : float
        The SCR of the reconstructed image
    scr_uncty : float
        The uncertainty in the SCR value
    """

    # Rotate and flip image to facilitate use with indexing breast
    img = img**2

    # If you want to normalize the reconstruction before computing the
    # SCR
    if normalize:
        temp_val = (ant_rad - 0.14) / 0.97 + 0.106
        img[np.isnan(img)] = np.min(img[~np.isnan(img)])

        in_ant_trajectory = get_roi(temp_val - 0.1,
                                    np.size(img, axis=0),
                                    ant_rad)
        img[np.logical_not(in_ant_trajectory)] = np.min(img)
        img = img - np.min(img)
        img = img / np.max(img)

    # Create a model of the reconstruction, segmented by the various
    # tissue types
    indexing_breast = get_breast(np.size(img, 0), ant_rad=ant_rad,
                                 adi_rad=adi_rad, adi_x=0, adi_y=0,
                                 fib_rad=0, fib_x=0, fib_y=0,
                                 tum_rad=tum_rad + _tum_rad_increase,
                                 tum_x=tum_x, tum_y=tum_y, skin_thickness=0,
                                 adi_perm=2, fib_perm=3, tum_perm=4,
                                 skin_perm=1, air_perm=1)

    # Determine the max values in the tumor region and the clutter
    # region
    sig_val = np.max(img[indexing_breast == 4])
    max_clut_val = np.max(img[np.logical_and(indexing_breast != 4,
                                             indexing_breast != 1)])

    # Compute the SCR value
    scr = 20 * np.log10(sig_val / max_clut_val)

    # Estimate the uncertainties in the tumor and clutter responses
    _, tum_uncty = _get_top_percent_tum(img, indexing_breast)
    _, clut_uncty = _get_top_percent_clut(img, indexing_breast)

    # Compute the estimated uncertainty in the SCR
    scr_uncty = np.sqrt((20 * tum_uncty / (np.log(10) * sig_val))**2 +
                        (20 * clut_uncty / (np.log(10) * max_clut_val))**2)

    return scr, scr_uncty


def get_loc_err(img, ant_rad, tum_x, tum_y):
    """Return the localization error of the tumor response in the image

    Compute the localization error for the reconstructed image in meters

    Parameters
    ----------
    img : array_like
        The reconstructed image
    ant_rad : float
        The radius of the antenna trajectory during the scan, in meters
    tum_x : float
        The x-position of the tumor during the scan, in meters
    tum_y : float
        The y-position of the tumor during the scan, in meters

    Returns
    -------
    loc_err : float
        The localization error in meters
    """

    # Rotate image to properly compute the distances
    img = np.fliplr(img)

    # Find the conversion factor to convert pixel index to distance
    pix_to_dist = 2 * ant_rad / np.size(img, 0)

    # Set any NaN values to zero
    img[np.isnan(img)] = 0

    # Find the index of the maximum response in the reconstruction
    max_loc = np.argmax(img)

    # Find the x/y-indices of the max response in the reconstruction
    max_x_pix, max_y_pix = np.unravel_index(max_loc, np.shape(img))

    # Convert this to the x/y-positions
    max_x_pos = (max_x_pix - np.size(img, 0) // 2) * pix_to_dist
    max_y_pos = (max_y_pix - np.size(img, 0) // 2) * pix_to_dist

    # Compute the localization error
    loc_err = np.sqrt((max_x_pos - tum_x)**2 + (max_y_pos - tum_y)**2)

    return loc_err


def round_uncertainty(value, uncty):
    """Properly rounds a value and its uncertainty

    Parameters
    ----------
    value : float
        The value that will be rounded to match the uncertainty
    uncty : float
        The uncertainty that will be rounded to have one sig-fig

    Returns
    -------
    rounded_val : float
        The rounded value, rounded to the same decimal-place as the
        uncertainty
    rounded_uncty : float
        The rounded uncertainty, rounded to have one sig-fig
    """

    # Find the decimal place after rounding to have one sig-fig
    place_to_round = -int(np.floor(np.log10(np.abs(uncty))))

    # Round the value and its uncertainty to that decimal-place
    rounded_val = round(value, place_to_round)
    rounded_uncty = round(uncty, place_to_round)

    return rounded_val, rounded_uncty


###############################################################################


def _get_top_percent_tum(img, indexing_breast):
    """Find the mean and stdev of the top % of the tum-pixels in the img


    Finds the mean and standard deviation of the top _top_tum_percent of
    pixels in the tumor region

    Parameters
    ----------
    img : array_like
        The reconstructed image
    indexing_breast : array_like
        An arr for indexing the reconstructed image; segmenting the
        different regions

    Returns
    -------
    top_tum_mean : float
        The mean intensity in the _top_tum_percent of tumor pixels
    top_tum_uncty : float
        The stdev of the intensity in the _top_tum_percent of tumor
        pixels
    """

    # Find the pixels that belong to the tumor response
    tum_pixs = img[indexing_breast == 4]

    # Find the pixels in the tumor region that belong to the
    # _top_tum_percent
    # of tumor pixels
    top_tum_pixs = tum_pixs[tum_pixs > np.percentile(tum_pixs,
                                                     _top_tum_percent)]

    # Find the mean and standard deviation of intensity values among
    # these _top_tum_percent of pixels
    top_tum_uncty = np.std(top_tum_pixs)
    top_tum_mean = np.mean(top_tum_pixs)

    return top_tum_mean, top_tum_uncty


def _get_top_percent_clut(img, indexing_breast):
    """Find the mean and stdev of the top % of the clutter pixels

    Finds the mean and standard deviation of the top
    _top_clutter_percent of pixels in the clutter region

    Parameters
    ----------
    img : array_like
        The reconstructed image
    indexing_breast : array_like
        An arr for indexing the reconstructed image; segmenting the
        different regions

    Returns
    -------
    clut_top_mean : float
        The mean intensity in the _top_clutter_percent of clutter pixels
    clut_top_uncty : float
        The stdev of the intensity in the _top_clutter_percent of
        clutter pixels
    """

    # Find the pixels in the reconstruction which belong to the
    # clutter region
    clut_pixs = img[np.logical_and(indexing_breast != 4, indexing_breast != 1)]

    # Find the pixels in the clutter region in the _top_clutter_percent
    # of pixels
    top_clut_pixs = clut_pixs[clut_pixs > np.percentile(clut_pixs,
                                                        _top_clutter_percent)]

    # Find the mean and standard deviation of the intensity values
    # among these _top_clutter_percent of pixels
    clut_top_uncty = np.std(top_clut_pixs)
    clut_top_mean = np.mean(top_clut_pixs)

    return clut_top_mean, clut_top_uncty
