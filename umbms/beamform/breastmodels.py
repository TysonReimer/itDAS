"""
Tyson Reimer
University of Manitoba
November 8th, 2018
"""

import numpy as np
from umbms.beamform.extras import get_xy_arrs

###############################################################################

# The vacuum permeability and permittivity
vac_permeability = 1.256637e-6
vac_permittivity = 8.85e-12
vac_speed = 3e8  # The speed of light in vacuum

# The permittivities of the breast tissue analogs used in the lab at the
# central frequency (glycerin for fat, 30% Triton X-100 solution for
# fibroglandular, and saline solution for tumor)
measured_air_perm = 1
measured_adi_perm = 7.08
measured_fib_perm = 44.94
measured_tum_perm = 77.11

###############################################################################


def get_roi(roi_rad, m_size, ant_rad):
    """Return binary mask for central circular region of interest

    Returns a binary mask in which a circular region of interest is set
    to True and the region outside is set to False

    Parameters
    ----------
    roi_rad : float
        The radius (in meters) of the inner circular region of interest
    m_size : int
        The number of pixels along one dimension used to define the
        model-space
    ant_rad : float
        The radius of the antenna scan trajectory in meters
    """

    # Get arrays for the x,y positions of each pixel
    pix_xs, pix_ys = get_xy_arrs(m_size, ant_rad)

    # Find the distance from each pixel to the center of the model space
    pix_dist_from_center = np.sqrt(pix_xs**2 + pix_ys**2)

    # Get the region of interest as all the pixels inside the
    # circle-of-interest
    roi = np.zeros([m_size, m_size], dtype=np.bool)
    roi[pix_dist_from_center < roi_rad] = True

    return roi


def get_breast(m_size=500, ant_rad=0.21, adi_rad=0.00, adi_x=0.0, adi_y=0.0,
               fib_rad=0.0, fib_x=0.0, fib_y=0.0, tum_rad=0.0, tum_x=0.0375,
               tum_y=0.0375, skin_thickness=0.0, adi_perm=6.4, fib_perm=42.2,
               tum_perm=75.4, skin_perm=40, air_perm=1.0):
    """Returns a 2D breast model

    Returns a breast model containing selected tissue components.
    Each tissue (excluding skin) is modeled using a circular region
    and assigned a permittivity corresponding to the measured
    permittivity at the central scan frequency of the corresponding
    tissue surrogate.

    Parameters
    ----------
    m_size : int
        The number of pixels along one dimension in the model space
    ant_rad : float
        The radius (in meters) of the antenna trajectory during the scan
    adi_rad : float
        The radius of the adipose tissue component in meters
    adi_x : float
        The offset of the adipose tissue component in the x-direction in
        meters
    adi_y : float
        The offset of the adipose tissue component in the y-direction in
        meters
    fib_rad : float
        The radius of the fibroglandular tissue component in meters
    fib_x : float
        The offset of the fibroglandular tissue component in the x-direction in
        meters
    fib_y : float
        The offset of the fibroglandular tissue component in the y-direction in
        meters
    tum_rad : float
        The radius of the tumor tissue component in meters
    tum_x : float
        The offset of the tumor tissue component in the x-direction in
        meters
    tum_y : float
        The offset of the tumor tissue component in the y-direction in
        meters
    skin_thickness : float
        The thickness of the skin tissue component in meters
    adi_perm : float
        The permittivity of the adipose component
    fib_perm : float
        The permittivity of the fibroglandular component
    tum_perm : float
        The permittivity of the tumor component
    skin_perm : float
        The permittivity of the skin component
    air_perm : float
        The permittivity of the surrounding medium (assumed to be air)

    Returns
    -------
    breast_model : array_like
        2D arr containing the breast model
    """

    # Get the pixel x,y-positions
    pix_xs, pix_ys = get_xy_arrs(m_size, ant_rad)

    # Compute the pixel distances from the center of each tissue
    # component (excluding skin)
    pix_dist_from_adi = np.sqrt((pix_xs - adi_x)**2 + (pix_ys - adi_y)**2)
    pix_dist_from_fib = np.sqrt((pix_xs - fib_x)**2 + (pix_ys - fib_y)**2)
    pix_dist_from_tum = np.sqrt((pix_xs - tum_x)**2 + (pix_ys - tum_y)**2)

    # Initialize breast model to be uniform background medium
    breast_model = air_perm * np.ones([m_size, m_size])

    # Assign the tissue components
    breast_model[pix_dist_from_adi < adi_rad + skin_thickness] = skin_perm
    breast_model[pix_dist_from_adi < adi_rad] = adi_perm
    breast_model[pix_dist_from_fib < fib_rad] = fib_perm
    breast_model[pix_dist_from_tum < tum_rad] = tum_perm

    return breast_model
