"""
Tyson Reimer
University of Manitoba
June 3rd, 2019
"""

import numpy as np

from umbms.beamform.breastmodels import get_roi

###############################################################################


def fwd_proj(img_model, pix_angs, pix_dists, prop_speed, ini_t,
             fin_t, n_time_pts, ant_rad):
    """Forward-projects the img_model into the time-domain

    Computes the forward projection of the image_model - projecting
    from the image-space to the data-domain / sinogram space.

    Parameters
    ----------
    img_model : array_like
        The model of the object in the image-space
    pix_angs : array_like
        The angle off of the central axis of the antenna of each pixel,
        for each antenna position, in degrees
    pix_dists : array_like
        The distances between each pixel and the antenna, for each
        antenna position, in meters
    prop_speed : float
        The estimated propagation speed of the microwave signal, m/s
    ini_t : float
        The initial time-point used to represent the radar signals in
        the time-domain, in seconds (i.e., the first point in the
        fwd_proj corresponds to the ini_t time-of-response)
    fin_t : float
        The final time-point used to represent the radar signals in the
        time-domain, in seconds (i.e., the last point in the
        fwd_proj corresponds to the fin_t time-of-response)
    n_time_pts : int
        The number of points used to represent the radar signals in the
        time-domain
    ant_rad : float
        The radius of the antenna trajectory during the scan

    Returns
    -------
    fwd_projection : array_like
        The forward projection of the image_model in the time-domain
        (i.e., sinogram-space)
    """

    # Find the number of antenna positions used in the scan
    n_ant_pos = pix_angs.shape[0]

    # Make the vector representing the time-points of the radar signals
    # in the time domain
    scan_times = np.linspace(ini_t, fin_t, n_time_pts)

    # Init arr to return
    fwd_projection = np.zeros([n_time_pts, n_ant_pos], dtype=np.complex64)

    # Get ROI (within antenna trajectory)
    roi = get_roi(roi_rad=ant_rad, ant_rad=ant_rad,
                  m_size=np.size(img_model, axis=0))

    # Apply window to set pixels outside of ROI to zero
    img_model[np.logical_not(roi)] = 0

    # Find the physical width of each pixel in meters
    pix_dist_width = 2 * ant_rad / pix_angs.shape[0]

    # Convert this physical width to a time-of-response width
    pix_time_width = pix_dist_width / prop_speed

    # Find the pixels that are in front of the antenna, for each antenna
    # position
    pix_in_front_ant = np.abs(pix_angs) < 90

    # Find the response times of every pixel, and the upper and lower
    # bounds on these estimates
    pix_times = pix_dists * pix_in_front_ant / prop_speed

    # Find the upper and lower bounds on the time-of-response estimates
    # for each pixel
    upper_pix_times = pix_times + 0.5 * pix_time_width
    lower_pix_times = pix_times - 0.5 * pix_time_width

    # For every antenna position in the scan
    for ant_pos in range(n_ant_pos):

        # For every time-point in the time-domain radar signal
        for time_pt in range(n_time_pts):

            # Find the coordinates of the pixels that correspond to this
            # time_pt time-of-response
            resp_idxs = np.logical_and(scan_times[time_pt]
                                       < upper_pix_times[ant_pos, :, :],
                                       scan_times[time_pt]
                                       > lower_pix_times[ant_pos, :, :])

            # Find the values in the img_model that will be contributing
            # to the forward projection here
            contributing_values = img_model[resp_idxs]

            # Sum the contributing values after applying any correction
            # factors
            fwd_projection[time_pt, ant_pos] += np.sum(contributing_values)

    return fwd_projection
