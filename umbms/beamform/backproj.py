"""
Tyson Reimer
University of Manitoba
June 3rd, 2019
"""

import numpy as np

###############################################################################


def back_proj(td_data, pix_angs, pix_dists, prop_speed, ini_t, fin_t,
              use_dmas=False):
    """Back-projects time-domain data onto the image-space

    Performs back-projection using time-domain data. Uses either the
    DAS or DMAS beamformer as its basis.

    Parameters
    ----------
    td_data : array_lke
        The time-domain data (sinogram) that will be back-projected into
        the image space
    pix_angs : array_like
        The angles off of the central axis of the antenna of every
        pixel, for each antenna position, in degrees
    pix_dists : array_like
        The distances between each pixel and the antenna, for each
        antenna position, in meters
    prop_speed : float
        The estimated propagation speed of the microwave signal in m/s
    ini_t : float
        The initial time (s) of the td_data (i.e., the first element
        in each radar signal corresponds to the ini_t of response)
    fin_t : float
        The final time (s) of the td_data (i.e., the last element in
        each radar signal corresponds to the fin_t of response)
    use_dmas : bool
        If True, uses the delay-multiply-and-sum beamformer as the
        back-projection operator. Otherwise, uses the delay-and-sum
        beamformer

    Returns
    -------
    back_projection : array_like
        The back-projection, in the image-space, of the td_data
    """

    td_data = np.abs(td_data)  # Take the abs-value

    # Find the number of points in the time-domain used to represent the
    # signal, and the number of antenna positions
    n_time_pts, n_ant_pos = td_data.shape

    # Find the vector of the time-points used to represent the time
    # domain signal, and the incremental step-size
    scan_times = np.linspace(ini_t, fin_t, n_time_pts)
    scan_time_step = np.abs(scan_times[1] - scan_times[0])

    # Init arr for storing the back-projection from each antenna
    # position
    back_projections = np.zeros_like(pix_angs)

    # Init arr to return
    back_projection = np.zeros(np.shape(pix_angs)[1:])

    # Find the pixels in front of the antenna, for each antenna position
    pix_front_of_ant = np.abs(pix_angs) < 90

    # Find the response-times for every pixel
    pix_times = pix_dists * pix_front_of_ant / prop_speed

    # Find the upper and lower bounds on the time-of-response estimates
    # for each pixel
    upper_pix_times = scan_times + 0.5 * scan_time_step
    lower_pix_times = scan_times - 0.5 * scan_time_step

    # For every antenna position used in the scan
    for ant_pos in range(n_ant_pos):

        # Find the response times of each pixel, for this antenna
        # position
        pix_times_here = pix_times[ant_pos, :, :]

        # For every time-point of response
        for time_pt in range(np.size(td_data, axis=0)):

            # Find the coordinates of the pixels that correspond to this
            # time-of-response
            resp_idxs = np.logical_and(pix_times_here
                                       > lower_pix_times[time_pt],
                                       pix_times_here
                                       < upper_pix_times[time_pt])

            # Get the value that will be back-projected to the pixels at
            # the resp_idxs
            back_proj_val = td_data[time_pt, ant_pos]

            # Back-project the value to the image-space
            back_projections[ant_pos, :, :][resp_idxs] += back_proj_val

    # If using the DMAS beamformer as the foundation of the
    # back-projection method
    if use_dmas:

        # For every antenna position, perform pair-wise signal
        # multiplication with every other antenna position
        for ant_pos in range(n_ant_pos):
            for second_ant_pos in range(ant_pos + 1, n_ant_pos):

                # Perform pair-wise signal multiplication before
                # summation
                back_projection += (back_projections[ant_pos, :, :]
                                    * back_projections[second_ant_pos, :, :])

    else:  # Use the DAS as the foundation of the back-projection method

        # Sum the back-projections from each antenna position
        back_projection = np.sum(back_projections, axis=0)

    return back_projection
