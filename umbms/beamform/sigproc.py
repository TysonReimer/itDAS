"""
Tyson Reimer
University of Manitoba
November 7, 2018
"""

import numpy as np
from umbms.beamform.extras import get_scan_times

###############################################################################


def iczt(fd_data, ini_t, fin_t, n_time_pts, ini_f, fin_f, axis=0):
    """Computes the ICZT of a 1D or 2D array

    Computes the inverse chirp z-transform (ICZT) on an array in the
    frequency domain, evaluating the transform at the specified
    n_time_pts between ini_t and fin_t.

    Parameters
    ----------
    fd_data : array_like
        The frequency-domain arr to be transformed via the ICZT to the
        time-domain
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        in seconds
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        in seconds
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The initial frequency used in the scan, in Hz
    fin_f : float
        The final frequency used in the scan, in Hz
    axis : int
        The axis along which to compute the transform

    Returns
    -------
    iczt_data : array_like
        Array of the transformed data, after applying the ICZT to the
        input fd_data
    """

    # Number of dimensions of fd_dat
    n_dimensions = len(np.shape(fd_data))

    # Assert fd_data is 1D or 2D
    assert n_dimensions in [1, 2], 'Error: fd_data not 1D or 2D array'

    if n_dimensions == 2:

        iczt_data = _iczt_two_dimension(fd_data, ini_t, fin_t, n_time_pts,
                                        ini_f, fin_f, axis=axis)
    else:

        # Find the number of frequencies use
        n_freqs = np.size(fd_data)

        # Find the conversion factor to convert from time-of-response to
        # angle around the unit circle
        time_to_angle = (2 * np.pi) / np.max(get_scan_times(ini_f, fin_f,
                                                            n_freqs))

        # Find the parameters for computing the ICZT over the specified
        # time window
        theta_naught = ini_t * time_to_angle
        phi_naught = (fin_t - ini_t) * time_to_angle / (n_time_pts - 1)

        # Compute the exponential values only once
        exp_theta_naught = np.exp(1j * theta_naught)
        exp_phi_naught = np.exp(1j * phi_naught)

        # Make a dummy vector to facilitate vectorized computation
        dummy_vec = np.arange(n_freqs)

        time_pts = np.arange(n_time_pts)  # Get time-index vector

        # Find the z-value matrix, to facilitate vectorized computation
        z_vals = exp_theta_naught * exp_phi_naught ** time_pts
        zs_power = np.power(z_vals[None, :], dummy_vec[:, None])

        # Compute the 1D ICZT, converting this frequency domain data
        # to the time domain
        iczt_data = _iczt_one_dimension(fd_data[:], zs_power, n_freqs)

    # Apply phase compensation
    iczt_data = phase_compensate(iczt_data, ini_f=ini_f, ini_t=ini_t,
                                 fin_t=fin_t, n_time_pts=n_time_pts)

    return iczt_data


###############################################################################


def _iczt_two_dimension(fd_data, ini_t, fin_t, n_time_pts, ini_f, fin_f,
                        axis=0):
    """Computes the ICZT of a 2D-array

    Computes the inverse chirp z-transform (ICZT) on a 2D array in the
    frequency domain, evaluating the transform at the specified
    n_time_pts between ini_t and fin_t.

    Parameters
    ----------
    fd_data : array_like
        The frequency-domain arr to be transformed via the ICZT to the
        time-domain
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        in seconds
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        in seconds
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The initial frequency used in the scan, in Hz
    fin_f : float
        The final frequency used in the scan, in Hz
    axis : int
        The axis along which to compute the transform

    Returns
    -------
    iczt_data : array_like
        Array of the transformed data, after applying the ICZT to the
        input fd_data
    """

    # Assert that this is only for 2D arrays
    assert axis in [0, 1], 'Axis error: axis must be in [0, 1]'

    # Find the number of frequencies use
    n_freqs = np.size(fd_data, axis=0)

    # Find the conversion factor to convert from time-of-response to
    # angle around the unit circle
    time_to_angle = (2 * np.pi) / np.max(get_scan_times(ini_f, fin_f, n_freqs))

    # Find the parameters for computing the ICZT over the specified time
    # window
    theta_naught = ini_t * time_to_angle
    phi_naught = (fin_t - ini_t) * time_to_angle / (n_time_pts - 1)

    # Compute the exponential values only once
    exp_theta_naught = np.exp(1j * theta_naught)
    exp_phi_naught = np.exp(1j * phi_naught)

    # Make a dummy vector to facilitate vectorized computation
    dummy_vec = np.arange(n_freqs)

    time_pts = np.arange(n_time_pts)  # Get time-index vector

    # Find the z-value matrix, to facilitate vectorized computation
    z_vals = exp_theta_naught * exp_phi_naught**time_pts
    zs_power = np.power(z_vals[None, :], dummy_vec[:, None])

    # If wanting to compute the transform along the 0th axis
    if axis == 0:

        # Init return arr
        iczt_data = np.zeros([n_time_pts, np.size(fd_data, axis=1)],
                             dtype=np.complex64)

        # For every point along the other dimension
        for ii in range(np.size(fd_data, axis=1)):

            # Compute the 1D ICZT, converting this frequency domain data
            # to the time domain
            iczt_data[:, ii] = _iczt_one_dimension(fd_data[:, ii], zs_power,
                                                   n_freqs)

    else:  # If wanting to compute along the 1st axis
        iczt_data = np.zeros([np.size(fd_data, axis=0), n_time_pts],
                             dtype=np.complex64)

        # For every point along the other dimension
        for ii in range(np.size(fd_data, axis=0)):

            # Compute the 1D ICZT, converting this frequency domain data
            # to the time domain
            iczt_data[ii, :] = _iczt_one_dimension(fd_data[ii, :], zs_power,
                                                   n_freqs)

    return iczt_data


def _iczt_one_dimension(fd_data, zs_power, n_freqs):
    """Computes the ICZT of a 1D array

    Computes the inverse chirp z-transform (ICZT) on a 1D vector in the
    frequency domain, evaluating the transform at the specified
    n_time_pts between ini_t and fin_t, defined by zs_power

    Parameters
    ----------
    fd_data : array_like
        The frequency-domain vector to be transformed via the ICZT to
        the time-domain
    zs_power : array_like
        2D z-matrix, created in _iczt_two_dimension to facilitate
        vectorized computation
    n_freqs : int
        The number of frequencies used

    Returns
    -------
    iczt_data : array_like
        1D arr of the transformed data, after applying the ICZT to the
        input fd_data
    """

    # Compute the ICZT
    iczt_data = np.sum(fd_data[:, None] * zs_power, axis=0) / n_freqs

    return iczt_data


def phase_compensate(td_data, ini_f, ini_t, fin_t, n_time_pts):
    """Applies phase compensation to TD signals obtained with the ICZT

    Parameters
    ----------
    td_data : array_like
        Time-domain signals obtained via the ICZT
    ini_f : float
        Initial frequency used in the scan, in Hz
    ini_t : float
        Initial time point of the td_data, in seconds
    fin_t : float
        Final time point of the td_data, in seconds
    n_time_pts : int
        Number of time-points used to create the td_data

    Returns
    -------
    compensated_td_data : array_like
        Time-domain signals after phase compensation
    """

    n_dim = len(np.shape(td_data))

    assert n_dim in [1, 2], "td_data must be 1D or 2D arr"

    # Create vector of the time points used to represent the td_data
    time_vec = np.linspace(ini_t, fin_t, n_time_pts)

    # Phase correction factor
    phase_fac = np.exp(1j * 2 * np.pi * ini_f * time_vec)

    if n_dim == 1:  # If td_data was 1D arr

        compensated_td_data = td_data * phase_fac  # Apply to measured data

    else:  # If td_data was 2D arr

        compensated_td_data = td_data * phase_fac[:, None]

    return compensated_td_data
