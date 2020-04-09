"""
Tyson Reimer
University of Manitoba
November 7, 2018
"""

import numpy as np

###############################################################################

__GHz = 1e9  # Conversion factor from Hz to GHz

__VACUUM_SPEED = 3e8  # Speed of light in a vacuum

###############################################################################


def get_xy_arrs(m_size, ant_rad):
    """Finds the x/y position of each pixel in the image-space

    Returns arrays that contain the x-distances and y-distances of every
    pixel in the model.

    Parameters
    ----------
    m_size : int
        The number of pixels along one dimension of the model
    ant_rad : float
        The radius of the antenna trajectory during the scan in meters

    Returns
    -------
    x_dists : array_like
        A 2D arr. Each element in the arr contains the x-position of
        that pixel in the model, in meters
    y_dists : array_like
        A 2D arr. Each element in the arr contains the y-position of
        that pixel in the model, in meters
    """

    # Initialize the arrays
    x_coords = np.ones([m_size, m_size])
    y_coords = np.ones_like(x_coords)

    # Assign each pixel the 1-based index of that pixel
    for pix in range(m_size):
        x_coords[pix, :] = pix + 1
        y_coords[:, pix] = pix + 1

    # Find the distances for these pixels
    x_dists = (x_coords - m_size // 2) * 2 * ant_rad / m_size
    y_dists = -(y_coords - m_size // 2) * 2 * ant_rad / m_size

    return x_dists, y_dists


def apply_ant_t_delay(scan_rad):
    """Accounts for antenna time delay by extending the scan rad

    Gets the "true" antenna radius used during a scan. Assumes ant_rad
    is measured from the black-line radius on the plastic antenna stand.
    Uses formula described in the M.Sc. thesis of Diego
    Rodriguez-Herrera [1]

    1. D. Rodriguez-Herrera, "Antenna characterisation and optimal
       sampling constraints for breast microwave imaging systems with a
       novel wave speed propagation algorithm," M.Sc. Thesis,
       University of Manitoba, Winnipeg, Manitoba, 2016.

    Parameters
    ----------
    scan_rad : float
        The radius of the antenna trajectory during the scan in meters,
        as measured from the black-line on the plastic antenna stand.

    Returns
    -------
    ant_rad : float
        The adjusted radius of the antenna trajectory, using the formula
        described in the M.Sc. thesis of Diego Rodriguez-Herrera [1]
    """

    # Use empirical formula from the thesis by Diego Rodriguez-Herrera,
    # 2016, [1] to find the compensated antenna radius
    ant_rad = 0.97 * (scan_rad - 0.106) + 0.148

    return ant_rad


def get_pixdist_ratio(m_size, ant_rad):
    """Get the ratio between pixel number and physical distance

    Returns the pixel-to-distance ratio (physical distance, in meters)

    Parameters
    ----------
    m_size : int
        The number of pixels used along one-dimension for the model
        (the model is assumed to be square)
    ant_rad : float
        The radius of the antenna trajectory during the scan, in meters

    Returns
    -------
    pix_to_dist_ratio : float
        The number of pixels per physical meter
    """

    # Get the ratio between pixel and physical length
    pix_to_dist_ratio = m_size / (2 * ant_rad)

    return pix_to_dist_ratio


def get_scan_freqs(ini_f, fin_f, n_freqs):
    """Returns the linearly-separated frequencies used in a scan

    Returns the vector of frequencies used in the scan.

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    scan_freq_vector : array_like
        Vector containing each frequency used in the scan
    """

    # Get the vector of the frequencies used
    scan_freq_vector = np.linspace(ini_f, fin_f, n_freqs)

    return scan_freq_vector


def get_freq_step(ini_f, fin_f, n_freqs):
    """Gets the incremental frequency step df in Hz used in the scan

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    df : float
        The incremental frequency step-size used in the scan, in Hz
    """

    # Get the vector of the frequencies used in the scan
    freqs = get_scan_freqs(ini_f, fin_f, n_freqs)

    # Find the incremental frequency step-size
    df = freqs[1] - freqs[0]

    return df


def get_scan_times(ini_f, fin_f, n_freqs):
    """Returns the times-of-response obtained after using the IDFT

    Gets the vector of time-points for the time-domain representation
    of the radar signals, when the IDFT is used to convert the data
    from the frequency to the time-domain.

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    scan_times : array_like
        Vector of the time-points used to represent the radar signal
        when the IDFT is used to convert the data from the
        frequency-to-time domain, in seconds
    """

    # Get the incremental frequency step-size
    freq_step = get_freq_step(ini_f, fin_f, n_freqs)

    # Compute the corresponding time-domain step-size when the IDFT is
    # used to convert the measured data to the time domain
    time_step = 1 / (n_freqs * freq_step)

    # Get the vector of the time-points used to represent the
    # IDFT-obtained signal
    scan_times = np.linspace(0, n_freqs * time_step, n_freqs)

    return scan_times


def get_freqs(ini_t, fin_t, n_times):
    """Get the freq points obtained via the DFT

    Gets the vector of freq-points for the freq-domain representation
    of the radar signals, when the DFT is used to convert the data
    from the time domain to the freq domain

    Parameters
    ----------
    ini_t : float
        The initial time, in seconds, used in the time vector
    fin_t : float
        The final time, in seconds, used in the time vector
    n_times : int
        The number of time points used in the time vector.

    Returns
    -------
    freqs : array_like
        Vector of the freq-points used to represent the radar signal
        when the IDFT is used to convert the data from the
        frequency-to-time domain, in Hz
    """

    # Get the vector of time-points used to represent the time-domain
    # signal, assuming it was obtained via the IDFT
    times = np.linspace(ini_t, fin_t, n_times)

    # Find the incremental time step-size
    time_step = times[1] - times[0]

    # Convert to the incremental frequency step-size
    freq_step = 1 / (n_times * time_step)

    # Get the vector of the frequencies
    freqs = np.linspace(0, n_times * freq_step, n_times)

    return freqs


def get_time_step(ini_f, fin_f, n_freqs):
    """Returns the time-step obtained after IDFT time domain conversion

    Returns the time-step dt in seconds used when representing the
    radar signal after frequency-to-time domain conversion with the IDFT

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    dt : float
        The incremental time-step used to represent the time-domain
        signal after frequency-to-time domain conversion with the
        IDFT, in seconds
    """

    # Get the vector of time-points used to represent the radar signals
    # when the IDFT is used for time-domain conversion
    scan_time_vector = get_scan_times(ini_f, fin_f, n_freqs)

    # Find the incremental time step-size
    dt = (scan_time_vector[1] - scan_time_vector[0])

    return dt


def get_ant_scan_xys(ant_rad, n_ant_pos, ini_ant_ang=-130.0):
    """Returns the x,y positions of each antenna position in the scan

    Returns two vectors, containing the x- and y-positions in meters of
    the antenna during a scan.

    Parameters
    ----------
    ant_rad : float
        The radius of the trajectory of the antenna during the scan,
        in meters
    n_ant_pos : int
        The number of antenna positions used in the scan
    ini_ant_ang : float
        The initial angle offset (in deg) of the antenna from the
        negative x-axis

    Returns
    -------
    ant_xs : array_like
        The x-positions in meters of each antenna position used in the
        scan
    ant_ys : array_like
        The y-positions in meters of each antenna position used in the
        scan
    """

    # Find the polar angles of each of the antenna positions used in the
    # scan
    ant_angles = (np.linspace(0, (355 / 360) * 2 * np.pi, n_ant_pos) +
                  np.deg2rad(ini_ant_ang))
    ant_angles = np.flip(ant_angles)

    ant_xs = np.cos(ant_angles) * ant_rad  # Find the x-positions

    ant_ys = np.sin(ant_angles) * ant_rad  # Find the y-positions

    return ant_xs, ant_ys


def get_ant_xy_idxs(ant_rad, n_ant_pos, m_size, ini_ant_ang=-130.0):
    """Returns the x,y-pixel indices of each antenna position

    Returns two vectors, containing the x- and y-coordinates
    (pixel indices) of the antenna positions during a scan.

    Parameters
    ----------
    ant_rad : float
        The radius of the trajectory of the antenna during the scan,
        in meters
    n_ant_pos : int
        The number of antenna positions used in the scan
    m_size : int
        The number of pixels along one dimension used to define the
        model
    ini_ant_ang : float
        The initial angle offset (in deg) of the antenna from the
        negative x-axis

    Returns
    -------
    ant_x_idxs : array_like
        The x-coordinates (pixel indices) of each antenna position used
        in the scan
    ant_y_idxs : array_like
        The y-coordinates (pixel indices) of each antenna position used
        in the scan
    """

    # Get ratio between pixel width and distance
    pixdist_ratio = get_pixdist_ratio(m_size, ant_rad)

    # Get the ant x/y positions
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad, n_ant_pos,
                                      ini_ant_ang=ini_ant_ang)

    # Convert the antenna x,y positions to x,y coordinates, store as ints so
    # they can be used for indexing later
    ant_x_idxs = np.floor(ant_xs * pixdist_ratio + m_size // 2).astype(int)
    ant_y_idxs = np.floor(ant_ys * pixdist_ratio + m_size // 2).astype(int)

    return ant_x_idxs, ant_y_idxs


def get_pix_dists_angs(m_size, n_ant_pos, ant_rad, ini_ant_ang=-130.0):
    """Returns the distance and angle between each pixel and antenna

    Returns arrays in which each pixel is assigned its distance from the
    antenna and its angle off of the central axis of the antenna in
    degrees, for each antenna position used in a scan

    Parameters
    ----------
    m_size : int
        The number of pixels along one dimension used to define the
        image-space
    n_ant_pos : int
        The number of antenna positions used in the scan
    ant_rad : float
        The radius of the antenna trajectory during the scan in meters
    ini_ant_ang : float
        The initial angle offset (in deg) of the antenna from the
        negative x-axis

    Returns
    -------
    pix_dists : array_like
        3D arr in which each pixel is assigned the value corresponding
        to its distance from the antenna for that antenna position
        (i.e., for NxMxL arr, N is the number of antenna positions,
        MxL represents the model space)
    pix_angs : array_like
        3D arr in which each pixel is assigned the value corresponding
        to its angle off of the central angle of the antenna for that
        antenna position (i.e., for NxMxL arr, N is the number of
        antenna positions, MxL represents the model space)
    """

    # Get the antenna x,y positions
    ant_xs, ant_ys = get_ant_scan_xys(ant_rad, n_ant_pos,
                                      ini_ant_ang=ini_ant_ang)

    # Initialize arrays
    pix_dists = np.ones([n_ant_pos, m_size, m_size])
    pix_angs = np.ones_like(pix_dists)

    # Get the x,y positions of every pixel in the model
    pix_xs, pix_ys = get_xy_arrs(m_size, ant_rad)

    for ant_pos in range(n_ant_pos):

        # Find the x,y-position differences between each pixel and
        # the antenna for this ant_pos
        x_diffs = pix_xs - ant_xs[ant_pos]
        y_diffs = pix_ys - ant_ys[ant_pos]

        # Convert these x,y-position differences into an absolute
        # geometric difference
        pix_dists[ant_pos, :, :] = 2 * np.sqrt(x_diffs**2 + y_diffs**2)

        # Obtain the angles for these pixels
        pix_angs[ant_pos, :, :] = -np.arctan2(ant_xs[ant_pos] * y_diffs
                                              - ant_ys[ant_pos] * x_diffs,
                                              ant_xs[ant_pos] * x_diffs
                                              + ant_ys[ant_pos] * y_diffs)

    pix_angs = np.rad2deg(pix_angs)

    # Convert the angles so that they have the proper range (pixels
    # to the left of the antennas central axis have negative angles,
    # pixels to the right of the antennas central axis have positive
    # angles; i.e., the central axis of the antenna lies at the
    # angle 0deg).
    pix_angs[np.abs(pix_angs) < 90] = 0
    indexing_array = np.zeros_like(pix_angs)
    indexing_array[pix_angs < 0] = -1
    indexing_array[pix_angs > 0] = 1
    pix_angs[indexing_array == -1] += 180
    pix_angs[indexing_array == 1] -= 180
    pix_angs[indexing_array == 0] = 180

    return pix_dists, pix_angs
