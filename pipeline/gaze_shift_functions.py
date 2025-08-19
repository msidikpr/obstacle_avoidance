# import modules 

import os
import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm
import scipy as sp
import scipy.interpolate
from matplotlib import pyplot as plt
from datetime import datetime
from time import time
import matplotlib.patches as mpatches
import cv2
import subprocess
from scipy import stats
import sklearn.neighbors
import sklearn.linear_model
from scipy.ndimage import gaussian_filter

import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
from pipeline.helper_functions import interpolate_array



def drop_nearby_events(thin, avoid, win=0.25):
    """Drop events that fall near others.

    When eliminating compensatory eye/head movements which fall right after
    gaze-shifting eye/head movements, `thin` should be the compensatory event
    times.

    Parameters
    ----------
    thin : np.array
        Array of timestamps (as float in units of seconds) that
        should be thinned out, removing any timestamps that fall
        within `win` seconds of timestamps in `avoid`.
    avoid : np.array
        Timestamps to avoid being near.
    win : np.array
        Time (in seconds) that times in `thin` must fall before or
        after items in `avoid` by.
    
    """

    to_drop = np.array([c for c in thin for g in avoid if ((g>(c-win)) & (g<(c+win)))])
    thinned = np.delete(thin, np.isin(thin, to_drop))

    return thinned


def drop_repeat_events(eventT, onset=True, win=0.040):
    """Eliminate saccades repeated over sequential camera frames.

    Saccades sometimes span sequential camera frames, so that two or
    three sequential camera frames are labaled as saccade events, despite
    only being a single eye/head movement. This function keeps only a
    single frame from the sequence, either the first or last in the
    sequence.

    Parameters
    ----------
    eventT : np.array
        Array of saccade times (in seconds as float).
    onset : bool
        If True, a sequence of back-to-back frames labeled as a saccade will
        be reduced to only the first/onset frame in the sequence. If false, the
        last in the sequence will be used.
    win : float
        Distance in time (in seconds) that frames must follow each other to be
        considered repeating. Frames are 0.016 ms, so the default value, 0.020
        requires that frames directly follow one another.

    Returns
    -------
    thinned : np.array
        Array of saccade times, with repeated labels for single events removed.

    """

    duplicates = set([])

    for t in eventT:

        if onset:
            # keep first
            new = eventT[((eventT-t)<win) & ((eventT-t)>0)]
        
        else:
            # keep last
            new = eventT[((t-eventT)<win) & ((t-eventT)>0)]
        duplicates.update(list(new))

    thinned = np.sort(np.setdiff1d(eventT, np.array(list(duplicates)), assume_unique=True))
    
    return thinned


def read_timestamp_series(s):
    """ Read timestamps as a pd.Series and format time.

    Parameters
    ----------
    s : pd.Series
        Timestamps as a Series. Expected to be formated as
        hours:minutes:seconds.microsecond

    Returns
    -------
    output_time : np.array
        Returned as the number of seconds that have passed since the
        previous midnight, with microescond precision, e.g. 700.000000

    """

    # Expected string format for timestamps.
    fmt = '%H:%M:%S.%f'

    output_time = []

    if s.dtype != np.float64:

        for current_time in s:

            str_time = str(current_time).strip()

            try:
                t = datetime.strptime(str_time, fmt)

            except ValueError as v:
                # If the string had unexpected characters (too much precision) for
                # one timepoint, drop the extra characters.

                ulr = len(v.args[0].partition('unconverted data remains: ')[2])
                
                if ulr:
                    str_time = str_time[:-ulr]
            
            try:
                output_time.append(
                        (datetime.strptime(str_time, '%H:%M:%S.%f')
                            - datetime.strptime('00:00:00.000000', '%H:%M:%S.%f')
                            ).total_seconds())

            except ValueError:
                output_time.append(np.nan)

        output_time = np.array(output_time)

    else:
        output_time = s.values

    return output_time


def deinterlace(vidfile, savepath, exp_fps=30, quiet=True):
    """ Deinterlace a video.

    Parameters
    ----------
    path : str
        Path to the video file.
    savepath : str
        Path to save the new video file. Default is None.
    rotate : bool
        Whether to rotate the video 180 degrees. Default is True.
    exp_fps : int
        Expected frame rate of the video. If the video matches
        this frame rate (in Hz), it will be deinterlaced. Otherwise,
        it will be skipped. Default is 30 Hz.
    quiet : bool
        Whether to suppress the output from ffmpeg. Default is True.
    
    Returns
    -------
    savepath : str
        Path to the new video file.

    """

    # Open video, get frame count and rate
    cap = cv2.VideoCapture(vidfile)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Skip this video if it doesn't match the expected frame rate.
    if fps != exp_fps:
        return
        
    # Create the FFMPEG commands for video rotation.
    vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'

    # Create the full FFMPEG command
    cmd = ['ffmpeg', '-i', vidfile, '-vf', vf_val, '-c:v', 'libx264',
          '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
          '256k', '-y', savepath]
    
    # Set the log level
    if quiet is True:
        cmd.extend(['-loglevel', 'quiet'])

    # Run the FFMPEG command.
    subprocess.call(cmd)

    return savepath


def interp_timestamps(camT, use_medstep=False):
    """ Interpolate timestamps for double the number of
    frames. Compensates for video deinterlacing.
    
    Parameters
    ----------
    camT : np.array
        Camera timestamps aquired at 30Hz
    use_medstep : bool
        When True, the median diff(camT) will be used as the timestep
        in interpolation. If False, the timestep between each frame
        will be used instead.

    Returns
    -------
    camT_out : np.array
        Timestamps of camera interpolated so that there are twice the
        number of timestamps in the array. Each timestamp in camT will
        be replaced by two, set equal distances from the original.

    """

    camT_out = np.zeros(np.size(camT, 0)*2)
    medstep = np.nanmedian(np.diff(camT, axis=0))

    if use_medstep:
        
        # Shift each deinterlaced frame by 0.5 frame periods
        # forward/backwards assuming a constant framerate

        camT_out[::2] = camT - 0.25 * medstep
        camT_out[1::2] = camT + 0.25 * medstep
    
    elif not use_medstep:

        # Shift each deinterlaced frame by the actual time between
        # frames. If a camera frame was dropped, this approach will
        # be more accurate than `medstep` above.
        
        steps = np.diff(camT, axis=0, append=camT[-1]+medstep)
        camT_out[::2] = camT
        camT_out[1::2] = camT + 0.5 * steps

    return camT_out


def nanxcorr(x, y, maxlag=25):
    """ Cross correlation ignoring NaNs.

    Parameters
    ----------
    x : array
        Array of values.
    y : array
        Array of values to shift. Must be same length as x.
    maxlag : int
        Number of lags to shift y prior to testing correlation.
    
    Returns
    -------
    cc_out : array
        Cross correlation.
    lags : range
        Lag vector.

    """

    lags = range(-maxlag, maxlag)
    cc = []

    for i in range(0,len(lags)):
        
        # shift data
        yshift = np.roll(y, lags[i])
        
        # get index where values are usable in both x and yshift
        use = ~pd.isnull(x + yshift)
        
        # some restructuring
        x_arr = np.asarray(x, dtype=object)
        yshift_arr = np.asarray(yshift, dtype=object)

        x_use = x_arr[use]
        yshift_use = yshift_arr[use]
        
        # normalize
        x_use = (x_use - np.mean(x_use)) / (np.std(x_use) * len(x_use))

        yshift_use = (yshift_use - np.mean(yshift_use)) / np.std(yshift_use)
        
        # get correlation
        cc.append(np.correlate(x_use, yshift_use))

    cc_out = np.hstack(np.stack(cc))
    
    return cc_out, lags

def read_timestamp_file(timestamp_path,position_data_length=None,
                        force_timestamp_shift=False):
    """ Read timestamps from a .csv file.

    Parameters
    ----------
    position_data_length : None or int
        Number of timesteps in data from deeplabcut. This is used to
        determine whether or not the number of timestamps is too short
        for the number of video frames.
        Eyecam and Worldcam will have half the number of timestamps as
        the number of frames, since they are aquired as an interlaced
        video and deinterlaced in analysis. To fix this, timestamps need
        to be interpolated.
    force_timestamp_shift : bool
        When True, the timestamps will be interpolated regardless of
        whether or not the number of timestamps is too short for the
        number of frames. Default is False.

    Returns
    -------
    camT : np.array
        Timestamps of camera interpolated so that there are twice the
        number of timestamps in the array than there were in the provided
        csv file.

    """

    # Read data and set up format
    s = pd.read_csv(timestamp_path, encoding='utf-8',
                    engine='c', header=None).squeeze()
    
    # If the csv file has a header name for the column, (which is
    # is the int 0 for some early recordings), remove it.
    if s[0] == 0:
        s = s[1:]
    
    # Read the timestamps as a series and format them
    camT = read_timestamp_series(s)
    
    # Auto check if vids were deinterlaced
    if position_data_length is not None:

        if position_data_length > len(camT):

            # If the number of timestamps is too short for the number
            # of frames, interpolate the timestamps.

            camT = interp_timestamps(camT, use_medstep=False)
    
    # Force the times to be shifted if the user is sure it should be done
    if force_timestamp_shift is True:

        camT = interp_timestamps(camT, use_medstep=False)
    
    return camT


class Kalman():
    """ Kalman filter.

    From https://github.com/wehr-lab/autopilot/tree/parallax
    """

    def __init__(self, dim_state: int, dim_measurement: int = None, dim_control: int=0,
                 *args, **kwargs):

        self.dim_state = dim_state # type: int
        if dim_measurement is None:
            self.dim_measurement = self.dim_state # type: int
        else:
            self.dim_measurement = dim_measurement # type: int
        self.dim_control = dim_control # type: int

        self._init_arrays()

    def _init_arrays(self, state=None):
        """
        Initialize the arrays
        """
        # State arrays
        if state is not None:
            # TODO: check it's the right shape
            self.x_state = state
        else:
            self.x_state = np.zeros((self.dim_state, 1))

        # initialize kalman arrays
        self.P_cov               = np.eye(self.dim_state)                           # uncertainty covariance
        self.Q_proc_var          = np.eye(self.dim_state)                           # process uncertainty
        self.B_control           = np.eye(self.dim_control)                         # control transition matrix
        self.F_state_trans       = np.eye(self.dim_state)                           # x_state transition matrix
        if self.dim_state == self.dim_measurement:
            self.H_measure = np.eye(self.dim_measurement)
        else:
            self.H_measure           = np.zeros((self.dim_measurement, self.dim_state)) # measurement function
        self.R_measure_var       = np.eye(self.dim_measurement)                     # measurement uncertainty
        self._alpha_sq           = 1.                                               # fading memory control
        self.M_proc_measure_xcor = np.zeros((self.dim_state, self.dim_measurement)) # process-measurement cross correlation
        self.z_measure           = np.array([[None] * self.dim_measurement]).T

        # gain and residual are computed during the innovation step. We
        # save them so that in case you want to inspect them for various
        # purposes
        self.K = np.zeros((self.dim_state, self.dim_measurement)) # kalman gain
        self.y = np.zeros((self.dim_measurement, 1))
        self.S = np.zeros((self.dim_measurement, self.dim_measurement)) # system uncertainty
        self.SI = np.zeros((self.dim_measurement, self.dim_measurement)) # inverse system uncertainty

        # identity matrix. Do not alter this.
        self._I = np.eye(self.dim_state)

        # these will always be a copy of x_state,P_cov after predict() is called
        self.x_prior = self.x_state.copy()
        self.P_prior = self.P_cov.copy()

        # these will always be a copy of x_state,P_cov after update() is called
        self.x_post = self.x_state.copy()
        self.P_post = self.P_cov.copy()

    def predict(self, u=None, B=None, F=None, Q=None):
        """
        Predict next x_state (prior) using the Kalman filter x_state propagation
        equations.

        Parameters
        ----------

        u : np.array, default 0
            Optional control vector.

        B : np.array(dim_state, dim_u), or None
            Optional control transition matrix; a value of None
            will cause the filter to use `self.B_control`.

        F : np.array(dim_state, dim_state), or None
            Optional x_state transition matrix; a value of None
            will cause the filter to use `self.F_state_trans`.

        Q : np.array(dim_state, dim_state), scalar, or None
            Optional process noise matrix; a value of None will cause the
            filter to use `self.Q_proc_var`.
        """
        if B is None:
            B = self.B_control
        if F is None:
            F = self.F_state_trans
        if Q is None:
            Q = self.Q_proc_var
        elif np.isscalar(Q):
            Q = np.eye(self.dim_state) * Q

        # x_state = Fx + Bu
        if B is not None and u is not None:
            # make sure control vector is column
            u = np.atleast_2d(u)
            if u.shape[1] > u.shape[0]:
                u = u.T
            self.x_state = np.dot(F, self.x_state) + np.dot(B, u)
        else:
            self.x_state = np.dot(F, self.x_state)

        # P_cov = FPF' + Q_proc_var
        self.P_cov = self._alpha_sq * np.dot(np.dot(F, self.P_cov), F.T) + Q

        # save prior
        np.copyto(self.x_prior, self.x_state)
        np.copyto(self.P_prior, self.P_cov)

    def update(self, z, R=None, H=None):
        """
        Add a new measurement (z_measure) to the Kalman filter.

        If z_measure is None, nothing is computed. However, x_post and P_post are
        updated with the prior (x_prior, P_prior), and self.z_measure is set to None.

        Parameters
        ----------
        z : (dim_measurement, 1): array_like
            measurement for this update. z_measure can be a scalar if dim_measurement is 1,
            otherwise it must be convertible to a column vector.

            If you pass in a value of H_measure, z_measure must be a column vector the
            of the correct size.

        R : np.array, scalar, or None
            Optionally provide R_measure_var to override the measurement noise for this
            one call, otherwise  self.R_measure_var will be used.

        H : np.array, or None
            Optionally provide H_measure to override the measurement function for this
            one call, otherwise self.H_measure will be used.
        """
        # set to None to force recompute
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z_measure = np.array([[None] * self.dim_measurement]).T
            np.copyto(self.x_post, self.x_state)
            np.copyto(self.P_post, self.P_cov)
            self.y = np.zeros((self.dim_measurement, 1))
            return

        if R is None:
            R = self.R_measure_var
        elif np.isscalar(R):
            R = np.eye(self.dim_measurement) * R

        if H is None:
            z = self._reshape_z(z, self.dim_measurement, self.x_state.ndim)
            H = self.H_measure

        # y = z_measure - Hx
        # error (residual) between measurement and prediction
        self.y = z - np.dot(H, self.x_state)

        # common subexpression for speed
        PHT = np.dot(self.P_cov, H.T)

        # S = HPH' + R_measure_var
        # project system uncertainty into measurement space
        self.S = np.dot(H, PHT) + R
        self.SI = np.linalg.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = np.dot(PHT, self.SI)

        # x_state = x_state + Ky
        # predict new x_state with residual scaled by the kalman gain
        self.x_state = self.x_state + np.dot(self.K, self.y)

        # P_cov = (I-KH)P_cov(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P_cov = (I-KH)P_cov usually seen in the literature.

        I_KH = self._I - np.dot(self.K, H)
        self.P_cov = np.dot(np.dot(I_KH, self.P_cov), I_KH.T) + np.dot(np.dot(self.K, R), self.K.T)

        # save measurement and posterior x_state
        np.copyto(self.z_measure, z)
        np.copyto(self.x_post, self.x_state)
        np.copyto(self.P_post, self.P_cov)
        return self.x_state

    def _reshape_z(self, z, dim_z, ndim):
        """ ensure z is a (dim_z, 1) shaped vector"""

        z = np.atleast_2d(z)
        if z.shape[1] == dim_z:
            z = z.T

        if z.shape != (dim_z, 1):
            raise ValueError('z (shape {}) must be convertible to shape ({}, 1)'.format(z.shape, dim_z))

        if ndim == 1:
            z = z[:, 0]

        if ndim == 0:
            z = z[0, 0]

        return z

    def process(self, z, **kwargs):
        """
        Call predict and update, passing the relevant kwargs

        Args:
            z ():
            **kwargs ():

        Returns:
            np.ndarray: self.x_state
        """

        # prepare args for predict and call
        predict_kwargs = {k:kwargs.get(k, None) for k in ("u", "B", "F", "Q")}
        self.predict(**predict_kwargs)

        # same thing for update
        update_kwargs = {k: kwargs.get(k, None) for k in ('R', 'H')}
        return self.update(z, **update_kwargs)

    def residual_of(self, z):
        """
        Returns the residual for the given measurement (z_measure). Does not alter
        the x_state of the filter.
        """
        return z - np.dot(self.H_measure, self.x_prior)

    def measurement_of_state(self, x):
        """
        Helper function that converts a x_state into a measurement.

        Parameters
        ----------

        x : np.array
            kalman x_state vector

        Returns
        -------

        z_measure : (dim_measurement, 1): array_like
            measurement for this update. z_measure can be a scalar if dim_measurement is 1,
            otherwise it must be convertible to a column vector.
        """

        return np.dot(self.H_measure, x)

    @property
    def alpha(self):
        """
        Fading memory setting. 1.0 gives the normal Kalman filter, and
        values slightly larger than 1.0 (such as 1.02) give a fading
        memory effect - previous measurements have less influence on the
        filter's estimates. This formulation of the Fading memory filter
        (there are many) is due to Dan Simon [1]_.
        """
        return self._alpha_sq**.5

    @alpha.setter
    def alpha(self, value):
        if not np.isscalar(value) or value < 1:
            raise ValueError('alpha must be a float greater than 1')

        self._alpha_sq = value**2

class ImuOrientation():
    """ IMU sensor fusion

    From https://github.com/wehr-lab/autopilot/tree/parallax

    Compute absolute orientation (roll, pitch) from accelerometer and gyroscope measurements
    (eg from :class:`.hardware.i2c.I2C_9DOF` )

    Uses a :class:`.timeseries.Kalman` filter, and implements :cite:`patonisFusionMethodCombining2018a` to fuse
    the sensors

    Can be used with accelerometer data only, or with combined accelerometer/gyroscope data for
    greater accuracy

    Arguments:
        invert_gyro (bool): if the gyroscope's orientation is inverted from accelerometer measurement, multiply
            gyro readings by -1 before using
        use_kalman (bool): Whether to use kalman filtering (True, default), or return raw trigonometric
            transformation of accelerometer readings (if provided, gyroscope readings will be ignored)

    Attributes:
        kalman (:class:`.transform.timeseries.Kalman`): If ``use_kalman == True`` , the Kalman Filter.

    References:
        :cite:`patonisFusionMethodCombining2018a`
        :cite:`abyarjooImplementingSensorFusion2015`
    """

    def __init__(self, use_kalman:bool = True, invert_gyro:bool=False, *args, **kwargs):

        self.invert_gyro = invert_gyro # type: bool
        self._last_update = None # type: typing.Optional[float]
        self._dt = 0 # type: float
        # preallocate orientation array for filtered values
        self.orientation = np.zeros((2), dtype=float) # type: np.ndarray
        # and for unfiltered values so they aren't ambiguous
        self._orientation = np.zeros((2), dtype=float)  # type: np.ndarray

        self.kalman = None # type: typing.Optional[Kalman]
        if use_kalman:
            self.kalman = Kalman(dim_state=2, dim_measurement=2, dim_control=2)  # type: typing.Optional[Kalman]

    def process(self, accelgyro):
        """

        Args:
            accelgyro (tuple, :class:`numpy.ndarray`): tuple of (accelerometer[x,y,z], gyro[x,y,z]) readings as arrays, or
                an array of just accelerometer[x,y,z]

        Returns:
            :class:`numpy.ndarray`: filtered [roll, pitch] calculations in degrees
        """
        # check what we were given...
        if isinstance(accelgyro, (tuple, list)) and len(accelgyro) == 2:
            # combined accelerometer and gyroscope readings
            accel, gyro = accelgyro
        elif isinstance(accelgyro, np.ndarray) and np.squeeze(accelgyro).shape[0] == 3:
            # just accelerometer readings
            accel = accelgyro
            gyro = None
        else:
            # idk lol
            # self.logger.exception(f'Need input to be a tuple of accelerometer and gyroscope readings, or an array of accelerometer readings. got {accelgyro}')
            print('Error')
            return

        # convert accelerometer readings to roll and pitch
        pitch = 180*np.arctan2(accel[0], np.sqrt(accel[1]**2 + accel[2]**2))/np.pi
        roll = 180*np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2))/np.pi

        if self.kalman is None:
            # store orientations in external attribute if not using kalman filter
            self.orientation[:] = (roll, pitch)
            return self.orientation.copy()
        else:
            # if using kalman filter, use private array to store raw orientation
            self._orientation[:] = (roll, pitch)

        # TODO: Don't assume that we're fed samples instantatneously -- ie. once data representations are stable, need to accept a timestamp here rather than making one
        if self._last_update is None or gyro is None:
            # first time through don't have dt to scale gyro by
            self.orientation[:] = np.squeeze(self.kalman.process(self._orientation))
            self._last_update = time()
        else:
            if self.invert_gyro:
                gyro *= -1

            # get dt for time since last update
            update_time = time()
            self._dt = update_time-self._last_update
            self._last_update = update_time

            if self._dt>1:
                # if it's been really long, the gyro read is pretty much useless and will give ridiculous reads
                self.orientation[:] = np.squeeze(self.kalman.process(self._orientation))
            else:
                # run predict and update stages separately to incorporate gyro
                self.kalman.predict(u=gyro[0:2]*self._dt)
                self.orientation[:] = np.squeeze(self.kalman.update(self._orientation))

        return self.orientation.copy()
    

def open_dlc_h5(dlc_path):
    """ Open the .h5 file generated by DLC.

    Parameters
    ----------
    h5key : str
        The key to the .h5 file. Default is None.

    """
    
    pts = pd.read_hdf(dlc_path)
    
    # organize columns
    pts.columns = [' '.join(col[:][1:3]).strip() for col in pts.columns.values]
    
    pts = pts.rename(columns={pts.columns[n]: pts.columns[n].replace(' ', '_') for n in range(len(pts.columns))})
    pt_loc_names = pts.columns.values
    return pts, pt_loc_names


def fit_ellipse(x, y):
    """ Fit an ellipse to points labeled around the perimeter of pupil.

    Parameters
    ----------
    x : np.array
        Positions of points along the x-axis for a single video frame.
    y : np.array
        Positions of labeled points along the y-axis for a single video frame.

    Returns
    -------
    ellipse_dict : dict
        Parameters of the ellipse...
        X0 : center at the x-axis of the non-tilt ellipse
        Y0 : center at the y-axis of the non-tilt ellipse
        a : radius of the x-axis of the non-tilt ellipse
        b : radius of the y-axis of the non-tilt ellipse
        long_axis : radius of the long axis of the ellipse
        short_axis : radius of the short axis of the ellipse
        angle_to_x : angle from long axis to horizontal plane
        angle_from_x : angle from horizontal plane to long axis
        X0_in : center at the x-axis of the tilted ellipse
        Y0_in : center at the y-axis of the tilted ellipse
        phi : tilt orientation of the ellipse in radians

    """

    # Remove bias of the ellipse
    meanX = np.mean(x)
    meanY = np.mean(y)
    x = x - meanX
    y = y - meanY

    # Estimation of the conic equation
    X = np.array([x**2, x*y, y**2, x, y])
    X = np.stack(X).T
    a = np.dot(np.sum(X, axis=0), np.linalg.pinv(np.matmul(X.T,X)))

    # Extract parameters from the conic equation
    a, b, c, d, e = a[0], a[1], a[2], a[3], a[4]

    # Eigen decomp
    Q = np.array([[a, b/2],[b/2, c]])
    eig_val, eig_vec = np.linalg.eig(Q)

    # Get angle to long axis
    if eig_val[0] < eig_val[1]:
        angle_to_x = np.arctan2(eig_vec[1,0], eig_vec[0,0])
    else:
        angle_to_x = np.arctan2(eig_vec[1,1], eig_vec[0,1])

    angle_from_x = angle_to_x
    orientation_rad = 0.5 * np.arctan2(b, (c-a))
    cos_phi = np.cos(orientation_rad)
    sin_phi = np.sin(orientation_rad)

    a, b, c, d, e = [a*cos_phi**2 - b*cos_phi*sin_phi + c*sin_phi**2,
                    0,
                    a*sin_phi**2 + b*cos_phi*sin_phi + c*cos_phi**2,
                    d*cos_phi - e*sin_phi,
                    d*sin_phi + e*cos_phi]

    meanX, meanY = [cos_phi*meanX - sin_phi*meanY,
                    sin_phi*meanX + cos_phi*meanY]

    # Check if conc expression represents an ellipse
    test = a*c

    if test > 0:

        # Make sure coefficients are positive
        if a<0:
            a, c, d, e = [-a, -c, -d, -e]

        # Final ellipse parameters
        X0 = meanX - d/2/a
        Y0 = meanY - e/2/c
        F = 1 + (d**2)/(4*a) + (e**2)/(4*c)
        a = np.sqrt(F/a)
        b = np.sqrt(F/c)
        long_axis = 2*np.maximum(a,b)
        short_axis = 2*np.minimum(a,b)

        # Rotate axes backwards to find center point of
        # original tilted ellipse
        R = np.array([[cos_phi, sin_phi], [-sin_phi, cos_phi]])
        P_in = R @ np.array([[X0],[Y0]])
        X0_in = P_in[0][0]
        Y0_in = P_in[1][0]

        # Organize parameters in dictionary to return
        ellipse_dict = {
            'X0':X0,
            'Y0':Y0,
            'F':F,
            'a':a,
            'b':b,
            'long_axis':long_axis/2,
            'short_axis':short_axis/2,
            'angle_to_x':angle_to_x,
            'angle_from_x':angle_from_x,
            'cos_phi':cos_phi,
            'sin_phi':sin_phi,
            'X0_in':X0_in,
            'Y0_in':Y0_in,
            'phi':orientation_rad
        }

    else:

        # If the conic equation didn't return an ellipse, do not
        # return any real values and fill the dictionary with NaNs.
        dict_keys = ['X0','Y0','F','a','b','long_axis',
                        'short_axis','angle_to_x','angle_from_x',
                        'cos_phi','sin_phi','X0_in','Y0_in','phi']
        dict_vals = list(np.ones([len(dict_keys)]) * np.nan)

        ellipse_dict = dict(zip(dict_keys, dict_vals))
    
    return ellipse_dict

def classify_gaze_movements(dHead_trial, dGaze_trial, eyeT_trial, 
                            shifted_head=0, still_gaze=120, shifted_gaze=200):
    """
    Classify gaze movements into left/right gaze shifts and left/right compensatory movements.
    
    Parameters:
    - dHead_trial: Array of head movement values
    - dGaze_trial: Array of gaze movement values
    - eyeT_trial: Array of eye timestamps
    - shifted_head: Threshold for head movement (default 60)
    - still_gaze: Threshold for stationary gaze (default 120)
    - shifted_gaze: Threshold for gaze shift (default 200)
    
    Returns:
    - gazeL_trial_event: Left gaze shift events
    - gazeR_trial_event: Right gaze shift events
    - compL_trial_event: Left compensatory events
    - compR_trial_event: Right compensatory events
    - gazeL_idx: Indices of left gaze shifts in original array
    - gazeR_idx: Indices of right gaze shifts in original array
    - compL_idx: Indices of left compensatory movements in original array
    - compR_idx: Indices of right compensatory movements in original array
    """
    # Classify movements
    #gazeL_trial = eyeT_trial[(dHead_trial > shifted_head) &         \
    #             (dGaze_trial > shifted_gaze)]
    #gazeR_trial = eyeT_trial[(dHead_trial < -shifted_head) &        \
    #                 (dGaze_trial < -shifted_gaze)]
#
    #compL_trial = eyeT_trial[(dHead_trial > shifted_head) &         \
    #                 (dGaze_trial < still_gaze) &           \
    #                 (dGaze_trial > -still_gaze)]
    #compR_trial = eyeT_trial[(dHead_trial < -shifted_head) &        \
    #                 (dGaze_trial > -still_gaze) &          \
    #                 (dGaze_trial < still_gaze)]
    
    ## new criterian 
    gazeL_trial = eyeT_trial[(dGaze_trial > shifted_gaze)]
    gazeR_trial = eyeT_trial[(dGaze_trial < -shifted_gaze)]

    compL_trial = eyeT_trial[(dHead_trial > shifted_head) &         \
                     (dGaze_trial < still_gaze) &           \
                     (dGaze_trial > -still_gaze)]
    compR_trial = eyeT_trial[(dHead_trial < -shifted_head) &        \
                     (dGaze_trial > -still_gaze) &          \
                     (dGaze_trial < still_gaze)]
                


    # Eliminating compensatory eye/head movements which fall right after gaze-shifting eye/head movements
    #compL_trial = drop_nearby_events(compL_trial, gazeL_trial)  
    #compR_trial = drop_nearby_events(compR_trial, gazeR_trial)

    # Eliminate saccades repeated over sequential camera frames
    gazeL_trial_event = drop_repeat_events(gazeL_trial)
    gazeR_trial_event = drop_repeat_events(gazeR_trial)
    compL_trial_event = drop_repeat_events(compL_trial)
    compR_trial_event = drop_repeat_events(compL_trial)

    #gazeL_trial_event = gazeL_trial
    #gazeR_trial_event = gazeR_trial
    #compL_trial_event = compL_trial
    #compR_trial_event = compL_trial

    # Extract idx

    _, gazeL_trial_idx, _ = np.intersect1d(eyeT_trial,gazeL_trial_event, return_indices=True)
    _, gazeR_trial_idx, _ = np.intersect1d(eyeT_trial,gazeR_trial_event, return_indices=True)
    _, compL_trial_idx, _ = np.intersect1d(eyeT_trial,compL_trial_event, return_indices=True)
    _, compR_trial_idx, _ = np.intersect1d(eyeT_trial,compR_trial_event, return_indices=True)
    #print('gaze')
    return gazeL_trial_event, gazeR_trial_event, compL_trial_event, compR_trial_event,gazeL_trial_idx, gazeR_trial_idx, compL_trial_idx, compR_trial_idx



def calculate_speed(df): 
    for ind, row in df.iterrows():
        if row['odd'] == 'left': 
            head_cen_list = row['head_cen_x_cm'] 
            ts_odd_ind = np.argmax(head_cen_list>(row.leftspout_x_cm+5))
            ts_temp_time = np.diff(row['trial_timestamps'][ts_odd_ind:])
            temp_time = np.diff(row['trial_timestamps'])
        if row['odd'] == 'right':
            head_cen_list = row['head_cen_x_cm'] 
            ts_even_ind = np.argmax(head_cen_list<(row.rightspout_x_cm-5))
            ts_temp_time = np.diff(row['trial_timestamps'][ts_even_ind:])
            temp_time = np.diff(row['trial_timestamps'])
            #temp_time = np.diff(row['trial_timestamps'])
        ts_x = np.diff(row['ts_head_cen_x_cm']); ts_y = np.diff(row['ts_head_cen_y_cm'])
        x = np.diff(row['head_cen_x_cm']); y = np.diff(row['head_cen_y_cm'])
        if len(x) == len(temp_time):
            xspeed = list((x/temp_time)**2)
            ts_xspeed = list((ts_x/ts_temp_time)**2)
        elif len(x) > len(temp_time):
            xspeed = list((x[:len(temp_time)]/temp_time)**2)
            ts_xspeed = list((ts_x[:len(ts_temp_time)]/ts_temp_time)**2)
        elif len(x) < len(temp_time):
            xspeed = list((x/temp_time[:len(x)])**2)
            ts_xspeed = list((ts_x/ts_temp_time[:len(ts_x)])**2)
        if len(y) == len(temp_time):
            yspeed = list((y/temp_time)**2)
            ts_yspeed = list((ts_y/ts_temp_time)**2)
        elif len(y) > len(temp_time):
            yspeed = list((y[:len(temp_time)]/temp_time)**2)
            ts_yspeed = list((ts_y[:len(ts_temp_time)]/ts_temp_time)**2)
        elif len(y) < len(temp_time):
            yspeed = list((y/temp_time[:len(y)])**2)
            ts_yspeed = list((ts_y/ts_temp_time[:len(ts_y)])**2)

        df.at[ind, 'speed']  = gaussian_filter(np.sqrt(np.sum([xspeed, yspeed],axis=0)),3).astype(object)
        df.at[ind, 'ts_speed']  = gaussian_filter(np.sqrt(np.sum([ts_xspeed, ts_yspeed],axis=0)),3).astype(object)
        df.at[ind, 'avg_speed']  = np.nanmean(gaussian_filter(np.sqrt(np.sum([xspeed, yspeed],axis=0)),3).astype(object))
        df.at[ind, 'avg_ts_speed']  = np.nanmean(gaussian_filter(np.sqrt(np.sum([ts_xspeed, ts_yspeed],axis=0)),3).astype(object))
        distance = np.sqrt((x.astype(float))**2) + np.sqrt((y.astype(float))**2)
        ts_distance = np.sqrt((ts_x.astype(float))**2) + np.sqrt((ts_y.astype(float))**2)
        df.at[ind, 'distance'] = distance.astype(object)
        df.at[ind, 'ts_distance'] = ts_distance.astype(object)
        df.at[ind, 'total_distance'] = np.nansum(distance).astype(object)
        df.at[ind, 'ts_total_distance'] = np.nansum(ts_distance).astype(object)




def calculate_relative_distance(df):
    """calculates relavtive distance of nose to point on obstacle"""
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                try:
                    nose_x = row['head_cen_x_cm']
                    nose_y = row['head_cen_y_cm']
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']]) 
                        if np.isnan(e)==True:
                            distances.append(np.nan)
                        elif nose_y[i] > row['gt_obstacleTR_y_cm'] and nose_y[i] < row['gt_obstacleBR_y_cm'] :
                            obstalce_y = nose_y[i]
                            if nose_x[i] < obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) * -1
                                distances.append(distance)
                            if nose_x[i] > obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) 
                                distances.append(distance)
                            
                        else:
                            distance_to_top = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleTR_y_cm']))
                            distance_to_bottom = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleBR_y_cm']))
                            if distance_to_top < distance_to_bottom:
                                distance = distance_to_top
                                if nose_x[i] < obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] > obstalce_x:
                                    distance = distance 
                                    distances.append(distance)
                            if distance_to_top > distance_to_bottom:
                                distance = distance_to_bottom
                                if nose_x[i] < obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] > obstalce_x:
                                    distance = distance 
                                    distances.append(distance)

                            
                except IndexError:
                    distances.append(np.nan)

                
                df.at[ind,'distance_from_edge'] = interpolate_array(np.array(distances)).astype(object)
                df.at[ind,'len_distance_from_edge'] =interpolate_array(np.array(distances)).astype(object).size
            if direction =='left':
                try:
                    nose_x = row['head_cen_x_cm']
                    nose_y = row['head_cen_y_cm']
                    distances = []
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
                        if np.isnan(e)==True:
                            distances.append(np.nan) 
                        elif nose_y[i] > row['gt_obstacleTL_y_cm'] and nose_y[i] < row['gt_obstacleBL_y_cm']:
                            obstalce_y = nose_y[i]
                            if nose_x[i] > obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) * -1
                                distances.append(distance)
                            if nose_x[i] < obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) 
                                distances.append(distance)
                        else:
                            distance_to_top = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleTL_y_cm']))
                            distance_to_bottom = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleBL_y_cm']))
                            if distance_to_top < distance_to_bottom:
                                distance = distance_to_top
                                if nose_x[i] > obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] < obstalce_x:
                                    distance = distance 
                                    distances.append(distance)
                            if distance_to_top > distance_to_bottom:
                                distance = distance_to_bottom
                                if nose_x[i] > obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] < obstalce_x:
                                    distance = distance 
                                    distances.append(distance)
                except IndexError:
                    distances.append(np.nan)
            
                df.at[ind,'distance_from_edge'] = interpolate_array(np.array(distances)).astype(object)
                df.at[ind,'len_distance_from_edge'] =interpolate_array(np.array(distances)).astype(object).size


def ts_calculate_relative_distance(df):
    """calculates relavtive distance of nose to point on obstacle"""
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                try:
                    nose_x = row['ts_head_cen_x_cm']
                    nose_y = row['ts_head_cen_y_cm']
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']]) 
                        if np.isnan(e)==True:
                            distances.append(np.nan)
                        elif nose_y[i] > row['gt_obstacleTR_y_cm'] and nose_y[i] < row['gt_obstacleBR_y_cm'] :
                            obstalce_y = nose_y[i]
                            if nose_x[i] < obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) * -1
                                distances.append(distance)
                            if nose_x[i] > obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) 
                                distances.append(distance)
                            
                        else:
                            distance_to_top = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleTR_y_cm']))
                            distance_to_bottom = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleBR_y_cm']))
                            if distance_to_top < distance_to_bottom:
                                distance = distance_to_top
                                if nose_x[i] < obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] > obstalce_x:
                                    distance = distance 
                                    distances.append(distance)
                            if distance_to_top > distance_to_bottom:
                                distance = distance_to_bottom
                                if nose_x[i] < obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] > obstalce_x:
                                    distance = distance 
                                    distances.append(distance)

                            
                except IndexError:
                    distances.append(np.nan)

                
                df.at[ind,'ts_distance_from_edge'] = interpolate_array(np.array(distances)).astype(object)
                df.at[ind,'len_ts_distance_from_edge'] =interpolate_array(np.array(distances)).astype(object).size
            if direction =='left':
                try:
                    nose_x = row['ts_head_cen_x_cm']
                    nose_y = row['ts_head_cen_y_cm']
                    distances = []
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
                        if np.isnan(e)==True:
                            distances.append(np.nan) 
                        elif nose_y[i] > row['gt_obstacleTL_y_cm'] and nose_y[i] < row['gt_obstacleBL_y_cm']:
                            obstalce_y = nose_y[i]
                            if nose_x[i] > obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) * -1
                                distances.append(distance)
                            if nose_x[i] < obstalce_x:
                                distance = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,obstalce_y)) 
                                distances.append(distance)
                        else:
                            distance_to_top = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleTL_y_cm']))
                            distance_to_bottom = np.abs(calculate_distances(nose_x[i],nose_y[i],obstalce_x,row['gt_obstacleBL_y_cm']))
                            if distance_to_top < distance_to_bottom:
                                distance = distance_to_top
                                if nose_x[i] > obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] < obstalce_x:
                                    distance = distance 
                                    distances.append(distance)
                            if distance_to_top > distance_to_bottom:
                                distance = distance_to_bottom
                                if nose_x[i] > obstalce_x:
                                    distance = distance * -1
                                    distances.append(distance)
                                if nose_x[i] < obstalce_x:
                                    distance = distance 
                                    distances.append(distance)
                except IndexError:
                    distances.append(np.nan)
            
                df.at[ind,'ts_distance_from_edge'] = interpolate_array(np.array(distances)).astype(object)
                df.at[ind,'len_ts_distance_from_edge'] =interpolate_array(np.array(distances)).astype(object).size


def calculate_distances(x_points, y_points, x_reference, y_reference):
    """
    Calculate the distances of a set of (x, y) points from a reference point.

    Args:
        x_points (array-like): Array of x-coordinates of the points.
        y_points (array-like): Array of y-coordinates of the points.
        x_reference (float): The x-coordinate of the reference point.
        y_reference (float): The y-coordinate of the reference point.

    Returns:
        list: List of distances from the reference point to each point in the set.
    """
    # Convert input arrays to NumPy arrays
    x_points = np.array(x_points)
    y_points = np.array(y_points)

    # Calculate the distances
    distances = np.sqrt((x_points - x_reference)**2 + (y_points - y_reference)**2)

    return distances



def compute_onset_ratios( eye_velocity, head_velocity, onset_indices):
    """
    Compute normalized eye/head ratio [-1, 1] at exact gaze shift onsets.
    
    Args:
        gaze_velocity (np.array): 1D array of gaze velocities (°/s)
        eye_velocity (np.array): 1D array of eye velocities (°/s)
        head_velocity (np.array): 1D array of head velocities (°/s)
        onset_indices (list): List of gaze shift onset indices
        
    Returns:
        np.array: Normalized ratios (1=eye, -1=head) at each onset
    """
    ratios = []
    
    for idx in onset_indices:
        # Get velocities at onset
        eye_velocity = np.abs(eye_velocity)
        head_velocity = np.abs(head_velocity)

        eye_vel = eye_velocity[idx]
        head_vel = head_velocity[idx]

        #print(eye_vel,head_vel,np.max(eye_velocity) ,np.max(head_velocity))
        #norm_eye = eye_vel/np.nanmax(eye_velocity)
        #norm_head = head_vel/np.nanmax(head_velocity)

        #ratio = (norm_eye-norm_head)/(norm_eye+norm_head) # returns value -1 to 1 -1 to 0 head bias. 0 to 1 eye bias


        norm_eye = eye_vel / np.nanmax(eye_velocity) #* eye_vel  
        norm_head = head_vel / np.nanmax(head_velocity)  #* head_vel
        ratio = (norm_eye-norm_head)/(norm_eye+norm_head)  # returns value -1 to 1 -1 to 0 head bias. 0 to 1 eye bias
      
        ##print(norm_eye,norm_head)
        #
        ## Compute normalized ratio [-1, 1]
        #if (norm_eye + norm_head) == 0:
        #    ratio = 0.0  # Avoid division by zero
        #else:
        #    raw_ratio = norm_eye / (norm_eye + norm_head)  # [0, 1]
        #    ratio = 2 * raw_ratio - 1  # Map to [-1, 1]
        
        ratios.append(ratio)
    
    return np.array(ratios)


def calculate_relative_distance_goal_ts(df):
    '''calculates change in distance from nose to goal port using trial start trace '''
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                nose_x = row['ts_head_cen_x_cm'].astype(float)
                nose_y = row['ts_head_cen_y_cm'].astype(float)
                port_x_tar = row.leftspout_x_cm
                port_y_tar = row.leftspout_y_cm
                port_x_start = row.rightspout_x_cm
                port_y_start = row.rightspout_y_cm
                distances_tar = []
                distances_start = []
                for i,e in enumerate(nose_x):
                  distance_tar = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_tar,port_y_tar))
                  distance_start = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_start,port_y_start))
                  distances_tar.append(distance_tar)
                  distances_start.append(distance_start)
                df.at[ind,'ts_distance_from_target_port'] = np.array(distances_tar).astype(object)
                df.at[ind,'ts_distance_from_start_port'] = np.array(distances_start).astype(object)

            else:
                nose_x = row['ts_head_cen_x_cm'].astype(float)
                nose_y = row['ts_head_cen_y_cm'].astype(float)
                port_x_tar = row.rightspout_x_cm
                port_y_tar = row.rightspout_y_cm
                port_x_start = row.leftspout_x_cm
                port_y_start = row.leftspout_y_cm
                distances_tar = []
                distances_start = []
                for i,e in enumerate(nose_x):
                  distance_tar = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_tar,port_y_tar))
                  distance_start = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_start,port_y_start))
                  distances_tar.append(distance_tar)
                  distances_start.append(distance_start)
                df.at[ind,'ts_distance_from_target_port'] = np.array(distances_tar).astype(object)
                df.at[ind,'ts_distance_from_start_port'] = np.array(distances_start).astype(object)
                
def calculate_relative_distance_goal(df):
    '''calculates change in distance from nose to goal port using trial start trace '''
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                nose_x = row['head_cen_x_cm'].astype(float)
                nose_y = row['head_cen_y_cm'].astype(float)
                port_x_tar = row.leftspout_x_cm
                port_y_tar = row.leftspout_y_cm
                port_x_start = row.rightspout_x_cm
                port_y_start = row.rightspout_y_cm
                distances_tar = []
                distances_start = []
                for i,e in enumerate(nose_x):
                  distance_tar = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_tar,port_y_tar))
                  distance_start = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_start,port_y_start))
                  distances_tar.append(distance_tar)
                  distances_start.append(distance_start)
                df.at[ind,'distance_from_target_port'] = np.array(distances_tar).astype(object)
                df.at[ind,'distance_from_start_port'] = np.array(distances_start).astype(object)

            else:
                nose_x = row['head_cen_x_cm'].astype(float)
                nose_y = row['head_cen_y_cm'].astype(float)
                port_x_tar = row.rightspout_x_cm
                port_y_tar = row.rightspout_y_cm
                port_x_start = row.leftspout_x_cm
                port_y_start = row.leftspout_y_cm
                distances_tar = []
                distances_start = []
                for i,e in enumerate(nose_x):
                  distance_tar = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_tar,port_y_tar))
                  distance_start = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_start,port_y_start))
                  distances_tar.append(distance_tar)
                  distances_start.append(distance_start)
                df.at[ind,'distance_from_target_port'] = np.array(distances_tar).astype(object)
                df.at[ind,'distance_from_start_port'] = np.array(distances_start).astype(object)


def find_event_frames(video_timestamps, event_timestamps):
    """
    Find which events occurred during the video and their corresponding video frames.
    Returns (None, None) if no events are found within the video duration.
    
    Args:
        video_timestamps (array-like): Monotonically increasing array of timestamps for each video frame.
        event_timestamps (array-like): Array of timestamps when events occurred.
        
    Returns:
        tuple: (event_indices, frame_indices) or (None, None) if no events found
               - event_indices: Indices of the original event array that fall within video duration
               - frame_indices: Corresponding video frame indices for these events
    """
    video_ts = np.asarray(video_timestamps)
    event_ts = np.asarray(event_timestamps)
    
    if len(video_ts) == 0 or len(event_ts) == 0:
        return None, None
    
    # Find which events are within video duration and their original indices
    valid_mask = (event_ts >= video_ts[0]) & (event_ts <= video_ts[-1])
    event_indices = np.where(valid_mask)[0]
    valid_events = event_ts[valid_mask]
    
    # Return None if no valid events found
    if len(valid_events) == 0:
        return None, None
    
    # Find corresponding frame indices
    frame_indices = np.searchsorted(video_ts, valid_events, side='right') - 1
    frame_indices = np.clip(frame_indices, 0, len(video_ts) - 1)
    
    return event_indices.tolist(), frame_indices.tolist()