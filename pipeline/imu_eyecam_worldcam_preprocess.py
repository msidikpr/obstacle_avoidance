import json, os, cv2
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import xarray as xr
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import glob
import h5py
import subprocess
import deeplabcut
from datetime import datetime


from scipy.ndimage import gaussian_filter1d


import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')

import warnings
warnings.filterwarnings('ignore')

from src.base import BaseInput
from src.topcam import Topcam
from src.utils.auxiliary import find_index_in_list,flatten_series
from src.utils.path import find
#from utils.base_functions import *
from pipeline.helper_functions import list_columns,interpolate_array
from pipeline.gaze_shift_functions import *
from pipeline.oa_calculations import distance_calcs,heading_calcs,cluster_obstacle,deveation,lateral_error,df_tortuosity,head_angle_velocity,start


class IMU_eyecam_world_ephys_Processing(BaseInput):
    def __init__(self, metadata_path):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.path = self.metadata['path']
        self.dates_list = [i for i in list(self.metadata.keys()) if i != 'path']
        self.dlc_project = r"D:\obstacle_avoidance\deeplabcut\eye_tracking_050825-Mike-2025-05-08\config.yaml"
        self.dlc_project_topdown = r"D:\obstacle_avoidance\deeplabcut\obstacle_avoidance_withcam_041825-Mike-2025-04-18\config.yaml"

    def preprocess_topdown(self):
        data_path = Path(self.path).expanduser()
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in tqdm(use_animals,'animal'):
                for task in os.listdir(data_path / date / ani):
                    # get the reye video and reye time stamp paths 
                    Top_vid = [str(i) for i in list((data_path / date / ani/ task).rglob('*.avi')) if 'TOP1.avi' in str(i)]
                    
                    if self.metadata[date][ani][task]['dlc'] == False:
                        video = Top_vid[0]
                        deeplabcut.analyze_videos(config=self.dlc_project_topdown,videos=[video])
                    else:
                        continue
    def preprocess_eyecam(self):
        data_path = Path(self.path).expanduser()
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in tqdm(use_animals,'animal'):
                for task in os.listdir(data_path / date / ani):
                    # get the reye video and reye time stamp paths
                    #if 'cam' not in task:
                    #    print('non cam')
                    #else: 
                    REYE_vid = [str(i) for i in list((data_path / date / ani/ task).rglob('*.avi')) if 'REYE.avi' in str(i)]
                    #print(REYE_vid)
                    REYE_ts = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv')) if 'REYE_BonsaiTS.csv' in str(i)]
                    #get the strings for namiing
                    print(REYE_vid)
                    vid_name =  os.path.split(REYE_vid[0])[1]
                    key_pieces = vid_name.split('.')[:-1]
                    key = '.'.join(key_pieces)
                    avi_out_path = os.path.join((data_path / date / ani/ task), (key + 'deinter.avi'))
                    csv_out_path = os.path.join((data_path / date / ani/ task), (key + '_BonsaiTSformatted.csv'))
                    #dienterlace video
                    if os.path.exists(avi_out_path):
                        print('already deinterlace')
                    else:
                        deinterlace(REYE_vid[0], avi_out_path, exp_fps=30, quiet=True)
                        print('deinterlace finish')
                        #get new timestamps 
                        cap = cv2.VideoCapture(REYE_vid[0])
                        # get some info about the video
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # number of total frames
                        frame_count_deinter = frame_count * 2
                        fps = cap.get(cv2.CAP_PROP_FPS) # frame rate
                    # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                        self.timestamp_path = REYE_ts[0]
                        csv_out = pd.DataFrame(self.read_timestamp_file(int(frame_count_deinter)))
                     # save new timestamps
                        print('save new timestamps')
                        csv_out.to_csv(csv_out_path, index=False) 
                    if self.metadata[date][ani][task]['dlc'] == False:
                        print('dlc')
                        deeplabcut.analyze_videos(config=self.dlc_project,videos=[avi_out_path])
                    if self.metadata[date][ani][task]['labeled'] == False:
                        print('labeled')
                        deeplabcut.create_labeled_video(self.dlc_project,videos=[avi_out_path])
                    else:
                        print('already preprocessed')
                        continue
    def process_gazeshift(self):
        """save dictonary of gazeshift params as npz file"""
        data_path = Path(self.path).expanduser()
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in tqdm(use_animals,'animal'):
                for task in os.listdir(data_path / date / ani):
                    ## imu 
                    #import IMU data
                    imu_timestamps_path = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv')) if '_Ephys_BonsaiBoardTS' in str(i)][0]
                    imu_path = [str(i) for i in list((data_path / date / ani/ task).rglob('*.bin')) if '_IMU' in str(i)][0]
                    # Set up datatypes and names for each channel
                    dtypes = np.dtype([
                        ("acc_x",np.uint16),
                        ("acc_y",np.uint16),
                        ("acc_z",np.uint16),
                        ("none1",np.uint16),
                        ("gyro_x",np.uint16),
                        ("gyro_y",np.uint16),
                        ("gyro_z",np.uint16),
                        ("none2",np.uint16)
                    ])
                    binary_in = pd.DataFrame(np.fromfile(imu_path, dtypes, -1, ''))
                    binary_in = binary_in.drop(columns=['none1','none2'])
                    # Convert to - 5V to + 5V (from ints)
                    data = 10 * (binary_in.astype(float)/(2**16) - 0.5)

                    # Downsample in time dimension
                    imu_samprate = 30000 # fixed constant
                    imu_dwnsmpl = 100 # imu_samprate is 30000, and we are using 100, therefore making it 30000/100 = 300hz. 
                                      # if you want to boost up the temporal resolution up to 500hz, you can use 60.

                    data = data.iloc[::imu_dwnsmpl]

                    # Calculate new sample rate
                    samp_freq = imu_samprate / imu_dwnsmpl

                    # Alphabetize columns
                    data = data.reindex(sorted(data.columns), axis=1)

                    # Read in timestamps
                    csv_data = pd.read_csv(imu_timestamps_path).squeeze()
                    pdtime = pd.DataFrame(read_timestamp_series(csv_data))

                    # Get first/last timepoint, num_samples
                    t0 = pdtime.iloc[0,0]
                    num_samp = np.size(data,0)

                    # Samples start at t0, and are acquired at rate of
                    # 'ephys_sample_rate'/ 'imu_downsample'
                    newtime = list(np.array(t0 + np.linspace(0, num_samp-1, num_samp) / samp_freq))


                    IMU = ImuOrientation()
                    # Convert accelerometer to g
                    zero_reading = 2.9
                    sensitivity = 1.6
                    acc = pd.DataFrame.to_numpy((data[['acc_x', 'acc_y', 'acc_z']]-zero_reading)*sensitivity)

                    # Convert gyro to deg/sec
                    gyro = pd.DataFrame.to_numpy((data[['gyro_x', 'gyro_y', 'gyro_z']] - pd.DataFrame.mean(data[['gyro_x', 'gyro_y', 'gyro_z']]))*400)

                    # Collect roll & pitch
                    roll_pitch = np.zeros([len(acc),2])
                    for x in range(len(acc)):
                        roll_pitch[x,:] = IMU.process((acc[x],gyro[x])) # update by row
                    roll_pitch = pd.DataFrame(roll_pitch, columns=['roll','pitch'])

                    # Collect the data together
                    all_data = pd.concat([
                        data.reset_index(),
                        pd.DataFrame(acc).reset_index(),
                        pd.DataFrame(gyro).reset_index(),
                        roll_pitch
                    ], axis=1).drop(labels='index',axis=1)

                    # Set up column names
                    all_data.columns = [
                        'acc_x_raw','acc_y_raw','acc_z_raw',
                        'gyro_x_raw','gyro_y_raw','gyro_z_raw',
                        'acc_x','acc_y','acc_z',
                        'gyro_x', 'gyro_y', 'gyro_z',
                        'roll', 'pitch'
                    ]
                    output_data = xr.DataArray(all_data, dims=['sample','channel'])
                    imu_data = output_data.assign_coords({'sample':newtime})

                    imuT_raw = imu_data.sample # imu timestamps

                    # Raw gyro values
                    gyro_x_raw = imu_data.sel(channel='gyro_x_raw').values
                    gyro_y_raw = imu_data.sel(channel='gyro_y_raw').values
                    gyro_z_raw = imu_data.sel(channel='gyro_z_raw').values

                    # Gyro values in degrees
                    gyro_x = imu_data.sel(channel='gyro_x').values
                    gyro_y = imu_data.sel(channel='gyro_y').values
                    gyro_z = imu_data.sel(channel='gyro_z').values

                    # Pitch and roll in deg
                    roll = imu_data.sel(channel='roll').values
                    pitch = imu_data.sel(channel='pitch').values

                    #process eyecam
                    dlc_h5 = [str(i) for i in list((data_path / date / ani/ task).rglob('*.h5')) if 'REYEdeinterDLC' in str(i)][0]
                    eye_timestamp_path = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv')) if 'REYE_BonsaiTS.csv' in str(i)][0]
                    pts, pt_names =open_dlc_h5(dlc_h5)
                    eyeT = read_timestamp_file(eye_timestamp_path, len(pts))
                    xrpts = xr.DataArray(pts, dims=['frame', 'point_loc'])

                    data = xrpts
                    names = list(data['point_loc'].values)
                    thresh = 0.99 #configuration value

                    x_locs = []
                    y_locs = []
                    likeli_locs = []

                    # seperate the lists of point names into x, y, and likelihood
                    for loc_num in range(0, len(names)):
                        loc = names[loc_num]
                        if '_x' in loc:
                            x_locs.append(loc)
                        elif '_y' in loc:
                            y_locs.append(loc)
                        elif 'likeli' in loc:
                            likeli_locs.append(loc)
                    # get the xarray, split up into x, y,and likelihood
                    for loc_num in range(0, len(likeli_locs)):
                        pt_loc = likeli_locs[loc_num]
                        if loc_num == 0:
                            likeli_pts = data.sel(point_loc=pt_loc)
                        elif loc_num > 0:
                            likeli_pts = xr.concat([likeli_pts, data.sel(point_loc=pt_loc)],
                                                    dim='point_loc', fill_value=np.nan)

                    for loc_num in range(0, len(x_locs)):
                        pt_loc = x_locs[loc_num]
                        # threshold from likelihood
                        data.sel(point_loc=pt_loc)[data.sel(point_loc=pt_loc) < thresh] = np.nan
                        if loc_num == 0:
                            x_pts = data.sel(point_loc=pt_loc)
                        elif loc_num > 0:
                            x_pts = xr.concat([x_pts, data.sel(point_loc=pt_loc)],
                                                dim='point_loc', fill_value=np.nan)

                    for loc_num in range(0, len(y_locs)):
                        pt_loc = y_locs[loc_num]
                        # threshold from likelihood
                        data.sel(point_loc=pt_loc)[data.sel(point_loc=pt_loc) < thresh] = np.nan
                        if loc_num == 0:
                            y_pts = data.sel(point_loc=pt_loc)
                        elif loc_num > 0:
                            y_pts = xr.concat([y_pts, data.sel(point_loc=pt_loc)],
                                                dim='point_loc', fill_value=np.nan)

                    x_pts = xr.DataArray.squeeze(x_pts)
                    y_pts = xr.DataArray.squeeze(y_pts)
                    likeli_pts = xr.DataArray.squeeze(likeli_pts)

                    # convert to dataframe, transpose so points are columns
                    x_vals = xr.DataArray.to_pandas(x_pts).T
                    y_vals = xr.DataArray.to_pandas(y_pts).T
                    likeli_vals = xr.DataArray.to_pandas(likeli_pts).T
                    likelihood_in = likeli_vals.values

                    # Subtract center of IR light reflection from points around the pupil.

                    spot_xcent = np.mean(x_vals.iloc[:,-5:], 1)
                    spot_ycent = np.mean(y_vals.iloc[:,-5:], 1)
                    spot_likelihood = likelihood_in[:,-5:].copy()

                    likelihood = likelihood_in[:,:-5]

                    x_vals = x_vals.iloc[:,:-5].subtract(spot_xcent, axis=0)
                    y_vals = y_vals.iloc[:,:-5].subtract(spot_ycent, axis=0)

                    x_vals = x_vals.iloc[:,:-2]
                    y_vals = y_vals.iloc[:,:-2]
                    likelihood = likelihood[:,:-2]

                    pupil_count = np.sum(likelihood >= thresh, 1)

                    spot_count = np.sum(spot_likelihood >= thresh, 1)

                    usegood_eye = (pupil_count >= 7) &               \
                                    (spot_count >= 5)

                    usegood_eyecalib = (pupil_count >= 8) &          \
                                        (spot_count >= 5)

                    usegood_reflec = (spot_count >=5)


                    # Threshold out pts more than a given distance away from nanmean of that point
                    std_thresh_x = np.empty(np.shape(x_vals))

                    for point_loc in range(0,np.size(x_vals, 1)):
                        _val = x_vals.iloc[:,point_loc]
                        std_thresh_x[:,point_loc] = (np.abs(np.nanmean(_val) - _val)                \
                                        / 24) > 24

                    std_thresh_y = np.empty(np.shape(y_vals))

                    for point_loc in range(0,np.size(x_vals, 1)):
                        _val = y_vals.iloc[:,point_loc]
                        std_thresh_y[:,point_loc] = (np.abs(np.nanmean(_val) - _val)                \
                                        / 24) > 4.1

                    std_thresh_x = np.nanmean(std_thresh_x, 1)
                    std_thresh_y = np.nanmean(std_thresh_y, 1)

                    x_vals[std_thresh_x > 0] = np.nan
                    y_vals[std_thresh_y > 0] = np.nan

                    cols = [
                        'X0',           # 0
                        'Y0',           # 1
                        'F',            # 2
                        'a',            # 3
                        'b',            # 4
                        'long_axis',    # 5
                        'short_axis',   # 6
                        'angle_to_x',   # 7
                        'angle_from_x', # 8
                        'cos_phi',      # 9
                        'sin_phi',      # 10
                        'X0_in',        # 11
                        'Y0_in',        # 12
                        'phi'           # 13
                    ]
                    #Fit ellipse

                    ellipse = np.empty([len(usegood_eye), 14])

                    # Step through each frame, fit an ellipse to points, and add ellipse
                    # parameters to array with data for all frames together.
                    linalgerror = 0
                    for step in tqdm(range(0,len(usegood_eye))):

                        if usegood_eye[step] == True:

                            try:
                            
                                e_t = fit_ellipse(x_vals.iloc[step].values,
                                                        y_vals.iloc[step].values)

                                ellipse[step] = [
                                    e_t['X0'],              # 0
                                    e_t['Y0'],              # 1
                                    e_t['F'],               # 2
                                    e_t['a'],               # 3
                                    e_t['b'],               # 4
                                    e_t['long_axis'],       # 5
                                    e_t['short_axis'],      # 6
                                    e_t['angle_to_x'],      # 7
                                    e_t['angle_from_x'],    # 8
                                    e_t['cos_phi'],         # 9
                                    e_t['sin_phi'],         # 10
                                    e_t['X0_in'],           # 11
                                    e_t['Y0_in'],           # 12
                                    e_t['phi']              # 13
                                ]

                            except np.linalg.LinAlgError as e:
                            
                                linalgerror = linalgerror + 1
                                ellipse[step] = list(np.ones([len(cols)]) * np.nan)

                        elif usegood_eye[step] == False:
                        
                            ellipse[step] = list(np.ones([len(cols)]) * np.nan)

                    print('LinAlg error count = ' + str(linalgerror))

                    # List of all places where the ellipse meets threshold
                    R = np.linspace(0, 2*np.pi, 100)

                    # (short axis / long axis) < thresh
                    usegood_ellipcalb = np.where((usegood_eyecalib == True)                     \
                            & ((ellipse[:,6] / ellipse[:,5]) < 0.85))

                    # Limit number of frames used for calibration
                    f_lim = 50000
                    if np.size(usegood_ellipcalb,1) > f_lim:
                        shortlist = sorted(np.random.choice(usegood_ellipcalb[0],
                                            size=f_lim, replace=False))
                    else:
                        shortlist = usegood_ellipcalb

                    # Find camera center
                    A = np.vstack([np.cos(ellipse[shortlist,7]),
                                    np.sin(ellipse[shortlist,7])])

                    b = np.expand_dims(np.diag(A.T @ np.squeeze(ellipse[shortlist, 11:13].T)), axis=1)

                    cam_cent = np.linalg.inv(A @ A.T) @ A @ b

                    # Ellipticity and scale
                    ellipticity = (ellipse[shortlist,6] / ellipse[shortlist,5]).T

                    try:
                        scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *                       \
                        (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=0)))       \
                        / np.sum(1 - (ellipticity)**2)

                    except ValueError:
                    
                        scale = np.nansum(np.sqrt(1 - (ellipticity)**2) *                       \
                        (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=1)))       \
                        / np.sum(1 - (ellipticity)**2)


                    # Horizontal orientation (THETA)
                    theta = np.arcsin((ellipse[:,11] - cam_cent[0]) / scale)

                    # Vertical orientation (PHI)
                    phi = np.arcsin((ellipse[:,12] - cam_cent[1]) / np.cos(theta) / scale)
                    # Organize data to return as an xarray of most essential parameters
                    ellipse_df = pd.DataFrame({
                        'theta':list(theta),
                        'phi':list(phi),
                        'longaxis':list(ellipse[:,5]),
                        'shortaxis':list(ellipse[:,6]),
                        'X0':list(ellipse[:,11]),
                        'Y0':list(ellipse[:,12]),
                        'ellipse_phi':list(ellipse[:,7])
                    })

                    ellipse_param_names = [
                        'theta',
                        'phi',
                        'longaxis',
                        'shortaxis',
                        'X0',
                        'Y0',
                        'ellipse_phi'
                    ]

                    ellipse_out = xr.DataArray(ellipse_df,
                        coords=[('frame', range(0, len(ellipse_df))),
                                ('ellipse_params', ellipse_param_names)],
                        dims=['frame', 'ellipse_params'])

                    ellipse_out.attrs['cam_center_x'] = cam_cent[0,0]
                    ellipse_out.attrs['cam_center_y'] = cam_cent[1,0]

                    ## top1 dlc 
                    top_dlc_path = [str(i) for i in list((data_path / date / ani/ task).rglob('*.h5')) if '_TOP1DLC' in str(i)][0]
                    top_timestamp_path = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv')) if '_TOP1_BonsaiTS' in str(i)][0]
                    topT = read_timestamp_file(top_timestamp_path, len(pts))

                    ## gaze shif criterion 
                    shifted_head = 0
                    still_gaze = 120
                    shifted_gaze = 200

                    th = np.rad2deg(ellipse_out.sel(ellipse_params = 'theta').values)
                    theta = th - np.nanmean(th)
                    dEye = np.diff(theta)
                    ##gyro x for timestamp correction 
                    ## gyro z for dHead 
                    Openephys_time = read_timestamp_file(imu_timestamps_path)
                    OpenephysT0 = Openephys_time[0]

                    # match timestamps
                    pts, pt_names = open_dlc_h5(dlc_h5)
                    eyeT = read_timestamp_file(eye_timestamp_path, len(pts))
                    eyeT = eyeT - OpenephysT0
                    imuT_raw = imu_data.sample
                    imuT_raw = imuT_raw - OpenephysT0
                    topT = read_timestamp_file(top_timestamp_path, len(pts))
                    topT = topT - OpenephysT0

                    # account for the drift (it is a system known issue that open ephys box has a weird drift phenomenon)

                    lag_range = np.arange(-0.2, 0.2, 0.002)
                    cc = np.zeros(np.shape(lag_range))

                    t1 = np.arange(5, len(dEye)/60 - 120, 20).astype(int)
                    t2 = t1 + 60
                    offset = np.zeros(np.shape(t1))
                    ccmax = np.zeros(np.shape(t1))
                    imu_interp = scipy.interpolate.interp1d(imuT_raw, gyro_x)

                    for tstart in tqdm(range(len(t1))):
                    
                        for l in range(len(lag_range)):
                            try:
                                c, lag = nanxcorr(-dEye[t1[tstart]*60 : t2[tstart]*60],
                                            imu_interp(eyeT[t1[tstart]*60 : t2[tstart]*60]+lag_range[l]),
                                            1)
                                cc[l] = c[1]

                            except:
                                cc[l] = np.nan

                        offset[tstart] = lag_range[np.argmax(cc)]    
                        ccmax[tstart] = np.max(cc)

                    offset[ccmax<0.2] = np.nan

                    # Fit regression to timing drift
                    model = sklearn.linear_model.LinearRegression()
                    dataT = np.array(eyeT[t1*60 + 30*60])

                    model.fit(dataT[~np.isnan(offset)].reshape(-1,1),
                                offset[~np.isnan(offset)]) 

                    ephys_offset = model.intercept_
                    ephys_drift_rate = model.coef_

                    imuT = imuT_raw - (ephys_offset + imuT_raw * ephys_drift_rate)

                    tmp_eyeT = eyeT.flatten()[:-1]

                    # calculating eye movements

                    eye_use_thresh = 50
                    theta[np.where(abs(theta) > eye_use_thresh)] = np.nan

                    dEye_dps = dEye / np.diff(eyeT) # deg/sec

                    dEye_dps[np.where(abs(dEye_dps) >1500)] = np.nan

                    # Extract gaze/compensatory trials
                    dHead = scipy.interpolate.interp1d(imuT,
                                                        gyro_z,
                                                        bounds_error=False)(eyeT)[:-1]
                    dGaze = dHead + dEye_dps

                    ## old criterian 
                    #gazeL = tmp_eyeT[(dHead > shifted_head) &         \
                    #                 (dGaze > shifted_gaze)]
                    #gazeR = tmp_eyeT[(dHead < -shifted_head) &        \
                    #                 (dGaze < -shifted_gaze)]

                    #compL = tmp_eyeT[(dHead > shifted_head) &         \
                    #                 (dGaze < still_gaze) &           \
                    #                 (dGaze > -still_gaze)]
                    #compR = tmp_eyeT[(dHead < -shifted_head) &        \
                    #                 (dGaze > -still_gaze) &          \
                    #                 (dGaze < still_gaze)]
                    
                    ## new criterian 
                    gazeL = tmp_eyeT[(dGaze > shifted_gaze)]
                    gazeR = tmp_eyeT[(dGaze < -shifted_gaze)]

                    compL = tmp_eyeT[(dHead > shifted_head) &         \
                                     (dGaze < still_gaze) &           \
                                     (dGaze > -still_gaze)]
                    compR = tmp_eyeT[(dHead < -shifted_head) &        \
                                     (dGaze > -still_gaze) &          \
                                     (dGaze < still_gaze)]
                    



                    # Eliminating compensatory eye/head movements which fall right after gaze-shifting eye/head movements
                    #compL = drop_nearby_events(compL, gazeL)  
                    #compR = drop_nearby_events(compR, gazeR)

                    # Eliminate saccades repeated over sequential camera frames # 5/21 changed window to be over two frames 
                    gazeL_event = drop_repeat_events(gazeL)
                    gazeR_event = drop_repeat_events(gazeR)
                    compL_event = drop_repeat_events(compL)
                    compR_event = drop_repeat_events(compR)

                    # Extract idx

                    _, gazeL_idx, _ = np.intersect1d(tmp_eyeT,gazeL_event, return_indices=True)
                    _, gazeR_idx, _ = np.intersect1d(tmp_eyeT,gazeR_event, return_indices=True)
                    _, compL_idx, _ = np.intersect1d(tmp_eyeT,compL_event, return_indices=True)
                    _, compR_idx, _ = np.intersect1d(tmp_eyeT,compR_event, return_indices=True)

                    animal_directory = Path(data_path / date / ani/ task)
                    imu_eye_dict,_ = save_imu_eye_data(directory=animal_directory,Openephys_time =Openephys_time,
                                      animal=ani, date=date,Usegood_reflec=usegood_reflec,spot_count=spot_count,usegood_eyecalib = usegood_eyecalib, ellipticity=ellipticity,
                                      usegood_ellipcalb=usegood_ellipcalb,ellipse=ellipse,cam_cent=cam_cent,pupil_count=pupil_count,
                                      offset=offset,ccmax=ccmax,ellipse_out=ellipse_out,scale = scale,
                                      dEye_dps=dEye_dps,dHead=dHead,imuT=imuT,dGaze=dGaze,gazeL_event=gazeL_event,gazeL_idx=gazeL_idx,gazeR_event= gazeR_event,gazeR_idx=gazeR_idx,eyeT = eyeT,
                                      compL_event=compL_event,compL_idx=compL_idx,compR_event = compR_event, compR_idx=compR_idx, filename= str(date)+ "_"+str(ani)+"_imu_eyecam_dict")
                    create_summary_figure(data_dict =imu_eye_dict,save_path =  animal_directory)
                    


    def preprocess_world(self):
        data_path = Path(self.path).expanduser()
        print(data_path)
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in tqdm(use_animals,'animal'):
                for task in os.listdir(data_path / date / ani):
                    # get the reye video and reye time stamp paths
                    #if 'cam' not in task:
                    #    print('non cam')
                    #else: 
                    WORLD_vid = [str(i) for i in list((data_path / date / ani/ task).rglob('*.avi')) if 'WORLD.avi' in str(i)]
                    #print(REYE_vid)
                    WORLD_ts = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv')) if 'WORLD_BonsaiTS.csv' in str(i)]
                    #get the strings for namiing
                    print(WORLD_vid)
                    vid_name =  os.path.split(WORLD_vid[0])[1]
                    key_pieces = vid_name.split('.')[:-1]
                    key = '.'.join(key_pieces)
                    avi_out_path = os.path.join((data_path / date / ani/ task), (key + 'deinter.avi'))
                    csv_out_path = os.path.join((data_path / date / ani/ task), (key + '_BonsaiTSformatted.csv'))
                    #dienterlace video
                    if os.path.exists(avi_out_path):
                        print('already deinterlace')
                    else:
                        deinterlace(WORLD_vid[0], avi_out_path, exp_fps=30, quiet=True)
                        print('deinterlace finish')
                        #get new timestamps 
                        cap = cv2.VideoCapture(WORLD_vid[0])
                        # get some info about the video
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) # number of total frames
                        frame_count_deinter = frame_count * 2
                        fps = cap.get(cv2.CAP_PROP_FPS) # frame rate
                    # read in the exiting timestamps, interpolate to match the new number of steps, and format as dataframe
                        self.timestamp_path = WORLD_ts[0]
                        csv_out = pd.DataFrame(self.read_timestamp_file(int(frame_count_deinter)))
                     # save new timestamps
                        print('save new timestamps')
                        csv_out.to_csv(csv_out_path, index=False) 


                    

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
        vf_val = 'yadif=1:-1:, scale=640:480'

        print(os.path.normpath(vidfile))
        print(os.path.normpath(savepath))

        # Create the full FFMPEG command
        cmd = ['ffmpeg', '-i', os.path.normpath(vidfile), '-vf', vf_val, '-c:v', 'libx264',
              '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
              '256k', '-y', os.path.normpath(savepath)]

        # Set the log level
        if quiet is True:
            cmd.extend(['-loglevel', 'quiet'])

        # Run the FFMPEG command.
        subprocess.run(cmd, capture_output=True, text=True)

        return savepath

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


def drop_repeat_events(eventT, onset=True, win=0.020):
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

                    
def save_imu_eye_data( ellipse_out,
    directory: str,
    animal: str,
    date: str = None,
    Usegood_reflec: bool = None,
    spot_count: int = None,
    ellipticity: int = None,
    usegood_ellipcalb: int = None,
    ellipse:np.ndarray= None,
    scale:float=None,
    cam_cent:np.ndarray = None,
    pupil_count:np.ndarray = None,
    offset:np.ndarray = None,
    ccmax:np.ndarray = None,
    Openephys_time: np.ndarray = None,
    dEye_dps: np.ndarray = None,
    dHead: np.ndarray = None,
    imuT: np.ndarray = None,
    dGaze: np.ndarray = None,
    eyeT: np.ndarray = None,
    gazeL_event: np.ndarray = None,
    gazeL_idx: np.ndarray = None,
    gazeR_event: np.ndarray = None,
    gazeR_idx: np.ndarray = None,
    compL_event: np.ndarray = None,
    compL_idx: np.ndarray = None,
    compR_event: np.ndarray = None,
    compR_idx: np.ndarray = None,
    filename: str = None,
    **extra_data
) -> str:
    """
    Save sensor data in NumPy .npz format without timestamp.
    
    Args:
        directory: Path to save directory
        animal: Animal ID (required)
        date: Recording date (optional, defaults to current date)
        All parameters can be scalars or numpy arrays
        **extra_data: Additional data to include
        
    Returns:
        str: Full path to saved .npz file
    """
    # Create base dictionary
    data_dict = {
        'animal': np.array(animal),
        'date': np.array(date if date else datetime.now().strftime("%Y-%m-%d")),
    }
    
    # Add all sensor data
    sensor_params = {'animal': animal,
        'Usegood_reflec': Usegood_reflec,
        'spot_count':spot_count,
        'ellipticity':ellipticity,
        'usegood_ellipcalb':usegood_ellipcalb,
        'ellipse':ellipse,
        'scale':scale,
        'cam_cent':cam_cent,
        'pupil_count':pupil_count,
        'offset':offset,
        'ccmax':ccmax,
        'Openephys_time':Openephys_time,
        'dEye_dps': dEye_dps,
        'dHead': dHead,
        'imuT': imuT,
        'dGaze': dGaze,
        'eyeT': eyeT,
        'gazeL_event': gazeL_event,
        'gazeL_idx': gazeL_idx,
        'gazeR_event': gazeR_event,
        'gazeR_idx': gazeR_idx,
        'compL_event': compL_event,
        'compL_idx': compL_idx,
        'compR_event': compR_event,
        'compR_idx': compR_idx,
        **extra_data
    }
    
    # Convert all values to numpy arrays
    for key, value in sensor_params.items():
        if value is not None:
            data_dict[key] = np.array(value)
    
    # Create directory if needed
    os.makedirs(directory, exist_ok=True)
    
    # Set default filename
    if filename is None:
        filename = f"{animal}_{data_dict['date'].item().replace('-', '')}.npz"
    elif not filename.endswith('.npz'):
        filename += '.npz'
    
    # Save in compressed numpy format
    filepath = os.path.join(directory, filename)
    np.savez_compressed(filepath, **data_dict)
    ellipse_out.to_netcdf(os.path.join(directory,str(animal)+'_elipse_params.nc'))

    
    return data_dict,filepath   


def create_summary_figure(data_dict, fig_dwnsmpl=100, save_path=None):
    """
    Create a 3x4 summary figure from eye tracking data (10 subplots, 2 empty) and optionally save as PDF.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing the eye tracking data with expected keys:
        ['pupil_count', 'ellipticity', 'usegood_ellipcalb', 'ellipse', 'cam_cent',
         'usegood_eyecalib', 'dGaze', 'dEye_dps', 'dHead', 'offset', 'ccmax', 'scale']
    fig_dwnsmpl : int
        Downsampling factor for some plots (default=100)
    save_path : str, optional
        Path to save the figure as PDF. If None, figure is not saved.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The created figure
    ax : array of matplotlib.axes.Axes
        Array of subplot axes
    """
    # Extract variables from dictionary
    pupil_count = data_dict['pupil_count']
    ellipticity = data_dict['ellipticity']
    usegood_ellipcalb = data_dict['usegood_ellipcalb']
    ellipse = data_dict['ellipse']
    cam_cent = data_dict['cam_cent']
    usegood_eyecalib = data_dict['usegood_eyecalib']
    dGaze = data_dict['dGaze']
    dEye_dps = data_dict['dEye_dps']
    dHead = data_dict['dHead']
    offset = data_dict['offset']
    ccmax = data_dict['ccmax']
    animal = data_dict['animal']
    # Calculate scale with improved error handling
    try:
        f_lim = 50000
        if np.size(usegood_ellipcalb, 1) > f_lim:
            shortlist = sorted(np.random.choice(usegood_ellipcalb[0],
                             size=f_lim, replace=False))
        else:
            shortlist = usegood_ellipcalb
        
        try:
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) * \
                   (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=0))) / \
                   np.sum(1 - (ellipticity)**2)
        except ValueError:
            scale = np.nansum(np.sqrt(1 - (ellipticity)**2) * \
                   (np.linalg.norm(ellipse[shortlist, 11:13] - cam_cent.T, axis=1))) / \
                   np.sum(1 - (ellipticity)**2)
    except Exception as e:
        print(f"Error calculating scale: {str(e)}")
        scale = 1.0  # Default value if calculation fails

    # Create figure and subplots (3 rows, 4 columns)
    fig, ax = plt.subplots(3, 4, figsize=(20, 15))
    
    # Plot 1: Histogram of pupil_count
    ax[0, 0].hist(pupil_count, bins=9, range=(0,9), density=True)
    ax[0, 0].set_xlabel('num good eye points')
    ax[0, 0].set_ylabel('fraction of frames')
    
    # Plot 2: Ellipticity histogram
    try:
        ax[0, 1].hist(ellipticity, density=True)
        ax[0, 1].set_title(f'ellipticity; thresh=0.85')
        ax[0, 1].set_ylabel('ellipticity')
        ax[0, 1].set_xlabel('fraction of frames')
    except Exception as e:
        print(f'Figure error in ellipticity plot: {str(e)}')
    
    # Plot 3: Eye axes relative to center
    try:
        w = ellipse[:,7]
        for i in range(0, len(usegood_ellipcalb)):
            _show = usegood_ellipcalb[0][i::fig_dwnsmpl]
            ax[0, 2].plot((ellipse[_show,11] + [-5 * np.cos(w[_show]), 5 * np.cos(w[_show])]),
                         (ellipse[_show,12] + [-5*np.sin(w[_show]), 5*np.sin(w[_show])]))
        
        ax[0, 2].plot(cam_cent[0], cam_cent[1], 'r*')
        ax[0, 2].set_title('eye axes relative to center')
    except Exception as e:
        print(f'Figure error in eye axes plot: {str(e)}')
    
    # Plot 4: Calibration check
    try:
        xvals = np.linalg.norm(ellipse[usegood_eyecalib, 11:13].T - cam_cent, axis=0)
        yvals = scale * np.sqrt(1 - (ellipse[usegood_eyecalib, 6] / ellipse[usegood_eyecalib, 5])**2)
        calib_mask = ~np.isnan(xvals) & ~np.isnan(yvals)
        
        if np.sum(calib_mask) > 0:
            slope, _, r_value, _, _ = scipy.stats.linregress(xvals[calib_mask], yvals[calib_mask].T)
        else:
            slope, r_value = np.nan, np.nan
        
        ax[0, 3].plot(xvals[::fig_dwnsmpl], yvals[::fig_dwnsmpl], '.', markersize=1)
        ax[0, 3].plot(np.linspace(0,50), np.linspace(0,50), 'r')
        ax[0, 3].set_title(f'scale={scale:.3f} r={r_value:.3f} m={slope:.3f}')
        ax[0, 3].set_xlabel('pupil camera dist')
        ax[0, 3].set_ylabel('scale * ellipticity')
    except Exception as e:
        print(f'Error in calibration plot: {str(e)}')
    
    # Plot 5: dHead histogram
    ax[1, 0].set_xlim(-750,750)
    ax[1, 0].hist(dHead, density=True,range = (-750,750),bins=30 )
    ax[1, 0].set_title('dHead')
    
    # Plot 6: dEye_dps histogram
    ax[1, 1].set_xlim(-750,750)
    ax[1, 1].hist(dEye_dps, density=True,range = (-750,750),bins=30)
    ax[1, 1].set_title('dEye_dps')
    
    # Plot 7: dGaze histogram
    ax[1, 2].set_xlim(-750,750)
    ax[1, 2].hist(dGaze, density=True,range = (-750,750),bins=30)
    ax[1, 2].set_title('dGaze')
    
    # Plot 8: dEye vs dHead scatter
    ax[1, 3].plot(dEye_dps[::10], dHead[::10], 'k.', markersize=.7)
    ax[1, 3].set_xlabel('dEye')
    ax[1, 3].set_ylabel('dHead')
    ax[1, 3].set_xlim((-850,850))
    ax[1, 3].set_ylim((-850,850))
    ax[1, 3].plot([-850,850], [850,-850], 'r:')
    
    # Plot 9: Offset plot
    ax[2, 0].plot(offset)
    ax[2, 0].set_title('offset')
    
    # Plot 10: ccmax plot
    ax[2, 1].plot(ccmax)
    ax[2, 1].set_title('ccmax')
    
    # Turn off unused subplots
    ax[2, 2].axis('off')
    ax[2, 3].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save to PDF if path is provided
    if save_path is not None:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(os.path.join(os.path.join(save_path,str(animal)+'_diagnostic.pdf')), format='pdf', bbox_inches='tight',dpi=300)
            print(f"Figure successfully saved to: {save_path}")
        except Exception as e:
            print(f"Error saving figure to {save_path}: {str(e)}")
    
    return fig, ax