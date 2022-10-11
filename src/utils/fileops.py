import xarray as xr
from scipy.io import savemat
import os, json
import cv2
import pandas as pd
import numpy as np
import PySimpleGUI as sg

def read_ephys_binary(path, n_ch, probe_name=None, chmap_path=None):
    """ Read in ephys binary and remap channels.

    Parameters:
    if a probe name is given, the binary file will be remapped. otherwise, channels will be kept in the same order

    Returns:
    ephys (pd.DataFrame): ephys data with shape (time, channel)
    """
    # set up data types
    dtypes = np.dtype([('ch'+str(i),np.uint16) for i in range(0,n_ch)])
    # read in binary file
    ephys_arr = pd.DataFrame(np.fromfile(path, dtypes, -1, ''))
    if probe_name is not None:
        # open channel map file
        if chmap_path is None:
            chmap_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),'/config/channel_maps.json')
        with open(chmap_path, 'r') as fp:
            all_maps = json.load(fp)
        # get channel map for the current probe
        ch_map = all_maps[probe_name]
        # remap with known order of channels
        ephys_arr = ephys_arr.iloc[:,[i-1 for i in list(ch_map)]]
    return ephys_arr

def nc2mat():
    f = sg.popup_get_file('Choose .nc file.')
    data = xr.open_dataset(f)
    data_dict = dict(zip(list(data.REYE_ellipse_params['ellipse_params'].values), [data.REYE_ellipse_params.sel(ellipse_params=p).values for p in list(data.REYE_ellipse_params['ellipse_params'].values)]))
    save_name = os.path.join(os.path.split(f)[0], os.path.splitext(os.path.split(f)[1])[0])+'.mat'
    print('saving {}'.format(save_name))
    savemat(save_name, data_dict)

def avi_to_arr(path, ds=0.25):
    vid = cv2.VideoCapture(path)
    # array to put video frames into
    # will have the shape: [frames, height, width] and be returned with dtype=int8
    arr = np.empty([int(vid.get(cv2.CAP_PROP_FRAME_COUNT)),
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)*ds),
                        int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)*ds)], dtype=np.uint8)
    # iterate through each frame
    for f in range(0,int(vid.get(cv2.CAP_PROP_FRAME_COUNT))):
        # read the frame in and make sure it is read in correctly
        ret, img = vid.read()
        if not ret:
            break
        # convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # downsample the frame by an amount specified in the config file
        img_s = cv2.resize(img, (0,0), fx=ds, fy=ds, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        arr[f,:,:] = img_s.astype(np.int8)
    return arr

