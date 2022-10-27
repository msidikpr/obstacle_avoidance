import os, subprocess, math, cv2
import numpy as np
import pandas as pd



## Load Frames from video returns array(frames, width, height)

def format_frames(vid_path):
    # open the .avi file
    vidread = cv2.VideoCapture(vid_path)
    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                           int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                           int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))], dtype=np.uint8)
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = frame.astype(np.int8)  
    return all_frames