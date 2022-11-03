import os, subprocess, math, cv2
import numpy as np
import pandas as pd
import itertools 



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



## make serries from multiple objects from df column 
def flatten_column(df,column):
  column_list = []
  for ind, row in df.iterrows(): 
    #pts=
    series = row[column]
    column_list.append(series)
  column_list = list(itertools.chain(*column_list))
  column_list = np.array(column_list)
  return column_list


##create list of columns from df based on list of keys 
#keys as [] exp ['nose','leftear','rightear','spine','midspine']
def list_columns(df,keys): 
  columns_list = []
  for key in keys:
    columns = [col for col in df.columns if key in col]
    columns_list.append(columns)
  columns_list = list(itertools.chain(*columns_list))
  columns_list = np.array(columns_list)


  return columns_list