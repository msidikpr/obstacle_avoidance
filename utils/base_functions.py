import os, subprocess, math, cv2
import numpy as np
import pandas as pd
import itertools 
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec



## Load Frames from video returns array(frames, width, height)

def format_frames(vid_path, dwnsmpl):
    # open the .avi file
    vidread = cv2.VideoCapture(vid_path)
    # empty array that is the target shape
    # should be number of frames x downsampled height x downsampled width
    all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                           int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)*dwnsmpl),
                           int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH)*dwnsmpl)], dtype=np.uint8)
    # iterate through each frame
    for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # read the frame in and make sure it is read in correctly
        ret, frame = vidread.read()
        if not ret:
            break
        # convert to grayyscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ## took out downsampled frames in dylans code mike 10/19/22
        # downsample the frame by an amount specified in the config file
        sframe = cv2.resize(frame, (0,0), fx=dwnsmpl, fy=dwnsmpl, interpolation=cv2.INTER_NEAREST)
        # add the downsampled frame to all_frames as int8
        all_frames[frame_num,:,:] = sframe.astype(np.uint8)  
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

def intersect(p1, p2, p3, p4):
    x1,y1 = p1
    x2,y2 = p2
    x3,y3 = p3
    x4,y4 = p4
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        x,y = 0,0
        return (x,y)
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        x,y = 0,0

        


        return (x,y)
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        x,y = 0,0
        return (x,y)
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return (x,y)





def nearestX_roundup(num, x):
  d = num // x 
  a = d * x
  b = a + x

  if (num/x).is_integer() == True:
    return num
  else:  
    return b

def midpoint(x1, y1, x2, y2):
    return ((x1 + x2)/2, (y1 + y2)/2)

def calculate_angle(p1, p2, p3):
    """
    Calculates the angle made by connecting three points.
    Assumes that p1 is the vertex of the angle.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    
    # Calculate the vectors
    v1 = (x2 - x1, y2 - y1)
    v2 = (x3 - x1, y3 - y1)
    
    # Calculate the dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    
    # Calculate the magnitudes of the vectors
    v1_mag = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    v2_mag = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    
    # Calculate the angle (in radians) using the dot product and vector magnitudes
    angle = math.acos(dot_product / (v1_mag * v2_mag))
    
    # Convert the angle to degrees and return it
    return math.degrees(angle)


def find_consecutive_repeats(series):

    """
    finds consective repeats in a pd.series
    Used to find trials that have repeat of the same  obstacle location
    """
    consecutive_repeats = []
    count = 1
    prev_value = None

    for index, value in series.iteritems():
        if value == prev_value:
            count += 1
        else:
            if count > 3:
                count = 1
                prev_value = value
            
            count = 1
            prev_value = value

        if count == 3:
            consecutive_repeats.append((index - count+1, index, value,count))
    
 
    for count,row in enumerate(consecutive_repeats):
        if consecutive_repeats[count][-1] != 3:
            del consecutive_repeats[count]
    

    return consecutive_repeats

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        #>>> # linear interpolation of NaNs
        #>>> nans, x= nan_helper(y)
        #>>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """
    y=y.astype(float)

    return np.isnan(y), lambda z: z.nonzero()[0]


def interpolate_array(array):
    """takes in np array to interp across nans"""
    nans, x= nan_helper(array)
    array[nans]= np.interp(x(nans), x(~nans), array[~nans].astype(float))

    return array.astype(float)

