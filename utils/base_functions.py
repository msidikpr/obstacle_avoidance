import os, subprocess, math, cv2
import numpy as np
import pandas as pd
import itertools 
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
from plots.plots2 import *

import seaborn as sns



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

"""create color dict for unique items in list"""
def create_color_dict(df,key,color_pallete,sort =False):
    if sort == True:
        color_labels = df[key].unique().astype(int)
        color_labels.sort()
        color_labels = color_labels.astype(str)
        rgb_values = sns.color_palette(color_pallete, len(color_labels))
        color_map = dict(zip(color_labels, rgb_values))
        color_map = {'0'+k: v for k, v in color_map.items()}



    else:    
        color_labels = df[key].unique().astype(str)
        rgb_values = sns.color_palette(color_pallete, len(color_labels))
        color_map = dict(zip(color_labels, rgb_values))
    return color_map

def plot_arena(df,axis,obstacle = False):
    df =df.copy(deep=True)
    #arena_x = pd.unique(df[['arenaTL_x_cm',
    #'arenaTR_x_cm','arenaBR_x_cm',
    #'arenaBL_x_cm',
    #'arenaTL_x_cm']].values.ravel('K'))
#
    #arena_y = pd.unique(df[['arenaTL_y_cm',
    #'arenaTR_y_cm','arenaBR_y_cm',
    #'arenaBL_y_cm',
    #'arenaTL_y_cm']].values.ravel('K'))

    arena_x = df[['arenaTL_x_cm',
    'arenaTR_x_cm','arenaBR_x_cm',
    'arenaBL_x_cm',
    'arenaTL_x_cm']].median().values.ravel('K')

    arena_y = df[['arenaTL_y_cm',
    'arenaTR_y_cm','arenaBR_y_cm',
    'arenaBL_y_cm',
    'arenaTL_y_cm']].median().values.ravel('K')
    

    #left_port =  pd.unique(df[['leftportT_x_cm','leftportT_y_cm']].values.ravel('K'))

    #right_port = pd.unique(df[['rightportT_x_cm','rightportT_y_cm']].values.ravel('K'))

    left_port =  df[['leftportT_x_cm','leftportT_y_cm']].median().values.ravel('K')

    right_port = df[['rightportT_x_cm','rightportT_y_cm']].median().values.ravel('K')


    
    
    axis.plot([arena_x[0],arena_x[1],arena_x[2],arena_x[3],arena_x[0]],
                          [arena_y[0],arena_y[1],arena_y[2],arena_y[3],arena_y[0]],c='k')

    axis.scatter(left_port[0],left_port[1],c='purple',s=200,marker = 's')
    axis.vlines(ymax=arena_y[0],ymin=arena_y[3],x=left_port[0],colors='k')
    axis.scatter(right_port[0],right_port[1],c='r',s=200,marker = 's')
    axis.vlines(ymax=arena_y[1],ymin=arena_y[2],x=right_port[0],colors='k')

    if obstacle == True:
        obstacle_x = pd.unique(df[['mean_gt_obstacleTL_x_cm',
        'mean_gt_obstacleTR_x_cm','mean_gt_obstacleBR_x_cm',
        'mean_gt_obstacleBL_x_cm',
        'mean_gt_obstacleTL_x_cm']].median().values.ravel('K'))

        obstacle_y =  pd.unique(df[['mean_gt_obstacleTL_y_cm',
        'mean_gt_obstacleTR_y_cm','mean_gt_obstacleBR_y_cm',
        'mean_gt_obstacleBL_y_cm',
        'mean_gt_obstacleTL_y_cm']].median().values.ravel('K'))

        axis.plot([obstacle_x[0],obstacle_x[1],obstacle_x[2],obstacle_x[3],obstacle_x[0]],
                              [obstacle_y[0],obstacle_y[1],obstacle_y[2],obstacle_y[3],obstacle_y[0]],c='k')

    
    axis.set_ylim([51,0]); axis.set_xlim([0, 61])

def plot_arena_single(df,axis,obstacle = False):
    df =df.copy(deep=True)
    arena_x = pd.unique(df[['arenaTL_x_cm',
    'arenaTR_x_cm','arenaBR_x_cm',
    'arenaBL_x_cm',
    'arenaTL_x_cm']].values.ravel('K'))

    arena_y = pd.unique(df[['arenaTL_y_cm',
    'arenaTR_y_cm','arenaBR_y_cm',
    'arenaBL_y_cm',
    'arenaTL_y_cm']].values.ravel('K'))

    

    left_port =  pd.unique(df[['leftportT_x_cm','leftportT_y_cm']].values.ravel('K'))

    right_port = pd.unique(df[['rightportT_x_cm','rightportT_y_cm']].values.ravel('K'))

    
    
    axis.plot([arena_x[0],arena_x[1],arena_x[2],arena_x[3],arena_x[0]],
                          [arena_y[0],arena_y[1],arena_y[2],arena_y[3],arena_y[0]],c='k')

    axis.scatter(left_port[0],left_port[1],c='purple',s=200,marker = 's')
    axis.vlines(ymax=arena_y[0],ymin=arena_y[2],x=left_port[0],colors='k')
    axis.scatter(right_port[0],right_port[1],c='r',s=200,marker = 's')
    axis.vlines(ymax=arena_y[0],ymin=arena_y[2],x=right_port[0],colors='k')

    if obstacle == True:
        obstacle_x = pd.unique(df[['gt_obstacleTL_x_cm',
        'gt_obstacleTR_x_cm','gt_obstacleBR_x_cm',
        'gt_obstacleBL_x_cm',
        'gt_obstacleTL_x_cm']].values.ravel('K'))

        obstacle_y =  pd.unique(df[['gt_obstacleTL_y_cm',
        'gt_obstacleTR_y_cm','gt_obstacleBR_y_cm',
        'gt_obstacleBL_y_cm',
        'gt_obstacleTL_y_cm']].values.ravel('K'))

        axis.plot([obstacle_x[0],obstacle_x[1],obstacle_x[2],obstacle_x[3],obstacle_x[0]],
                              [obstacle_y[0],obstacle_y[1],obstacle_y[2],obstacle_y[3],obstacle_y[0]],c='k')

    
    axis.set_ylim([51,0]); axis.set_xlim([0, 61])


"""input is df of single obstacle cluster"""
def plot_obstacle(df,axis,cluster):
    df = df.copy()
    keys = list_columns(df,['gt'])
    keys = [key for key in keys if 'cen' not in key]
    for key in keys:
        df.loc[df.obstacle_cluster ==cluster,key] = df.loc[df.obstacle_cluster ==cluster,key].mean()
    
    obstacle_x = pd.unique(df[['mean_gt_obstacleTL_x_cm',
        'mean_gt_obstacleTR_x_cm','mean_gt_obstacleBR_x_cm',
        'mean_gt_obstacleBL_x_cm',
        'mean_gt_obstacleTL_x_cm']].values.ravel('K'))

    obstacle_y =  pd.unique(df[['mean_gt_obstacleTL_y_cm',
    'mean_gt_obstacleTR_y_cm','mean_gt_obstacleBR_y_cm',
    'mean_gt_obstacleBL_y_cm',
    'mean_gt_obstacleTL_y_cm']].values.ravel('K'))

    axis.plot([obstacle_x[0],obstacle_x[1],obstacle_x[2],obstacle_x[3],obstacle_x[0]],
                              [obstacle_y[0],obstacle_y[1],obstacle_y[2],obstacle_y[3],obstacle_y[0]],c='k')
    axis.set_ylim([51,0]); axis.set_xlim([0, 61])


def plot_orginal_obstacle(df,axis,cluster, correct = False, corect_x= 0,corect_y = 0):
    keys = list_columns(df,['gt'])
    keys = [key for key in keys if 'cen' not in key]
    for key in keys:
        df.loc[df.obstacle_cluster ==cluster,key] = df.loc[df.obstacle_cluster ==cluster,key].mean()
    obstacle_x = pd.unique(df[['gt_obstacleTL_x_cm',
        'gt_obstacleTR_x_cm','gt_obstacleBR_x_cm',
        'gt_obstacleBL_x_cm',
        'gt_obstacleTL_x_cm']].values.ravel('K'))

    obstacle_y =  pd.unique(df[['gt_obstacleTL_y_cm',
    'gt_obstacleTR_y_cm','gt_obstacleBR_y_cm',
    'gt_obstacleBL_y_cm',
    'gt_obstacleTL_y_cm']].values.ravel('K'))

    if correct == False:
        axis.plot([obstacle_x[0],obstacle_x[1],obstacle_x[2],obstacle_x[3],obstacle_x[0]],
                              [obstacle_y[0],obstacle_y[1],obstacle_y[2],obstacle_y[3],obstacle_y[0]],c='k')
        axis.set_ylim([51,0]); axis.set_xlim([0, 61])
    else:
        axis.plot([obstacle_x[0]+corect_x,obstacle_x[1]+corect_x,obstacle_x[2]+corect_x,obstacle_x[3]+corect_x,obstacle_x[0]+corect_x],
                              [obstacle_y[0]+corect_y,obstacle_y[1]+corect_y,obstacle_y[2]+corect_y,obstacle_y[3]+corect_y,obstacle_y[0]+corect_y],c='k')
        axis.set_ylim([51,0]); axis.set_xlim([0, 61])



    

    



def create_sublists(lst):
    """
    Create a list of sublists, where each sublist contains the n and n+1 index elements from the original list.

    Args:
        lst (list): The original list of integers.

    Returns:
        list: A list of sublists, where each sublist contains the n and n+1 index elements from the original list.
    """
    sublists = []
    for i in range(len(lst)-1):
        sublists.append([lst[i], lst[i+1]])
    return sublists



def split_range_into_parts(start, end, n):
    if start >= end:
        raise ValueError("Start value must be less than end value.")
    
    if n <= 0:
        raise ValueError("Number of parts (n) must be greater than 0.")
    
    total_range = end - start
    part_size = total_range / n
    
    ranges = []
    current_start = start
    
    for _ in range(n):
        current_end = current_start + part_size
        ranges.append((current_start, current_end))
        current_start = current_end
    
    return ranges

def column_to_array(column,df):
    to_array = df[str(column)].to_numpy()
    #print(test_array[0])
    array= np.zeros([len(df),  len(to_array[0])])
    count = 0
    for row in to_array:
        array[count,:] = row
        count += 1
    return array


def get_mean_median_by_variable(df,key): 
    for key_name,frame in df.groupby([str(key)]):
        for direction, direction_frame in frame.groupby(['odd']):
            for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
                for start, start_frame in cluster_frame.groupby(['start']):
                    array = np.zeros([len(start_frame), 50])
                    count = 0
                    for ind,row in start_frame.iterrows():
                        array[count,:] = row['interp_ts_nose_y_cm']
                        count += 1
                    mean_trace = np.nanmean(array,axis=0)
                    median_trace = np.nanmedian(array,axis = 0)
                    std_trace = np.nanstd(array,axis=0)
                    mad_trace = stats.median_abs_deviation(array,axis = 0,nan_policy='omit')
                    x = frame.loc[(frame['obstacle_cluster'] ==cluster) & (frame['start']==start)&(frame['odd'] ==direction)]
                    for ind,row in x.iterrows():
                        df.at[ind,key+'_''mean_interp_ts_nose_y_cm']= mean_trace.astype(object)
                        df.at[ind,key+'_''median_interp_ts_nose_y_cm']= median_trace.astype(object)
                        df.at[ind,key+'_''std_interp_ts_nose_y_cm']= std_trace.astype(object)
                        df.at[ind,key+'_''mad_interp_ts_nose_y_cm']= mad_trace.astype(object)
def by_start_obstalce_average_df(df,date):
        savepath = "D:/obstacle_avoidance/recordings"
        savepath_session = os.path.join(*[savepath,'figures'])
        #savepath_session = os.path.join(*[savepath,str(pd.unique(self.df.date).item()),str(pd.unique(self.df.animal).item()),str(pd.unique(self.df.task).item())])
        pdf = PdfPages(os.path.join((savepath_session), str(date)+ 'by_' +'_start_'+'obstacle' + 'consecutive.pdf'))
        key='start'
        fig = plt.figure(constrained_layout=False, figsize=(20, 10),dpi=90)
        fig.suptitle('by ' + key + ' '+ 'and ' +'obstacle ')
        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)


        """Right"""
        panel_1 = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=spec2[0])
        ax1 = fig.add_subplot(panel_1[0,0])
        plot_arena(df,ax1)
        ax2 = fig.add_subplot(panel_1[0,1])
        plot_arena(df,ax2)
        ax2.set_title('right')
        ax3 = fig.add_subplot(panel_1[1,0])
        plot_arena(df,ax3)
        ax4 = fig.add_subplot(panel_1[1,1])
        plot_arena(df,ax4)
        ax5 = fig.add_subplot(panel_1[2,0])
        plot_arena(df,ax5)
        ax6 = fig.add_subplot(panel_1[2,1])
        plot_arena(df,ax6)

        right_axs = [ax1,ax2,ax3,ax4,ax5,ax6]


        """Left """
        panel_2 = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=spec2[1])
        ax7 = fig.add_subplot(panel_2[0,0])
        plot_arena(df,ax7)
        ax8 = fig.add_subplot(panel_2[0,1])
        plot_arena(df,ax8)
        ax8.set_title('left')
        ax9 = fig.add_subplot(panel_2[1,0])
        plot_arena(df,ax9)
        ax10 = fig.add_subplot(panel_2[1,1])
        plot_arena(df,ax10)
        ax11= fig.add_subplot(panel_2[2,0])
        plot_arena(df,ax11)
        ax12 = fig.add_subplot(panel_2[2,1])
        plot_arena(df,ax12)

        left_axs = [ax7,ax8,ax9,ax10,ax11,ax12]




        """ plot trials"""
        right_obstacle_dict = dict(zip(pd.unique(df['obstacle_cluster'].sort_values().to_list()),right_axs))
        left_obstacle_dict = dict(zip(pd.unique(df['obstacle_cluster'].sort_values().to_list()),left_axs))

        for direction, direction_frame in df.groupby(['odd']):
            for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
                #cluster_frame = cluster_frame.sample(num_sample)
                right_obstacle_axis = right_obstacle_dict.get(cluster)
                left_obstacle_axis = left_obstacle_dict.get(cluster)
                plot_obstacle(cluster_frame,right_obstacle_axis,cluster)
                plot_obstacle(cluster_frame,left_obstacle_axis,cluster)
                right_obstacle_axis.set_title(str(cluster))
                left_obstacle_axis.set_title(str(cluster))

                for start, start_frame in cluster_frame.groupby(['start']):

                    if direction == 'right':
                        if start == 'top':
                            if cluster in [2,3]:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)

                            else:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)
                        if start == 'bottom':
                            if cluster in [2,3]:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)
                            else:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)


                    if direction == 'left':
                        if start == 'top':
                             if cluster in [2,3]:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)
                             else:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)
                        if start == 'bottom':
                            if cluster in [2,3]:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)

                            else:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)

        pdf.savefig(); plt.close()
        pdf.close()
def by_start_obstalce_average_df_key(df,date,key):
        savepath = "D:/obstacle_avoidance/recordings"
        savepath_session = os.path.join(*[savepath,'figures'])
        #savepath_session = os.path.join(*[savepath,str(pd.unique(self.df.date).item()),str(pd.unique(self.df.animal).item()),str(pd.unique(self.df.task).item())])
        pdf = PdfPages(os.path.join((savepath_session), str(date)+ 'by_' +'_start_'+'obstacle' + 'consecutive.pdf'))
        #key='start'
        fig = plt.figure(constrained_layout=False, figsize=(20, 10),dpi=90)
        fig.suptitle('by ' + key + ' '+ 'and ' +'obstacle ')
        spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)


        """Right"""
        panel_1 = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=spec2[0])
        ax1 = fig.add_subplot(panel_1[0,0])
        plot_arena(df,ax1)
        ax2 = fig.add_subplot(panel_1[0,1])
        plot_arena(df,ax2)
        ax2.set_title('right')
        ax3 = fig.add_subplot(panel_1[1,0])
        plot_arena(df,ax3)
        ax4 = fig.add_subplot(panel_1[1,1])
        plot_arena(df,ax4)
        ax5 = fig.add_subplot(panel_1[2,0])
        plot_arena(df,ax5)
        ax6 = fig.add_subplot(panel_1[2,1])
        plot_arena(df,ax6)

        right_axs = [ax1,ax2,ax3,ax4,ax5,ax6]


        """Left """
        panel_2 = gridspec.GridSpecFromSubplotSpec(3,2,subplot_spec=spec2[1])
        ax7 = fig.add_subplot(panel_2[0,0])
        plot_arena(df,ax7)
        ax8 = fig.add_subplot(panel_2[0,1])
        plot_arena(df,ax8)
        ax8.set_title('left')
        ax9 = fig.add_subplot(panel_2[1,0])
        plot_arena(df,ax9)
        ax10 = fig.add_subplot(panel_2[1,1])
        plot_arena(df,ax10)
        ax11= fig.add_subplot(panel_2[2,0])
        plot_arena(df,ax11)
        ax12 = fig.add_subplot(panel_2[2,1])
        plot_arena(df,ax12)

        left_axs = [ax7,ax8,ax9,ax10,ax11,ax12]




        """ plot trials"""
        right_obstacle_dict = dict(zip(pd.unique(df['obstacle_cluster'].sort_values().to_list()),right_axs))
        left_obstacle_dict = dict(zip(pd.unique(df['obstacle_cluster'].sort_values().to_list()),left_axs))

        for direction, direction_frame in df.groupby(['odd']):
            for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
                #cluster_frame = cluster_frame.sample(num_sample)
                right_obstacle_axis = right_obstacle_dict.get(cluster)
                left_obstacle_axis = left_obstacle_dict.get(cluster)
                plot_obstacle(cluster_frame,right_obstacle_axis,cluster)
                plot_obstacle(cluster_frame,left_obstacle_axis,cluster)
                right_obstacle_axis.set_title(str(cluster))
                left_obstacle_axis.set_title(str(cluster))

                for start, start_frame in cluster_frame.groupby(['start']):

                    if direction == 'right':
                        if start == 'top':

                            if cluster in [2,3]:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)

                            else:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)
                        if start == 'bottom':
                            if cluster in [2,3]:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)
                            else:
                                right_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                right_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)


                    if direction == 'left':
                        if start == 'top':
                             if cluster in [2,3]:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)
                             else:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'black')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='black', alpha=0.5)
                        if start == 'bottom':
                            if cluster in [2,3]:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['median_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['mad_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)

                            else:
                                left_obstacle_axis.plot(np.linspace(10,50,50),start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0],c = 'red')
                                left_obstacle_axis.fill_between(np.linspace(10,50,50), start_frame[key + '_'+'mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)+start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), start_frame['mean_interp_ts_nose_y_cm'].to_numpy()[0].astype(float)-start_frame['std_interp_ts_nose_y_cm'].to_numpy()[0].astype(float), facecolor='red', alpha=0.5)

        pdf.savefig(); plt.close()
        pdf.close()   

def check_trial_for_obstalce_cross(row): 
    """function checks if the trail trace crosses through the obsacle and returns boolean
    True is dose not cross through obstacle. False mean at least one point is inside the obstalce"""
    
    obstacle_x = [row['gt_obstacleTL_x_cm'],row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm'],row['gt_obstacleBL_x_cm']]
    obstacle_y = [row['gt_obstacleTL_y_cm'],row['gt_obstacleTR_y_cm'],row['gt_obstacleBR_y_cm'],row['gt_obstacleBL_y_cm']]
    nose_x = row['nose_x_cm'].astype(float)
    nose_y = row['nose_y_cm'].astype(float)

    def are_points_inside_polygon(x_points, y_points, obstacle_x, obstacle_y):
        n = len(obstacle_x)
        results = []

        for x, y in zip(x_points, y_points):
            inside = False
            p1x, p1y = obstacle_x[0], obstacle_y[0]

            for i in range(n + 1):
                p2x, p2y = obstacle_x[i % n], obstacle_y[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y

            results.append(inside)
        return results
    results = are_points_inside_polygon(nose_x,nose_y,obstacle_x,obstacle_y)
    if sum(results) == 0:
        return True  
    else:
        return False 

def drop_nans_in_columns(df,column):
    copy =df.copy(deep=True) 
    nan_inds = copy.index[np.where(copy[column].isnull())[0]]
    copy = copy.drop(index=nan_inds)
    return copy


def reject_outliers(data, m = 2):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else np.zeros(len(d))
    inds = np.argwhere(data[s<m])
    return data[s<m],inds

def create_consective_df_new(df):
        """get df from data of groups of 3 trials that are consecutive """
        con_df = pd.DataFrame()
        copy = df.copy(deep=True)
        copy = copy.reset_index(drop=True)
        for animal,animal_frame in copy.groupby('animal'):
            for date, date_frame in animal_frame.groupby('date'):
                repeats_list = find_consecutive_repeats(date_frame['obstacle_cluster'])
                for i in range(len(repeats_list)):
                    check = date_frame.loc[repeats_list[i][0]:repeats_list[i][1]]
                    #print(np.diff(check['index'].to_numpy()).sum())
                    if np.diff(check['index'].to_numpy()).sum()==2:
                        trial_df = pd.DataFrame()
                        trial_df = trial_df.append(date_frame.loc[repeats_list[i][0]:repeats_list[i][1]])
                        trial_df['consective_inds'] = str(list(range(repeats_list[i][0], repeats_list[i][1]+1)))
                        trial_df['consective_type'] = [1,2,3]
                        con_df = con_df.append(trial_df) 
                    else:
                        continue
                       
        return con_df
def save_dataframe_to_hdf(dataframe, directory, filename):
    """
    Save a Pandas DataFrame to an HDF5 file in the specified directory.

    Parameters:
    dataframe (pandas.DataFrame): The DataFrame to be saved.
    directory (str): The directory where the HDF5 file will be saved.
    filename (str): The name of the HDF5 file.

    Returns:
    str: The full path of the saved HDF5 file.
    """
    # Ensure the directory exists, create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Concatenate directory and filename to get the full path
    full_path = os.path.join(directory, filename)

    # Save the DataFrame to HDF5 file
    dataframe.to_hdf(full_path, key='data', mode='w')

    return full_path

def flatten_list_of_arrays(array_list):
    flatten_array_list = np.concatenate(array_list).ravel().tolist()
    return np.asarray(flatten_array_list)

def get_ob_distance_from_mean_center(df):
    df['mean_center_x'] = np.nan
    df['mean_center_y'] = np.nan
    for cluster,cluter_df in df.groupby(['obstacle_cluster']):
        gt_label = cluter_df['obstacle_cluster'].astype(int).tolist()
        obstacle_xpos = np.array(cluter_df.loc[:,'gt_obstacle_cen_x_cm'])
        obstacle_ypos = np.array(cluter_df.loc[:,'gt_obstacle_cen_y_cm'])
        obstacle_xypos_ar = np.stack((obstacle_xpos, obstacle_ypos))
        mean_x =np.nanmean(obstacle_xpos)
        mean_y = np.nanmean(obstacle_ypos)
        df['mean_center_x'][df['obstacle_cluster']==cluster]  = mean_x
        df['mean_center_y'][df['obstacle_cluster']==cluster]  = mean_y
    for ind,row in df.iterrows():
        distance_from_mean_center = calculate_distances(np.array(row.gt_obstacle_cen_x_cm),
                                                        np.array(row.gt_obstacle_cen_y_cm),row.mean_center_x,row.mean_center_y)
        df.at[ind,'distance_from_mean_center'] = distance_from_mean_center

def ts_lateral_error_to_target_port(df):
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():
            if direction == 'right':
                nose_y = row['ts_nose_y_cm'].astype(float)
                port_y = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
                lateral_error = []
                for i in nose_y:
                    err = np.abs(i -port_y)
                    lateral_error.append(err)
                lateral_error = np.array(lateral_error)        
                df.at[ind,'ts_lateral_error_to_target_port'] = lateral_error.astype(object) 
            else:
                nose_y = row['ts_nose_y_cm'].astype(float)
                port_y = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
                lateral_error = []
                for i in nose_y:
                    err = np.abs(i -port_y)
                    lateral_error.append(err)
                lateral_error = np.array(lateral_error)        
                df.at[ind,'ts_lateral_error_to_target_port'] = lateral_error.astype(object) 
                
def compute_obstacle_tortuosity_distance_threshold(df,thresh):
    drop_nans_in_columns(df,'obstacle_ind')
    for ind,row in df.iterrows():
        nose_x = row.ts_nose_x_cm
        nose_y = row.ts_nose_y_cm
        try:
            distance_thresh = np.argwhere(row.ts_distance_from_edge>=thresh).max()
            obstacle_ind = int(row.obstacle_ind)
        except ValueError:
            continue
        try:
            thresh_tor, thresh_lin = compute_tortuosity(nose_x[distance_thresh:obstacle_ind],nose_y[distance_thresh:obstacle_ind])
            df.at[ind,'ob_tortuosity'+'_'+str(thresh)] =thresh_tor
            df.at[ind,'ob_linearity'+'_'+str(thresh)] = thresh_lin
            df.at[ind,'percent_change_ob_tortuosity'+'_'+str(thresh)] =(thresh_tor - row.ob_tortuosity) * 100
            df.at[ind,'percent_change_ob_linearity'+'_'+str(thresh)] = (thresh_lin - row.ob_linearity) * 100
        except IndexError:
            df.at[ind,'ob_tortuosity'+'_'+str(thresh)] =np.nan
            df.at[ind,'ob_linearity'+'_'+str(thresh)] = np.nan
            df.at[ind,'percent_change_ob_tortuosity'+'_'+str(thresh)] =np.nan
            df.at[ind,'percent_change_ob_linearity'+'_'+str(thresh)] = np.nan

def calculate_obstalce_and_edge_vector_angle(df,thresh,thresh1):
    """only use long df"""
    for ind,row in df.iterrows():
        if row.obstacle_cluster == 0:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleBR_x_cm,row.gt_obstacleBR_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue

            
        elif row.obstacle_cluster == 1:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleBL_x_cm,row.gt_obstacleBL_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue
            
        elif row.obstacle_cluster == 4:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleTR_x_cm,row.gt_obstacleTR_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue
            
            
        elif row.obstacle_cluster == 5:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleTL_x_cm,row.gt_obstacleTL_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue
            
def calculate_obstalce_and_edge_vector_angle(df,thresh,thresh1):
    """only use long df"""
    for ind,row in df.iterrows():
        if row.obstacle_cluster == 0:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleBR_x_cm,row.gt_obstacleBR_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue

            
        elif row.obstacle_cluster == 1:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleBL_x_cm,row.gt_obstacleBL_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue
            
        elif row.obstacle_cluster == 4:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleTR_x_cm,row.gt_obstacleTR_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue
            
            
        elif row.obstacle_cluster == 5:
            nose_x = row.ts_nose_x_cm
            nose_y = row.ts_nose_y_cm
            edge_x,edge_y = row.gt_obstacleTL_x_cm,row.gt_obstacleTL_y_cm
            try:
                distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
                distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
                if distance_thresh == distance_thresh1:
                    continue
                else:

                    obstacle_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,nose_y[distance_thresh]))
                    edge_vector = calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(edge_x,edge_y))
                    mouse_vector=calculate_vector_between_points((nose_x[distance_thresh],nose_y[distance_thresh]),(nose_x[distance_thresh1],nose_y[distance_thresh1]))
                    obstacle_rad,obstacle_ang = angle_between_vectors(mouse_vector,obstacle_vector)
                    edge_rad,edge_ang =angle_between_vectors(mouse_vector,edge_vector)
                    df.at[ind,'obstacle_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = obstacle_ang
                    df.at[ind,'edge_vector_ang'+'_'+str(thresh)+'_'+str(thresh1)] = edge_ang
            except ValueError:
                continue
            
def variable_by_distance_threshold(df,var,thresh,thresh1):
    for ind,row in df.iterrows(): 
        try:
            distance_thresh = np.argwhere(row.ts_distance_from_edge<=thresh).min()
            distance_thresh1 = np.argwhere(row.ts_distance_from_edge<=thresh1).min()
            if distance_thresh == distance_thresh1:
                continue
            else:
                df.at[ind,var + '_'+str(thresh)+'_'+str(thresh1)] = row[var][distance_thresh:distance_thresh1]  
                df.at[ind, var + '_' +str(thresh)+'_'+str(thresh1)+ '_mean'] = np.nanmean(row[var][distance_thresh:distance_thresh1])  
        except ValueError:
            continue  
def calculate_quartiles(time_series_list):
    quartiles = np.nanpercentile(time_series_list, [25, 50, 75], axis=0)
    q1, median, q3 = quartiles[0], quartiles[1], quartiles[2]
    iqr = q3 - q1
    return median, q1, q3, iqr
            
        