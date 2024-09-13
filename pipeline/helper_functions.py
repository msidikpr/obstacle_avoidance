"""helper functions for pipline """
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import itertools 
from scipy.interpolate import interp1d
import seaborn as sns
from scipy.ndimage import gaussian_filter


def list_columns(df,keys): 
  """keys == list"""
  columns_list = []
  for key in keys:
    columns = [col for col in df.columns if key in col]
    columns_list.append(columns)
  columns_list = list(itertools.chain(*columns_list))
  columns_list = np.array(columns_list)


  return columns_list

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

def flatten_list_of_arrays(array_list):
    """input: list of arrays
    output 1d np.array"""
    flatten_array_list = np.concatenate(array_list).ravel().tolist()
    return np.asarray(flatten_array_list)

def calculate_quartiles(time_series_list):
    quartiles = np.nanpercentile(time_series_list, [25, 50, 75], axis=0)
    q1, median, q3 = quartiles[0], quartiles[1], quartiles[2]
    iqr = q3 - q1
    return median, q1, q3, iqr

def create_df_by_type(df):
    '''creates df of long short middle approaches'''
    cluster_0_long = df[(df['obstacle_cluster']==0)&(df['start']=='top')&(df['odd']== 'right')]
    cluster_1_long = df[(df['obstacle_cluster']==1)&(df['start']=='top')&(df['odd']== 'left')]
    cluster_4_long = df[(df['obstacle_cluster']==4)&(df['start']=='bottom')&(df['odd']== 'right')]
    cluster_5_long = df[(df['obstacle_cluster']==5)&(df['start']=='bottom')&(df['odd']== 'left')]

    cluster_0_short = df[(df['obstacle_cluster']==0)&(df['start']=='top')&(df['odd']== 'left')]
    cluster_1_short = df[(df['obstacle_cluster']==1)&(df['start']=='top')&(df['odd']== 'right')]
    cluster_4_short = df[(df['obstacle_cluster']==4)&(df['start']=='bottom')&(df['odd']== 'left')]
    cluster_5_short = df[(df['obstacle_cluster']==5)&(df['start']=='bottom')&(df['odd']== 'right')]

    cluster_2 = df[(df['obstacle_cluster']==2)]
    cluster_3 = df[(df['obstacle_cluster']==3)]

    
    

    long_df = pd.concat([cluster_0_long,cluster_1_long,cluster_4_long,cluster_5_long])
    short_df = pd.concat([cluster_0_short,cluster_1_short,cluster_4_short,cluster_5_short])
    middle_df = pd.concat([cluster_2,cluster_3])
    return long_df,short_df,middle_df

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
    


def get_mean_of_df( df,key:str, bins:int,matx = False):
        fake_time = np.linspace(0,1,bins)
        mat = np.zeros([len(df), bins])
        count = 0
        df = drop_nans_in_columns(df,key)
        for ind, row in df.iterrows():
            xT = np.linspace(0,1,len(row[key]))
            mat[count,:] = interp1d(xT, row[key], bounds_error=False)(fake_time)
            count += 1
        mean = np.nanmean(mat, axis=0)

        df = pd.Series(mean)
        df.fillna(method='bfill', axis=0, inplace=True)
        df.fillna(method='ffill', axis=0, inplace=True)
        mean = df.to_numpy()
        if matx == True:

            return mean,mat
        else:
            return mean




def get_median_of_df( df,key:str, bins:int,matx = False):
        fake_time = np.linspace(0,1,bins)
        mat = np.zeros([len(df), bins])
        count = 0
        df = drop_nans_in_columns(df,key)
        for ind, row in df.iterrows():
            xT = np.linspace(0,1,len(row[key]))
            mat[count,:] = interp1d(xT, row[key], bounds_error=False)(fake_time)
            count += 1
        median = np.nanmedian(mat, axis=0)

        df = pd.Series(median)
        df.fillna(method='bfill', axis=0, inplace=True)
        df.fillna(method='ffill', axis=0, inplace=True)
        median = df.to_numpy()
        if matx == True:

            return median,mat
        else:
            return median


def drop_nans_in_columns(df,column):
    copy =df.copy(deep=True) 
    nan_inds = copy.index[np.where(copy[column].isnull())[0]]
    copy = copy.drop(index=nan_inds)
    return copy

def create_consective_df_new(df):
        """get df from data of groups larger than 3 trials that are consecutive """
        con_df = pd.DataFrame()
        copy = df.copy(deep=True)
        copy = copy.reset_index(drop=True)
        for animal,animal_frame in copy.groupby('animal'):
            for date, date_frame in animal_frame.groupby('date'):
                repeats_list = find_consecutive_repeats(date_frame['obstacle_cluster'])
                for i in range(len(repeats_list)):
                    check = date_frame.loc[repeats_list[i][0]:repeats_list[i][1]]
                    #print(np.diff(check['index'].to_numpy()).sum())
                    
                    trial_df = pd.DataFrame()
                    trial_df = trial_df.append(date_frame.loc[repeats_list[i][0]:repeats_list[i][1]])
                    trial_df['consective_inds'] = str(list(range(repeats_list[i][0], repeats_list[i][1]+1)))
                    trial_df['consective_type'] = list(range( (repeats_list[i][1]+1) - (repeats_list[i][0])))
                    con_df = con_df.append(trial_df) 
                    
                       
        return con_df
def find_consecutive_repeats(series):

    """
    finds consective repeats in a pd.series
    Used to find trials that have repeat of the same  obstacle location
    """
    consecutive_repeats = []
    edit_list = []
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

        if count >= 3:
            consecutive_repeats.append((index - count+1, index, value,count))
    consecutive_repeats = np.asarray(consecutive_repeats)
    for i,row in enumerate(consecutive_repeats):
        edit_list.append(consecutive_repeats[consecutive_repeats[:,0] == row[0]][-1]) 
    edit_array = np.asarray(edit_list)
    edit_array = np.unique(edit_array,axis=0)
    
 
    #for count,row in enumerate(consecutive_repeats):
    #    if consecutive_repeats[count][-1] != 3:
    #        del consecutive_repeats[count]
    

    return edit_array

def assign_date_index(df, train = False):
    '''assign dates with index 0 == first day'''
    df = df.copy(deep = True)
    date_list = df.date.unique().tolist() 
    date_list = sorted([int(x) for x in date_list])
    date_list = list(map(str, date_list))
    date_list = ['0'+ date for date in date_list]
    index_list = list(range((len(date_list))))
    if train == True:
        index_list = [(i+1)*-1 for i in index_list][::-1]
    date_index_dict = dict(zip(date_list,index_list))
    for ind,row in df.iterrows():
        df.at[ind,'date_index']= int(date_index_dict.get(row.date))

    return df
def symmetrize_y_axis(axes):
    y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=-y_max, ymax=y_max)
    
def edge_outliers(array,thresh):
    "take out edege outliers in interpolations"
    diff = np.diff(array)
    if np.abs(diff[-1]) > thresh:
        array[-1] = (array[-2] * 1.25)
    else:
        array = array
    return array


def create_df_by_type_nostart(df):
    '''creates df of only long aprochaes'''
    cluster_0_long = df[(df['obstacle_cluster']==0)&(df['odd']== 'right')]
    cluster_1_long = df[(df['obstacle_cluster']==1)&(df['odd']== 'left')]
    cluster_4_long = df[(df['obstacle_cluster']==4)&(df['odd']== 'right')]
    cluster_5_long = df[(df['obstacle_cluster']==5)&(df['odd']== 'left')]

    cluster_0_short = df[(df['obstacle_cluster']==0)&(df['odd']== 'left')]
    cluster_1_short = df[(df['obstacle_cluster']==1)&(df['odd']== 'right')]
    cluster_4_short = df[(df['obstacle_cluster']==4)&(df['odd']== 'left')]
    cluster_5_short = df[(df['obstacle_cluster']==5)&(df['odd']== 'right')]

    cluster_2 = df[(df['obstacle_cluster']==2)]
    cluster_3 = df[(df['obstacle_cluster']==3)]

    
    

    long_df = pd.concat([cluster_0_long,cluster_1_long,cluster_4_long,cluster_5_long])
    short_df = pd.concat([cluster_0_short,cluster_1_short,cluster_4_short,cluster_5_short])
    middle_df = pd.concat([cluster_2,cluster_3])
    return long_df,short_df,middle_df


def largest_sequentially_increasing_by_one_subarray(arr):
    max_len = 1  # To store the length of the largest sequential segment
    max_start = 0  # To store the start index of the largest segment
    start = 0  # Start index of the current sequential segment

    # Iterate over the array to find sequentially increasing by 1 segments
    for i in range(1, len(arr)):
        # If the current element is not exactly 1 greater than the previous one
        if arr[i] != arr[i - 1] + 1:
            # Calculate the length of the current segment
            length = i - start
            if length > max_len:
                max_len = length
                max_start = start
            # Start a new segment from the current element
            start = i

    # Handle the case if the largest segment is at the end of the array
    if len(arr) - start > max_len:
        max_start = start
        max_len = len(arr) - start

    # Return the largest sequentially increasing by 1 subarray
    return arr[max_start:max_start + max_len]

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

def smooth(points,sigma = 3):
    filtered = gaussian_filter(points.astype(float),sigma = sigma)
    return filtered