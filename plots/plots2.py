"""plots for light_dark and visual condition data"""
import cv2
import json, os, cv2
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import xarray as xr
import seaborn as sns
import h5py as hf
from tqdm import tqdm
from tqdm import tqdm
import itertools 
from scipy.interpolate import interp1d
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import os, fnmatch
from scipy.spatial.distance import cdist
import copy

import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
#sys.path.append(r'C:\Users\nlab\Documents\GitHub\obstacle_avoidance\plots') # go to parent dir 

from utils.base_functions import *

#from plots.plots import *
import warnings
warnings.filterwarnings('ignore')

def calculate_relative_distance_goal_ts(df):
    '''calculates change in distance from nose to goal port '''
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                nose_x = row['ts_nose_x_cm'].astype(float)
                nose_y = row['ts_nose_y_cm'].astype(float)
                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
                port_x_tar = np.nanmean([np.nanmean(row['leftportB_x_cm']),row['leftportT_x_cm']])
                port_y_tar = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
                port_x_start = np.nanmean([np.nanmean(row['rightportB_x_cm']),row['rightportT_x_cm']])
                port_y_start = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
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
                nose_x = row['ts_nose_x_cm'].astype(float)
                nose_y = row['ts_nose_y_cm'].astype(float)
                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
                port_x_tar = np.nanmean([np.nanmean(row['rightportB_x_cm']),row['rightportT_x_cm']])
                port_y_tar = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
                port_x_start = np.nanmean([np.nanmean(row['leftportB_x_cm']),row['leftportT_x_cm']])
                port_y_start = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
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
    '''calculates change in distance from nose to goal port '''
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                nose_x = row['nose_x_cm'].astype(float)
                nose_y = row['nose_y_cm'].astype(float)
                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
                port_x_tar = np.nanmean([np.nanmean(row['leftportB_x_cm']),row['leftportT_x_cm']])
                port_y_tar = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
                port_x_start = np.nanmean([np.nanmean(row['rightportB_x_cm']),row['rightportT_x_cm']])
                port_y_start = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
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
                nose_x = row['nose_x_cm'].astype(float)
                nose_y = row['nose_y_cm'].astype(float)
                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
                port_x_tar = np.nanmean([np.nanmean(row['rightportB_x_cm']),row['rightportT_x_cm']])
                port_y_tar = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
                port_x_start = np.nanmean([np.nanmean(row['leftportB_x_cm']),row['leftportT_x_cm']])
                port_y_start = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
                distances_tar = []
                distances_start = []
                for i,e in enumerate(nose_x):
                  distance_tar = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_tar,port_y_tar))
                  distance_start = np.abs(calculate_distances(nose_x[i],nose_y[i],port_x_start,port_y_start))
                  distances_tar.append(distance_tar)
                  distances_start.append(distance_start)
                df.at[ind,'distance_from_target_port'] = np.array(distances_tar).astype(object)
                df.at[ind,'distance_from_start_port'] = np.array(distances_start).astype(object)

def ts_calculate_relative_distance(df):
    """calculates relavtive distance of nose to point on obstacle"""
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                try:
                    nose_x = row['ts_nose_x_cm']
                    nose_y = row['ts_nose_y_cm']
                    distances = []
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']]) 
                        if nose_y[i] > row['gt_obstacleTR_y_cm'] and nose_y[i] < row['gt_obstacleBR_y_cm'] :
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

                
                df.at[ind,'ts_distance_from_edge'] = np.array(distances).astype(object)
                df.at[ind,'ts_len_distance_from_edge'] = np.array(distances).astype(object).size
            if direction =='left':
                try:
                    nose_x = row['ts_nose_x_cm']
                    nose_y = row['ts_nose_y_cm']
                    distances = []
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']]) 
                        if nose_y[i] > row['gt_obstacleTL_y_cm'] and nose_y[i] < row['gt_obstacleBL_y_cm']:
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
            
                
                df.at[ind,'ts_distance_from_edge'] = np.array(distances).astype(object)
                df.at[ind,'ts_len_distance_from_edge'] = np.array(distances).astype(object).size

def calculate_relative_distance(df):
    """calculates relavtive distance of nose to point on obstacle"""
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():  
            if direction == 'right':
                try:
                    nose_x = row['nose_x_cm']
                    nose_y = row['nose_y_cm']
                    distances = []
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']]) 
                        if nose_y[i] > row['gt_obstacleTR_y_cm'] and nose_y[i] < row['gt_obstacleBR_y_cm'] :
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

                
                df.at[ind,'distance_from_edge'] = np.array(distances).astype(object)
                df.at[ind,'len_distance_from_edge'] = np.array(distances).astype(object).size
            if direction =='left':
                try:
                    nose_x = row['nose_x_cm']
                    nose_y = row['nose_y_cm']
                    distances = []
                    distances = []
                    for i,e in enumerate(nose_x):
                        obstalce_x = np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']]) 
                        if nose_y[i] > row['gt_obstacleTL_y_cm'] and nose_y[i] < row['gt_obstacleBL_y_cm']:
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
            
                
                df.at[ind,'distance_from_edge'] = np.array(distances).astype(object)
                df.at[ind,'len_distance_from_edge'] = np.array(distances).astype(object).size


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
def ts_angle_to_open_corner(df):
    """get angel of nose to open corner at a hold"""
    for direction, direction_frame in df.groupby(['odd']):
        for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
            for ind,row in cluster_frame.iterrows():
                    nose_x = row['ts_nose_x_cm'].astype(float)
                    nose_y = row['ts_nose_y_cm'].astype(float)
                    head_center_x,head_center_y = np.mean([row.nose_x_cm,row.rightear_x_cm,row.leftear_x_cm],axis=0),np.mean([row.nose_y_cm,row.rightear_y_cm,row.leftear_y_cm],axis=0)
                    reye_x,reye_y = np.mean([row.nose_x_cm,row.rightear_x_cm],axis=0),np.mean([row.nose_y_cm,row.rightear_y_cm],axis=0)
                    leye_x,leye_y = np.mean([row.nose_x_cm,row.leftear_x_cm],axis=0),np.mean([row.nose_y_cm,row.leftear_y_cm],axis=0)
                    ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
                    ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
                    spine_x, spine_y = row.ts_spine_x_cm.astype(float),row.ts_spine_y_cm.astype(float)
                    tailbase_x, tailbase_y = row.ts_tailbase_x_cm.astype(float),row.ts_tailbase_y_cm.astype(float) 
                    if direction == 'right':
                        if cluster == 0 or cluster == 1:
                            corner_x = row['gt_obstacleBR_x_cm']
                            corner_y = row['gt_obstacleBR_y_cm']
                            degs = []
                            
                            reye_degs = []
                            
                            leye_degs = []

                            body_degs = []
                            
                            
                            for i in list(range(len(nose_x))):
                                vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                rad,deg = angle_between_vectors(vector1,vector2)
                                degs.append(deg)
                                
                                reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                reye_degs.append(reye_deg)
                                
                                leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                leye_degs.append(leye_deg)

                                body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                body_degs.append(body_deg)




                                
                            degs = np.array(degs)
                            leye_degs = np.array(leye_degs)
                            reye_degs = np.array(reye_degs)
                            body_degs = np.array(body_degs)
                            
                        
                            #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                            try:
                             df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
                             df.at[ind,'ts_leye_angle_to_corner'] = leye_degs.astype(object)
                             df.at[ind,'ts_reye_angle_to_corner'] = reye_degs.astype(object)
                             df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                            except ValueError:
                                print(degs)
                            except IndexError:
                                df.at[ind,'ts_angle_to_corner'] = np.nan
                                df.at[ind,'ts_leye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_reye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_body_angle_to_corner'] = np.nan
                        if cluster == 2 or cluster == 3:
                                obstalce_edge= np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']])
                                nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
                                ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
                                distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTR_y_cm']))
                                distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBR_y_cm']))
                                if distance_to_top < distance_to_bottom:
                                    corner_x = row['gt_obstacleTR_x_cm']
                                    corner_y = row['gt_obstacleTR_y_cm']
                                    degs = []
                                    
                                    reye_degs = []
                                    
                                    leye_degs = []

                                    body_degs = []
                                    
                                    for i in list(range(len(nose_x))):
                                        vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                        vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        rad,deg = angle_between_vectors(vector1,vector2)
                                        degs.append(deg)
                                        
                                        reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                        reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                        reye_degs.append(reye_deg)
                                        
                                        leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                        leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                        leye_degs.append(leye_deg)

                                        body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                        body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                        body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                        body_degs.append(body_deg)

                                        
                                    degs = np.array(degs)
                                    leye_degs = np.array(leye_degs)
                                    reye_degs = np.array(reye_degs)
                                    body_degs = np.array(body_degs)
                                    
                                if distance_to_top > distance_to_bottom:
                                    corner_x = row['gt_obstacleBR_x_cm']
                                    corner_y = row['gt_obstacleBR_y_cm']
                                    degs = []
                                    
                                    reye_degs = []
                                    
                                    leye_degs = []

                                    body_degs = []
                                    
                                    for i in list(range(len(nose_x))):
                                        vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                        vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        rad,deg = angle_between_vectors(vector1,vector2)
                                        degs.append(deg)
                                        
                                        reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                        reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                        reye_degs.append(reye_deg)
                                        
                                        leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                        leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                        leye_degs.append(leye_deg)

                                        body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                        body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                        body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                        body_degs.append(body_deg)
                                        
                                    degs = np.array(degs)
                                    leye_degs = np.array(leye_degs)
                                    reye_degs = np.array(reye_degs)
                                    body_degs = np.array(body_degs)
                                    
                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                                try:
                                    df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
                                    df.at[ind,'ts_leye_angle_to_corner'] = leye_degs.astype(object)
                                    df.at[ind,'ts_reye_angle_to_corner'] = reye_degs.astype(object)
                                    df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                                except IndexError:
                                    df.at[ind,'ts_angle_to_corner'] = np.nan
                                    df.at[ind,'ts_leye_angle_to_corner'] = np.nan
                                    df.at[ind,'ts_reye_angle_to_corner'] = np.nan
                                    df.at[ind,'ts_body_angle_to_corner'] = np.nan
                                    
                                    
                                    
                        if cluster == 4 or cluster == 5:
                            corner_x = row['gt_obstacleTR_x_cm']
                            corner_y = row['gt_obstacleTR_y_cm']
                            degs = []
                            
                            reye_degs = []
                            
                            leye_degs = []

                            body_degs = []
                            
                            for i in list(range(len(nose_x))):
                                vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                rad,deg = angle_between_vectors(vector1,vector2)
                                degs.append(deg)
                                
                                reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                reye_degs.append(reye_deg)
                                
                                leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                                leye_degs.append(leye_deg)

                                body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                body_degs.append(body_deg)
                                
                            degs = np.array(degs)
                            leye_degs = np.array(leye_degs)
                            reye_degs = np.array(reye_degs)
                            body_degs = np.array(body_degs)
                            
                            #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                            try:
                                df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
                                df.at[ind,'ts_leye_angle_to_corner'] = leye_degs.astype(object)
                                df.at[ind,'ts_reye_angle_to_corner'] = reye_degs.astype(object)
                                df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                            except IndexError:
                                df.at[ind,'ts_angle_to_corner'] = np.nan
                                df.at[ind,'ts_leye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_reye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                             
                    if direction == 'left':
                        if cluster == 0 or cluster == 1:
                            corner_x = row['gt_obstacleBL_x_cm']
                            corner_y = row['gt_obstacleBL_y_cm']
                            degs = []
                            
                            reye_degs = []
                            
                            leye_degs = []

                            body_degs = []
                            
                            for i in list(range(len(nose_x))):
                                vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                rad,deg = angle_between_vectors(vector1,vector2)
                                degs.append(deg)
                                
                                reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                reye_degs.append(reye_deg)
                                
                                leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                                leye_degs.append(leye_deg)
                                
                                body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                body_degs.append(body_deg)
                            degs = np.array(degs)
                            leye_degs = np.array(leye_degs)
                            reye_degs = np.array(reye_degs)
                            body_degs = np.array(body_degs)
                            
                            #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                            try:
                                df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
                                df.at[ind,'ts_leye_angle_to_corner'] = leye_degs.astype(object)
                                df.at[ind,'ts_reye_angle_to_corner'] = reye_degs.astype(object)
                                df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                            except IndexError:
                                df.at[ind,'ts_angle_to_corner'] = np.nan
                                df.at[ind,'ts_leye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_reye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_body_angle_to_corner'] = np.nan
                        if cluster == 2 or cluster == 3:
                                obstalce_edge= np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
                                nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
                                ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
                                distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTL_y_cm']))
                                distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBL_y_cm']))
                                if distance_to_top < distance_to_bottom:
                                    corner_x = row['gt_obstacleTL_x_cm']
                                    corner_y = row['gt_obstacleTL_y_cm']
                                    degs = []
                                    
                                    reye_degs = []
                                    
                                    leye_degs = []

                                    body_degs = []
                                    
                                    for i in list(range(len(nose_x))):
                                        vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                        vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        rad,deg = angle_between_vectors(vector1,vector2)
                                        degs.append(deg)
                                        
                                        reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                        reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                        reye_degs.append(reye_deg)
                                        
                                        leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                        leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                        leye_degs.append(leye_deg)

                                        body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                        body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                        body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                        body_degs.append(body_deg)
                                        
                                    degs = np.array(degs)
                                    leye_degs = np.array(leye_degs)
                                    reye_degs = np.array(reye_degs)
                                    body_degs = np.array(body_degs)
                                    
                                if distance_to_top > distance_to_bottom:
                                    corner_x = row['gt_obstacleBL_x_cm']
                                    corner_y = row['gt_obstacleBL_y_cm']
                                    degs = []
                                    
                                    reye_degs = []
                                    
                                    leye_degs = []

                                    body_degs = []
                                    
                                    for i in list(range(len(nose_x))):
                                        vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                        vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        rad,deg = angle_between_vectors(vector1,vector2)
                                        degs.append(deg)
                                        
                                        reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                        reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                        reye_degs.append(reye_deg)
                                        
                                        leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                        leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                        leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                        leye_degs.append(leye_deg)

                                        body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                        body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                        body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                        body_degs.append(body_deg)
                                        
                                    degs = np.array(degs)
                                    leye_degs = np.array(leye_degs)
                                    reye_degs = np.array(reye_degs)
                                    body_degs = np.array(body_degs)
                                    
                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                                try:
                                    df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
                                    df.at[ind,'ts_leye_angle_to_corner'] = leye_degs.astype(object)
                                    df.at[ind,'ts_reye_angle_to_corner'] = reye_degs.astype(object)
                                    df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                                except IndexError:
                                    df.at[ind,'ts_angle_to_corner'] = np.nan
                                    df.at[ind,'ts_leye_angle_to_corner'] = np.nan
                                    df.at[ind,'ts_reye_angle_to_corner'] = np.nan
                                    df.at[ind,'ts_body_angle_to_corner'] = np.nan
                            
                            
                        if cluster == 4 or cluster == 5:
                            corner_x = row['gt_obstacleTL_x_cm']
                            corner_y = row['gt_obstacleTL_y_cm']
                            degs = []
                            
                            reye_degs = []
                            
                            leye_degs = []

                            body_degs = []
                            
                            for i in list(range(len(nose_x))):
                                vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                rad,deg = angle_between_vectors(vector1,vector2)
                                degs.append(deg)
                                
                                reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                reye_degs.append(reye_deg)
                            
                                leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                                leye_degs.append(leye_deg)

                                body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                body_degs.append(body_deg)
                                
                            degs = np.array(degs)
                            leye_degs = np.array(leye_degs)
                            reye_degs = np.array(reye_degs)
                            body_degs = np.array(body_degs)
                            
                            #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                            try:
                                df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
                                df.at[ind,'ts_leye_angle_to_corner'] = leye_degs.astype(object)
                                df.at[ind,'ts_reye_angle_to_corner'] = reye_degs.astype(object)
                                df.at[ind,'ts_body_angle_to_corner'] = body_degs.astype(object)
                            except IndexError:
                                df.at[ind,'ts_angle_to_corner'] = np.nan
                                df.at[ind,'ts_leye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_reye_angle_to_corner'] = np.nan
                                df.at[ind,'ts_body_angle_to_corner'] = np.nan


#def ts_angle_to_open_corner(df):
#    """get angel of nose to open corner at a hold"""
#    for direction, direction_frame in df.groupby(['odd']):
#            for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
#                  for ind,row in cluster_frame.iterrows():
#                        if direction == 'right':
#                            if cluster == 0 or cluster == 1:
#                                nose_x = row['ts_nose_x_cm'].astype(float)
#                                nose_y = row['ts_nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleBR_x_cm']
#                                corner_y = row['gt_obstacleBR_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                 df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
#                                except ValueError:
#                                    print(degs)
#                                except IndexError:
#                                    df.at[ind,'ts_angle_to_corner'] = np.nan
#                            if cluster == 2 or cluster == 3:
#                                    nose_x = row['ts_nose_x_cm'].astype(float)
#                                    nose_y = row['ts_nose_y_cm'].astype(float)
#                                    ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
#                                    ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
#                                    obstalce_edge= np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']])
#                                    nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
#                                    ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
#                                    distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTR_y_cm']))
#                                    distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBR_y_cm']))
#                                    if distance_to_top < distance_to_bottom:
#                                        corner_x = row['gt_obstacleTR_x_cm']
#                                        corner_y = row['gt_obstacleTR_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    if distance_to_top > distance_to_bottom:
#                                        corner_x = row['gt_obstacleBR_x_cm']
#                                        corner_y = row['gt_obstacleBR_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                    try:
#                                        df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
#                                    except IndexError:
#                                        df.at[ind,'ts_angle_to_corner'] = np.nan
#                                        
#                                        
#
#                                        
#                            if cluster == 4 or cluster == 5:
#                                nose_x = row['ts_nose_x_cm'].astype(float)
#                                nose_y = row['ts_nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleTR_x_cm']
#                                corner_y = row['gt_obstacleTR_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                    df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
#                                except IndexError:
#                                    df.at[ind,'ts_angle_to_corner'] = np.nan
#                                 
#                        if direction == 'left':
#                            if cluster == 0 or cluster == 1:
#                                nose_x = row['ts_nose_x_cm'].astype(float)
#                                nose_y = row['ts_nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleBL_x_cm']
#                                corner_y = row['gt_obstacleBL_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                    df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
#                                except IndexError:
#                                    df.at[ind,'ts_angle_to_corner'] = np.nan
#                            if cluster == 2 or cluster == 3:
#                                    nose_x = row['ts_nose_x_cm'].astype(float)
#                                    nose_y = row['ts_nose_y_cm'].astype(float)
#                                    ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
#                                    ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
#                                    obstalce_edge= np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
#                                    nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
#                                    ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
#                                    distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTL_y_cm']))
#                                    distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBL_y_cm']))
#                                    if distance_to_top < distance_to_bottom:
#                                        corner_x = row['gt_obstacleTL_x_cm']
#                                        corner_y = row['gt_obstacleTL_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    if distance_to_top > distance_to_bottom:
#                                        corner_x = row['gt_obstacleBL_x_cm']
#                                        corner_y = row['gt_obstacleBL_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                    try:
#                                        df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
#                                    except IndexError:
#                                        df.at[ind,'ts_angle_to_corner'] = np.nan
#                                
#                                
#                            if cluster == 4 or cluster == 5:
#                                nose_x = row['ts_nose_x_cm'].astype(float)
#                                nose_y = row['ts_nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleTL_x_cm']
#                                corner_y = row['gt_obstacleTL_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                    df.at[ind,'ts_angle_to_corner'] = degs.astype(object)
#                                except IndexError:
#                                    df.at[ind,'ts_angle_to_corner'] = np.nan
def angle_to_open_corner(df):
    """get angel of nose to open corner at a hold"""
    for direction, direction_frame in df.groupby(['odd']):
            
        for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
            for ind,row in cluster_frame.iterrows():
                nose_x = row['nose_x_cm'].astype(float)
                nose_y = row['nose_y_cm'].astype(float)
                head_center_x,head_center_y = np.mean([row.nose_x_cm,row.rightear_x_cm,row.leftear_x_cm],axis=0),np.mean([row.nose_y_cm,row.rightear_y_cm,row.leftear_y_cm],axis=0)
                reye_x,reye_y = np.mean([row.nose_x_cm,row.rightear_x_cm],axis=0),np.mean([row.nose_y_cm,row.rightear_y_cm],axis=0)
                leye_x,leye_y = np.mean([row.nose_x_cm,row.leftear_x_cm],axis=0),np.mean([row.nose_y_cm,row.leftear_y_cm],axis=0)
                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
                spine_x, spine_y = row.spine_x_cm.astype(float),row.spine_y_cm.astype(float)
                tailbase_x, tailbase_y = row.tailbase_x_cm.astype(float),row.tailbase_y_cm.astype(float) 

                if direction == 'right':
                    if cluster == 0 or cluster == 1:
                        corner_x = row['gt_obstacleBR_x_cm']
                        corner_y = row['gt_obstacleBR_y_cm']
                        degs = []
                        
                        reye_degs = []
                        
                        leye_degs = []
                        
                        body_degs = []
                        
                        for i in list(range(len(nose_x))):
                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            rad,deg = angle_between_vectors(vector1,vector2)
                            degs.append(deg)
                            
                            reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                            reye_degs.append(reye_deg)
                            
                            leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                            leye_degs.append(leye_deg)
                            
                            body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                            body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                            body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                            body_degs.append(body_deg)

                        degs = np.array(degs)
                        leye_degs = np.array(leye_degs)
                        reye_degs = np.array(reye_degs)
                        body_degs = np.array(body_degs) 
                        
                    
                        #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                        try:
                         df.at[ind,'angle_to_corner'] = degs.astype(object)
                         df.at[ind,'leye_angle_to_corner'] = leye_degs.astype(object)
                         df.at[ind,'reye_angle_to_corner'] = reye_degs.astype(object)
                         df.at[ind,'body_angle_to_corner'] = body_degs.astype(object)
                         
                        except ValueError:
                            print(degs)
                        except IndexError:
                            df.at[ind,'angle_to_corner'] = np.nan
                            df.at[ind,'leye_angle_to_corner'] = np.nan
                            df.at[ind,'reye_angle_to_corner'] = np.nan
                            df.at[ind,'body_angle_to_corner'] = np.nan
                            
                    if cluster == 2 or cluster == 3:
                            obstalce_edge= np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']])
                            nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
                            ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
                            distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTR_y_cm']))
                            distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBR_y_cm']))
                            if distance_to_top < distance_to_bottom:
                                corner_x = row['gt_obstacleTR_x_cm']
                                corner_y = row['gt_obstacleTR_y_cm']
                                degs = []
                                
                                reye_degs = []
                                
                                leye_degs = []

                                body_degs = []
                                
                                for i in list(range(len(nose_x))):
                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    rad,deg = angle_between_vectors(vector1,vector2)
                                    degs.append(deg)
                                    
                                    reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                    reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                    reye_degs.append(reye_deg)
                                    
                                    leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                    leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                    leye_degs.append(leye_deg)

                                    body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                    body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                    body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                    body_degs.append(body_deg)
                                    
                                degs = np.array(degs)
                                leye_degs = np.array(leye_degs)
                                reye_degs = np.array(reye_degs)
                                body_degs = np.array(body_degs)
                                
                            if distance_to_top > distance_to_bottom:
                                corner_x = row['gt_obstacleBR_x_cm']
                                corner_y = row['gt_obstacleBR_y_cm']
                                degs = []
                                
                                reye_degs = []
                                
                                leye_degs = []

                                body_degs = []
                                
                                for i in list(range(len(nose_x))):
                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    rad,deg = angle_between_vectors(vector1,vector2)
                                    degs.append(deg)
                                    
                                    reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                    reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                    reye_degs.append(reye_deg)
                                    
                                    leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                    leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                    leye_degs.append(leye_deg)

                                    body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                    body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                    body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                    body_degs.append(body_deg)

                                    
                                    
                                degs = np.array(degs)
                                leye_degs = np.array(leye_degs)
                                reye_degs = np.array(reye_degs)
                                body_degs = np.array(body_degs)
                                
                            #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                            try:
                                df.at[ind,'angle_to_corner'] = degs.astype(object)
                                df.at[ind,'leye_angle_to_corner'] = leye_degs.astype(object)
                                df.at[ind,'reye_angle_to_corner'] = reye_degs.astype(object)
                                df.at[ind,'body_angle_to_corner'] = body_degs.astype(object)
                            except IndexError:
                                df.at[ind,'angle_to_corner'] = np.nan
                                df.at[ind,'leye_angle_to_corner'] = np.nan
                                df.at[ind,'reye_angle_to_corner'] = np.nan
                                df.at[ind,'body_angle_to_corner'] = np.nan
                                
                                
                                
                    if cluster == 4 or cluster == 5:
                        corner_x = row['gt_obstacleTR_x_cm']
                        corner_y = row['gt_obstacleTR_y_cm']
                        degs = []
                        
                        reye_degs = []
                        
                        leye_degs = []

                        body_degs = []
                        
                        for i in list(range(len(nose_x))):
                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            rad,deg = angle_between_vectors(vector1,vector2)
                            degs.append(deg)
                            
                            reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                            reye_degs.append(reye_deg)
                            
                            leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                            leye_degs.append(leye_deg)

                            body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                            body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                            body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                            body_degs.append(body_deg)
                            
                        degs = np.array(degs)
                        leye_degs = np.array(leye_degs)
                        reye_degs = np.array(reye_degs)
                        body_degs = np.array(body_degs)
                        
                        #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                        try:
                            df.at[ind,'angle_to_corner'] = degs.astype(object)
                            df.at[ind,'leye_angle_to_corner'] = leye_degs.astype(object)
                            df.at[ind,'reye_angle_to_corner'] = reye_degs.astype(object)
                            df.at[ind,'body_angle_to_corner'] = body_degs.astype(object)
                        except IndexError:
                            df.at[ind,'angle_to_corner'] = np.nan
                            df.at[ind,'leye_angle_to_corner'] = np.nan
                            df.at[ind,'reye_angle_to_corner'] = np.nan
                            df.at[ind,'body_angle_to_corner'] = np.nan
                         
                if direction == 'left':
                    if cluster == 0 or cluster == 1:
                        corner_x = row['gt_obstacleBL_x_cm']
                        corner_y = row['gt_obstacleBL_y_cm']
                        degs = []
                        
                        reye_degs = []
                        
                        leye_degs = []

                        body_degs = []
                        
                        for i in list(range(len(nose_x))):
                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            rad,deg = angle_between_vectors(vector1,vector2)
                            degs.append(deg)
                            
                            reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                            reye_degs.append(reye_deg)
                            
                            leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                            leye_degs.append(leye_deg)

                            body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                            body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                            body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                            body_degs.append(body_deg)
                            
                        degs = np.array(degs)
                        leye_degs = np.array(leye_degs)
                        reye_degs = np.array(reye_degs)
                        body_degs = np.array(body_degs)
                        
                        #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                        try:
                            df.at[ind,'angle_to_corner'] = degs.astype(object)
                            df.at[ind,'leye_angle_to_corner'] = leye_degs.astype(object)
                            df.at[ind,'reye_angle_to_corner'] = reye_degs.astype(object)
                            df.at[ind,'body_angle_to_corner'] = body_degs.astype(object)
                        except IndexError:
                            df.at[ind,'angle_to_corner'] = np.nan
                            df.at[ind,'leye_angle_to_corner'] = np.nan
                            df.at[ind,'reye_angle_to_corner'] = np.nan
                            df.at[ind,'body_angle_to_corner'] = np.nan
                    if cluster == 2 or cluster == 3:
                            obstalce_edge= np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
                            nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
                            ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
                            distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTL_y_cm']))
                            distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBL_y_cm']))
                            if distance_to_top < distance_to_bottom:
                                corner_x = row['gt_obstacleTL_x_cm']
                                corner_y = row['gt_obstacleTL_y_cm']
                                degs = []
                                
                                reye_degs = []
                                
                                leye_degs = []

                                body_degs = []
                                
                                for i in list(range(len(nose_x))):
                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    rad,deg = angle_between_vectors(vector1,vector2)
                                    degs.append(deg)
                                    
                                    reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                    reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                    reye_degs.append(reye_deg)
                                    
                                    leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                    leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                    leye_degs.append(leye_deg)

                                    body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                    body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                    body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                    body_degs.append(body_deg)
                                    
                                degs = np.array(degs)
                                leye_degs = np.array(leye_degs)
                                reye_degs = np.array(reye_degs)
                                body_degs = np.array(body_degs)
                                
                            if distance_to_top > distance_to_bottom:
                                corner_x = row['gt_obstacleBL_x_cm']
                                corner_y = row['gt_obstacleBL_y_cm']
                                degs = []
                                
                                reye_degs = []
                                
                                leye_degs = []

                                body_degs = []
                                
                                for i in list(range(len(nose_x))):
                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    rad,deg = angle_between_vectors(vector1,vector2)
                                    degs.append(deg)
                                    
                                    reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                                    reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                                    reye_degs.append(reye_deg)
                                    


                                    leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                                    leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                                    leye_rad,leye_deg = angle_between_vectors(leye_vector1,leye_vector2)
                                    leye_degs.append(leye_deg)

                                    body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                                    body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                                    body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                                    body_degs.append(body_deg)
                                    
                                degs = np.array(degs)
                                leye_degs = np.array(leye_degs)
                                reye_degs = np.array(reye_degs)
                                body_degs = np.array(body_degs)
                                
                            #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                            try:
                                df.at[ind,'angle_to_corner'] = degs.astype(object)
                                df.at[ind,'leye_angle_to_corner'] = leye_degs.astype(object)
                                df.at[ind,'reye_angle_to_corner'] = reye_degs.astype(object)
                                df.at[ind,'body_angle_to_corner'] = body_degs.astype(object)
                            except IndexError:
                                df.at[ind,'angle_to_corner'] = np.nan
                                df.at[ind,'leye_angle_to_corner'] = np.nan
                                df.at[ind,'reye_angle_to_corner'] = np.nan
                                df.at[ind,'body_angle_to_corner'] = np.nan
                        
                        
                    if cluster == 4 or cluster == 5:
                        corner_x = row['gt_obstacleTL_x_cm']
                        corner_y = row['gt_obstacleTL_y_cm']
                        degs = []
                        
                        reye_degs = []
                        
                        leye_degs = []

                        body_degs = []
                        
                        for i in list(range(len(nose_x))):
                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            rad,deg = angle_between_vectors(vector1,vector2)
                            degs.append(deg)
                            
                            reye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(reye_x[i],reye_y[i]))# vector from ear to nose
                            reye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            reye_rad,reye_deg = angle_between_vectors(reye_vector1,reye_vector2)
                            reye_degs.append(reye_deg)
                            
                            leye_vector1 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(leye_x[i],leye_y[i]))# vector from ear to nose
                            leye_vector2 = calculate_vector_between_points((head_center_x[i],head_center_y[i]),(corner_x,corner_y))# vector from nose to open corner
                            leye_rad,leye_deg = angle_between_vectors(leye_vector1,reye_vector2)
                            leye_degs.append(leye_deg)

                            body_vector1 = calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(spine_x[i],spine_y[i]))
                            body_vector2 =  calculate_vector_between_points((tailbase_x[i],tailbase_y[i]),(corner_x,corner_y))
                            body_rad,body_deg = angle_between_vectors(body_vector1,body_vector2)
                            body_degs.append(body_deg)


                            
                        degs = np.array(degs)
                        leye_degs = np.array(leye_degs)
                        reye_degs = np.array(reye_degs)
                        body_degs = np.array(body_degs)
                        
                        #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
                        try:
                            df.at[ind,'angle_to_corner'] = degs.astype(object)
                            df.at[ind,'leye_angle_to_corner'] = leye_degs.astype(object)
                            df.at[ind,'reye_angle_to_corner'] = reye_degs.astype(object)
                            df.at[ind,'body_angle_to_corner'] = body_degs.astype(object)
                        except IndexError:
                            df.at[ind,'angle_to_corner'] = np.nan
                            df.at[ind,'leye_angle_to_corner'] = np.nan
                            df.at[ind,'reye_angle_to_corner'] = np.nan
                            df.at[ind,'body_angle_to_corner'] = np.nan
#def angle_to_open_corner(df):
#    """get angel of nose to open corner at a hold"""
#    for direction, direction_frame in df.groupby(['odd']):
#            for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
#                  for ind,row in cluster_frame.iterrows():
#                        if direction == 'right':
#                            if cluster == 0 or cluster == 1:
#                                nose_x = row['nose_x_cm'].astype(float)
#                                nose_y = row['nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleBR_x_cm']
#                                corner_y = row['gt_obstacleBR_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                 df.at[ind,'angle_to_corner'] = degs.astype(object)
#                                except ValueError:
#                                    print(degs)
#                                except IndexError:
#                                    df.at[ind,'angle_to_corner'] = np.nan
#                            if cluster == 2 or cluster == 3:
#                                    nose_x = row['nose_x_cm'].astype(float)
#                                    nose_y = row['nose_y_cm'].astype(float)
#                                    ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
#                                    ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
#                                    obstalce_edge= np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']])
#                                    nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
#                                    ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
#                                    distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTR_y_cm']))
#                                    distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBR_y_cm']))
#                                    if distance_to_top < distance_to_bottom:
#                                        corner_x = row['gt_obstacleTR_x_cm']
#                                        corner_y = row['gt_obstacleTR_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    if distance_to_top > distance_to_bottom:
#                                        corner_x = row['gt_obstacleBR_x_cm']
#                                        corner_y = row['gt_obstacleBR_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                    try:
#                                        df.at[ind,'angle_to_corner'] = degs.astype(object)
#                                    except IndexError:
#                                        df.at[ind,'angle_to_corner'] = np.nan
#                                        
#                                        
#
#                                        
#                            if cluster == 4 or cluster == 5:
#                                nose_x = row['nose_x_cm'].astype(float)
#                                nose_y = row['nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleTR_x_cm']
#                                corner_y = row['gt_obstacleTR_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                    df.at[ind,'angle_to_corner'] = degs.astype(object)
#                                except IndexError:
#                                    df.at[ind,'angle_to_corner'] = np.nan
#                                 
#                        if direction == 'left':
#                            if cluster == 0 or cluster == 1:
#                                nose_x = row['nose_x_cm'].astype(float)
#                                nose_y = row['nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleBL_x_cm']
#                                corner_y = row['gt_obstacleBL_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                    df.at[ind,'angle_to_corner'] = degs.astype(object)
#                                except IndexError:
#                                    df.at[ind,'angle_to_corner'] = np.nan
#                            if cluster == 2 or cluster == 3:
#                                    nose_x = row['nose_x_cm'].astype(float)
#                                    nose_y = row['nose_y_cm'].astype(float)
#                                    ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
#                                    ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
#                                    obstalce_edge= np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
#                                    nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
#                                    ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
#                                    distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTL_y_cm']))
#                                    distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBL_y_cm']))
#                                    if distance_to_top < distance_to_bottom:
#                                        corner_x = row['gt_obstacleTL_x_cm']
#                                        corner_y = row['gt_obstacleTL_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    if distance_to_top > distance_to_bottom:
#                                        corner_x = row['gt_obstacleBL_x_cm']
#                                        corner_y = row['gt_obstacleBL_y_cm']
#                                        degs = []
#                                        
#                                        for i in list(range(len(nose_x))):
#                                            vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                            vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                            rad,deg = angle_between_vectors(vector1,vector2)
#                                            degs.append(deg)
#                                            
#                                        degs = np.array(degs)
#                                    #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                    try:
#                                        df.at[ind,'angle_to_corner'] = degs.astype(object)
#                                    except IndexError:
#                                        df.at[ind,'angle_to_corner'] = np.nan
#                                
#                                
#                            if cluster == 4 or cluster == 5:
#                                nose_x = row['nose_x_cm'].astype(float)
#                                nose_y = row['nose_y_cm'].astype(float)
#                                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
#                                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
#                                corner_x = row['gt_obstacleTL_x_cm']
#                                corner_y = row['gt_obstacleTL_y_cm']
#                                degs = []
#                                
#                                for i in list(range(len(nose_x))):
#                                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
#                                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(corner_x,corner_y))# vector from nose to open corner
#                                    rad,deg = angle_between_vectors(vector1,vector2)
#                                    degs.append(deg)
#                                    
#                                degs = np.array(degs)
#                                #df.at[ind,'angle_to_corner_' + str()] = degs.astype(object)
#                                try:
#                                    df.at[ind,'angle_to_corner'] = degs.astype(object)
#                                except IndexError:
#                                    df.at[ind,'angle_to_corner'] = np.nan


def angle_to_target_port(df):
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():
            if direction == 'right':
                nose_x = row['nose_x_cm'].astype(float)
                nose_y = row['nose_y_cm'].astype(float)
                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
                port_x = np.nanmean([np.nanmean(row['leftportB_x_cm']),row['leftportT_x_cm']])
                port_y = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
                degs = []
                #
                for i in list(range(len(nose_x))):
                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(port_x,port_y))# vector from nose to open corner
                    rad,deg = angle_between_vectors(vector1,vector2)
                    degs.append(deg)
                    #
                degs = np.array(degs)
                df.at[ind,'angle_to_target_port'] = degs.astype(object)
            else:
                nose_x = row['nose_x_cm'].astype(float)
                nose_y = row['nose_y_cm'].astype(float)
                ear_x = np.mean([row['rightear_x_cm'],row['leftear_x_cm']],axis=0)
                ear_y = np.mean([row['rightear_y_cm'],row['leftear_y_cm']],axis=0)
                port_x = np.nanmean([np.nanmean(row['rightportB_x_cm']),row['rightportT_x_cm']])
                port_y = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
                degs = []
                #
                for i in list(range(len(nose_x))):
                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(port_x,port_y))# vector from nose to open corner
                    rad,deg = angle_between_vectors(vector1,vector2)
                    degs.append(deg)
                    #
                degs = np.array(degs)
                df.at[ind,'angle_to_target_port'] = degs.astype(object)

def ts_angle_to_target_port(df):
    for direction, direction_frame in df.groupby(['odd']):
        for ind,row in direction_frame.iterrows():
            if direction == 'right':
                nose_x = row['ts_nose_x_cm'].astype(float)
                nose_y = row['ts_nose_y_cm'].astype(float)
                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
                port_x = np.nanmean([np.nanmean(row['leftportB_x_cm']),row['leftportT_x_cm']])
                port_y = np.nanmean([np.nanmean(row['leftportB_y_cm']),row['leftportT_y_cm']])
                degs = []
                #
                for i in list(range(len(nose_x))):
                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(port_x,port_y))# vector from nose to open corner
                    rad,deg = angle_between_vectors(vector1,vector2)
                    degs.append(deg)
                    #
                degs = np.array(degs)
                df.at[ind,'ts_angle_to_target_port'] = degs.astype(object)
            else:
                nose_x = row['ts_nose_x_cm'].astype(float)
                nose_y = row['ts_nose_y_cm'].astype(float)
                ear_x = np.mean([row['ts_rightear_x_cm'],row['ts_leftear_x_cm']],axis=0)
                ear_y = np.mean([row['ts_rightear_y_cm'],row['ts_leftear_y_cm']],axis=0)
                port_x = np.nanmean([np.nanmean(row['rightportB_x_cm']),row['rightportT_x_cm']])
                port_y = np.nanmean([np.nanmean(row['rightportB_y_cm']),row['rightportT_y_cm']])
                degs = []
                #
                for i in list(range(len(nose_x))):
                    vector1 = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
                    vector2 = calculate_vector_between_points((ear_x[i],ear_y[i]),(port_x,port_y))# vector from nose to open corner
                    rad,deg = angle_between_vectors(vector1,vector2)
                    degs.append(deg)
                    #
                degs = np.array(degs)
                df.at[ind,'ts_angle_to_target_port'] = degs.astype(object)



def calculate_vector_between_points(point1, point2):
    """
    Calculate the vector between two (x, y) points.

    Args:
        point1 (tuple or list): The coordinates of the first point (x1, y1).
        point2 (tuple or list): The coordinates of the second point (x2, y2).

    Returns:
        np.ndarray: The vector as a NumPy array [dx, dy].
    """
    x1, y1 = point1
    x2, y2 = point2
    
    dx = x2 - x1
    dy = y2 - y1

    
    vector = np.array([dx, dy])
    
    return vector


def angle_between_vectors(vector1, vector2):
    # Convert input lists to NumPy arrays for vector operations
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Calculate the dot product of the two vectors
    dot_product = np.dot(vector1, vector2)
    
    # Calculate the magnitudes (norms) of each vector
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    
    # Calculate the cosine of the angle between the vectors
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    
    # Use arccosine to calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)
    
    # Calculate the angle in degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_radians, angle_degrees


def compute_tortuosity(x_coordinates, y_coordinates):
    if len(x_coordinates) != len(y_coordinates):
        raise ValueError("Input arrays must have the same length.")

    # Calculate the path length
    path_length = 0
    for i in range(1, len(x_coordinates)):
        dx = x_coordinates[i] - x_coordinates[i-1]
        dy = y_coordinates[i] - y_coordinates[i-1]
        path_length += np.sqrt(dx**2 + dy**2)

    # Calculate the Euclidean distance
    start_point = (x_coordinates[0], y_coordinates[0])
    end_point = (x_coordinates[-1], y_coordinates[-1])
    euclidean_distance = np.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

    # Compute tortuosity as the ratio of path length to Euclidean distance
    tortuosity = path_length / euclidean_distance
    linearity = 1/tortuosity 

    return tortuosity,linearity
def df_tortuosity(df):
    for ind,row in df.iterrows():
        try:
            target =  np.nanmax(np.argwhere((-1< row['distance_from_edge']) ))
            tor,lin= compute_tortuosity(row['nose_x_cm'][:target+1],row['nose_y_cm'][:target+1])
            df.at[ind,'tortuosity']=tor
            df.at[ind,'linearity']=lin
        except ValueError:
            df.at[ind,'tortuosity']=np.nan
            df.at[ind,'linearity']=np.nan
            
def lateral_error_open_corner(df):
    """get lateral error of nose to open corner at a hold"""
    for direction, direction_frame in df.groupby(['odd']):
        for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
            for ind,row in cluster_frame.iterrows():
                if direction == 'right':
                    if cluster == 0 or cluster == 1:
                        nose_y = row['ts_nose_y_cm'].astype(float)
                        corner_y = row['gt_obstacleBR_y_cm']
                        lateral_error = []
                        for i in nose_y:
                            if i > corner_y:  
                                err = np.abs(i -corner_y) 
                                lateral_error.append(err)
                            if i < corner_y: 
                                err = np.abs(i -corner_y) *-1
                                lateral_error.append(err)
                            
                        lateral_error = np.array(lateral_error)
                        
                        df.at[ind,'lateral_error'] = lateral_error.astype(object)
                        
                    if cluster == 2 or cluster == 3:
                        nose_x = row['ts_nose_x_cm'].astype(float)
                        nose_y = row['ts_nose_y_cm'].astype(float)
                        obstalce_edge= np.mean([row['gt_obstacleTR_x_cm'],row['gt_obstacleBR_x_cm']])
                        nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
                        ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
                        distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTR_y_cm']))
                        distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBR_y_cm']))
                        lateral_error = []
                        if distance_to_top < distance_to_bottom:
                            corner_y = row['gt_obstacleTR_y_cm']
                            for i in nose_y:
                                if i > corner_y:  
                                    err = np.abs(i -corner_y) * -1 
                                    lateral_error.append(err)
                                if i < corner_y: 
                                    err = np.abs(i -corner_y)
                                    lateral_error.append(err)
                            lateral_error = np.array(lateral_error)
                            df.at[ind,'lateral_error'] = lateral_error.astype(object)
                        if distance_to_top > distance_to_bottom:
                            corner_y = row['gt_obstacleBR_y_cm']
                            for i in nose_y:
                                if i > corner_y:  
                                    err = np.abs(i -corner_y) 
                                    lateral_error.append(err)
                                if i < corner_y: 
                                    err = np.abs(i -corner_y)* -1 
                                    lateral_error.append(err)
                            lateral_error = np.array(lateral_error)
                            df.at[ind,'lateral_error'] = lateral_error.astype(object)
                            
                                
                    if cluster == 4 or cluster == 5:
                        nose_y = row['ts_nose_y_cm'].astype(float)
                        corner_y = row['gt_obstacleTR_y_cm']
                        lateral_error = []
                        for i in nose_y:
                            if i > corner_y:  
                                err = np.abs(i -corner_y) * -1 
                                lateral_error.append(err)
                            if i < corner_y: 
                                err = np.abs(i -corner_y)
                                lateral_error.append(err)
                            
                        lateral_error = np.array(lateral_error)
                        df.at[ind,'ts_lateral_error'] = lateral_error.astype(object)
                        
                         
                if direction == 'left':
                    if cluster == 0 or cluster == 1:
                        nose_y = row['ts_nose_y_cm'].astype(float)
                        corner_y = row['gt_obstacleBL_y_cm']
                        lateral_error = []
                        for i in nose_y:
                            if i > corner_y:  
                                err = np.abs(i -corner_y) 
                                lateral_error.append(err)
                            if i < corner_y: 
                                err = np.abs(i -corner_y)*-1
                                lateral_error.append(err)
                            
                        lateral_error = np.array(lateral_error)
                        df.at[ind,'lateral_error'] = lateral_error.astype(object)
                    if cluster == 2 or cluster == 3:
                        nose_x = row['ts_nose_x_cm'].astype(float)
                        nose_y = row['ts_nose_y_cm'].astype(float)
                        obstalce_edge= np.mean([row['gt_obstacleTL_x_cm'],row['gt_obstacleBL_x_cm']])
                        nose_near_edge = nose_x[np.nanargmin(np.abs(np.array(nose_x) - obstalce_edge))]
                        try:
                            ind_nose_near_edge = np.argwhere(nose_x==nose_near_edge)[0][0]
                        except:
                            print(nose_x,obstalce_edge,nose_near_edge)
                        distance_to_top = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleTL_y_cm']))
                        distance_to_bottom = np.abs(calculate_distances(nose_x[ind_nose_near_edge],nose_y[ind_nose_near_edge],obstalce_edge,row['gt_obstacleBL_y_cm']))
                        lateral_error = []
                        if distance_to_top < distance_to_bottom:
                            corner_y = row['gt_obstacleTL_y_cm']
                            for i in nose_y:
                                if i > corner_y:  
                                    err = np.abs(i -corner_y) * -1 
                                    lateral_error.append(err)
                                if i < corner_y: 
                                    err = np.abs(i -corner_y)
                                    lateral_error.append(err)
                            lateral_error = np.array(lateral_error)
                        if distance_to_top > distance_to_bottom:
                            corner_y = row['gt_obstacleBL_y_cm']
                            for i in nose_y:
                                if i > corner_y:  
                                    err = np.abs(i -corner_y) 
                                    lateral_error.append(err)
                                if i < corner_y: 
                                    err = np.abs(i -corner_y)* -1 
                                    lateral_error.append(err)
                            lateral_error = np.array(lateral_error)
                            df.at[ind,'lateral_error'] = lateral_error.astype(object)
                        
                    if cluster == 4 or cluster == 5:
                        nose_y = row['ts_nose_y_cm'].astype(float)
                        corner_y = row['gt_obstacleTL_y_cm']
                        lateral_error = []
                        for i in nose_y:
                            if i > corner_y:  
                                err = np.abs(i -corner_y) * -1 
                                lateral_error.append(err)
                            if i < corner_y: 
                                err = np.abs(i -corner_y) 
                                lateral_error.append(err)
                            
                        lateral_error = np.array(lateral_error)
                        df.at[ind,'lateral_error'] = lateral_error.astype(object)
"""utils"""

def symmetrize_y_axis(axes):
    y_max = np.abs(axes.get_ylim()).max()
    axes.set_ylim(ymin=-y_max, ymax=y_max)

def find_consective_trials(df):
    by_animal = df.groupby(['animal'])
    for animal,animal_frame in by_animal:
        by_date = animal_frame.groupby(['date'])
        for date,date_frame in by_date:
            by_cluster = date_frame.groupby(['obstacle_cluster'])
            for cluster,cluster_frame in by_cluster:
                by_direction = cluster_frame.groupby(['odd'])
                for direction, direction_frame in by_direction:
                    by_start = direction_frame.groupby(['start'])
                    for start,start_frame in by_start:
                        for num,row in enumerate(start_frame.iterrows()):
                            if num == 0:
                                df.at[row[0],'consecutive'] = int(1) 
                            else:
                                try:
                                    if (row[1]['index']) - (df.at[row[0]-2,'index' ]) == 2:
                                        df.at[row[0],'consecutive'] = int(2) 
                                    else:
                                        df.at[row[0],'consecutive'] = int(1)
                                except KeyError:
                                    df.at[row[0],'consecutive'] = int(1)

def create_df_by_type_out_ofway(df):
    '''creates df of only long aprochaes'''
    cluster_0_long = df[(df['obstacle_cluster']==0)&(df['start']=='bottom')&(df['odd']== 'right')]
    cluster_1_long = df[(df['obstacle_cluster']==1)&(df['start']=='bottom')&(df['odd']== 'left')]
    cluster_4_long = df[(df['obstacle_cluster']==4)&(df['start']=='top')&(df['odd']== 'right')]
    cluster_5_long = df[(df['obstacle_cluster']==5)&(df['start']=='top')&(df['odd']== 'left')]

    cluster_0_short = df[(df['obstacle_cluster']==0)&(df['start']=='bottom')&(df['odd']== 'left')]
    cluster_1_short = df[(df['obstacle_cluster']==1)&(df['start']=='bottom')&(df['odd']== 'right')]
    cluster_4_short = df[(df['obstacle_cluster']==4)&(df['start']=='top')&(df['odd']== 'left')]
    cluster_5_short = df[(df['obstacle_cluster']==5)&(df['start']=='top')&(df['odd']== 'right')]

    cluster_2 = df[(df['obstacle_cluster']==2)]
    cluster_3 = df[(df['obstacle_cluster']==3)]

    
    

    long_df = pd.concat([cluster_0_long,cluster_1_long,cluster_4_long,cluster_5_long])
    short_df = pd.concat([cluster_0_short,cluster_1_short,cluster_4_short,cluster_5_short])
    middle_df = pd.concat([cluster_2,cluster_3])
    return long_df,short_df,middle_df                                    
def create_df_by_type(df):
    '''creates df of only long aprochaes'''
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
def zero_out_angle(df):
    '''sets angles to target corner to zero after first time angles to corner'''
    for ind, row in df.iterrows():
        angle_array = copy.deepcopy(row['angle_to_corner'])
        ts_angle_array = copy.deepcopy(row['ts_angle_to_corner'])
        try:
            zero = angle_array[np.nanargmin(np.abs(angle_array - 0))]
            zero_ind = np.where(angle_array == zero)[0][0]
            angle_array[zero_ind:] = 0 

            ts_zero = ts_angle_array[np.nanargmin(np.abs(ts_angle_array - 0))]
            ts_zero_ind = np.where(ts_angle_array == ts_zero)[0][0]
            ts_angle_array[ts_zero_ind:] = 0 


            df.at[ind,'zero_out_angle_to_corner'] = angle_array.astype(object)
            df.at[ind,'ts_zero_out_angle_to_corner'] = ts_angle_array.astype(object)
        except:
            continue


def zero_out_angle_target_port(df):
    for ind, row in df.iterrows():
        angle_array = copy.deepcopy(row['angle_to_target_port'])
        ts_angle_array = copy.deepcopy(row['ts_angle_to_target_port'])
        try:
            zero = angle_array[np.nanargmax(np.abs(angle_array - 0))]
            zero_ind = np.where(angle_array == zero)[0][0]

            angle_array[zero_ind:] = 0 

            ts_zero = angle_array[np.nanargmax(np.abs(ts_angle_array - 0))]
            ts_zero_ind = np.where(ts_angle_array == ts_zero)[0][0]
            

            ts_angle_array[ts_zero_ind:] = 0 
            df.at[ind,'zero_out_angle_to_target_port'] = angle_array.astype(object)
            df.at[ind,'ts_zero_out_angle_to_target_port'] = ts_angle_array.astype(object)
        except:
            continue
#def calculate_speed(df):
#    for ind, row in df.iterrows():
#        if row['odd'] == 'left': 
#            nose_list = row['nose_x_cm'] 
#            odd_ind = np.argmax(nose_list>(df.leftportT_x_cm.unique()+5))
#            temp_time = np.diff(row['trial_timestamps'][odd_ind:])
#        if row['odd'] == 'right':
#            nose_list = row['nose_x_cm'] 
#            even_ind = np.argmax(nose_list<(df.rightportT_x_cm.unique()-5))
#            temp_time = np.diff(row['trial_timestamps'][even_ind:])
#            #temp_time = np.diff(row['trial_timestamps'])
#        x = np.diff(row['ts_nose_x_cm']); y = np.diff(row['ts_nose_y_cm'])
#        if len(x) == len(temp_time):
#            xspeed = list((x/temp_time)**2)
#        elif len(x) > len(temp_time):
#            xspeed = list((x[:len(temp_time)]/temp_time)**2)
#        elif len(x) < len(temp_time):
#            xspeed = list((x/temp_time[:len(x)])**2)
#        if len(y) == len(temp_time):
#            yspeed = list((y/temp_time)**2)
#        elif len(y) > len(temp_time):
#            yspeed = list((y[:len(temp_time)]/temp_time)**2)
#        elif len(y) < len(temp_time):
#            yspeed = list((y/temp_time[:len(y)])**2)
#        df.at[ind, 'speed']  = np.sqrt(np.sum([xspeed, yspeed],axis=0)).astype(object)
#        distance = np.sqrt((x.astype(float))**2) + np.sqrt((y.astype(float))**2)
#        df.at[ind, 'distance'] = distance.astype(object)
#        df.at[ind, 'total_distance'] = np.nansum(distance).astype(object)

def calculate_speed(df):
    for ind, row in df.iterrows():
        if row['odd'] == 'left': 
            nose_list = row['nose_x_cm'] 
            ts_odd_ind = np.argmax(nose_list>(df.leftportT_x_cm.unique()+5))
            ts_temp_time = np.diff(row['trial_timestamps'][ts_odd_ind:])
            temp_time = np.diff(row['trial_timestamps'])
        if row['odd'] == 'right':
            nose_list = row['nose_x_cm'] 
            ts_even_ind = np.argmax(nose_list<(df.rightportT_x_cm.unique()-5))
            ts_temp_time = np.diff(row['trial_timestamps'][ts_even_ind:])
            temp_time = np.diff(row['trial_timestamps'])
            #temp_time = np.diff(row['trial_timestamps'])
        ts_x = np.diff(row['ts_nose_x_cm']); ts_y = np.diff(row['ts_nose_y_cm'])
        x = np.diff(row['nose_x_cm']); y = np.diff(row['nose_y_cm'])
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

        df.at[ind, 'speed']  = np.sqrt(np.sum([xspeed, yspeed],axis=0)).astype(object)
        df.at[ind, 'ts_speed']  = np.sqrt(np.sum([ts_xspeed, ts_yspeed],axis=0)).astype(object)
        distance = np.sqrt((x.astype(float))**2) + np.sqrt((y.astype(float))**2)
        ts_distance = np.sqrt((ts_x.astype(float))**2) + np.sqrt((ts_y.astype(float))**2)
        df.at[ind, 'distance'] = distance.astype(object)
        df.at[ind, 'ts_distance'] = ts_distance.astype(object)
        df.at[ind, 'total_distance'] = np.nansum(distance).astype(object)
        df.at[ind, 'ts_total_distance'] = np.nansum(ts_distance).astype(object)


def plot_trials_sample(df,sample):
    """Direction by key obstalce trials"""
   
    fig = plt.figure(constrained_layout=False, figsize=(20, 10),dpi=90)
    
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
            cluster_frame = cluster_frame.sample(sample)
            right_obstacle_axis = right_obstacle_dict.get(cluster)
            left_obstalce_axis = left_obstacle_dict.get(cluster)
            plot_obstacle(cluster_frame,right_obstacle_axis,cluster)
            plot_obstacle(cluster_frame,left_obstalce_axis,cluster)
            right_obstacle_axis.set_title(str(cluster))
            left_obstalce_axis.set_title(str(cluster))
            for ind,row in cluster_frame.iterrows():
                if direction == 'right':
                    which_axis = right_obstacle_dict.get(cluster)
                    which_axis.plot(row['nose_x_cm'],row['nose_y_cm'],alpha = .5)
                if direction == 'left':
                    which_axis = left_obstacle_dict.get(cluster)
                    which_axis.plot(row['nose_x_cm'],row['nose_y_cm'],alpha = .5)


#def deveation(df):
#    for ind,row in df.iterrows():
#        obstacle_ind = int(np.argwhere(row.ts_distance_from_edge>= -2).max())
#        goal_ind = int(np.argwhere(row.ts_distance_from_edge<= -2).min())
#        df.at[ind,'obstacle_ind'] = obstacle_ind
#        df.at[ind,'goal_ind'] = goal_ind#

#        obstacle_basis_start_x, obstacle_basis_start_y = row.ts_nose_x_cm[0],row.ts_nose_y_cm[0]
#        obstacle_basis_end_x, obstacle_basis_end_y = row.ts_nose_x_cm[obstacle_ind],row.ts_nose_y_cm[obstacle_ind]
#        obstacle_basis_vector = calculate_vector_between_points((obstacle_basis_start_x,obstacle_basis_start_y),(obstacle_basis_end_x,obstacle_basis_end_y))
#        obstacle_nose_x,obstacle_nose_y = row.ts_nose_x_cm[:obstacle_ind],row.ts_nose_y_cm[:obstacle_ind]
#        df.at[ind,'obstacle_nose_x'] = obstacle_nose_x.astype(object)
#        df.at[ind,'obstacle_nose_y'] = obstacle_nose_y.astype(object)
#        df.at[ind,'obstacle_basis_start_x'] = row.ts_nose_x_cm[0]
#        df.at[ind,'obstacle_basis_start_y'] = row.ts_nose_y_cm[0]
#        df.at[ind,'obstacle_basis_end_x'] = row.ts_nose_y_cm[obstacle_ind]
#        df.at[ind,'obstacle_basis_end_y'] = row.ts_nose_y_cm[obstacle_ind]#
#

#        goal_basis_start_x, goal_basis_start_y = row.ts_nose_x_cm[goal_ind],row.ts_nose_y_cm[goal_ind]
#        goal_basis_end_x, goal_basis_end_y = row.ts_nose_x_cm[-1],row.ts_nose_y_cm[-1]
#        goal_basis_vector = calculate_vector_between_points((goal_basis_start_x,goal_basis_start_y),(goal_basis_end_x,goal_basis_end_y))
#        goal_nose_x,goal_nose_y = row.ts_nose_x_cm[goal_ind:],row.ts_nose_y_cm[goal_ind:]
#        df.at[ind,'goal_nose_x'] = goal_nose_x.astype(object)
#        df.at[ind,'goal_nose_y'] = goal_nose_y.astype(object)
#        df.at[ind,'goal_basis_start_x'] = row.ts_nose_x_cm[goal_ind]
#        df.at[ind,'goal_basis_start_y'] = row.ts_nose_y_cm[goal_ind]
#        df.at[ind,'goal_basis_end_x'] = row.ts_nose_y_cm[-1]
#        df.at[ind,'goal_basis_end_y'] = row.ts_nose_y_cm[-1]#

#        obstacle_devations = []
#        for i in list(range(len(obstacle_nose_x))):
#            obstacle_nose_vector = calculate_vector_between_points((obstacle_basis_start_x,obstacle_basis_start_y),(obstacle_nose_x[i],obstacle_nose_y[i]))
#            _,dev = angle_between_vectors(obstacle_nose_vector,obstacle_basis_vector)
#            #devations.append(dev) 
#            #dev = calculate_angle((nose_x[i],nose_y[i]),(origin_x,origin_y),(end_x,end_y))
#            obstacle_devations.append(dev)
#        df.at[ind,'obstacle_devations'] = np.array(obstacle_devations).astype(object)
#        
#        goal_devations = []
#        for i in list(range(len(goal_nose_x))):
#            goal_nose_vector = calculate_vector_between_points((goal_basis_start_x,goal_basis_start_y),(goal_nose_x[i],goal_nose_y[i]))
#            _,dev = angle_between_vectors(goal_nose_vector,goal_basis_vector)
#            #devations.append(dev) 
#            #dev = calculate_angle((nose_x[i],nose_y[i]),(origin_x,origin_y),(end_x,end_y))
#            goal_devations.append(dev)
#        df.at[ind,'goal_devations'] = np.array(goal_devations).astype(object)

def deveation(df):
    for ind,row in df.iterrows():
        try:
            obstacle_ind = int(np.argwhere(row.ts_distance_from_edge>= -2).max())
        except ValueError:
            continue
        try:
            goal_ind = int(np.argwhere(row.ts_distance_from_edge<= -2).min())
        except ValueError:
            continue
        df.at[ind,'obstacle_ind'] = obstacle_ind
        df.at[ind,'goal_ind'] = goal_ind

        obstacle_basis_start_x, obstacle_basis_start_y = row.ts_nose_x_cm[0],row.ts_nose_y_cm[0]
        obstacle_basis_end_x, obstacle_basis_end_y = row.ts_nose_x_cm[obstacle_ind],row.ts_nose_y_cm[obstacle_ind]
        obstacle_basis_vector = calculate_vector_between_points((obstacle_basis_start_x,obstacle_basis_start_y),(obstacle_basis_end_x,obstacle_basis_end_y))
        obstacle_nose_x,obstacle_nose_y = row.ts_nose_x_cm[:obstacle_ind],row.ts_nose_y_cm[:obstacle_ind]
        df.at[ind,'obstacle_nose_x'] = obstacle_nose_x.astype(object)
        df.at[ind,'obstacle_nose_y'] = obstacle_nose_y.astype(object)
        df.at[ind,'obstacle_basis_start_x'] = obstacle_basis_start_x
        df.at[ind,'obstacle_basis_start_y'] = obstacle_basis_start_y
        df.at[ind,'obstacle_basis_end_x'] = obstacle_basis_end_x
        df.at[ind,'obstacle_basis_end_y'] = obstacle_basis_end_y


        goal_basis_start_x, goal_basis_start_y = row.ts_nose_x_cm[goal_ind],row.ts_nose_y_cm[goal_ind]
        goal_basis_end_x, goal_basis_end_y = row.ts_nose_x_cm[-1],row.ts_nose_y_cm[-1]
        goal_basis_vector = calculate_vector_between_points((goal_basis_start_x,goal_basis_start_y),(goal_basis_end_x,goal_basis_end_y))
        goal_nose_x,goal_nose_y = row.ts_nose_x_cm[goal_ind:],row.ts_nose_y_cm[goal_ind:]
        df.at[ind,'goal_nose_x'] = goal_nose_x.astype(object)
        df.at[ind,'goal_nose_y'] = goal_nose_y.astype(object)
        df.at[ind,'goal_basis_start_x'] = goal_basis_start_x
        df.at[ind,'goal_basis_start_y'] = goal_basis_start_y
        df.at[ind,'goal_basis_end_x'] = goal_basis_end_x
        df.at[ind,'goal_basis_end_y'] = goal_basis_end_y

        obstacle_devations = []
        for i in list(range(len(obstacle_nose_x))):
            obstacle_nose_vector = calculate_vector_between_points((obstacle_basis_start_x,obstacle_basis_start_y),(obstacle_nose_x[i],obstacle_nose_y[i]))
            _,dev = angle_between_vectors(obstacle_nose_vector,obstacle_basis_vector)
            #devations.append(dev) 
            #dev = calculate_angle((nose_x[i],nose_y[i]),(origin_x,origin_y),(end_x,end_y))
            obstacle_devations.append(dev)
        df.at[ind,'obstacle_devations'] = np.array(obstacle_devations).astype(object)
        
        goal_devations = []
        for i in list(range(len(goal_nose_x))):
            goal_nose_vector = calculate_vector_between_points((goal_basis_start_x,goal_basis_start_y),(goal_nose_x[i],goal_nose_y[i]))
            _,dev = angle_between_vectors(goal_nose_vector,goal_basis_vector)
            #devations.append(dev) 
            #dev = calculate_angle((nose_x[i],nose_y[i]),(origin_x,origin_y),(end_x,end_y))
            goal_devations.append(dev)
        df.at[ind,'goal_devations'] = np.array(goal_devations).astype(object)

def get_mean_of_df( df,key:str, bins:int):
        fake_time = np.linspace(0,1,bins)
        mat = np.zeros([len(df), bins])
        count = 0
        for ind, row in df.iterrows():
            xT = np.linspace(0,1,len(row[key]))
            mat[count,:] = interp1d(xT, row[key], bounds_error=False)(fake_time)
            count += 1
        mean = np.nanmean(mat, axis=0)

        df = pd.Series(mean)
        df.fillna(method='bfill', axis=0, inplace=True)
        mean = df.to_numpy()

        return mean

def get_median_of_df( df,key:str, bins:int):
        fake_time = np.linspace(0,1,bins)
        mat = np.zeros([len(df), bins])
        count = 0
        for ind, row in df.iterrows():
            xT = np.linspace(0,1,len(row[key]))
            mat[count,:] = interp1d(xT, row[key], bounds_error=False)(fake_time)
            count += 1
        median = np.nanmedian(mat, axis=0)

        df = pd.Series(median)
        df.fillna(method='bfill', axis=0, inplace=True)
        median = df.to_numpy()
        
        return median
def goal_and_obstacle_tortuosity(df):
    for ind,row in df.iterrows():
        try:
            ob_tor,ob_lin= compute_tortuosity(row['obstacle_nose_x'],row['obstacle_nose_y'])
            df.at[ind,'ob_tortuosity']=ob_tor
            df.at[ind,'ob_linearity']=ob_lin

            go_tor,go_lin= compute_tortuosity(row['goal_nose_x'],row['goal_nose_y'])
            df.at[ind,'go_tortuosity']=go_tor
            df.at[ind,'go_linearity']=go_lin
        except (ValueError,TypeError) as error:
            df.at[ind,'ob_tortuosity']=np.nan
            df.at[ind,'ob_linearity']=np.nan

            df.at[ind,'go_tortuosity']=np.nan
            df.at[ind,'go_linearity']=np.nan
def head_body_angle(df):
    for ind,row in df.iterrows():
        nose_x,nose_y = row.nose_x_cm,row.nose_y_cm 
        ear_x, ear_y = np.mean([row.leftear_x_cm, row.rightear_x_cm],axis=0) ,np.mean([row.leftear_y_cm, row.rightear_y_cm],axis=0) 
        spine_x, spine_y = row.spine_x_cm,row.spine_y_cm 
        tailbase_x, tailbase_y = row.tailbase_x_cm,row.tailbase_y_cm 

        degs = []
        for i in list(range(len(nose_x))):
            head_vector = calculate_vector_between_points((ear_x[i],ear_y[i]),(nose_x[i],nose_y[i]))# vector from ear to nose
            body_vector = calculate_vector_between_points((spine_x[i],spine_y[i]),(tailbase_x[i],tailbase_y[i]))# vector from nose to open corner
            rad,deg = angle_between_vectors(head_vector.astype(float),body_vector.astype(float))
            degs.append(deg)
        degs = np.array(degs)
        df.at[ind,'head_body_angle'] = degs.astype(object)