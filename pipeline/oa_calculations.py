"""function needed for oa calculations"""
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import itertools 
from sklearn.cluster import KMeans
from scipy.ndimage import gaussian_filter
from pipeline.helper_functions import list_columns,interpolate_array,split_range_into_parts,largest_sequentially_increasing_by_one_subarray
import copy
from scipy.signal import find_peaks

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

def calculate_relative_distance_goal_ts(df):
    '''calculates change in distance from nose to goal port using trial start trace '''
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

def distance_calcs(df):
    calculate_relative_distance(df)
    calculate_relative_distance_goal(df)
    calculate_relative_distance_goal_ts(df)
    ts_calculate_relative_distance(df)


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
                            corner_y = row['gt_obstacleBR_y_cm']+2
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
                                    corner_y = row['gt_obstacleTR_y_cm']-2
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
                                    corner_y = row['gt_obstacleBR_y_cm']+2
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
                            corner_y = row['gt_obstacleTR_y_cm']-2
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
                            corner_y = row['gt_obstacleBL_y_cm']+2
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
                                    corner_y = row['gt_obstacleTL_y_cm']-2
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
                                    corner_y = row['gt_obstacleBL_y_cm']+2
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
                            corner_y = row['gt_obstacleTL_y_cm']-2
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


def zero_out_angle(df):
    '''sets angles to target corner to zero after first time angles to corner'''
    for ind, row in df.iterrows():
        #angle_array = copy.deepcopy(row['angle_to_corner'])
        ts_angle_array = copy.deepcopy(row['ts_angle_to_corner'])
        try:
            #zero = angle_array[np.nanargmin(np.abs(angle_array - 0))]
            #zero_ind = np.where(angle_array == zero)[0][0]
            #angle_array[zero_ind:] = 0 

            ts_zero = ts_angle_array[np.nanargmin(np.abs(ts_angle_array - 0))]
            ts_zero_ind = np.where(ts_angle_array == ts_zero)[0][0]
            ts_angle_array[ts_zero_ind:] = ts_zero 


            #df.at[ind,'zero_out_angle_to_corner'] = angle_array.astype(object)
            df.at[ind,'ts_zero_out_angle_to_corner'] = ts_angle_array.astype(object)
        except:
            continue


def zero_out_angle_target_port(df):
    for ind, row in df.iterrows():
        #angle_array = copy.deepcopy(row['angle_to_corner'])
        ts_angle_array = copy.deepcopy(row['ts_angle_to_corner'])
        try:
            #zero = angle_array[np.nanargmin(np.abs(angle_array - 0))]
            #zero_ind = np.where(angle_array == zero)[0][0]
            #angle_array[zero_ind:] = 0 

            ts_zero = ts_angle_array[np.nanargmin(np.abs(ts_angle_array - 0))]
            ts_zero_ind = np.where(ts_angle_array == ts_zero)[0][0]
            ts_angle_array[ts_zero_ind:] = 0 


            #df.at[ind,'zero_out_angle_to_corner'] = angle_array.astype(object)
            df.at[ind,'ts_zero_out_angle_to_corner'] = ts_angle_array.astype(object)
        except:
            continue



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

def heading_calcs(df):
    ts_angle_to_open_corner(df)
    print('corner')
    ts_angle_to_target_port(df)
    print('port')
    zero_out_angle(df)
    


def cluster_obstacle(df,numcluster):

    """cluster obstacle position"""
    df = df[df['gt_obstacle_cen_x_cm'].notna()]


    kmeans_input = np.vstack([df['gt_obstacle_cen_x_cm'].values, df['gt_obstacle_cen_y_cm'].values])

    kmeans_input = np.transpose(kmeans_input)

    labels = KMeans(n_clusters=numcluster).fit(kmeans_input).labels_
    df['obstacle_cluster'] = labels

    #get mean of obstacle center
    for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
        x=df.loc[df['obstacle_cluster']==cluster_name]
        mean_cenx = np.nanmean(x['gt_obstacle_cen_x_cm'])
        mean_ceny = np.nanmean(x['gt_obstacle_cen_y_cm'])
  
        for ind,row in df.iterrows(): 
            if row['obstacle_cluster'] == cluster_name:
                df.at[ind,'mean_gt_obstacle_cen_x_cm'] = mean_cenx
                df.at[ind,'mean_gt_obstacle_cen_y_cm'] = mean_ceny
    #label cluster by position 
    if numcluster == 9:
        print(numcluster)
        df['cluster_label'] = np.nan
        x_pos,y_pos  = np.sort(df['mean_gt_obstacle_cen_x_cm'].unique()),np.sort(df['mean_gt_obstacle_cen_y_cm'].unique())
        col_1, col_2, col_3 = x_pos[0:3],x_pos[3:6],x_pos[6:9]
        row_1, row_2, row_3 = y_pos[0:3],y_pos[3:6],y_pos[6:9]
        for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
            #label cluster by obstacle post
            x=df.loc[df['obstacle_cluster']==cluster_name]
            for ind,row in x.iterrows():
                if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_1: 
                     df.at[ind,'cluster_label'] = 0

                if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                    df.at[ind,'cluster_label'] = 1

                if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                    df.at[ind,'cluster_label'] = 2

                if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                    df.at[ind,'cluster_label'] = 3

                if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                    df.at[ind,'cluster_label'] = 4

                if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                    df.at[ind,'cluster_label'] = 5

                if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                    df.at[ind,'cluster_label'] = 6

                if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                    df.at[ind,'cluster_label'] = 7

                if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                    df.at[ind,'cluster_label'] = 8
    elif numcluster == 6:
        print(numcluster)
        df['cluster_label'] = np.nan
        x_pos,y_pos  = np.sort(df['mean_gt_obstacle_cen_x_cm'].unique()),np.sort(df['mean_gt_obstacle_cen_y_cm'].unique())
        col_1, col_2 = x_pos[0:3],x_pos[3:6]
        row_1, row_2, row_3 = y_pos[0:2],y_pos[2:4],y_pos[4:6]
        for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
            #label cluster by obstacle post
            x=df.loc[df['obstacle_cluster']==cluster_name]
            for ind,row in x.iterrows():
                if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_1: 
                     df.at[ind,'cluster_label'] = 0

                if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                    df.at[ind,'cluster_label'] = 1

                #if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                 #   df.at[ind,'cluster_label'] = 2

                if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                    df.at[ind,'cluster_label'] = 2

                if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                    df.at[ind,'cluster_label'] = 3

                #if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                    #df.at[ind,'cluster_label'] = 5

                if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                    df.at[ind,'cluster_label'] = 4

                if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                    df.at[ind,'cluster_label'] = 5

            
           #    df.at[ind,'cluster_label'] = 8
    df['obstacle_cluster'] = df['cluster_label'].astype(int)
    return df



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



def lateral_error_open_corner(df):
    """get lateral error of nose to open corner"""
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
                        df.at[ind,'lateral_error'] = lateral_error.astype(object)
                        
                         
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

def ts_lateral_error_to_target_port(df):
    """lateral error to target port"""
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

def lateral_error(df):
    lateral_error_open_corner(df)
    ts_lateral_error_to_target_port(df)


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
            #target =  np.nanmax(np.argwhere((-1< row['ts_distance_from_edge']) ))
            tor,lin= compute_tortuosity(row['ts_nose_x_cm'],row['ts_nose_y_cm'])
            df.at[ind,'tortuosity']=tor
            df.at[ind,'linearity']=lin
        except ValueError:
            df.at[ind,'tortuosity']=np.nan
            df.at[ind,'linearity']=np.nan


def get_head_angle(df):
    """ ego centric head angle using trial"""
    for ind, row in df.iterrows():
        leftear_x = row['ts_leftear_x_cm']
        leftear_y = row['ts_leftear_y_cm']
        rightear_x = row['ts_rightear_x_cm']
        rightear_y = row['ts_rightear_y_cm']
        nose_x = row['ts_nose_x_cm']
        nose_y = row['ts_nose_y_cm']
        angs = []
        if row.odd == 'left':
            for step in range(len(leftear_x)):
                ang = np.arctan2(np.mean([leftear_y[step],rightear_y[step]])-nose_y[step],(np.mean([leftear_x[step],rightear_x[step]])-nose_x[step])*-1)
                angs.append(ang)
        else:
            for step in range(len(leftear_x)):
                ang = np.arctan2(np.mean([leftear_y[step],rightear_y[step]])-nose_y[step],(np.mean([leftear_x[step],rightear_x[step]])-nose_x[step]))
                angs.append(ang)
        df.at[ind, 'head_angle'] = gaussian_filter(np.array(np.degrees(angs)),2,mode = 'reflect').astype(object)

def angular_velocity_head(df):
    for ind,row in df.iterrows():
        #filtered_head_angle = gaussian_filter(row.head_angle.astype(float),2)
        head_angle_velocity = calculate_angular_velocity(row.head_angle.astype(float),60).astype(object)
        df.at[ind,'head_angle_velocity'] = head_angle_velocity
        df.at[ind,'intial_head_angle_velocity'] = head_angle_velocity.astype(float)[np.isfinite(head_angle_velocity.astype(float))][0]

def calculate_angular_velocity(angles, frame_rate):
    # Convert angles to radians if they are given in degrees
    angles = np.radians(angles)

    # Calculate angular velocity using NumPy's array operations
    angular_displacements = np.diff(angles)
    angular_velocities = angular_displacements / (1 / frame_rate)
    np.angle(angular_velocities.astype(float))
    return angular_velocities


def head_angle_velocity(df):
    """calculate the egocentric head angle and velocity"""
    get_head_angle(df)
    angular_velocity_head(df)


def start(df):
    labels = ['top','bottom']
    top_bottom = split_range_into_parts(np.nanmedian(pd.unique(df.arenaTL_y_cm)),np.nanmedian(pd.unique(df.arenaBL_y_cm)),2)
    top_bottom_dict = dict(zip(labels,top_bottom))
    for ind, row in df.iterrows():
        if top_bottom_dict.get('top')[0]<= np.nanmean(row['ts_nose_y_cm'][:20]) <= top_bottom_dict.get('top')[1]:
            df.at[ind,'start'] = 'top'
        if top_bottom_dict.get('bottom')[0]<= np.nanmean(row['ts_nose_y_cm'][:20]) <= top_bottom_dict.get('bottom')[1]:
            df.at[ind,'start'] = 'bottom'
        if np.nanmean(row['ts_nose_y_cm'][:5]) == np.nan:
            df.drop(df.iloc[ind])
    df = df[df['start'].notna()]

def calculate_speed(df): 
    for ind, row in df.iterrows():
        if row['odd'] == 'left': 
            nose_list = row['nose_x_cm'] 
            ts_odd_ind = np.argmax(nose_list>(row.leftportT_x_cm+5))
            ts_temp_time = np.diff(row['trial_timestamps'][ts_odd_ind:])
            temp_time = np.diff(row['trial_timestamps'])
        if row['odd'] == 'right':
            nose_list = row['nose_x_cm'] 
            ts_even_ind = np.argmax(nose_list<(row.rightportT_x_cm-5))
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


def angular_velocity_head(df):
    for ind,row in df.iterrows():
        #filtered_head_angle = gaussian_filter(row.head_angle.astype(float),2)
        head_angle_velocity = calculate_angular_velocity(row.head_angle.astype(float),60).astype(object)
        df.at[ind,'head_angle_velocity'] = head_angle_velocity
        df.at[ind,'intial_head_angle_velocity'] = head_angle_velocity.astype(float)[np.isfinite(head_angle_velocity.astype(float))][0]

def calculate_angular_velocity(angles, frame_rate):
    # Convert angles to radians if they are given in degrees
    angles = np.radians(angles)

    # Calculate angular velocity using NumPy's array operations
    angular_displacements = np.diff(angles)
    angular_velocities = angular_displacements / (1 / frame_rate)
    np.angle(angular_velocities.astype(float))
    return angular_velocities


#def turn_direction(df):
#    for ind,row in df.iterrows():
#        if np.nanmean(row.head_angle_velocity[:5]) < 0:
#            df.at[ind,'turn_direction'] = 'up'
#        elif np.nanmean(row.head_angle_velocity[:5])> 0:
#            df.at[ind,'turn_direction'] = 'down'

def turn_direction(df):
    for ind,row in df.iterrows():
        peaks, _ = find_peaks(np.abs(row.head_angle_velocity), width=2,height=(1,20),distance=10)
        head_direction = row.head_angle_velocity[peaks[0]]
        head_direction_end = row.head_angle_velocity[peaks[-1]]
        
        if head_direction < 0:
            df.at[ind,'turn_direction'] = 'up'
        elif head_direction> 0:
            df.at[ind,'turn_direction'] = 'down'

        if head_direction_end < 0:
            df.at[ind,'turn_direction_end'] = 'up'
        elif head_direction_end> 0:
            df.at[ind,'turn_direction_end'] = 'down'

def turn_direction_left_right(df):
    for ind,row in df.iterrows():

        peaks, _ = find_peaks(np.abs(row.head_angle_velocity), width=2,height=(1,20),distance=10)
        head_direction = row.head_angle_velocity[peaks[0]]
        head_direction_end = row.head_angle_velocity[peaks[-1]]

        if row.odd == 'left':
        
            if head_direction < 0:
                df.at[ind,'turn_direction_left_right'] = 'right'
            elif head_direction> 0:
                df.at[ind,'turn_direction_left_right'] = 'left'

        elif row.odd == 'right':
        
            if head_direction < 0:
                df.at[ind,'turn_direction_left_right'] = 'left'
            elif head_direction> 0:
                df.at[ind,'turn_direction_left_right'] = 'right'

           

def turn_to_obstacle(df):
    for ind,row in df.iterrows():
        if (row.obstacle_cluster ==  2) or  (row.obstacle_cluster ==  3):
            if (row.start=='top') & (row.turn_direction=='up'):
                df.at[ind,'turn_to_obstacle'] = 'away'
            if (row.start=='top') & (row.turn_direction=='down'):
                df.at[ind,'turn_to_obstacle'] = 'towards'
            if (row.start=='bottom') & (row.turn_direction=='down'):
                df.at[ind,'turn_to_obstacle'] = 'away'
            if (row.start=='bottom') & (row.turn_direction=='up'):
                df.at[ind,'turn_to_obstacle'] = 'towards'
        elif (row.obstacle_cluster ==  0) or  (row.obstacle_cluster ==  1):
            if row.turn_direction == 'up':
                df.at[ind,'turn_to_obstacle'] = 'towards'
            else:
                df.at[ind,'turn_to_obstacle'] = 'away'
        elif (row.obstacle_cluster ==  4) or  (row.obstacle_cluster ==  5):
            if row.turn_direction == 'up':
                df.at[ind,'turn_to_obstacle'] = 'away'
            else:
                df.at[ind,'turn_to_obstacle'] = 'towards'



def redo_ts_trace(df,thresh = 5):
    "correct for tracking jitter"
    
    #points = ['ts_nose_x','ts_nose_y','ts_nose_x_cm','ts_nose_y_cm','ts_leftear_x','ts_leftear_y','ts_leftear_x_cm','ts_leftear_y_cm','ts_rightear_x',
    #              'ts_rightear_y','ts_rightear_x_cm','ts_rightear_y_cm','ts_spine_x','ts_spine_y','ts_midspine_x','ts_midspine_y','ts_spine_x_cm','ts_spine_y_cm','ts_midspine_x_cm',
    #             'ts_midspine_y_cm','ts_midspine_x','ts_midspine_y','ts_midspine_x_cm','ts_midspine_y_cm','ts_tailbase_x','ts_tailbase_y','ts_tailbase_x_cm','ts_tailbase_y_cm',]
    #for ind, row in df.iterrows():
    #    diff_array = np.diff(np.round(np.diff(row.ts_nose_y_cm).astype(float),5))
    #    zero_inds = np.argwhere(diff_array == 0).flatten()
    #    drop_inds = largest_sequentially_increasing_by_one_subarray(zero_inds)
#
    # 
#
    #    if len(drop_inds) <=thresh:
    #        continue
    #    else:
    #        if row.odd == 'left':
    #            nose =  row['ts_nose_x_cm'][drop_inds[-1]:]
    #            odd_ind = np.argmax(nose>(row.leftportT_x_cm+5))
    #            for point in points:
    #                df.at[ind,point] = row[point][drop_inds[-1]:][odd_ind:].astype(object)
    #        elif row.odd == 'right':
    #            nose =  row['ts_nose_x_cm'][drop_inds[-1]:]
    #            even_ind = np.argmax(nose<(row.rightportT_x_cm-5))
    #            for point in points:
#
    #                df.at[ind,point] = row[point][drop_inds[-1]:][even_ind:].astype(object)
    #heading_calcs(df)
    #distance_calcs(df)
    #deveation(df)
    #lateral_error(df)
    #df_tortuosity(df)
    #head_angle_velocity(df)
    #turn_direction(df)
    #turn_to_obstacle(df)

    thresh = 1 
    points = ['ts_nose_x','ts_nose_y','ts_nose_x_cm','ts_nose_y_cm','ts_leftear_x','ts_leftear_y','ts_leftear_x_cm','ts_leftear_y_cm','ts_rightear_x',
                      'ts_rightear_y','ts_rightear_x_cm','ts_rightear_y_cm','ts_spine_x','ts_spine_y','ts_midspine_x','ts_midspine_y','ts_spine_x_cm','ts_spine_y_cm','ts_midspine_x_cm',
                     'ts_midspine_y_cm','ts_midspine_x','ts_midspine_y','ts_midspine_x_cm','ts_midspine_y_cm','ts_tailbase_x','ts_tailbase_y','ts_tailbase_x_cm','ts_tailbase_y_cm',]

    for ind, row in df.iterrows():

        drop_inds = np.argwhere(np.round(np.abs(np.diff(np.diff(row.ts_nose_y_cm).astype(float)))) > 1).flatten()

    
        if len(drop_inds) <=thresh:
            continue
        else:
            if row.odd == 'left':
                nose =  row['ts_nose_x_cm'][drop_inds[-1]+1:]
                odd_ind = np.argmax(nose>(row.leftportT_x_cm+5))
                for point in points:
                    df.at[ind,point] = row[point][drop_inds[-1]+1:][odd_ind:].astype(object)
            elif row.odd == 'right':
                nose =  row['ts_nose_x_cm'][drop_inds[-1]+1:]
                even_ind = np.argmax(nose<(row.rightportT_x_cm-5))
                for point in points:
                    df.at[ind,point] = row[point][drop_inds[-1]+1:][even_ind:].astype(object)
    heading_calcs(df)
    distance_calcs(df)
    deveation(df)
    lateral_error(df)
    df_tortuosity(df)
    head_angle_velocity(df)
    turn_direction(df)


def intial_lateral_error(df):
    for ind,row in df.iterrows():
        if type(row.lateral_error)==float:
            df.at[ind,'intial_lateral_error'] = np.nan
        else:
            df.at[ind,'intial_lateral_error'] = row.lateral_error[0]
         
def avg_lateral_error(df):
    for ind,row in df.iterrows():
        if type(row.lateral_error)==float:
            df.at[ind,'avg_lateral_error'] = np.nan
        else:
            df.at[ind,'avg_lateral_error'] = np.nanmean(row.lateral_error[:int(row.obstacle_ind)])
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
def check_trial_for_obstalce_cross_df(df):
    for ind,row in df.iterrows():
        df.at[ind,'obstacle_cross'] = check_trial_for_obstalce_cross(row)


def angular_velocity_head_corner(df):
    for ind,row in df.iterrows():
        #filtered_head_angle = gaussian_filter(row.head_angle.astype(float),2)
        trace = gaussian_filter(np.array((row.ts_angle_to_corner.astype(float))),3,mode = 'reflect').astype(object)
        
        df.at[ind,'head_corner_angle_velocity'] = calculate_angular_velocity(trace.astype(float),60).astype(object)


def avg_lateral_error(df):
    for ind,row in df.iterrows():
        if type(row.lateral_error)==float:
            df.at[ind,'avg_lateral_error'] = np.nan
        else:
            df.at[ind,'avg_lateral_error'] = np.nanmean(row.lateral_error[:int(row.obstacle_ind)])
def avg_lateral_error_thresh(df,thresh = 5):
    for ind,row in df.iterrows():
        if type(row.lateral_error)==float:
            df.at[ind,'avg_lateral_error_thresh'+ '_' + str(thresh)] = np.nan
        else:
            first_dist_start,first_dist_end = np.where((row.ts_distance_from_edge>=0)&(row.ts_distance_from_edge<=thresh))[0][0],np.where((row.ts_distance_from_edge>=0)&(row.ts_distance_from_edge<=thresh))[0][-1]
            df.at[ind,'avg_lateral_error_thresh'+ '_' + str(thresh)] = np.nanmean(row.lateral_error[int(first_dist_start):int(row.first_dist_end)])

def distance_at_head_turn(df):
    for ind,row in df.iterrows():
        peaks, prop = find_peaks(np.abs(row.head_angle_velocity[:int(row.obstacle_ind)]), width=2,height=(3,20),distance=10)
        if len(peaks) == 0:
            df.at[ind,'distance_at_head_turn'] = np.nan
            df.at[ind,'distance_at_head_turn_index'] = np.nan
            df.at[ind,'distance_at_head_turn_indexs'] = np.nan
            df.at[ind,'angle_at_head_turn'] = np.nan
            df.at[ind,'num_turn'] = 0
        else:
            max_peak = peaks[prop['peak_heights'].argmax()]
            max_vel = prop['peak_heights'].max()
            end_movement = int(np.round(prop['right_ips'][prop['peak_heights'].argmax()]))
            start_movement = int(np.round(prop['left_ips'][prop['peak_heights'].argmax()]))
  
            df.at[ind,'distance_at_head_turn'] = row.ts_distance_from_edge[:int(row.obstacle_ind)][max_peak]
            df.at[ind,'distance_after_head_turn'] = row.ts_distance_from_edge[:int(row.obstacle_ind)][end_movement]
            df.at[ind,'distance_before_head_turn'] = row.ts_distance_from_edge[:int(row.obstacle_ind)][start_movement]
            df.at[ind,'indexs_of_head_turn'] = np.asarray([start_movement,max_peak,end_movement]).astype(object)
            df.at[ind,'head_turn_indexs'] = np.array(peaks).astype(object)
            df.at[ind,'before_head_turn_indexs'] = np.round(prop['left_ips']).astype(int).astype(object)
            df.at[ind,'after_head_turn_indexs'] = np.round(prop['right_ips']).astype(int).astype(object)
            df.at[ind,'angle_at_head_turn'] = row.ts_angle_to_corner[:int(row.obstacle_ind)][max_peak]
            df.at[ind,'angle_after_head_turn'] = row.ts_angle_to_corner[:int(row.obstacle_ind)][end_movement]
            df.at[ind,'angle_before_head_turn'] = row.ts_angle_to_corner[:int(row.obstacle_ind)][start_movement]
            df.at[ind,'num_turn'] = len(peaks)
            df.at[ind,'head_turn_velocity'] = max_vel

def intial_distance_to_obstacle(df):
    for ind,row in df.iterrows():
        df.at[ind,'intial_distance'] = np.round(row.ts_distance_from_edge[0])



def last_occurrence_indices(arr):
    # Reverse the array to find the last occurrence by position
    _, inverse_indices = np.unique(arr[::-1], return_inverse=True)
    
    # Calculate last occurrence indices
    last_indices = len(arr) - 1 - np.unique(inverse_indices, return_index=True)[1]
    
    # Sort to preserve order of appearance in the original array
    return np.sort(last_indices)

# Example usage
arr = np.array([4, 2, 3, 2, 4, 5, 3, 6])
result = last_occurrence_indices(arr)

def first_occurrence_indices(arr):
    unique_values, first_indices = np.unique(arr, return_index=True)
    sorted_indices = np.sort(first_indices)
    return sorted_indices




