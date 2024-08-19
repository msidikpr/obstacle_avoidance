

import numpy as np
from pathlib import Path
import pandas as pd 
from plots.plots2 import get_mean_of_df
from utils.base_functions import *
from scipy.interpolate import interp1d
from itertools import combinations


def generate_combinations(input_list):
    """
    Generate all combinations of strings from the input list.

    Parameters:
    input_list (list): The list of strings.

    Returns:
    list of lists: A list where each element is a list containing one or more strings.
    """
    result = []
    for r in range(1, len(input_list) + 1):  # Generate combinations of all lengths from 1 to len(input_list)
        result.extend(combinations(input_list, r))
    return [list(comb) for comb in result]  # Convert tuples to lists



def resample(array,bins):
    """interpolate array across fake time bins"""
    fake_time = np.linspace(0,1,bins)
    xT = np.linspace(0,1,len(array))
    resampled_array = interp1d(xT, array, bounds_error=False)(fake_time)
    resampled_array = pd.Series(resampled_array)
    resampled_array.fillna(method='ffill', axis=0, inplace=True)
    resampled_array.fillna(method='bfill', axis=0, inplace=True)


    return resampled_array.to_numpy()

def create_obstacle_vector(row,key):
    obstacle_vector = row[key][:int(row.obstacle_ind)]
    return obstacle_vector

def create_obstacle_vector_resampled(row,key,bins):
    obstacle_vector = row[key][:int(row.obstacle_ind)]
    return resample(obstacle_vector,bins)


def create_goal_vector_resampled(row,key,bins):
    obstacle_vector = row[key][int(row.obstacle_ind):]
    return resample(obstacle_vector,bins)

def create_feature_vector(row,feature_list):
    feature_array_list = []
    for i in feature_list:
        feature = row[i]
        feature_array_list.append(feature)
    return flatten_list_of_arrays(feature_array_list)

def create_feature_matrix(df,feature_list,chunking =False,chunk = None):
    feature_mat = []
    if chunking ==False:
        for ind, row in df.iterrows():
             feature_vector = create_feature_vector(row,feature_list)
             feature_mat.append(feature_vector)
        return np.vstack(feature_mat)
    else:
        for ind, row in df.iterrows():
             feature_vector = create_feature_vector_chunk(row,feature_list,chunk)
             feature_mat.append(feature_vector)
        return np.vstack(feature_mat)


        
def create_feature_vector_chunk(row,feature_list,chunk):
    feature_array_list = []
    for i in feature_list:
        feature = row[i][chunk[0]:chunk[1]]
        feature_array_list.append(feature)
    return flatten_list_of_arrays(feature_array_list)