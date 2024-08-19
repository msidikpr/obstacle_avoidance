"""code for clustering traces from obstacle location"""
import sys
import numpy as np
from pathlib import Path
import pandas as pd 
from dipy.segment.clustering import QuickBundles
sys.path.append(r'C:\Users\nlab\Documents\GitHub\obstacle_avoidance')
from plots.plots2 import get_mean_of_df
from utils.base_functions import *
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.segment.metric import *

def pair_elements(array1, array2):
    """
    Create an array of arrays where each array is a pair of the nth element from each input array.

    Parameters:
    array1 (numpy.ndarray): First input array.
    array2 (numpy.ndarray): Second input array.

    Returns:
    numpy.ndarray: Array of arrays where each array is a pair of the nth element from each input array.
    """
    # Check if the input arrays have the same length
    if len(array1) != len(array2):
        raise ValueError("Input arrays must have the same length")

    # Create an empty list to store pairs
    pairs = []

    # Iterate through the arrays and pair their elements
    for i in range(len(array1)):
        pairs.append(np.array([array1[i], array2[i]]))

    # Convert the list of pairs into a numpy array
    return np.array(pairs)


def create_streamlines(df,key_x,key_y):
    """input df of a single cluster"""
    _,matx = get_mean_of_df(df,key_x,50,matx=True)
    _,maty = get_mean_of_df(df,key_y,50,matx=True)

    cluster_array = []

    for i in list(range(len(matx))):
        trial_array = pair_elements(matx[i],maty[i])
        cluster_array.append(trial_array)
    streamlines = cluster_array

    return streamlines,matx,maty

def cluster_streamlines(streamlines, num_cluster = 4,threshold = 2):
    feature = ResampleFeature(nb_points=50)
    metric = AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(max_nb_clusters = num_cluster ,threshold=threshold,metric =metric )
    clusters = qb.cluster(streamlines)
    if len(clusters.get_small_clusters(5)) >=1:
        for i in list(range(len(clusters.get_small_clusters(5)))):
            clusters.remove_cluster(clusters.get_small_clusters(5)[0])
    else:
        None
    #clusters = sorted(clusters, key=len, reverse=True)

    return clusters
def create_cluster_distances(streamlines,clusters):
    cluster_distances = []
    for i in list(range(len(clusters))):
        streamline_distances = mean_euclidean_distance([streamlines[x] for x in clusters[i].indices], clusters.centroids[i])
        interpolate_array(streamline_distances)
        cluster_distances.append(streamline_distances)
    return cluster_distances


def create_cluster_dict(df, Date=False, Skill=False):
    if Date == True:
        var = 'date'
    else:
        var = 'skill'
    streamlines,matx,maty = create_streamlines(df)
    clusters = cluster_streamlines(streamlines)
    cluster_distances = create_cluster_distances(streamlines,clusters)
    
    cluster_dict = dict(type = df[var].unique(),obstacle_cluster= df['obstacle_cluster'].unique() ,cluster_object = clusters,obstalce_nose_x = matx, obstacle_nose_y = maty,cluster_distances = cluster_distances, dataframe = df)
    return cluster_dict



