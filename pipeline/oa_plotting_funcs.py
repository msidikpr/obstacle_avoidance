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
import seaborn as sns
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
from pipeline.helper_functions import list_columns,interpolate_array,smooth




def plot_arena(df,axis,obstacle = False,outer = False):
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
    if outer == False:
        arena_x = df[['arenaTL_x_cm',
        'arenaTR_x_cm','arenaTR_x_cm',
        'arenaTL_x_cm',
        'arenaTL_x_cm']].median().values.ravel('K')
        arena_y = df[['arenaTL_y_cm',
        'arenaTL_y_cm','arenaBL_y_cm',
        'arenaBL_y_cm',
        'arenaTL_y_cm']].median().values.ravel('K')

        axis.plot([arena_x[0],arena_x[1],arena_x[2],arena_x[3],arena_x[0]],
                              [arena_y[0],arena_y[1],arena_y[2],arena_y[3],arena_y[0]],c='k')
    if outer == True:
        


        arena_x_outer = df[['arenaTLouter_x_cm',
        'arenaTRouter_x_cm','arenaTRouter_x_cm',
        'arenaTLouter_x_cm',
        'arenaTLouter_x_cm']].median().values.ravel('K')
        arena_y_outer = df[['arenaTLouter_y_cm',
        'arenaTLouter_y_cm','arenaBRouter_y_cm',
        'arenaBRouter_y_cm',
        'arenaTLouter_y_cm']].median().values.ravel('K')

        axis.plot([arena_x_outer[0],arena_x_outer[1],arena_x_outer[2],arena_x_outer[3],arena_x_outer[0]],
                          [arena_y_outer[0],arena_y_outer[1],arena_y_outer[2],arena_y_outer[3],arena_y_outer[0]],c='k')
    

    #left_port =  pd.unique(df[['leftportT_x_cm','leftportT_y_cm']].values.ravel('K'))

    #right_port = pd.unique(df[['rightportT_x_cm','rightportT_y_cm']].values.ravel('K'))

    left_port =  df[['leftportT_x_cm','leftportT_y_cm']].median().values.ravel('K')

    right_port = df[['rightportT_x_cm','rightportT_y_cm']].median().values.ravel('K')

    axis.scatter(left_port[0],left_port[1],c='black',s=100,marker = 's')
    #axis.vlines(ymax=arena_y[0],ymin=arena_y[3],x=left_port[0],colors='k')
    axis.scatter(right_port[0],right_port[1],c='black',s=100,marker = 's')
   # axis.vlines(ymax=arena_y[1],ymin=arena_y[2],x=right_port[0],colors='k')

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



def plot_orginal_obstacle(df,axis,cluster, correct = False, corect_x= 0,corect_y = 0):
    keys = list_columns(df,['gt'])
    keys = [key for key in keys if 'cen' not in key]
    for key in keys:
        df.loc[df.obstacle_cluster ==cluster,key] = df.loc[df.obstacle_cluster ==cluster,key].median()
    obstacle_x = df[['gt_obstacleTL_x_cm',
        'gt_obstacleTR_x_cm','gt_obstacleTR_x_cm',
        'gt_obstacleTL_x_cm',
        'gt_obstacleTL_x_cm']].median().values.ravel('K')

    obstacle_y =  df[['gt_obstacleTL_y_cm',
    'gt_obstacleTL_y_cm','gt_obstacleBL_y_cm',
    'gt_obstacleBL_y_cm',
    'gt_obstacleTL_y_cm']].median().values.ravel('K')

    if correct == False:
        axis.plot([obstacle_x[0],obstacle_x[1],obstacle_x[2],obstacle_x[3],obstacle_x[0]],
                              [obstacle_y[0],obstacle_y[1],obstacle_y[2],obstacle_y[3],obstacle_y[0]],c='k')
        axis.set_ylim([51,0]); axis.set_xlim([0, 61])
    else:
        axis.plot([obstacle_x[0]+corect_x,obstacle_x[1]+corect_x,obstacle_x[2]+corect_x,obstacle_x[3]+corect_x,obstacle_x[0]+corect_x],
                              [obstacle_y[0]+corect_y,obstacle_y[1]+corect_y,obstacle_y[2]+corect_y,obstacle_y[3]+corect_y,obstacle_y[0]+corect_y],c='k')
        axis.set_ylim([51,0]); axis.set_xlim([0, 61])
    return obstacle_x,obstacle_y

def plot_obstacle_from_row(axis,row):
    obstacle_x = [row.gt_obstacleTL_x_cm,
        row.gt_obstacleTR_x_cm,row.gt_obstacleTR_x_cm,
        row.gt_obstacleTL_x_cm,
        row.gt_obstacleTL_x_cm]
    obstacle_y = [row.gt_obstacleTL_y_cm,
        row.gt_obstacleTL_y_cm,row.gt_obstacleBL_y_cm,
        row.gt_obstacleBL_y_cm,
        row.gt_obstacleTL_y_cm]
    
    axis.plot([obstacle_x[0],obstacle_x[1],obstacle_x[2],obstacle_x[3],obstacle_x[0]],
                              [obstacle_y[0],obstacle_y[1],obstacle_y[2],obstacle_y[3],obstacle_y[0]],c='k')
    axis.set_ylim([51,0]); axis.set_xlim([0, 61])

   
def correct_trace(row,ob_x,ob_y):
    input_ob_x, input_ob_y = row.gt_obstacleTL_x_cm ,row.gt_obstacleTL_y_cm
    direction = row.odd
    diff_x = input_ob_x - ob_x[0]
    diff_y = input_ob_y- ob_y[0]
    correct_y = 1*(diff_y)
    if direction == 'right':
        correct_x = -1*(diff_x)

    else:
        correct_x = diff_x
    return smooth(row.ts_nose_x_cm - diff_x) , smooth(row.ts_nose_y_cm- diff_y) 


