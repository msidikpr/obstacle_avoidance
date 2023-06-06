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
import itertools 
from scipy.interpolate import interp1d
from scipy import signal
from scipy import stats
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import os, fnmatch
from random import sample


import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')

import warnings
warnings.filterwarnings('ignore')

from utils.base_functions import *
from src.utils.auxiliary import flatten_series
from src.utils.path import find
from src.base import BaseInput





## take raw_df from multiple session and plt  
class plot_oa(BaseInput):

    def __init__(self,metadata_path,df):
        try:
            with open(metadata_path) as f:
                self.metadata = json.load(f)
                self.path = self.metadata['path']
                self.dates_list = [i for i in list(self.metadata.keys()) if i != 'path' ]
        except FileNotFoundError:
            pass

        self.df = df 
        
    ## append df's together
    def gather_session_df(self,tasktype,numcluster):
        # list data path files
        data_path = Path(self.path).expanduser()
        # find date
        hf_list = []
        df =pd.DataFrame()
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in use_animals:
                for task in os.listdir(data_path / date / ani):
                    h5_paths=[str(i) for i in list((data_path / date / ani/ task).rglob('*.h5'))]
                    #print(h5_paths)
                    if tasktype == 'non_obstalce':
                        raw_h5 = [i for i in h5_paths if 'non' in i]
                        hf_list.append(raw_h5)
                    if tasktype == 'obstacle':
                        raw_h5 = [i for i in h5_paths if 'processed_' in i]
                        hf_list.append(raw_h5)
            #hf_list.append(raw_h5)
        hf_list = list(itertools.chain(*hf_list))
        for h5 in hf_list:
            data = pd.read_hdf(h5)
            df=df.append(data,ignore_index=False)
        self.df=df
        self.df['orginal_index'] = self.df.index
        self.df =self.df.reset_index()

        """get average areana and port postition """
        keys = list_columns(self.df,['arena','port'])
        keys = [i for i in keys if 'cm' in i]
        keys = [i for i in keys if 'portB' not in i]
        keys
        for key in keys:
            for ind,row in self.df.iterrows():
                    self.df.at[ind,key] = np.mean(row[key])
        for key in keys:
            self.df[key] = self.df[key].mean()

        """calculate the nose x interp
        interp nose x is interpolation across the same time basis of 50 bins"""
        fake_time = np.linspace(0,1,50)
        for ind, row in self.df.iterrows():
            xT = np.linspace(0,1,len(row['ts_nose_x_cm']))
            yT = np.linspace(0,1,len(row['ts_nose_y_cm']))
            intx = interp1d(xT, row['ts_nose_x_cm'], bounds_error=False)(fake_time).astype(object)
            inty = interp1d(yT, row['ts_nose_y_cm'], bounds_error=False)(fake_time).astype(object)
            
            self.df.at[ind,'time_interp_ts_nose_x_cm'] = intx.astype(object)
            self.df.at[ind,'time_interp_ts_nose_y_cm'] = inty.astype(object)
        #print(len(self.df))

        """labe top or bottom start"""
        labels = ['top','bottom']
        top_bottom = split_range_into_parts(pd.unique(self.df.arenaTL_y_cm).item(),pd.unique(self.df.arenaBL_y_cm).item(),2)
        top_bottom_dict = dict(zip(labels,top_bottom))
        for ind, row in self.df.iterrows():
            if top_bottom_dict.get('top')[0]<= np.nanmean(row['time_interp_ts_nose_y_cm'][:5]) <= top_bottom_dict.get('top')[1]:
                self.df.at[ind,'start'] = 'top'
            if top_bottom_dict.get('bottom')[0]<= np.nanmean(row['time_interp_ts_nose_y_cm'][:5]) <= top_bottom_dict.get('bottom')[1]:
                self.df.at[ind,'start'] = 'bottom'
            if np.nanmean(row['time_interp_ts_nose_y_cm'][:5]) == np.nan:
                self.df.drop(df.iloc[ind])
        self.df = self.df[self.df['start'].notna()]

        if tasktype == 'obstacle': 
            self.cluster(numcluster)

            """get average obstacle postition"""
            keys = list_columns(self.df,['gt'])
            keys = [key for key in keys if 'cen' not in key]

            for key in keys:
                self.df['mean_'+key] = np.nan
            for cluster,cluster_frame in self.df.groupby('obstacle_cluster'):
                for key in keys:
                    mean_obstacle = cluster_frame[key].mean()

                    self.df.loc[self.df['obstacle_cluster'] ==cluster,['mean_'+key]] = mean_obstacle

                    #self.df.loc[self.df.obstacle_cluster == cluster,'mean_'+key] = cluster_frame[key].mean()






            """Calculate interp y which is interpolation of nose_x vs nose_y then ploted against the mean_interp_noseX """
            for ind,row in self.df.iterrows():

                interp = interp1d(row['ts_nose_x_cm'].astype(float), row['ts_nose_y_cm'].astype(float) ,bounds_error=False, fill_value=np.nan)
                x_basis = np.linspace(10,50,50)
                interp_y = interp(x_basis.astype(float))
                interp_y = interpolate_array(interp_y)
                self.df.at[ind,'interp_ts_nose_y_cm'] = interp_y.astype(object)






            for direction, direction_frame in self.df.groupby(['odd']):
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

                        x = self.df.loc[(self.df['obstacle_cluster'] ==cluster) & (self.df['start']==start)&(self.df['odd'] ==direction)]
                        for ind,row in x.iterrows():
                            self.df.at[ind,'mean_interp_ts_nose_y_cm']= mean_trace.astype(object)

                            self.df.at[ind,'median_interp_ts_nose_y_cm']= median_trace.astype(object)

                            self.df.at[ind,'std_interp_ts_nose_y_cm']= std_trace.astype(object)

                            self.df.at[ind,'mad_interp_ts_nose_y_cm']= mad_trace.astype(object)



            get_mean_median_by_variable(self.df,'animal')
            get_mean_median_by_variable(self.df,'date')
        
        
        
        
            
        
        
                
    
        
    ##cluster obstacle positions
    def cluster(self,numcluster):
        self.df = self.df[self.df['gt_obstacle_cen_x_cm'].notna()]


        kmeans_input = np.vstack([self.df['gt_obstacle_cen_x_cm'].values, self.df['gt_obstacle_cen_y_cm'].values])

        kmeans_input = np.transpose(kmeans_input)

        labels = KMeans(n_clusters=numcluster).fit(kmeans_input).labels_
        self.df['obstacle_cluster'] = labels

        #get mean of obstacle center
        for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
            x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
            mean_cenx = np.nanmean(x['gt_obstacle_cen_x_cm'])
            mean_ceny = np.nanmean(x['gt_obstacle_cen_y_cm'])
  
            for ind,row in self.df.iterrows(): 
                if row['obstacle_cluster'] == cluster_name:
                    self.df.at[ind,'mean_gt_obstacle_cen_x_cm'] = mean_cenx
                    self.df.at[ind,'mean_gt_obstacle_cen_y_cm'] = mean_ceny
        #label cluster by position 
        if numcluster == 9:
            print(numcluster)
            self.df['cluster_label'] = np.nan
            x_pos,y_pos  = np.sort(self.df['mean_gt_obstacle_cen_x_cm'].unique()),np.sort(self.df['mean_gt_obstacle_cen_y_cm'].unique())
            col_1, col_2, col_3 = x_pos[0:3],x_pos[3:6],x_pos[6:9]
            row_1, row_2, row_3 = y_pos[0:3],y_pos[3:6],y_pos[6:9]
            for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
                #label cluster by obstacle post
                x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
                for ind,row in x.iterrows():
                    if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_1: 
                         self.df.at[ind,'cluster_label'] = 0

                    if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                        self.df.at[ind,'cluster_label'] = 1

                    if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                        self.df.at[ind,'cluster_label'] = 2

                    if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                        self.df.at[ind,'cluster_label'] = 3

                    if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                        self.df.at[ind,'cluster_label'] = 4

                    if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                        self.df.at[ind,'cluster_label'] = 5

                    if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                        self.df.at[ind,'cluster_label'] = 6

                    if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                        self.df.at[ind,'cluster_label'] = 7

                    if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                        self.df.at[ind,'cluster_label'] = 8
        elif numcluster == 6:
            print(numcluster)
            self.df['cluster_label'] = np.nan
            x_pos,y_pos  = np.sort(self.df['mean_gt_obstacle_cen_x_cm'].unique()),np.sort(self.df['mean_gt_obstacle_cen_y_cm'].unique())
            col_1, col_2 = x_pos[0:3],x_pos[3:6]
            row_1, row_2, row_3 = y_pos[0:2],y_pos[2:4],y_pos[4:6]
            for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
                #label cluster by obstacle post
                x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
                for ind,row in x.iterrows():
                    if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_1: 
                         self.df.at[ind,'cluster_label'] = 0

                    if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                        self.df.at[ind,'cluster_label'] = 1

                    #if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_1:
                     #   self.df.at[ind,'cluster_label'] = 2

                    if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                        self.df.at[ind,'cluster_label'] = 2

                    if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                        self.df.at[ind,'cluster_label'] = 3

                    #if row['mean_gt_obstacle_cen_x_cm'] in col_3 and row['mean_gt_obstacle_cen_y_cm'] in row_2:
                        #self.df.at[ind,'cluster_label'] = 5

                    if row['mean_gt_obstacle_cen_x_cm'] in col_1 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                        self.df.at[ind,'cluster_label'] = 4

                    if row['mean_gt_obstacle_cen_x_cm'] in col_2 and row['mean_gt_obstacle_cen_y_cm'] in row_3:
                        self.df.at[ind,'cluster_label'] = 5

                
               #    self.df.at[ind,'cluster_label'] = 8
        self.df['obstacle_cluster'] = self.df['cluster_label'].astype(int)


    
    ## Calculate body angle, relative to spine to midspine
    def get_body_angle(self):
        #make cluster
        for ind, row in self.df.iterrows():
            spine_x = row['ts_midspine_x_cm']
            spine_y = row['ts_midspine_y_cm']
            tail_x = row['ts_tailbase_x_cm']
            tail_y = row['ts_tailbase_y_cm']
            angs = []
            for step in range(len(spine_x)):
                ang = np.arctan2(spine_y[step]-tail_y[step],spine_x[step]-tail_x[step])
                angs.append(ang)
            self.df.at[ind, 'body_angle'] = np.array(angs).astype(object)

       


    
    ## Calculate head angle relative to mean of ears to nose 
    def get_head_angle(self):
        for ind, row in self.df.iterrows():
            leftear_x = row['ts_leftear_x_cm']
            leftear_y = row['ts_leftear_y_cm']
            rightear_x = row['ts_rightear_x_cm']
            rightear_y = row['ts_rightear_y_cm']
            nose_x = row['ts_nose_x_cm']
            nose_y = row['ts_nose_y_cm']
            angs = []
            for step in range(len(leftear_x)):
                #ang = np.arctan2(np.mean([leftear_y[step],rightear_y[step]])-nose_y[step],np.mean([leftear_x[step],rightear_x[step]])-nose_x[step])
                ang = np.arctan2(nose_y[step]-np.mean([leftear_y[step],rightear_y[step]]),nose_x[step]-np.mean([leftear_x[step],rightear_x[step]]))

                angs.append(ang)
            self.df.at[ind, 'head_angle'] = np.array(angs).astype(object)

    ## find intersection of head angle to facing side of obstacle 
    def get_obstacle_intersect_nose(self):
        for ind,row in self.df.iterrows():
            if row['odd'] == 'right':
                points_x = np.zeros(len(row['head_angle']))
                points_y = np.zeros(len(row['head_angle']))
                obstacle_top= (row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'] -6)
                obstacle_bottom=(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'] +6)
                for indx,i in enumerate(row['head_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_nose_x_cm'][indx]
                    mouse_y1 = row['ts_nose_y_cm'][indx]
                    mouse_x2 = mouse_x1+200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1+200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            if row['odd'] == 'left':
                points_x = np.zeros(len(row['head_angle']))
                points_y = np.zeros(len(row['head_angle']))
                obstacle_top= (row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm']-6)
                obstacle_bottom=(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm']+6)
                for indx,i in enumerate(row['head_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_nose_x_cm'][indx]
                    mouse_y1 = row['ts_nose_y_cm'][indx]
                    mouse_x2 = mouse_x1+200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1+200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            points_x_all= np.nan_to_num(points_x)
            points_y_all= np.nan_to_num(points_y)
            points_x = points_x_all[points_x_all!=0]
            #points_x = points_x[~np.isnan(points_x)]
            points_y = points_y_all[points_y_all!=0]
            #points_y = points_y[~np.isnan(points_y)]
            self.df.at[ind,'all_obstacle_intersect_nose_x'] = points_x_all.astype(object)
            self.df.at[ind,'all_obstacle_intersect_nose_y'] = points_y_all.astype(object)
            self.df.at[ind,'obstacle_intersect_nose_x'] = points_x.astype(object)
            self.df.at[ind,'obstacle_intersect_nose_y'] = points_y.astype(object)
            

    ## find intersection of head angle
    def get_obstacle_intersect_body(self):
        for ind,row in self.df.iterrows():
            if row['odd'] == 'right':
                points_x = np.zeros(len(row['body_angle']))
                points_y = np.zeros(len(row['body_angle']))
                obstacle_top= (row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'])
                obstacle_bottom=(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'])
                for indx,i in enumerate(row['body_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_midspine_x_cm'][indx]
                    mouse_y1 = row['ts_midspine_y_cm'][indx]
                    mouse_x2 = mouse_x1+200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1+200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            if row['odd'] == 'left':
                points_x = np.zeros(len(row['body_angle']))
                points_y = np.zeros(len(row['body_angle']))
                obstacle_top= (row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm'])
                obstacle_bottom=(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm'])
                for indx,i in enumerate(row['body_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_midspine_x_cm'][indx]
                    mouse_y1 = row['ts_midspine_y_cm'][indx]
                    mouse_x2 = mouse_x1+200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1+200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            points_x = points_x[points_x!=0]
            points_x = points_x[~np.isnan(points_x)]
            points_y = points_y[points_y!=0]
            points_x = points_y[~np.isnan(points_y)]
            self.df.at[ind,'obstacle_intersect_body_x'] = points_x.astype(object)
            self.df.at[ind,'obstacle_intersect_body_y'] = points_y.astype(object) 
    ## find midpoint of facing edge 
    def get_midpoint_edge(self):
        for ind,row in self.df.iterrows():
         if row['odd'] == 'right':
            obstacle_top= (row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'])
            obstacle_bottom=(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'])
            edge_mid = midpoint(obstacle_top[0],obstacle_top[1],obstacle_bottom[0],obstacle_bottom[1])
            self.df.at[ind,'obstacle_edge_mid_x_cm'] = edge_mid[0]
            self.df.at[ind,'obstacle_edge_mid_y_cm'] = edge_mid[1]
         if row['odd'] == 'left':
             obstacle_top= (row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm'])
             obstacle_bottom=(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm'])
             edge_mid = midpoint(obstacle_top[0],obstacle_top[1],obstacle_bottom[0],obstacle_bottom[1])
             self.df.at[ind,'obstacle_edge_mid_x_cm'] = edge_mid[0]
             self.df.at[ind,'obstacle_edge_mid_y_cm'] = edge_mid[1]
    
    ## 
    def get_intersect_counts_bins(self):
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            for ind,row in animal_frame.iterrows():
                row_weights = np.ones_like(row['obstacle_intersect_nose_y'])/float(len(row['obstacle_intersect_nose_y']))
                counts,bins=np.histogram((row['obstacle_intersect_nose_y'] -  row['obstacle_edge_mid_y_cm']),range=(-15,15),bins = 10 ,weights=row_weights)
                self.df.at[ind,'normalized_counts_intersect_nose_y'] = counts.astype('object')
                self.df.at[ind,'sum_normalized_counts_intersect_nose_y'] = float(sum(counts))
                self.df.at[ind,'bins_intersect_nose_y'] = bins.astype('object')
    
    def get_intersect_mean_counts(self):
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            by_cluster =  animal_frame.groupby(['cluster_label'])
            for cluster,cluster_frame in by_cluster:
                by_direction = cluster_frame.groupby(['odd'])
                for direcetion,direction_frame in by_direction:
                    mean_hist = np.mean(direction_frame['normalized_counts_intersect_nose_y'])
                    for ind,row in direction_frame.iterrows():
                        self.df.at[ind,'mean_normalized_counts_intersect_nose_y']=mean_hist.astype('object')
    
    
    def facing_angle(self):
        for ind,row in self.df.iterrows():
            if np.mean(row['head_angle'][:10]) > 0:
                self.df.at[ind,'facing_angle'] = True ## True means facing dowm postive value
            elif np.mean(row['head_angle'][:10]) == 0:
                self.df.at[ind,'facing_angle'] = np.nan
            else: 
                self.df.at[ind,'facing_angle'] = False
    
    def get_angle_to_ports(self):

        for ind,row in self.df.iterrows():
            angle_to_rightport = []
            angle_to_leftport = []
            rightport = [row['rightportT_x_cm'],row['rightportT_y_cm']]
            leftport = [row['leftportT_x_cm'],row['leftportT_y_cm']]
            for indx in range(len(row['ts_nose_x_cm'])):
                center = [np.mean([row['ts_rightear_x_cm'][indx],row['ts_leftear_x_cm'][indx]]),np.mean([row['ts_rightear_y_cm'][indx],row['ts_leftear_y_cm'][indx]])]
                nose_points = [row['ts_nose_x_cm'][indx],row['ts_nose_y_cm'][indx]]
                angleright = calculate_angle(center, nose_points, rightport)
                angleleft = calculate_angle(center, nose_points, leftport)
                angle_to_rightport.append(angleright)
                angle_to_leftport.append(angleleft)
            self.df.at[ind,'angle_to_rightport'] = np.array(angle_to_rightport).astype(object)
            self.df.at[ind,'angle_to_leftport'] = np.array(angle_to_leftport).astype(object)

        right_left = ['angle_to_leftport','angle_to_rightport','ts_nose_x_cm','ts_nose_y_cm']
        for ind,row in self.df.iterrows():
            for direction in right_left:
                interp = pd.Series(row[direction].astype(float)).interpolate().values
                resample = signal.resample(interp,200)
                self.df.at[ind,'resample_'+ direction] = resample.astype(object)

    def create_consective_df(self):
        con_df = pd.DataFrame()
        df = self.df.reset_index(drop=True)
        for animal,animal_frame in df.groupby('animal'):
            for date, date_frame in animal_frame.groupby('date'):
                repeats_list = find_consecutive_repeats(date_frame['obstacle_cluster'])
                for i in range(len(repeats_list)):
                    con_df = con_df.append(date_frame.loc[repeats_list[i][0]:repeats_list[i][1]])
        self.con_df = con_df
        




    def process_df(self,numcluster):
        self.cluster(numcluster)
        self.get_body_angle()
        self.get_head_angle()
        self.get_midpoint_edge()
        self.get_obstacle_intersect_body()
        self.get_obstacle_intersect_nose()
        self.get_intersect_counts_bins()
        self.get_intersect_mean_counts()
        self.facing_angle()
        self.get_angle_to_ports()
        

         


    ## plot the traces by cluster
    def plot_trace_cluster_single_animal(self):
        savepath = "D:/obstacle_avoidance/recordings"
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            by_date = animal_frame.groupby(['date'])
            for date, date_frame in by_date:
                df = date_frame
                savepath_session = os.path.join(*[savepath,str(pd.unique(df.date).item()),str(pd.unique(df.animal).item()),str(pd.unique(df.task).item())])
                pdf = PdfPages(os.path.join((savepath_session),(str(pd.unique(df.date).item()) + '_' + str(pd.unique(df.animal).item()))+ 'cluster.pdf'))

            ##Left ward trials
                fig, ax = plt.subplots(3,2, figsize=(25,21),dpi=50)
                fig.suptitle('Right Start'+'_'+str(df['animal'].unique()), size = 20)
                for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
                    x=df.loc[df['obstacle_cluster']==cluster_name]
                    for i, row in x.iterrows():
                        if row['odd'] == 'left':
                            plt.subplot(3,2,cluster_name+1)
                            plt.gca().set_aspect('equal', adjustable='box')
                            plt.gca().set_title(str(row['obstacle_cluster']))

                            plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                                [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='red')


                            plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                                [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')




                            plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                            plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                            plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                            sns.scatterplot(x=row['ts_nose_x_cm'],y=row['ts_nose_y_cm'],hue = enumerate(row['ts_nose_x_cm']), palette ='magma',legend=False,s =20 ) 
                            #plt.scatter(row['wobstacle_x_cm'], row['wobstacle_y_cm'], c = list(mcolors.TABLEAU_COLORS)[ row['obstacle_cluster']])
                            plt.ylim([52,0]); plt.xlim([0, 72])
                pdf.savefig(); plt.close()

                fig, ax = plt.subplots(3,3, figsize=(25,21),dpi = 50)
                fig.suptitle('Left Start'+'_'+str(df['animal'].unique()), size = 20)
                for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
                    x=df.loc[df['obstacle_cluster']==cluster_name]
                    for i, row in x.iterrows():
                        if row['odd'] == 'right':
                            plt.subplot(3,2,cluster_name+1)
                            plt.gca().set_aspect('equal', adjustable='box')
                            plt.gca().set_title(str(row['obstacle_cluster']))

                            plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                                [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='red')


                            plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                                [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')




                            plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                            plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                            plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                            sns.scatterplot(x=row['ts_nose_x_cm'],y=row['ts_nose_y_cm'],hue = enumerate(row['ts_nose_x_cm']), palette ='magma',legend=False, s = 7) 
                            #plt.scatter(row['wobstacle_x_cm'], row['wobstacle_y_cm'], c = list(mcolors.TABLEAU_COLORS)[ row['obstacle_cluster']])
                            plt.ylim([52,0]); plt.xlim([0, 72])
                pdf.savefig(); plt.close()
                pdf.close()


    ## plot the traces by cluster over multiple days
    def plot_trace_cluster_single_animal_multiday(self):
        savepath = "D:/obstacle_avoidance/recordings"
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            df = animal_frame
            savepath_session = os.path.join(*[savepath,'figures'])
            pdf = PdfPages(os.path.join((savepath_session),( str(pd.unique(df.animal).item())) + 'cluster_multiday.pdf'))

            ##Left ward trials
            fig, ax = plt.subplots(3,2, figsize=(25,21),dpi=50)
            fig.suptitle('Right Start'+'_'+str(df['animal'].unique()), size = 20)
            for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
                x=df.loc[df['obstacle_cluster']==cluster_name]
                for i, row in x.iterrows():
                    if row['odd'] == 'left':
                        plt.subplot(3,2,cluster_name+1)
                        plt.gca().set_aspect('equal', adjustable='box')
                        plt.gca().set_title(str(row['obstacle_cluster']))

                        plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                            [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='red')


                        plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                            [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')




                        plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                        plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                        plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                        sns.scatterplot(x=row['ts_nose_x_cm'],y=row['ts_nose_y_cm'],hue = enumerate(row['ts_nose_x_cm']), palette ='magma',legend=False,s =7 ) 
                        #plt.scatter(row['wobstacle_x_cm'], row['wobstacle_y_cm'], c = list(mcolors.TABLEAU_COLORS)[ row['obstacle_cluster']])
                        plt.ylim([52,0]); plt.xlim([0, 72])
            pdf.savefig(); plt.close()
            fig, ax = plt.subplots(3,3, figsize=(25,21),dpi = 50)
            fig.suptitle('Left Start'+'_'+str(df['animal'].unique()), size = 20)




            for clusters, cluster_name in enumerate(df['obstacle_cluster'].unique()):
                x=df.loc[df['obstacle_cluster']==cluster_name]
                for i, row in x.iterrows():
                    if row['odd'] == 'right':
                        plt.subplot(3,2,cluster_name+1)
                        plt.gca().set_aspect('equal', adjustable='box')
                        plt.gca().set_title(str(row['obstacle_cluster']))

                        plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                            [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='red')


                        plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                            [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')




                        plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                        plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                        plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                        sns.scatterplot(x=row['ts_nose_x_cm'],y=row['ts_nose_y_cm'],hue = enumerate(row['ts_nose_x_cm']), palette ='magma',legend=False, s = 7) 
                        #plt.scatter(row['wobstacle_x_cm'], row['wobstacle_y_cm'], c = list(mcolors.TABLEAU_COLORS)[ row['obstacle_cluster']])
                        plt.ylim([52,0]); plt.xlim([0, 72])
            pdf.savefig(); plt.close()
            pdf.close()


    ##plot headangle by cluster
    def plot_headangle(self):
        savepath = "D:/obstacle_avoidance/recordings"
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            df = animal_frame
            savepath_session = os.path.join(*[savepath,str(pd.unique(df.date).item()),str(pd.unique(df.animal).item()),str(pd.unique(df.task).item())])
            pdf = PdfPages(os.path.join((savepath_session),(str(pd.unique(df.date)) + '_' + str(pd.unique(df.animal).item()))+ 'head_angle.pdf'))

        
            for cluster_num,cluster in enumerate(df['obstacle_cluster'].unique()):
                #fig, ax = plt.subplots(,5, figsize=(25,21),dpi = 50)
                x = df.loc[df['obstacle_cluster']==cluster]
                x = x.reset_index()
                y = nearestX_roundup(len(x),4)
                fig, ax = plt.subplots(int((y/4)),4, figsize=(25,len(x)),dpi = 100)
                fig.suptitle(str(x['obstacle_cluster'].unique()) + '_' +str(x['animal'].unique()) , size = 20)
                for ind,row in x.iterrows():
                    plt.subplot(int((y/4)),4,ind+1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.gca().set_title(str(row['odd'])+str(ind))
                    for indx,i in enumerate(row['head_angle']):
                        current_ang = i
                        x1 = row['ts_nose_x_cm'][indx]
                        y1 = row['ts_nose_y_cm'][indx]
                        x2 = x1+3 * np.cos(current_ang)
                        y2 = y1+3* np.sin(current_ang)
                        plt.plot((x1,x2), (y1,y2), '-',color = 'black',alpha=0.3)
                    plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                            [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='orange')


                    plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                            [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')

                    sns.scatterplot(x=row['obstacle_intersect_nose_x'],y=row['obstacle_intersect_nose_y'],hue = row['obstacle_intersect_nose_x'], palette ='magma',legend=False)    
                    plt.scatter(row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm'],color = 'blue')
                    plt.scatter(row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'],color = 'red')
                    plt.scatter(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm'],color = 'orange')
                    plt.scatter(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'],color = 'green')
                    plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                    plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                    plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                    plt.ylim([52,0]); plt.xlim([0, 72])
                pdf.savefig(); plt.close()
            pdf.close()

    def plot_single_trial(self):
        savepath = "D:/obstacle_avoidance/recordings"
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
             by_date = animal_frame.groupby(['date'])
             for date,date_frame in by_date:
                df = date_frame

                savepath_session = os.path.join(*[savepath,str(pd.unique(df.date).item()),str(pd.unique(df.animal).item()),str(pd.unique(df.task).item())])
                pdf = PdfPages(os.path.join((savepath_session),(str(pd.unique(df.date)) + '_' + str(pd.unique(df.animal).item()))+ 'single_trial.pdf'))


                for cluster_num,cluster in enumerate(df['obstacle_cluster'].unique()):
                    #fig, ax = plt.subplots(,5, figsize=(25,21),dpi = 50)
                    x = df.loc[df['obstacle_cluster']==cluster]
                    x = x.reset_index()
                    y = nearestX_roundup(len(x),4)
                    fig, ax = plt.subplots(int((y/4)),4, figsize=(25,len(x)),dpi = 100)
                    fig.suptitle(str(x['obstacle_cluster'].unique()) + '_' +str(x['animal'].unique()) , size = 20)
                    for ind,row in x.iterrows():
                        plt.subplot(int((y/4)),4,ind+1)
                        plt.gca().set_aspect('equal', adjustable='box')
                        plt.gca().set_title(str(row['odd'])+str(ind))
                        plt.plot(row['ts_nose_x_cm'],row['ts_nose_y_cm'])
                        plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                                [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='orange')


                        plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                                [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')

                        #sns.scatterplot(x=row['obstacle_intersect_nose_x'],y=row['obstacle_intersect_nose_y'],hue = row['obstacle_intersect_nose_x'], palette ='magma',legend=False)    
                        plt.scatter(row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm'],color = 'blue')
                        plt.scatter(row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'],color = 'red')
                        plt.scatter(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm'],color = 'orange')
                        plt.scatter(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'],color = 'green')
                        plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                        plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                        plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                        plt.ylim([52,0]); plt.xlim([0, 72])
                    pdf.savefig(); plt.close()
                pdf.close()


    def intersect_histogram(self,savepath,filename,intersectpart):
        savepath = "D:/obstacle_avoidance/recordings"
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            df = animal_frame
            savepath_session = os.path.join(*[savepath,str(pd.unique(df.date).item()),str(pd.unique(df.animal).item()),str(pd.unique(df.task).item())])
            pdf = PdfPages(os.path.join((savepath_session),(str(pd.unique(df.date)) + '_' + str(pd.unique(df.animal).item()))+ '_intersect_histogram.pdf'))
            for cluster_num,cluster in enumerate(df['obstacle_cluster'].unique()):
                #fig, ax = plt.subplots(,5, figsize=(25,21),dpi = 50)
                x = df.loc[df['obstacle_cluster']==cluster]
                x = x.reset_index()
                y = nearestX_roundup(len(x),4)
                fig, ax = plt.subplots(int((y/4)),4, figsize=(25,len(x)),dpi = 100)
                fig.suptitle(str(x['obstacle_cluster'].unique()) + '_' +str(x['animal'].unique()) , size = 20)
                for ind,row in x.iterrows():
                    plt.subplot(int((y/4)),4,ind+1)
                    #plt.gca().set_aspect('equal', adjustable='box')
                    plt.gca().set_title(str(row['odd']) + str(ind))
                    plt.hist((row[intersectpart] -  row['obstacle_edge_mid_y_cm']) )
                    plt.xlim(8,-8)

                pdf.savefig(); plt.close()
            pdf.close()



#Summary Figure Single Day 
## set pdf 
# common save path
    def train_day_summary(self):
        savepath = "D:/obstacle_avoidance/recordings"
    # analyze by each animal and date 
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            by_date = animal_frame.groupby(['date'])
            for date, date_frame in by_date:
                df = date_frame
                #df = df.reset_index()
                # set up pdf page
                savepath_session = os.path.join(*[savepath,str(pd.unique(df.date).item()),str(pd.unique(df.animal).item()),str(pd.unique(df.task).item())])
                pdf = PdfPages(os.path.join((savepath_session),(str(pd.unique(df.date).item()) + '_' + str(pd.unique(df.animal).item()))+ '_summary.pdf'))

                fig = plt.figure(constrained_layout=False, figsize=(15, 15),dpi=90)
                spec2 = gridspec.GridSpec(ncols=2, nrows=4, figure=fig)
                plt.suptitle(str(pd.unique(df.animal)) + " " + str(pd.unique(df.date) ) + " " + str(len(df)))
                ax1 = fig.add_subplot(spec2[0,0])
                ax2 = fig.add_subplot(spec2[0,1])
                ax3 = fig.add_subplot(spec2[1,0])
                ax4 = fig.add_subplot(spec2[1,1])
                ax5 = fig.add_subplot(spec2[2,0])
                ax6 = fig.add_subplot(spec2[2,1])
                ax7 = fig.add_subplot(spec2[3,0])
                ax8 = fig.add_subplot(spec2[3,1])


                ##left and right trajectories

                right,left = df[df['odd']=='right'],df[df['odd']=='left']
                right = right.loc[right['dist']<70]
                left = left.loc[left['dist']<70]
                right_nose_x,right_nose_y=right['ts_nose_x_cm'].to_numpy(), right['ts_nose_y_cm'].to_numpy() 
                left_nose_x,left_nose_y=left['ts_nose_x_cm'].to_numpy(), left['ts_nose_y_cm'].to_numpy() 



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


                for ind in range(len(right)):
                    ax1.plot(right_nose_x[ind],right_nose_y[ind])
                ax1.set_ylim([51,0]); ax1.set_xlim([0, 71])
                ax1.title.set_text('Right Trials'+' '+ str(len(right)))

                ax1.plot([arena_x[0],arena_x[1],arena_x[2],arena_x[3],arena_x[0]],
                          [arena_y[0],arena_y[1],arena_y[2],arena_y[3],arena_y[0]],c='k')

                ax1.scatter(left_port[0],left_port[1],c='purple',s=200,marker = 's')
                ax1.scatter(right_port[0],right_port[1],c='r',s=200,marker = 's')

                for ind in range(len(left)):
                    ax2.plot(left_nose_x[ind],left_nose_y[ind])
                ax2.set_ylim([51,0]); ax2.set_xlim([0, 71])
                ax2.title.set_text('Left Trials'+' '+ str(len(left)))
                ax2.plot([arena_x[0],arena_x[1],arena_x[2],arena_x[3],arena_x[0]],
                          [arena_y[0],arena_y[1],arena_y[2],arena_y[3],arena_y[0]],c='k')

                ax2.scatter(left_port[0],left_port[1],c='purple',s=200,marker = 's')
                ax2.scatter(right_port[0],right_port[1],c='r',s=200,marker = 's')



                ## Trial time 
                #histogram
                ax3.hist(df['time'])
                ax3.set_xlabel('Time(sec)')
                ax3.set_ylabel('count')
                ax3.title.set_text('Trial Time Histogram')

                ax4.plot(range(len(df)),df['time'],'-o')
                ax4.set_xlabel('Trial Number')
                ax4.set_ylabel('Time(sec)')
                ax4.axline((0,df['time'].mean()),slope=0)
                ax4.legend(title ='Mean Trial Time(sec) ' + str(np.round(df['time'].mean())))
                ax4.title.set_text('Time per Trial')
                ## Trial distance

                ax5.hist(df['dist'])
                ax5.set_xlabel('Distance(cm)')
                ax5.set_ylabel('count')
                ax5.title.set_text('Trial Distance Histogram')

                ax6.plot(range(len(df)),df['dist'],'-o')
                ax6.set_xlabel('Trial Number')
                ax6.set_ylabel('Distance(cm)')
                ax6.axline((0,df['dist'].mean()),slope=0)
                ax6.legend(title ='Mean Trial Distance(cm) ' + str(np.round(df['dist'].mean())))
                ax6.title.set_text('Distance per Trial')

                ## angle to port

                #right starting 
                ax7.hist(np.mean(right['resample_angle_to_leftport']),color='b',label='leftport')
                ax7.hist(np.mean(right['resample_angle_to_rightport']),color = 'r',label='rightport')
                ax7.legend()
                ax7.set_xlim(0,180)
                ax7.title.set_text('Right Trials')

                #left starting 
                ax8.hist(np.mean(left['resample_angle_to_leftport']),color='b',label='leftport')
                ax8.hist(np.mean(left['resample_angle_to_rightport']),color = 'r',label='rightport')
                ax8.legend()
                ax8.set_xlim(0,180)
                ax8.title.set_text('Left Trials')

                pdf.savefig(); plt.close()
                pdf.close()


    def plot_consecutive_trials_singleday(self):
        savepath = "D:/obstacle_avoidance/recordings"
        # analyze by each animal and date 
        by_animal = self.df.groupby(['animal'])
        for animal,animal_frame in by_animal:
            by_date = animal_frame.groupby(['date'])
            for date, date_frame in by_date:
                df = date_frame
                df = df.reset_index()
                savepath_session = os.path.join(*[savepath,str(pd.unique(df.date).item()),str(pd.unique(df.animal).item()),str(pd.unique(df.task).item())])
                con_df = pd.DataFrame()
                obstacle_cluster_list = df['obstacle_cluster']
                con_list = find_consecutive_repeats(obstacle_cluster_list)

                for i in range(len(con_list)):
                    con_df = con_df.append(df.loc[con_list[i][0]:con_list[i][1]])
                con_df = con_df.reset_index()
                row_num = nearestX_roundup(len(con_df),3)/3
                fig, ax = plt.subplots(int(row_num),3, figsize=(15,int(len(con_df))),dpi = 90)
                pdf = PdfPages(os.path.join((savepath_session),(str(pd.unique(df.date).item()) + '_' + str(pd.unique(df.animal).item()))+ '_consective_trials.pdf')) 
                fig.suptitle(str(con_df['animal'].unique()) + '_' +str(con_df['date'].unique()) , size = 20)
                fig.tight_layout()
                plt.subplots_adjust(top=0.95)
                for ind,row in con_df.iterrows():
                    plt.subplot(int(row_num),3,ind+1)
                    plt.gca().set_aspect('equal', adjustable='datalim')
                    plt.gca().set_title(str(row['odd'])+str(row['obstacle_cluster']))
                    plt.plot(row['ts_nose_x_cm'],row['ts_nose_y_cm'])
                    plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                            [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='orange')
                    plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                            [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')
                    #sns.scatterplot(x=row['obstacle_intersect_nose_x'],y=row['obstacle_intersect_nose_y'],hue = row['obstacle_intersect_nose_x'], palette ='magma',legend=False)    
                    #plt.scatter(row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm'],color = 'blue')
                    #plt.scatter(row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'],color = 'red')
                    #plt.scatter(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm'],color = 'orange')
                    #plt.scatter(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'],color = 'green')
                    #plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                    plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                    plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                    plt.ylim([52,0]); plt.xlim([0, 72])
                pdf.savefig(); plt.close()
                pdf.close()


    def plot_consective_trials(self,key,color_pallete):
        """consecutive plots"""


        savepath = "D:/obstacle_avoidance/recordings"
        savepath_session = os.path.join(*[savepath,'figures'])
        key = key
        color_pallete = color_pallete
        color_map = create_color_dict(self.con_df,key,color_pallete)
        pdf = PdfPages(os.path.join((savepath_session), 'by_' + str(key)+ '_' 'consecutive.pdf'))
        for obstacle, obstalce_frame in self.con_df.groupby('obstacle_cluster'):

            df = obstalce_frame
            df = df.reset_index(drop=True)
            fig = plt.figure(constrained_layout=False, figsize=(15, 7.5),dpi=90)
            fig.suptitle('by_' + str(key) + str(pd.unique(df['obstacle_cluster'])))
            spec2 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
            #panel_1 = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=spec2[0])


            """Top Row"""
            ax1 = fig.add_subplot(spec2[0,0])
            ax1.set_title('right')
            plot_arena(df,ax1,obstacle=True)
            ax2 = fig.add_subplot(spec2[0,1])
            plot_arena(df,ax2,obstacle=True)
            ax3 = fig.add_subplot(spec2[0,2])
            plot_arena(df,ax3,obstacle=True)



            """Bottom Row"""
            ax4 = fig.add_subplot(spec2[1,0])
            ax4.set_title('left')
            plot_arena(df,ax4,obstacle=True)
            ax5 = fig.add_subplot(spec2[1,1])
            plot_arena(df,ax5,obstacle=True)
            ax6 = fig.add_subplot(spec2[1,2])
            plot_arena(df,ax6,obstacle=True)



            """Loop through data frame"""


            trial_list = list(range(0,len(df),3))
            trial_list = create_sublists(trial_list)
            trial_list = sample(trial_list,100)
            for sublist in trial_list:
                trial = df.iloc[sublist[0]:sublist[1]]
                trial = trial.reset_index(drop=True)
                color = color_map.get(pd.unique(trial[key]).item())
                if trial.at[0,'odd'] == 'right':
                    ax1.plot(trial.at[0,'ts_nose_x_cm'],trial.at[0,'ts_nose_y_cm'],c=color)
                    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
                    ax1.legend(markers, color_map.keys(), numpoints=1)
                    ax2.plot(trial.at[1,'ts_nose_x_cm'],trial.at[1,'ts_nose_y_cm'],c=color)
                    ax3.plot(trial.at[2,'ts_nose_x_cm'],trial.at[2,'ts_nose_y_cm'],c=color)
                if trial.at[0,'odd'] == 'left':
                    ax4.plot(trial.at[0,'ts_nose_x_cm'],trial.at[0,'ts_nose_y_cm'],c=color)
                    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
                    ax4.legend(markers, color_map.keys(), numpoints=1)
                    ax5.plot(trial.at[1,'ts_nose_x_cm'],trial.at[1,'ts_nose_y_cm'],c=color)
                    ax6.plot(trial.at[2,'ts_nose_x_cm'],trial.at[2,'ts_nose_y_cm'],c=color)
            pdf.savefig(); plt.close()
        pdf.close()

    def obstacle_by_variable(self,key,color_pallete):
        """Direction by key obstalce trials"""

        savepath = "D:/obstacle_avoidance/recordings"
        savepath_session = os.path.join(*[savepath,'figures'])


        key=key
        color_pallete = color_pallete
        color_map = create_color_dict(self.df,key,color_pallete)

        pdf = PdfPages(os.path.join((savepath_session), 'by ' + str(key)+ ' '+ 'and ' +' obstalce.pdf'))

        fig = plt.figure(constrained_layout=False, figsize=(15, 15),dpi=90)
        fig.suptitle('by ' + key + ' '+ 'and ' +'obstalce ')
        spec2 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)


        """Right"""
        panel_1 = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=spec2[0])
        ax1 = fig.add_subplot(panel_1[0,0])
        plot_arena(self.df,ax1)
        ax2 = fig.add_subplot(panel_1[0,1])
        plot_arena(self.df,ax2)

        ax3 = fig.add_subplot(panel_1[0,2])
        plot_arena(self.df,ax3)
        ax4 = fig.add_subplot(panel_1[1,0])
        plot_arena(self.df,ax4)
        ax5 = fig.add_subplot(panel_1[1,1])
        plot_arena(self.df,ax5)
        ax6 = fig.add_subplot(panel_1[1,2])
        plot_arena(self.df,ax6)

        right_axs = [ax1,ax2,ax3,ax4,ax5,ax6]
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
        ax1.legend(markers, color_map.keys(), numpoints=1,title = 'right')


        """Left """
        panel_2 = gridspec.GridSpecFromSubplotSpec(2,3,subplot_spec=spec2[1])
        ax7 = fig.add_subplot(panel_2[0,0])
        plot_arena(self.df,ax7)
        ax8 = fig.add_subplot(panel_2[0,1])
        plot_arena(self.df,ax8)
        ax8.set_title('left')
        ax9 = fig.add_subplot(panel_2[0,2])
        plot_arena(self.df,ax9)
        ax10 = fig.add_subplot(panel_2[1,0])
        plot_arena(self.df,ax10)
        ax11= fig.add_subplot(panel_2[1,1])
        plot_arena(self.df,ax11)
        ax12 = fig.add_subplot(panel_2[1,2])
        plot_arena(self.df,ax12)

        left_axs = [ax7,ax8,ax9,ax10,ax11,ax12]
        markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in color_map.values()]
        ax7.legend(markers, color_map.keys(), numpoints=1,title = 'left')



        """ plot trials"""
        right_obstacle_dict = dict(zip(pd.unique(self.df['obstacle_cluster'].sort_values().to_list()),right_axs))
        left_obstacle_dict = dict(zip(pd.unique(self.df['obstacle_cluster'].sort_values().to_list()),left_axs))

        for direction, direction_frame in self.df.groupby(['odd']):
            for cluster, cluster_frame in direction_frame.groupby(['obstacle_cluster']):
                cluster_frame = cluster_frame.sample(50)
                right_obstacle_axis = right_obstacle_dict.get(cluster)
                left_obstalce_axis = left_obstacle_dict.get(cluster)
                plot_obstacle(cluster_frame,right_obstacle_axis,cluster)
                plot_obstacle(cluster_frame,left_obstalce_axis,cluster)
                right_obstacle_axis.set_title(str(cluster))
                left_obstalce_axis.set_title(str(cluster))
                for ind,row in cluster_frame.iterrows():
                    color = color_map.get(pd.unique(row[key]).item())
                    if direction == 'right':
                        which_axis = right_obstacle_dict.get(cluster)
                        which_axis.plot(row['ts_nose_x_cm'],row['ts_nose_y_cm'],c = color)
                    if direction == 'left':
                        which_axis = left_obstacle_dict.get(cluster)
                        which_axis.plot(row['ts_nose_x_cm'],row['ts_nose_y_cm'],c = color)
        pdf.savefig(); plt.close()
        pdf.close()
