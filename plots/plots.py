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
from sklearn.cluster import KMeans
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mcolors
import os, fnmatch

import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')

from utils.base_functions import *
from src.utils.auxiliary import flatten_series
from src.utils.path import find
from src.base import BaseInput

import warnings
warnings.filterwarnings('ignore')



## take raw_df from multiple session and plt  
class plot_oa(BaseInput):

    def __init__(self,metadata_path,cluster=False,plot_trace=False):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.path = self.metadata['path']
        self.dates_list = [i for i in list(self.metadata.keys()) if i != 'path' ]
    ## append df's together
    def gather_session_df(self):
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
                    raw_h5 = [i for i in h5_paths if 'raw' in i]
                    hf_list.append(raw_h5)
        hf_list = list(itertools.chain(*hf_list))
        for h5 in hf_list:
            data = pd.read_hdf(h5)
            df=df.append(data,ignore_index=True)
        self.df=df
    ##cluster obstacle positions
    def cluster(self):
        self.df = self.df[self.df['gt_obstacle_cen_x_cm'].notna()]


        kmeans_input = np.vstack([self.df['gt_obstacle_cen_x_cm'].values, self.df['gt_obstacle_cen_y_cm'].values])

        kmeans_input = np.transpose(kmeans_input)

        labels = KMeans(n_clusters=9).fit(kmeans_input).labels_
        self.df['obstacle_cluster'] = labels

        #get mean of obstacle center
        for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
            x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
            mean_cenx = np.mean(x['gt_obstacle_cen_x_cm'])
            mean_ceny = np.mean(x['gt_obstacle_cen_y_cm'])
  
            for ind,row in self.df.iterrows(): 
                if row['obstacle_cluster'] == cluster_name:
                    self.df.at[ind,'mean_gt_obstacle_cen_x_cm'] = mean_cenx
                    self.df.at[ind,'mean_gt_obstacle_cen_y_cm'] = mean_ceny
        #label cluster by position 
        self.df['cluster_label'] = np.nan
        for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
            #label cluster by obstacle post
            x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
            for ind,row in x.iterrows():
                # position top left label 0
                if 26.511267901536936 <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 29.262555270323688 and 19.603003146893094 <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 21.48005617413041:
                    self.df.at[ind,'cluster_label'] = 0
                # postion top middle lable 1     
                if 34.224618031842425  <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 40.737819614353576  and 19.171765596431882  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 21.765871122032685:
                    self.df.at[ind,'cluster_label'] = 1
                # postion top middle lable 2     
                if 47.10153827545068   <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 51.15598067362944  and 19.386902039003104  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 21.36299006442639:
                    self.df.at[ind,'cluster_label'] = 2
                # postion top middle lable 3     
                if 24.622638543067332    <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 30.32366857885396   and 24.072987056842724  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 28.06565031743722:
                    self.df.at[ind,'cluster_label'] = 3 
                # postion top middle lable 4     
                if 34.33696422611072     <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 39.16315460138148   and 23.33051936216178  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 28.21568031983405:
                    self.df.at[ind,'cluster_label'] = 4
                # postion top middle lable 5     
                if 46.78662583258851     <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 51.45490492362184   and 23.30309610363196  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 26.340749658133035:
                    self.df.at[ind,'cluster_label'] = 5
                # postion top middle lable 6     
                if 23.672283873362495      <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 29.24383212681531  and 31.13010098995012  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 32.9010213861951:
                    self.df.at[ind,'cluster_label'] = 6
                # postion top middle lable 7     
                if 34.99528058841633      <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 38.68741237003255   and 30.937326612118863  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 32.92257553927523:
                    self.df.at[ind,'cluster_label'] = 7
                # postion top middle lable 8     
                if 47.0371762841755 <= x['mean_gt_obstacle_cen_x_cm'].unique() <= 52.09703145249776    and 30.463949607263373  <= x['mean_gt_obstacle_cen_y_cm'].unique() <= 32.749148951731954:
                    self.df.at[ind,'cluster_label'] = 8
        self.df['cluster_label'] = self.df['cluster_label'].astype('int')


    
    ## Calculate body angle, relative to spine to midspine
    def get_body_angle(self):
        #make cluster
        for ind, row in self.df.iterrows():
            spine_x = row['ts_spine_x_cm']
            spine_y = row['ts_spine_y_cm']
            midspine_x = row['ts_midspine_x_cm']
            midspine_y = row['ts_midspine_y_cm']
            angs = []
            for step in range(len(spine_x)):
                ang = np.arctan2(midspine_y[step] - spine_y[step],midspine_x[step] - spine_x[step])
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
                ang = np.arctan2(np.mean([leftear_y[step],rightear_y[step]])-nose_y[step],np.mean([leftear_x[step],rightear_x[step]])-nose_x[step])
                angs.append(ang)
            self.df.at[ind, 'head_angle'] = np.array(angs).astype(object)

    ## find intersection of head angle to facing side of obstacle 
    def get_obstacle_intersect_nose(self):
        for ind,row in self.df.iterrows():
            if row['odd'] == False:
                points_x = np.zeros(len(row['head_angle']))
                points_y = np.zeros(len(row['head_angle']))
                obstacle_top= (row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'] -6)
                obstacle_bottom=(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'] +6)
                for indx,i in enumerate(row['head_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_nose_x_cm'][indx]
                    mouse_y1 = row['ts_nose_y_cm'][indx]
                    mouse_x2 = mouse_x1-200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1-200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            if row['odd'] == True:
                points_x = np.zeros(len(row['head_angle']))
                points_y = np.zeros(len(row['head_angle']))
                obstacle_top= (row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm']-6)
                obstacle_bottom=(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm']+6)
                for indx,i in enumerate(row['head_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_nose_x_cm'][indx]
                    mouse_y1 = row['ts_nose_y_cm'][indx]
                    mouse_x2 = mouse_x1-200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1-200* np.sin(current_ang)
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
            if row['odd'] == False:
                points_x = np.zeros(len(row['body_angle']))
                points_y = np.zeros(len(row['body_angle']))
                obstacle_top= (row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'])
                obstacle_bottom=(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'])
                for indx,i in enumerate(row['body_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_spine_x_cm'][indx]
                    mouse_y1 = row['ts_spine_y_cm'][indx]
                    mouse_x2 = mouse_x1-200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1-200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            if row['odd'] == True:
                points_x = np.zeros(len(row['body_angle']))
                points_y = np.zeros(len(row['body_angle']))
                obstacle_top= (row['gt_obstacleTL_x_cm'],row['gt_obstacleTL_y_cm'])
                obstacle_bottom=(row['gt_obstacleBL_x_cm'],row['gt_obstacleBL_y_cm'])
                for indx,i in enumerate(row['body_angle']):
                    current_ang= i
                    mouse_x1 = row['ts_spine_x_cm'][indx]
                    mouse_y1 = row['ts_spine_y_cm'][indx]
                    mouse_x2 = mouse_x1-200 * np.cos(current_ang)
                    mouse_y2 = mouse_y1-200* np.sin(current_ang)
                    intersect_point=intersect((mouse_x1,mouse_y1),(mouse_x2,mouse_y2),obstacle_top,obstacle_bottom)
                    points_x[indx] = intersect_point[0]
                    points_y[indx] = intersect_point[1]
            points_x = points_x[points_x!=0]
            points_x = points_x[~np.isnan(points_x)]
            points_y = points_y[points_y!=0]
            points_x = points_y[~np.isnan(points_y)]
            self.df.at[ind,'obstacle_intersect_spine_x'] = points_x.astype(object)
            self.df.at[ind,'obstacle_intersect_spine_y'] = points_y.astype(object) 
    ## find midpoint of facing edge 
    def get_midpoint_edge(self):
        for ind,row in self.df.iterrows():
         if row['odd'] == False:
            obstacle_top= (row['gt_obstacleTR_x_cm'],row['gt_obstacleTR_y_cm'])
            obstacle_bottom=(row['gt_obstacleBR_x_cm'],row['gt_obstacleBR_y_cm'])
            edge_mid = midpoint(obstacle_top[0],obstacle_top[1],obstacle_bottom[0],obstacle_bottom[1])
            self.df.at[ind,'obstacle_edge_mid_x_cm'] = edge_mid[0]
            self.df.at[ind,'obstacle_edge_mid_y_cm'] = edge_mid[1]
         if row['odd'] == True:
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
        




    def process_df(self):
        self.cluster()
        #self.get_body_angle()
        self.get_head_angle()
        self.get_midpoint_edge()
        #self.get_obstacle_intersect_body()
        self.get_obstacle_intersect_nose()
        self.get_intersect_counts_bins()
        self.get_intersect_mean_counts()
        

         



##




    ## plot the traces by cluster
    def plot_trace_cluster_single_animal(self,savepath,filename):
        pdf = PdfPages(os.path.join(savepath,(filename) + '_figs.pdf'))
        ##Left ward trials
        fig, ax = plt.subplots(3,3, figsize=(25,21),dpi=50)
        fig.suptitle('Right Start'+'_'+str(self.df['animal'].unique()), size = 20)
        for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
            x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
            for i, row in x.iterrows():
                if row['odd'] == False:
                    plt.subplot(3,3,cluster_name+1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.gca().set_title(str(row['obstacle_cluster']))

                    plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                        [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='red')


                    plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                        [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')




                    plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                    plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                    plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                    sns.scatterplot(x=row['ts_nose_x_cm'],y=row['ts_nose_y_cm'],hue = enumerate(row['ts_nose_x_cm']), palette ='magma',legend=False) 
                    #plt.scatter(row['wobstacle_x_cm'], row['wobstacle_y_cm'], c = list(mcolors.TABLEAU_COLORS)[ row['obstacle_cluster']])
                    plt.ylim([52,0]); plt.xlim([0, 72])
        pdf.savefig(); plt.close()
        fig, ax = plt.subplots(3,3, figsize=(25,21),dpi = 50)
        fig.suptitle('Left Start'+'_'+str(self.df['animal'].unique()), size = 20)

        


        for clusters, cluster_name in enumerate(self.df['obstacle_cluster'].unique()):
            x=self.df.loc[self.df['obstacle_cluster']==cluster_name]
            for i, row in x.iterrows():
                if row['odd'] == True:
                    plt.subplot(3,3,cluster_name+1)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.gca().set_title(str(row['obstacle_cluster']))

                    plt.plot([row['arenaTL_x_cm'], row['arenaTR_x_cm'], row['arenaBR_x_cm'], row['arenaBL_x_cm'],row['arenaTL_x_cm']],
                        [row['arenaTL_y_cm'], row['arenaTR_y_cm'], row['arenaBR_y_cm'], row['arenaBL_y_cm'],row['arenaTL_y_cm']],color='red')


                    plt.plot([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBR_x_cm'], row['gt_obstacleBL_x_cm'],row['gt_obstacleTL_x_cm']],
                        [row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBR_y_cm'], row['gt_obstacleBL_y_cm'],row['gt_obstacleTL_y_cm']],color='green')




                    plt.scatter(row['gt_obstacle_cen_x_cm'],row['gt_obstacle_cen_y_cm'],color='blue')
                    plt.scatter(row['leftportT_x_cm'],row['leftportT_y_cm'],color='blue')
                    plt.scatter(row['rightportT_x_cm'],row['rightportT_y_cm'],color='black')
                    sns.scatterplot(x=row['ts_nose_x_cm'],y=row['ts_nose_y_cm'],hue = enumerate(row['ts_nose_x_cm']), palette ='magma',legend=False) 
                    #plt.scatter(row['wobstacle_x_cm'], row['wobstacle_y_cm'], c = list(mcolors.TABLEAU_COLORS)[ row['obstacle_cluster']])
                    plt.ylim([52,0]); plt.xlim([0, 72])
        pdf.savefig(); plt.close()
        pdf.close()
    ##plot headangle by cluster
    def plot_headangle(self,savepath,filename):
        pdf = PdfPages(os.path.join(savepath,(filename) + '_figs.pdf'))
        for cluster_num,cluster in enumerate(self.df['obstacle_cluster'].unique()):
            #fig, ax = plt.subplots(,5, figsize=(25,21),dpi = 50)
            x = self.df.loc[self.df['obstacle_cluster']==cluster]
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


    def intersect_histogram(self,savepath,filename,intersectpart):
        pdf = PdfPages(os.path.join(savepath,(filename) + '_figs.pdf'))
        for cluster_num,cluster in enumerate(self.df['obstacle_cluster'].unique()):
            #fig, ax = plt.subplots(,5, figsize=(25,21),dpi = 50)
            x = self.df.loc[self.df['obstacle_cluster']==cluster]
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


       