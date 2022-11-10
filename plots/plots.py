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

from utils.base_functions import format_frames,flatten_column,list_columns
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