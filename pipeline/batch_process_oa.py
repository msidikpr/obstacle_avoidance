"""class for batch processing multiple oa sessions"""
import json, os
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import h5py as hf
import sys
import itertools 
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')

import warnings
warnings.filterwarnings('ignore')

from pipeline.helper_functions import list_columns,interpolate_array,split_range_into_parts,check_trial_for_obstalce_cross,assign_date_index
from pipeline.oa_calculations import cluster_obstacle
from src.base import BaseInput


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
    def  gather_session_df(self,tasktype,numcluster):
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
                   
                    if tasktype == 'non_obstalce':
                        raw_h5 = [i for i in h5_paths if 'non' in i]
                        raw_h5 = [i for i in h5_paths if 'DLC' not in i]
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
        keys
        for key in keys:
            for ind,row in self.df.iterrows():
                    self.df.at[ind,key] = np.mean(row[key])
        #for key in keys:
        #    self.df[key] = self.df[key].mean()
        """redo ts_body parts"""
        keys = ['nose','leftear','rightear','spine','midspine','tailbase']
        for ind,row in self.df.iterrows():
            if row['odd']=='left':
                nose_list = row['nose_x_cm'] 
                odd_ind = np.argmax(nose_list>(row.leftportT_x_cm+5))
                ind_list =  list(range(len(row['nose_x_cm']))) 
                ts_inds = ind_list[odd_ind:]
                self.df.at[ind,'ts_inds'] = np.array(ts_inds).astype(object)
                for key in keys:
                    ts_part_x = row[key + '_x_cm'][odd_ind:]
                    ts_part_y = row[key + '_y_cm'][odd_ind:]
                    self.df.at[ind,'ts_'+key + '_x_cm'] = ts_part_x
                    self.df.at[ind,'ts_'+key + '_y_cm'] = ts_part_y


                

            else:
                nose_list = row['nose_x_cm']
                even_ind = np.argmax(nose_list<(row.rightportT_x_cm-5))
                ind_list =  list(range(len(row['nose_x_cm']))) 
                ts_inds = ind_list[even_ind:]
                self.df.at[ind,'ts_inds'] = np.array(ts_inds).astype(object)
                for key in keys:
                    ts_part_x = row[key + '_x_cm'][even_ind:]
                    ts_part_y = row[key + '_y_cm'][even_ind:]
                    self.df.at[ind,'ts_'+key + '_x_cm'] = ts_part_x
                    self.df.at[ind,'ts_'+key + '_y_cm'] = ts_part_y




    
        #self.get_angle_to_ports()

        if tasktype == 'obstacle': 
            cluster_obstacle(self.df,6)

            """get average obstacle postition"""
            keys = list_columns(self.df,['gt'])
            keys = [key for key in keys if 'cen' not in key]
            keys = [key for key in keys if 'mean' not in key]

            for key in keys:
                self.df['mean_'+key] = np.nan
            for cluster,cluster_frame in self.df.groupby('obstacle_cluster'):
                for key in keys:
                    mean_obstacle = cluster_frame[key].mean()

                    self.df.loc[self.df['obstacle_cluster'] ==cluster,['mean_'+key]] = mean_obstacle
                    

            #ind_to_drop = []
            #for ind,row in self.df.iterrows():
            # if check_trial_for_obstalce_cross(row) == False:
            #    ind_to_drop.append(ind)
            #self.df.drop(index=ind_to_drop,inplace=True)
    def  trial_min_df(self,training_ses = False):
        # per animal basis
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
                    raw_h5 = [i for i in h5_paths if 'test' in i]
                    hf_list.append(raw_h5)
        hf_list = list(itertools.chain(*hf_list))
        for h5 in hf_list:
            data = pd.read_hdf(h5)
            df=df.append(data,ignore_index=False)
        self.df=df
        self.df['orginal_index'] = self.df.index
        self.df =self.df.reset_index()
        if training_ses == True:
                self.df = assign_date_index(self.df,train=True)
        else:
            self.df = assign_date_index(self.df)
                

        dictists = {key: [] for key in ['animal','date','date_index','trialpermin','task','session_min',"trials",'trial_time']}
        for date, date_df in self.df .groupby(['date_index']):
            for ani, ani_df in date_df.groupby(['animal']):
                for task,task_df in ani_df.groupby('task'):
                    dictists['animal'].append(ani)
                    dictists['date_index'].append(date)
                    trialpermin = (len(task_df))/(task_df.len.sum()/60/60)
                    dictists['trialpermin'].append(trialpermin)
                    dictists['task'].append(task_df.task.unique()[0])
                    dictists['date'].append(task_df.date.unique()[0])
                    dictists['session_min'].append(task_df.len.sum()/60/60)
                    dictists['trials'].append(len(task_df))
                    dictists['trial_time'].append(task_df.len.median()/60)

        trialpermin_df = pd.DataFrame.from_dict(dictists)
        self.trialpermin_df = trialpermin_df

        


       