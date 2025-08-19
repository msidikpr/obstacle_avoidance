## script for processing obstacle avoidance session 
## Notes: 8/23/22(Works with ephys0 enviroment)
import json, os, cv2
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import glob
import h5py
import subprocess

from scipy.ndimage import gaussian_filter1d


import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')

import warnings
warnings.filterwarnings('ignore')

from src.base import BaseInput
from src.topcam import Topcam
from src.utils.auxiliary import find_index_in_list,flatten_series
from src.utils.path import find
#from utils.base_functions import *
from pipeline.helper_functions import list_columns,interpolate_array,flatten_list_of_arrays
from pipeline.oa_calculations import distance_calcs,heading_calcs,cluster_obstacle,deveation,lateral_error,df_tortuosity,head_angle_velocity,start
#from plots.plots import plot_oa


## create avoidance session object 
# metadata is json file that specifys the path to recording session for preprocessing
class AvoidanceProcessing(BaseInput):
    def __init__(self, metadata_path, task='oa'):
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        self.path = self.metadata['path']
        self.dates_list = [i for i in list(self.metadata.keys()) if i != 'path']
        self.tasktype = task
        if task=='oa':
            self.is_pillar_avoidance = True
            self.is_non_obstacle = False
        elif task =='non_obstalce':
            self.is_non_obstacle = True
            self.is_pillar_avoidance = False

        if self.is_pillar_avoidance:
            self.dlc_project = {'light':'/home/niell_lab/Documents/deeplabcut_projects/object_avoidance-Mike-2021-08-31/config.yaml',
                                'dark':None}

        elif self.is_non_obstacle:
            self.dlc_project = {'light':'/home/niell_lab/Documents/deeplabcut_projects/object_avoidance-Mike-2021-08-31/config.yaml',
                                'dark':None}
        self.camname = 'TOP1'
        self.generic_camconfig = {
            'paths': {
                'dlc_projects': {
                    self.camname: self.dlc_project
                },
            },
            'internals': {
                'follow_strict_naming': False,
                'crop_for_dlc': False,
                'filter_dlc_predictions': False,
                'multianimal_top_project': False,
                'likelihood_threshold': 0.75
            }
        }
    ## gather sessions from jason metadata
    def gather_all_sessions(self):
        data_dict = {'date': [],
                    'animal': [],
                    'task': [],             
                    'poke1_ts':[],
                    'poke2_ts': [],
                    'TOP1_ts': [],
                    'poke1_t0':[],
                    'poke2_t0': [],
                    'TOP1_t0': [],
                    'condition':[]}
        # list of dates for analysis
        data_path = Path(self.path).expanduser()
        # populate dict with metadata and timestamps
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in use_animals:
                for task in os.listdir(data_path / date / ani):
                    data_paths = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv'))]
                    data_paths = [i for i in data_paths if 'spout' not in i]
                    data_paths = [i for i in data_paths if 'REYE' not in i]
                    data_paths = [i for i in data_paths if 'Ephys' not in i]
                    data_paths = [i for i in data_paths if 'WORLD' not in i]
                    if data_paths != []:
                        _, name = os.path.split(data_paths[1])
                        split_name = name.split('_')
                        
                        data_dict['date'].append(split_name[0])
                        data_dict['animal'].append(split_name[1])
                        data_dict['task'].append(split_name[4])
                        data_dict['condition'].append(split_name[2])
                    for ind, csv in enumerate(data_paths):
                        self.timestamp_path = csv
                        time = self.read_timestamp_file()
                        _, name = os.path.split(data_paths[ind])
                        split_name = name.split('_')
                        data_dict[split_name[6] +'_ts'].append(time)
                        data_dict[split_name[6] +'_t0'].append(time[0])
        print('gather_all_sessions')
        self.all_sessions = pd.DataFrame.from_dict(data_dict)
## DLC        
    def change_dlc_project(self, project_paths):
        """
        project_paths should be a dict like... {'light':/path/to/config.yaml, 'dark':/path/to/config.yaml}
        """
        # update object
        self.dlc_project = project_paths
        # also update config
        self.generic_camconfig['paths']['dlc_projects'][self.camname] = self.dlc_project
##3 run dlc on sessions    
    def preprocess(self):
        for date in self.dates_list:
            date_dir = os.path.join(self.path, date)
            for animal in [i for i in list(self.metadata[date].keys())]:
                animal_dir = os.path.join(date_dir, animal)
                camconfig = self.generic_camconfig
                camconfig['animal_directory'] = animal_dir
                isdark = ('dark' in animal_dir)
                if not isdark:
                    print('using light network')
                    #camconfig['paths']['dlc_projects'][self.camname] = self.dlc_project['light']
                elif isdark:
                    print('using dark network')
                    #camconfig['paths']['dlc_projects'][self.camname] = self.dlc_project['dark']
                for recording_name in [k for k,v in self.metadata[date][animal].items()]:
                    recording_dir = os.path.join(animal_dir, recording_name)
                    print(recording_dir)
                    name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', recording_dir) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze','._'])][0])[1])[0].split('_')[:-2])
                    tc = Topcam(config=camconfig, recording_name=name, recording_path=recording_dir, camname=self.camname)
                    tc.pose_estimation()
                    #tc.gather_camera_files()
                    #tc.pack_position_data()
                    #tc.pack_video_frames()
                    #tc.pt_names = list(tc.xrpts['point_loc'].values)
                    #tc.filter_likelihood()
                    #tc.get_head_body_yaw()
                    #tc.save_params()    
## process each session
    def process(self, videos=False):
        self.gather_all_sessions()
        #print('check')
        for trial_ind, trial_row in tqdm(self.all_sessions.iterrows()):
            #print(trial_row)
            try:
                # analyze each trial
                #print(trial_row)
                trial = AvoidanceSession(s_input= trial_row, path_input=self.path, metadata_input=self.metadata,task=self.tasktype)
                print('trial')

            
                dlc_h5 = find('*'+str(trial_row['date'])+'*'+str(trial_row['animal'])+'*'+str(trial_row['task'])+'*.h5', self.path)
                dlc_h5 = [i for i in dlc_h5 if 'REYE' not in i]
                #print(dlc_h5)
                if dlc_h5 == []:
                    continue
                trial_path, _ = os.path.split(dlc_h5[0])
                trial_path = trial_path.replace(os.sep, '/')
                trial_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', trial_path) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze','._'])][0])[1])[0].split('_')[:-1])
                print(trial_name)
                trial.add_tracking(trial_name, trial_path)
                if self.is_pillar_avoidance:
                    trial.pillar_avoidance()
                elif self.is_non_obstacle:
                    trial.non_obstacle()
                # make short diagnostic video
                if videos:
                    trial.make_videos()
            except:
                pass


## Process individual recoring sessions

class AvoidanceSession(BaseInput):
    def __init__(self, s_input, path_input, metadata_input,task):
        self.s = s_input # series from dataframe of all trials
        self.tasktype = task
        self.likelihood_thresh = 0.75
        self.dist_across_arena = 60.96 # cm between bottom-right and bottom-left pillar new rig is 24 in long
        self.path = path_input
        self.camname = 'TOP1'
        self.shared_metadata = metadata_input

        #self.num_clusters_to_use = self.shared_metadata[self.s['date']][self.s['animal']][str(self.s['task'])]['num_positions']
        self.session_path = os.path.join(*[self.path, str(self.s['date']), str(self.s['animal']),str(self.s['task'])])
        #self.vidpath = find('*'+str(self.s['date'])+'*'+self.s['animal']+'*'+str(self.s['task'])+'*.avi', self.session_path)[0]

        self.generic_camconfig = {
            'internals': {
                'follow_strict_naming': False,
                'likelihood_threshold': 0.75,
                'multianimal_top_project':False
            }
        }
    
    def make_task_df(self):
        #len of object is one 
        num_odd_trials = np.min([len(self.s['poke1_ts']), len(self.s['poke2_ts'])])
        df1 = pd.DataFrame([])
        count =  -1 
        print(num_odd_trials)
        for c in range(num_odd_trials):
            
            # odd
            count += 1
            df1.at[count, 'first_poke'] = self.s['poke1_ts'][c]
            df1.at[count, 'second_poke'] = self.s['poke2_ts'][c]
            time = self.s['TOP1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
            vidframes = np.array(list(find_index_in_list(list(self.s['TOP1_ts']), list(time))))
            if len(vidframes) == 0:
                df1.drop(count)
                continue 
            df1.at[count, 'trial_timestamps'] = time.astype(object)
            df1.at[count, 'trial_vidframes'] = vidframes[0].astype(object)
            start_stop_inds = (int(np.where([self.s['TOP1_ts']==time[0]])[1]), int(np.where([self.s['TOP1_ts']==time[-1]])[1]))
            for pos in list(self.positions['point_loc'].values):
                df1.at[count, pos] = np.array(self.positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
            df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
          
            # even
            count += 1
            if c+1 < len(self.s['poke1_ts']):
                df1.at[count, 'first_poke'] = self.s['poke2_ts'][c]
                df1.at[count, 'second_poke'] = self.s['poke1_ts'][c+1]
                time = self.s['TOP1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
                vidframes = np.array(list(find_index_in_list(list(self.s['TOP1_ts']), list(time))))
                if len(vidframes) == 0:
                    df1.drop(count)
                    continue 
                df1.at[count, 'trial_timestamps'] = time.astype(object)
                df1.at[count, 'trial_vidframes'] = vidframes.astype(object)
                start_stop_inds = (int(np.where([self.s['TOP1_ts']==time[0]])[1]), int(np.where([self.s['TOP1_ts']==time[-1]])[1]))
                for pos in list(self.positions['point_loc'].values):
                    df1.at[count, pos] = np.array(self.positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
                df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
               
        df1['animal'] = self.s['animal']; df1['date'] = self.s['date']; df1['task'] = self.s['task'];  df1['condition'] = self.s['condition']
        if self.tasktype == 'non_obstalce':
            print('non_obstalce')


        print('df made')
        
        self.data = df1
        self.data.to_hdf(os.path.join(self.session_path,('test' + self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

        

    ## format frames 
    def format_frames_oa(self):
        #open avi file
        print(self.vidpath)
        vidread = cv2.VideoCapture(self.vidpath)
        # empty array that is the target shape
        # should be number of frames x downsampled height x downsampled width
        all_frames = np.empty([int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)),
                            int(vidread.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            int(vidread.get(cv2.CAP_PROP_FRAME_WIDTH))], dtype=np.uint8)
        # iterate through each frame
        for frame_num in tqdm(range(0,int(vidread.get(cv2.CAP_PROP_FRAME_COUNT)))):
            # read the frame in and make sure it is read in correctly
            ret, frame = vidread.read()
            if not ret:
                break
            # convert to grayyscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # add the downsampled frame to all_frames as int8
            all_frames[frame_num,:,:] = frame.astype(np.int8)
            self.video_frames = all_frames
        print('saved_frames')


        
    def add_tracking(self, name, path):
        tc = Topcam(self.generic_camconfig, name, path, self.camname)
        print('gather camera files')
        #print(self.camname)
        tc.gather_camera_files()
        print('pack_position_data')
        tc.pack_position_data()
        print('filter_likelihood')
        tc.filter_likelihood()
        #tc.pack_video_frames()
        self.positions = tc.xrpts
        #self.frames = tc.xrframes
        self.session_name = name
        self.session_path = path
    def deinterlace(vidfile, savepath, exp_fps=30, quiet=True):
        """ Deinterlace a video.

        Parameters
        ----------
        path : str
            Path to the video file.
        savepath : str
            Path to save the new video file. Default is None.
        rotate : bool
            Whether to rotate the video 180 degrees. Default is True.
        exp_fps : int
            Expected frame rate of the video. If the video matches
            this frame rate (in Hz), it will be deinterlaced. Otherwise,
            it will be skipped. Default is 30 Hz.
        quiet : bool
            Whether to suppress the output from ffmpeg. Default is True.

        Returns
        -------
        savepath : str
            Path to the new video file.

        """

        # Open video, get frame count and rate
        cap = cv2.VideoCapture(vidfile)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Skip this video if it doesn't match the expected frame rate.
        if fps != exp_fps:
            return

        # Create the FFMPEG commands for video rotation.
        vf_val = 'yadif=1:-1:0, vflip, hflip, scale=640:480'

        # Create the full FFMPEG command
        cmd = ['ffmpeg', '-i', vidfile, '-vf', vf_val, '-c:v', 'libx264',
              '-preset', 'slow', '-crf', '19', '-c:a', 'aac', '-b:a',
              '256k', '-y', savepath]

        # Set the log level
        if quiet is True:
            cmd.extend(['-loglevel', 'quiet'])

        # Run the FFMPEG command.
        subprocess.call(cmd)

        return savepath

    def convert_pxls_to_dist(self):
        x_cols = [i for i in self.data.columns.values if '_x' in i]
        y_cols = [i for i in self.data.columns.values if '_y' in i]
        if self.tasktype == 'non_obstalce':
            x_cols = [i for i in x_cols if 'obstacle' not in i]
            print(x_cols)
            y_cols = [i for i in y_cols if 'obstacle' not in i]
        for i in range(len(x_cols)):
            self.data[x_cols[i]+'_cm'] = self.data.loc[:,x_cols[i]] / self.pxls2cm
            self.data[y_cols[i]+'_cm'] = self.data.loc[:,y_cols[i]] / self.pxls2cm
    def get_median_trace(self,df):
        fake_time = np.linspace(0,1,100)
        all_nose_positions = np.zeros([len(df), 2, 100])
        count =  0
        for ind, row in df.iterrows():
            xT = np.linspace(0,1,len(row['nose_x_cm'])); yT = np.linspace(0,1,len(row['nose_y_cm']))
            all_nose_positions[count,0,:] = interp1d(xT, row['nose_x_cm'], bounds_error=False)(fake_time)
            all_nose_positions[count,1, :] = interp1d(yT, row['nose_y_cm'], bounds_error=False)(fake_time)
            count += 1
        median_trace = np.nanmedian(all_nose_positions, axis=0)
        for ind, row in df.iterrows():
            df.at[ind,'median_x_cm'] = median_trace[0,:].astype(object); df.at[ind,'median_y_cm'] = median_trace[1,:].astype(object)
    
    
    def non_obstacle(self):
        print(self.tasktype) 
        self.make_task_df()
        self.data = self.data[self.data['trial_vidframes'].notna()]
        dist_to_posts = np.nanmedian(self.data['arenaTR_x'].iloc[0],0) - np.nanmedian(self.data['arenaTL_x'].iloc[0],0)
        self.pxls2cm = dist_to_posts/self.dist_across_arena
        self.convert_pxls_to_dist()
        print('pxl')
        

        # label odd/even trials (i.e. moving leftwards or moving rightwards?)
        for ind,row in self.data.iterrows():
            nose_points = row['nose_x_cm'].astype(float)
            nose_points = nose_points[~np.isnan(nose_points)]
            if np.nanmean(nose_points[:10])<= 20:
                self.data.at[ind,'odd'] = 'left'
            elif np.nanmean(nose_points[:10]) >=20:
                self.data.at[ind,'odd'] = 'right'

        


        ## mean port and arena
        port_arena_list = list_columns(self.data,['arena','leftportT','rightportT'])
        port_arena_list = [i for i in port_arena_list if 'cm' in i]
        for pos in port_arena_list:
           for ind,row in self.data.iterrows():
                self.data.at[ind,pos] = np.mean(row[pos])

        port_arena_list = list_columns(self.data,['arena','leftportT','rightportT'])
        port_arena_list = [i for i in port_arena_list if 'cm' in i]
        for pos in port_arena_list:
           self.data[pos] = self.data[pos].mean()
           
        print('mean')
        self.data.to_hdf(os.path.join(self.session_path,('test' + self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        keys = ['nose','leftear','rightear','spine','midspine','tailbase']
        keys_list = list_columns(self.data,keys)
        keys_list= [col for col in keys_list if 'likelihood' not in col]
        keys_list= [col for col in keys_list if 'lind' not in col]
        for ind, row in self.data.iterrows(): 
            """interpolate and smooth key points
                interpolate across nans 
                gausian smooth sigma 3 """
            for key in keys_list:
                interp_smooth = gaussian_filter1d(interpolate_array(row[key]),3)
                self.data.at[ind,key] = interp_smooth.astype(float)
            


            if row['odd'] == 'left':
                nose_list = row['nose_x_cm'] 
                odd_ind = np.argmax(nose_list>(self.data.leftportT_x_cm.unique()+5)) ## change cut off to be based of port position
                for key in keys_list:
                    self.data.at[ind,'ts_' + key] = row[key][odd_ind:]
                #use odd_ind to index into obstacle 
                # iterate over columns list  

                #create gt_obstacle points
            else: 
                nose_list = row['nose_x_cm']
                even_ind = np.argmax(nose_list<(self.data.rightportT_x_cm.unique()-5))
                for key in keys_list:
                    self.data.at[ind,'ts_' + key] = row[key][even_ind:]
        print('trial_start_smooth')

        for ind,row in self.data.iterrows(): 
            dist = np.nansum(np.abs(np.diff(row['ts_nose_x_cm'])))
            self.data.at[ind,'dist'] = dist
        print('dist')

        

        

        self.data['time'] = self.data['len']/60
       

        self.data.to_hdf(os.path.join(self.session_path,('non_obstacle' + self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        print('saved_df')



    def pillar_avoidance(self):
        self.make_task_df()
      
        self.data = self.data[self.data['trial_vidframes'].notna()]
        

       
        ## convert pxl to cm 
        right = np.nanmedian([np.nanmedian(self.data['arenaTR_T_x'].iloc[0],0) ,np.nanmedian(self.data['arenaBR_T_x'].iloc[0],0) ])
        left = np.nanmedian([np.nanmedian(self.data['arenaTL_T_x'].iloc[0],0) ,np.nanmedian(self.data['arenaBL_T_x'].iloc[0],0) ])
        dist_to_posts = np.nanmedian([right - left],0)
        self.pxls2cm = dist_to_posts/self.dist_across_arena
        self.convert_pxls_to_dist()
        self.data.to_hdf(os.path.join(self.session_path, ('pxl_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        print('pxl')

        ## the median postition of arena,port,spout
        columns_list  = [col for col in self.data.columns.tolist()if 'likelihood' not in col]
        arena_list = [col for col in columns_list if 'arena' in col]

        for item in arena_list:
            self.data[item] = np.nanmedian(flatten_list_of_arrays(self.data[item].to_list()),0)

        port_list = [col for col in columns_list if 'port' in col]
        for item in port_list:
            self.data[item] = np.nanmedian(flatten_list_of_arrays(self.data[item].to_list()),0)

        spout_list = [col for col in columns_list if 'spout' in col]
        for item in spout_list:
            self.data[item] = np.nanmedian(flatten_list_of_arrays(self.data[item].to_list()),0)
        print('median postition of arena,port,spout')

        ## get center of port and spout
        # Define all column groups
        port_groups = {
            'leftportT': ['leftportT_1', 'leftportT_2', 'leftportT_3', 'leftportT_4'],
            'leftportB': ['leftportB_1', 'leftportB_2', 'leftportB_3', 'leftportB_4'],
            'rightportT': ['rightportT_1', 'rightportT_2', 'rightportT_3', 'rightportT_4'],
            'rightportB': ['rightportB_1', 'rightportB_2', 'rightportB_3', 'rightportB_4'],
            'rightspout': ['rightspout_1', 'rightspout_2', 'rightspout_3', 'rightspout_4'],
            'leftspout': ['leftspout_1', 'leftspout_2', 'leftspout_3', 'leftspout_4']
        }

        # Calculate means for each port group
        for port, prefixes in port_groups.items():
            x_cols = [f'{prefix}_x_cm' for prefix in prefixes]
            y_cols = [f'{prefix}_y_cm' for prefix in prefixes]

            # Convert to float and calculate means
            self.data[x_cols] = self.data[x_cols].astype(float)
            self.data[y_cols] = self.data[y_cols].astype(float)

            self.data[f'{port}_x_cm'] = self.data[x_cols].mean(axis=1, skipna=True)
            self.data[f'{port}_y_cm'] = self.data[y_cols].mean(axis=1, skipna=True)
        print('center of port and spout')

        # calculate head cen
        for ind,row in self.data.iterrows():
            head_cen_x = np.nanmedian([row.led_x_cm.astype(float),row.camera_1_x_cm.astype(float),
                                       row.camera_2_x_cm.astype(float),row.camera_3_x_cm.astype(float)],0) 
            head_cen_y = np.nanmedian([row.led_y_cm.astype(float),row.camera_1_y_cm.astype(float),
                                       row.camera_2_y_cm.astype(float),row.camera_3_y_cm.astype(float)],0) 
            self.data.at[ind,'head_cen_x_cm'] = head_cen_x.astype(object)
            self.data.at[ind,'head_cen_y_cm'] = head_cen_y.astype(object)
        print('head center')
    
    
        for ind,row in self.data.iterrows():
            nose_points = row['head_cen_x_cm'].astype(float)
            nose_points = nose_points[~np.isnan(nose_points)]

            if np.nanmean(nose_points[:10])<= 20:
                self.data.at[ind,'odd'] = 'left'
            elif np.nanmean(nose_points[:10]) >=20:
                self.data.at[ind,'odd'] = 'right'
        print('odd_even')
        self.data.to_hdf(os.path.join(self.session_path, ('odd_even_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

        

      

           


        
        ## get index of obstacle,bodyparts after mouse reaches a ceartin x postion Trial start

        # get list of columns need for re indexing
        keys = ['head_cen','leftear','rightear','spine','midspine','tailbase','trial_timestamps','trial_vidframes']
        keys_list = list_columns(self.data,keys)
        keys_list= [col for col in keys_list if 'likelihood' not in col]
        keys_list= [col for col in keys_list if 'lind' not in col]
 
        # check if odd or even trial
        #  get first index when nose crosses a distance thresh hold
        #trail start = ts
        ##odd tiral at 16 cm even at 56 cm     
        for ind, row in self.data.iterrows(): 
            if row['odd'] == 'left':
                nose_list = row['head_cen_x_cm'] 
                odd_ind = np.argmax(nose_list>(self.data.leftspout_x_cm.unique()+5))
                for key in keys_list:
                    if len(row[key].shape)==2:
                        self.data.at[ind,'ts_' + key] = row[key][0][odd_ind:]
                    else:


                        
                        self.data.at[ind,'ts_' + key] = row[key][odd_ind:]

            
                #use odd_ind to index into obstacle 
                # iterate over columns list  

                #create gt_obstacle points
            else: 
                nose_list = row['head_cen_x_cm']
                even_ind = np.argmax(nose_list<(self.data.rightspout_x_cm.unique()-5))
                for key in keys_list:
                    if len(row[key].shape)==2:
                        self.data.at[ind,'ts_' + key] = row[key][0][even_ind:]
                    else:
                        self.data.at[ind,'ts_' + key] = row[key][even_ind:]
                   
        print('trial_start')

        keys = ['head_cen','leftear','rightear','spine','midspine','tailbase']
        keys_list = list_columns(self.data,keys)
        keys_list= [col for col in keys_list if 'likelihood' not in col]
        keys_list= [col for col in keys_list if 'ts_'  in col]
        for ind, row in self.data.iterrows(): 
            """interpolate and smooth key points
                interpolate across nans 
                gausian smooth sigma 3 """
            for key in keys_list:
                interp_smooth = gaussian_filter1d(interpolate_array(row[key]),3)
                #resample = signal.resample(interp_smooth[~np.isnan(interp_smooth)],200)
                self.data.at[ind, key] = interp_smooth.astype(object)
        print('smooth_ts')

        for ind,row in self.data.iterrows(): 
            dist = np.nansum(np.abs(np.diff(row['ts_head_cen_x_cm'])))
            self.data.at[ind,'dist'] = dist
        #self.data = self.data.loc[self.data['dist']<80]
        print('dist')
        
        #self.data.to_hdf(os.path.join(self.session_path, ('test1_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

        ##  median point at gt obstacle 
        obstacle_cols = list_columns(self.data,['obstacle'])
        obstacle_cols = [col for col in obstacle_cols if 'likelihood' not in col]

        for ind, row in self.data.iterrows():
            nose_list = row['head_cen_x_cm']
            middle_time = np.where((nose_list > 25) & (nose_list < 50))
            if len(middle_time[0]) == 0:
                self.data = self.data.drop(ind)
            else:
                first,last = [middle_time[0][i] for i in (0, -1)] 
               
                for col in obstacle_cols:
                    trace = row[col][first:last]
                    #trace = trace.astype('float')
                    #kernel = np.ones(len(trace))
                    #trace = interpolate_replace_nans(trace,kernel)
                    self.data.at[ind,'gt_'+ col]= np.nanmean(trace)
        for ind, row in self.data.iterrows():
  
            xvals = np.stack([row['gt_obstacleTL_x'], row['gt_obstacleTR_x'], row['gt_obstacleBL_x'], row['gt_obstacleBR_x']])
            xvals_cm = np.stack([row['gt_obstacleTL_x_cm'], row['gt_obstacleTR_x_cm'], row['gt_obstacleBL_x_cm'], row['gt_obstacleBR_x_cm']])
            self.data.at[ind,'gt_obstacle_cen_x' ] = np.mean(xvals)
            self.data.at[ind,'gt_obstacle_cen_x_cm' ] = np.mean(xvals_cm)

            yvals = np.stack([row['gt_obstacleTL_y'], row['gt_obstacleTR_y'], row['gt_obstacleBL_y'], row['gt_obstacleBR_y']])
            yvals_cm = np.stack([row['gt_obstacleTL_y_cm'], row['gt_obstacleTR_y_cm'], row['gt_obstacleBL_y_cm'], row['gt_obstacleBR_y_cm']])
            self.data.at[ind,'gt_obstacle_cen_y' ] = np.mean(yvals)
            self.data.at[ind,'gt_obstacle_cen_y_cm' ] = np.mean(yvals_cm)
        print('ob_cen')
        self.data.to_hdf(os.path.join(self.session_path, ('test1_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

        #start(self.data)
        #print('start')
        ## distance calcs 
        #distance_calcs(self.data)
        #print('distance')
        #self.data.to_hdf(os.path.join(self.session_path, ('test1_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        
    
        self.data = cluster_obstacle(self.data,3)
        print('cluster')
        self.data.to_hdf(os.path.join(self.session_path, ('cluster_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        #self.data.to_hdf(os.path.join(self.session_path, ('test_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

        #heading_calcs(self.data)
        #print('heading')

        #deveation(self.data)
        #print('deveation')

        lateral_error(self.data)
        print('lateral_error')

        df_tortuosity(self.data)
        print('tortuosity')

        head_angle_velocity(self.data)
        print('head_angle_velocity')



        self.data.to_hdf(os.path.join(self.session_path, ('test_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        


        # drop any transits that were really slow (only drop slowest 10% of transits)
        #time_thresh = self.data['len'].quantile(0.9)
        self.data['time'] = self.data['len']/60
        #self.data = self.data[self.data['len']<time_thresh]
        self.processed_data =  self.data.drop(self.data.filter(regex='likelihood').columns,axis = 1)

        print('saving' + self.session_name + 'raw')
        self.raw_data =  self.data
        self.raw_data.to_hdf(os.path.join(self.session_path, ('raw_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        print('saving' + self.session_name + ' processed')
        print(self.session_path)
        self.processed_data.to_hdf(os.path.join(self.session_path,('processed_' + self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
      