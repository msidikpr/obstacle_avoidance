## script for processing obstacle avoidance session 
## Notes: 8/23/22(Works with ephys0 enviroment)
import json, os, cv2
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import glob
import h5py
from astropy.convolution import interpolate_replace_nans


import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
import warnings
warnings.filterwarnings('ignore')

from src.base import BaseInput
from src.topcam import Topcam
from src.utils.auxiliary import flatten_series, find_index_in_list
from src.utils.path import find
from utils.base_functions import format_frames,flatten_column,list_columns


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
        self.camname = 'top1'
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
                    'top1_ts': [],
                    'poke1_t0':[],
                    'poke2_t0': [],
                    'top1_t0': []}
        # list of dates for analysis
        data_path = Path(self.path).expanduser()
        # populate dict with metadata and timestamps
        for date in self.dates_list:
            use_animals = [k for k,v in self.metadata[date].items()]
            for ani in use_animals:
                for task in os.listdir(data_path / date / ani):
                    data_paths = [str(i) for i in list((data_path / date / ani/ task).rglob('*.csv'))]
                    data_paths = [i for i in data_paths if 'spout' not in i]
                    if data_paths != []:
                        _, name = os.path.split(data_paths[1])
                        split_name = name.split('_')
                        data_dict['date'].append(split_name[0])
                        data_dict['animal'].append(split_name[1])
                        data_dict['task'].append(split_name[4])
                    for ind, csv in enumerate(data_paths):
                        self.timestamp_path = csv
                        time = self.read_timestamp_file()
                        _, name = os.path.split(data_paths[ind])
                        split_name = name.split('_')
                        data_dict[split_name[5] +'_ts'].append(time)
                        data_dict[split_name[5] +'_t0'].append(time[0])
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
                trial = AvoidanceSession(trial_row, self.path, self.metadata,self.tasktype)

            
                dlc_h5 = find('*'+str(trial_row['date'])+'*'+str(trial_row['animal'])+'*'+str(trial_row['task'])+'*.h5', self.path)
                
                if dlc_h5 == []:
                    continue
                trial_path, _ = os.path.split(dlc_h5[0])
                trial_path = trial_path.replace(os.sep, '/')
                trial_name = '_'.join(os.path.splitext(os.path.split([i for i in find('*.avi', trial_path) if all(bad not in i for bad in ['plot','IR','rep11','betafpv','side_gaze','._'])][0])[1])[0].split('_')[:-1])
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
        self.dist_across_arena = 48.26 # cm between bottom-right and bottom-left pillar
        self.path = path_input
        self.camname = 'top1'
        self.shared_metadata = metadata_input

        #self.num_clusters_to_use = self.shared_metadata[self.s['date']][self.s['animal']][str(self.s['task'])]['num_positions']
        self.session_path = os.path.join(*[self.path, str(self.s['date']), str(self.s['animal']),str(self.s['task'])])
        self.vidpath = find('*'+str(self.s['date'])+'*'+self.s['animal']+'*'+str(self.s['task'])+'*.avi', self.session_path)[0]

        self.generic_camconfig = {
            'internals': {
                'follow_strict_naming': False,
                'likelihood_threshold': 0.75
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
            time = self.s['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
            vidframes = np.array(list(find_index_in_list(list(self.s['top1_ts']), list(time))))
            if len(vidframes) == 0:
                df1.drop(count)
                continue 
            df1.at[count, 'trial_timestamps'] = time.astype(object)
            df1.at[count, 'trial_vidframes'] = vidframes[0].astype(object)
            start_stop_inds = (int(np.where([self.s['top1_ts']==time[0]])[1]), int(np.where([self.s['top1_ts']==time[-1]])[1]))
            for pos in list(self.positions['point_loc'].values):
                df1.at[count, pos] = np.array(self.positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
            df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
          
            # even
            count += 1
            if c+1 < len(self.s['poke1_ts']):
                df1.at[count, 'first_poke'] = self.s['poke2_ts'][c]
                df1.at[count, 'second_poke'] = self.s['poke1_ts'][c+1]
                time = self.s['top1_ts']; time = time[time > df1.loc[count,'first_poke']]; time = time[time < df1.loc[count,'second_poke']]
                vidframes = np.array(list(find_index_in_list(list(self.s['top1_ts']), list(time))))
                if len(vidframes) == 0:
                    df1.drop(count)
                    continue 
                df1.at[count, 'trial_timestamps'] = time.astype(object)
                df1.at[count, 'trial_vidframes'] = vidframes.astype(object)
                start_stop_inds = (int(np.where([self.s['top1_ts']==time[0]])[1]), int(np.where([self.s['top1_ts']==time[-1]])[1]))
                for pos in list(self.positions['point_loc'].values):
                    df1.at[count, pos] = np.array(self.positions.loc[start_stop_inds[0]:start_stop_inds[1], pos]).astype(object)
                df1.at[count, 'len'] = start_stop_inds[1] - start_stop_inds[0]
               
        df1['animal'] = self.s['animal']; df1['date'] = self.s['date']; df1['task'] = self.s['task']
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
        tc.gather_camera_files()
        tc.pack_position_data()
        tc.filter_likelihood()
        #tc.pack_video_frames()
        self.positions = tc.xrpts
        #self.frames = tc.xrframes
        self.session_name = name
        self.session_path = path

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
        dist_to_posts = np.median(self.data['arenaTR_x'].iloc[0],0) - np.median(self.data['arenaTL_x'].iloc[0],0)
        self.pxls2cm = dist_to_posts/self.dist_across_arena
        self.convert_pxls_to_dist()
        print('pxl')
        # label odd/even trials (i.e. moving leftwards or moving rightwards?)
        for ind,row in self.data.iterrows():
            if np.nanmean(row['nose_x_cm'][:10]) <= 20:
                self.data.at[ind,'odd'] = True 
            elif np.nanmean(row['nose_x_cm'][:10]) >=20:
                self.data.at[ind,'odd'] = False

        

        ## take median of arena points
        arena_cols = [col for col in self.data.columns if 'arena' in col]
        arena_cols =[col for col in arena_cols if 'likelihood' not in col]
        for col in arena_cols:
            for ind,row in self.data.iterrows():
                self.data.at[ind,col] = np.mean(row[col])
        print('arena_median')

        ## take median of ports
        port_list =list_columns(self.data,['leftportT','rightportT'])
        port_list = [i for i in port_list if 'cm' in i]
        for pos in port_list:
            for ind,row in self.data.iterrows():
                self.data.at[ind,pos] = np.mean(row[pos])
        print('ports_median')

        ## mean port and arena
        port_arena_list = list_columns(self.data,['arena','leftportT','rightportT'])
        port_arena_list = [i for i in port_arena_list if 'cm' in i]
        for pos in port_arena_list:
           self.data['mean_'+pos] = self.data[pos].mean()
        print('mean')
        self.data.to_hdf(os.path.join(self.session_path,('test' + self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')


    def pillar_avoidance(self):
        self.make_task_df()
        #self.format_frames_oa()
        ##drop rows 
        self.data = self.data[self.data['trial_vidframes'].notna()]
        

       
        ## convert pxl to cm 
        dist_to_posts = np.median(self.data['arenaTR_x'].iloc[0],0) - np.median(self.data['arenaTL_x'].iloc[0],0)
        self.pxls2cm = dist_to_posts/self.dist_across_arena
        self.convert_pxls_to_dist()
        print('pxl')
        #self.data.to_hdf(os.path.join(self.session_path, ('df_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')

        # label odd/even trials (i.e. moving leftwards or moving rightwards?)
        for ind,row in self.data.iterrows():
            if np.nanmean(row['nose_x_cm'][:10]) <= 20:
                self.data.at[ind,'odd'] = True 
            elif np.nanmean(row['nose_x_cm'][:10]) >=20:
                self.data.at[ind,'odd'] = False

        

        ## take median of arena points
        arena_cols = [col for col in self.data.columns if 'arena' in col]
        arena_cols =[col for col in arena_cols if 'likelihood' not in col]
        for col in arena_cols:
            for ind,row in self.data.iterrows():
                self.data.at[ind,col] = np.mean(row[col])
        print('arena_median')

        ## take median of ports
        port_list =list_columns(self.data,['leftportT','rightportT'])
        port_list = [i for i in port_list if 'cm' in i]
        for pos in port_list:
            for ind,row in self.data.iterrows():
                self.data.at[ind,pos] = np.mean(row[pos])
        print('ports_median')

        ## mean port and arena
        port_arena_list = list_columns(self.data,['arena','leftportT','rightportT'])
        port_arena_list = [i for i in port_arena_list if 'cm' in i]
        for pos in port_arena_list:
           self.data['mean_'+pos] = self.data[pos].mean()
        print('mean')
        self.data.to_hdf(os.path.join(self.session_path, ('raw_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')


        
        ## get index of obstacle,bodyparts after mouse reaches a ceartin x postion Trial start

        # get list of columns need for re indexing
        keys = ['nose','leftear','rightear','spine','midspine','tailbase']
        keys_list = list_columns(self.data,keys)
        keys_list= [col for col in keys_list if 'likelihood' not in col]
        keys_list= [col for col in keys_list if 'lind' not in col]
 
        # check if odd or even trial
        #  get first index when nose crosses a distance thresh hold
        #trail start = ts
        ##odd tiral at 16 cm even at 56 cm     
        for ind, row in self.data.iterrows(): 
            if row['odd'] == True:
                nose_list = row['nose_x_cm'] 
                odd_ind = np.argmax(nose_list>16)
                for key in keys_list:
                    self.data.at[ind,'ts_' + key] = row[key][odd_ind:]
                #use odd_ind to index into obstacle 
                # iterate over columns list  

                #create gt_obstacle points
            else: 
                nose_list = row['nose_x_cm']
                even_ind = np.argmax(nose_list<56)
                for key in keys_list:
                    self.data.at[ind,'ts_' + key] = row[key][even_ind:]
        print('odd_even')
        self.data.to_hdf(os.path.join(self.session_path, ('test_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        print('test')

        ##  median point at gt obstacle 
        obstacle_cols = list_columns(self.data,['obstacle'])
        obstacle_cols = [col for col in obstacle_cols if 'likelihood' not in col]

        for ind, row in self.data.iterrows():
            nose_list = row['nose_x_cm']
            middle_time = np.where((nose_list > 25) & (nose_list < 50))
            if len(middle_time[0]) == 0:
                self.data = self.data.drop(ind)
            else:
                first,last = [middle_time[0][i] for i in (0, -1)] 
                # calculate median of each corner
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


        # drop any transits that were really slow (only drop slowest 10% of transits)
        time_thresh = self.data['len'].quantile(0.9)
        self.data = self.data[self.data['len']<time_thresh]
        self.raw_data =  self.data
        self.processed_data =  self.data.drop(self.data.filter(regex='likelihood').columns,axis = 1)
        print('saving' + self.session_name + 'raw')
        self.raw_data.to_hdf(os.path.join(self.session_path, ('raw_'+ self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')
        print('saving' + self.session_name + ' processed')
        self.processed_data.to_hdf(os.path.join(self.session_path,('processed_' + self.data['animal'].iloc[0]+'_'+str(self.data['date'].iloc[0])+'_'+str(self.data['task'].iloc[0])+'.h5')), 'w')