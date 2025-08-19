import os, subprocess, math, cv2
import numpy as np
import pandas as pd
import itertools 
from tqdm import tqdm
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy import stats
import sys
import seaborn as sns
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')

from pipeline.helper_functions import list_columns,interpolate_array,smooth
import warnings
warnings.filterwarnings('ignore')
from pipeline.gaze_shift_functions import *
from pipeline.helper_functions import interpolate_array,flatten_list_of_arrays



def plot_arena(df,axis,bottom = True,top = True,port=True):
    df =df.copy(deep=True)
    if bottom == True:
        arena_x_B = df[['arenaTL_B_x_cm', 'arenaTR_B_x_cm',
        'arenaBR_B_x_cm', 'arenaBL_B_x_cm','arenaTL_B_x_cm']].median().values.ravel('K')
        arena_y_B = df[['arenaTL_B_y_cm', 'arenaTR_B_y_cm',
        'arenaBR_B_y_cm', 'arenaBL_B_y_cm','arenaTL_B_y_cm']].median().values.ravel('K')
        axis.plot([arena_x_B[0],arena_x_B[1],arena_x_B[1],arena_x_B[0],arena_x_B[0]],
                  [arena_y_B[0],arena_y_B[0],arena_y_B[3],arena_y_B[3],arena_y_B[0]],c='k')
    if top == True:
        arena_x_T = df[['arenaTL_T_x_cm', 'arenaTR_T_x_cm',
        'arenaBR_T_x_cm', 'arenaBL_T_x_cm','arenaTL_T_x_cm']].median().values.ravel('K')
        arena_y_T = df[['arenaTL_T_y_cm', 'arenaTR_T_y_cm',
        'arenaBR_T_y_cm', 'arenaBL_T_y_cm','arenaTL_T_y_cm']].median().values.ravel('K')
        axis.plot([arena_x_T[0],arena_x_T[1],arena_x_T[1],arena_x_T[0],arena_x_T[3]],
                  [arena_y_T[0],arena_y_T[0],arena_y_T[3],arena_y_T[3],arena_y_T[3]],c='k')
    if port == True:
        port_x_T = df[['leftportT_x_cm', 'leftportB_x_cm',
        'rightportT_x_cm', 'rightportB_x_cm','leftspout_x_cm','rightspout_x_cm']].median().values.ravel('K')
        port_y_T = df[['leftportT_y_cm', 'leftportB_y_cm',
        'rightportT_y_cm', 'rightportB_y_cm','leftspout_y_cm','rightspout_y_cm']].median().values.ravel('K')
        axis.scatter([port_x_T[0],port_x_T[0],port_x_T[5],port_x_T[5],port_x_T[0],port_x_T[5]],
                  [port_y_T[0],port_y_T[1],port_y_T[0],port_y_T[1],port_y_T[4],port_y_T[4]],c=['k','k','k','k','red','purple'],s=[75,75,75,75,50,50],marker ='s' )
    

    
  
    

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
        axis.set_ylim([0,51]); axis.set_xlim([0, 61])
    return obstacle_x,obstacle_y

def gaze_shift_trials_figures_pdf(df, eyeT,dEye_dps,dHead,dGaze,gazeR_event, gazeL_event, OpenephysT0, 
                                 video_paths, output_filename="output.pdf", 
                                 output_dir=None):
    """
    Creates a PDF with combined figures for each row in the pandas DataFrame.
    Each figure includes:
    -1 page trail trajectory, dEye/dHead, dHead/dGaze, cumalitive head velocity 
    - 1/2 page 3 panels showing frames from 3 videos during R gaze shifts
    - 1/2 page 3 panels showing frames from 3 videos during L gaze shifts
    
    Parameters:
    - df: pandas DataFrame containing the data
    - gazeR_event, gazeL_event: Event data
    - OpenephysT0: Time offset
    - video_paths: List of 3 video file paths
    - output_filename: str, name of the output PDF file
    - output_dir: str, directory path to save the PDF
    - frames_to_avg: number of frames before/after to average (default ±2)
    """

     # Handle output directory
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_filename)
    else:
        output_path = output_filename
    
    # Create the PDF object
    with PdfPages(output_path) as pdf:
        fig,ax_trial = plt.subplots(2,2,figsize=(20, 15))
        
        # Iterate through each row of the DataFrame
        for ind, row in tqdm(df.iterrows()):
            # plot trial variables 
            fig,ax_trial = plt.subplots(2,2,figsize=(20, 15))

            gaze_data = process_original_plots(row, ax_trial[0,0], gazeR_event, gazeL_event, OpenephysT0,
                                               eyeT=eyeT,dEye_dps=dEye_dps,dHead=dHead,dGaze=dGaze)
            

            gazeR_trial_idx = gaze_data['gazeR_trial_idx']
            gazeL_trial_idx = gaze_data['gazeL_trial_idx']
            dEye_dps_trial = gaze_data['dEye_dps_trial']
            dHead_trial = gaze_data['dHead_trial']
            dGaze_trial = gaze_data['dGaze_trial']
            r_ind = gaze_data['r_ind']
            l_ind = gaze_data['l_ind']
            ts_trial_vidframes = gaze_data['ts_trial_vidframes']

          

            

            #ax_trial[0,0].plot(row.ts_head_cen_x_cm,row.ts_head_cen_y_cm)
            left_gaze = mlines.Line2D([], [], color='k',
                                  markersize=15, label='left_gaze')
            right_gaze = mlines.Line2D([], [], color='red',
                                  markersize=15, label='right_gaze')
            ax_trial[0,0].legend(handles = [left_gaze,right_gaze])
            ax_trial[0,1].plot(dHead_trial,c = 'blue')
            ax_trial[0,1].plot(dEye_dps_trial,c = 'purple')
            ax_trial[0,1].legend(["dHead", "dEye"])
            ax_trial[0,1].xaxis.set_major_formatter(ticker.FuncFormatter(frames_to_seconds))
            ax_trial[1,0].plot(dGaze_trial,c='green')
            ax_trial[1,0].plot(dHead_trial,c = 'blue')
            ax_trial[1,0].legend(["dGaze","dHead"])
            ax_trial[1,0].xaxis.set_major_formatter(ticker.FuncFormatter(frames_to_seconds))
            ax_trial[1,1].plot(np.cumsum(dHead_trial)/60)
            ax_trial[1,1].legend(['cumlative sum dHead'])
            ax_trial[1,1].xaxis.set_major_formatter(ticker.FuncFormatter(frames_to_seconds))

            if (r_ind == None) & (l_ind == None):
                print('no gaze')
                continue
            elif (r_ind != None) & (l_ind != None):
                print('gaze')
                #print(row.ts_head_cen_y_cm[np.array(l_ind)])
                ax_trial[0,0].scatter(row.ts_head_cen_x_cm[np.array(l_ind)],row.ts_head_cen_y_cm[np.array(l_ind)],c='black')
                ax_trial[0,0].scatter(row.ts_head_cen_x_cm[np.array(r_ind)],row.ts_head_cen_y_cm[np.array(r_ind)],c='red')
                ax_trial[0,1].scatter(gazeR_trial_idx,dEye_dps_trial[[gazeR_trial_idx]],c ='red')
                ax_trial[0,1].scatter(gazeL_trial_idx,dEye_dps_trial[[gazeL_trial_idx]],c ='black')
                ax_trial[1,0].scatter(gazeR_trial_idx,dGaze_trial[[gazeR_trial_idx]],c ='red')
                ax_trial[1,0].scatter(gazeL_trial_idx,dGaze_trial[[gazeL_trial_idx]],c ='black')
            elif r_ind == None:
                print('no gaze r')
                ax_trial[0,0].scatter(row.ts_head_cen_x_cm[np.array(l_ind)],row.ts_head_cen_y_cm[np.array(l_ind)],c='black')
                ax_trial[0,1].scatter(gazeL_trial_idx,dEye_dps_trial[[gazeL_trial_idx]],c ='black')
                ax_trial[1,0].scatter(gazeL_trial_idx,dGaze_trial[[gazeL_trial_idx]],c ='black')
            elif l_ind == None:
                print('no gaze 1')
                ax_trial[0,0].scatter(row.ts_head_cen_x_cm[np.array(r_ind)],row.ts_head_cen_y_cm[np.array(r_ind)],c='red')
                ax_trial[0,1].scatter(gazeR_trial_idx,dEye_dps_trial[[gazeR_trial_idx]],c ='red')
                ax_trial[1,0].scatter(gazeR_trial_idx,dGaze_trial[[gazeR_trial_idx]],c ='red')

            ax_trial[1,1].scatter(gazeR_trial_idx,np.cumsum(dHead_trial)[[gazeR_trial_idx]]/60,c ='red')
            ax_trial[1,1].scatter(gazeL_trial_idx,np.cumsum(dHead_trial)[[gazeL_trial_idx]]/60,c ='black')

            # Adjust layout to prevent overlap
            plt.tight_layout()
            plt.rcParams['font.size'] = 20  # Set the default font size to 12
        
        # Save this figure to the PDF
            pdf.savefig(fig, bbox_inches='tight')
        
        # Close the figure to free memory
            plt.close(fig)
            print(f"trial successfully created at: {os.path.abspath(output_path)}")

            if len(gazeR_trial_idx) == 0:
                Gaze_fig,lGaze_ax = get_averaged_frames_with_timeseries(title= 'lGaze',dEye_trial= dEye_dps_trial,dHeadTrial= dHead_trial,dGazeTrial=dGaze_trial,timeseries_idx=gazeL_trial_idx,
                                                                     timeseries_range=15, video_paths=video_paths,frame_numbers=ts_trial_vidframes[gazeL_trial_idx])
                pdf.savefig(lGaze_fig, bbox_inches='tight')
                plt.close(lGaze_fig)
                print(f"lGaze_trial successfully created at: {os.path.abspath(output_path)}")
            elif len(gazeL_trial_idx) == 0:
                rGaze_fig,rGaze_ax = get_averaged_frames_with_timeseries(title= 'rGaze',dEye_trial= dEye_dps_trial,dHeadTrial= dHead_trial,dGazeTrial=dGaze_trial,timeseries_idx=gazeR_trial_idx,timeseries_range=15,
                                                                      video_paths=video_paths,frame_numbers=ts_trial_vidframes[gazeR_trial_idx])
                pdf.savefig(rGaze_fig, bbox_inches='tight')
                plt.close(rGaze_fig)
                print(f"rGaze_trial successfully created at: {os.path.abspath(output_path)}")
            else:
        
            # rGaze 
                rGaze_fig,rGaze_ax = get_averaged_frames_with_timeseries(title= 'rGaze',dEye_trial= dEye_dps_trial,dHeadTrial= dHead_trial,dGazeTrial=dGaze_trial,timeseries_idx=gazeR_trial_idx,timeseries_range=15,
                                                                          video_paths=video_paths,frame_numbers=ts_trial_vidframes[gazeR_trial_idx])
                pdf.savefig(rGaze_fig, bbox_inches='tight')
                plt.close(rGaze_fig)
                print(f"rGaze_trial successfully created at: {os.path.abspath(output_path)}")
            # lGaze 
                lGaze_fig,lGaze_ax = get_averaged_frames_with_timeseries(title= 'lGaze',dEye_trial= dEye_dps_trial,dHeadTrial= dHead_trial,dGazeTrial=dGaze_trial,timeseries_idx=gazeL_trial_idx,
                                                                         timeseries_range=15, video_paths=video_paths,frame_numbers=ts_trial_vidframes[gazeL_trial_idx])
                pdf.savefig(lGaze_fig, bbox_inches='tight')
                plt.close(lGaze_fig)
                print(f"lGaze_trial successfully created at: {os.path.abspath(output_path)}")
            




def process_original_plots(row, ax, gazeR_event, gazeL_event, OpenephysT0,eyeT,dEye_dps,dHead,dGaze):
    """Process the original plots and return gaze shift indices"""
    ## correct timestamps to ephys board 
    trial_imu_corected = row.ts_trial_timestamps - OpenephysT0
    ts_trial_vidframes = row.ts_trial_vidframes
    
    ## find frames that gaze and comp movements 
    gazeR_ind, r_ind = find_event_frames(trial_imu_corected, gazeR_event)
    gazeL_ind, l_ind = find_event_frames(trial_imu_corected, gazeL_event)
    
    ## plot trial trajectories 
    ax.plot(row.ts_head_cen_x_cm, row.ts_head_cen_y_cm)
    left_gaze = mlines.Line2D([], [], color='k', markersize=15, label='left_gaze')
    right_gaze = mlines.Line2D([], [], color='red', markersize=15, label='right_gaze')
    ax.legend(handles=[left_gaze, right_gaze])
    ax.axis('off')

    ## plot arena and obstacle     
    plot_arena(row.to_frame().T, ax, bottom=True)
    obstacle_x = [row.gt_obstacleTL_x_cm,
                row.gt_obstacleTR_x_cm, row.gt_obstacleTR_x_cm,
                row.gt_obstacleTL_x_cm,
                row.gt_obstacleTL_x_cm]
    obstacle_y = [row.gt_obstacleTL_y_cm,
                row.gt_obstacleTL_y_cm, row.gt_obstacleBL_y_cm,
                row.gt_obstacleBL_y_cm,
                row.gt_obstacleTL_y_cm]

    ax.plot([obstacle_x[0], obstacle_x[1], obstacle_x[2], obstacle_x[3], obstacle_x[0]],
              [obstacle_y[0], obstacle_y[1], obstacle_y[2], obstacle_y[3], obstacle_y[0]], c='k')

    ## plot dgaze and head from trial     
    start_timestamp, start_index = find_closest(eyeT.flatten()[:-1], np.array(trial_imu_corected[0]))
    end_timestamp, end_index = find_closest(eyeT.flatten()[:-1], np.array(trial_imu_corected[-1]))
    dEye_dps_trial = dEye_dps[start_index:end_index]
    dHead_trial = dHead[start_index:end_index]
    dGaze_trial = dGaze[start_index:end_index]
    eyeT_trial = eyeT.flatten()[:-1][start_index:end_index]
    
    gazeL_trial_event, gazeR_trial_event, compL_trial_event, compR_trial_event, \
    gazeL_trial_idx, gazeR_trial_idx, compL_trial_idx, compR_trial_idx = classify_gaze_movements(
        dHead_trial, dGaze_trial, eyeT_trial)


    
    return {
        'dEye_dps_trial':dEye_dps_trial,
        'dHead_trial':dHead_trial,
        'dGaze_trial':dGaze_trial,
        'eyeT_trial':eyeT_trial,
        'gazeR_trial_idx': gazeR_trial_idx,
        'gazeL_trial_idx': gazeL_trial_idx,
        'trial_imu_corected': trial_imu_corected,
        'r_ind': r_ind,'l_ind':l_ind,
        'ts_trial_vidframes': ts_trial_vidframes


    }


def get_averaged_frames_with_timeseries(video_paths, frame_numbers, dEye_trial, dHeadTrial,dGazeTrial, title,
                                       timeseries_idx, timeseries_range,
                                       average_range=15, cols=4, figsize=(50, 50)):
    """
    Returns figure with video frames and time series plots using separate indexing for time series.
    
    Args:
        video_paths (list): List of video paths
        frame_numbers (list): Frame indices to plot (for videos)
        dEye_trial (np.array): Eye movement time series data
        dHeadTrial (np.array): Head movement time series data
        title (str): Title prefix for plots
        timeseries_idx (list): Indices of events in time series data
        timeseries_range (int): Range around timeseries_idx to plot
        average_range (int): Frames to average around target frame (for videos)
        cols (int): Number of columns (fixed to 4: videos + time series)
        figsize (tuple): Figure size
    
    Returns:
        fig: matplotlib figure object
        axes: 2D array of axes
    """
    if not video_paths:
        raise ValueError("At least one video path must be provided")
    if len(frame_numbers) != len(timeseries_idx):
        raise ValueError("frame_numbers and timeseries_idx must have same length")
    
    num_videos = len(video_paths)
    num_frames = len(frame_numbers)
    cols = 4  # Force 4 columns (3 videos + 1 time series)
    
    # Calculate rows needed
    rows = num_frames
    
    # Create figure with 4 columns
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Open video files
    caps = [cv2.VideoCapture(vp) for vp in video_paths]
    for i, cap in enumerate(caps):
        if not cap.isOpened():
            for c in caps:
                if c.isOpened():
                    c.release()
            raise ValueError(f"Could not open video at {video_paths[i]}")
    
    # Get video properties
    total_frames = [int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) for cap in caps]
    
    # Process each frame number
    for row_idx, (frame_num, ts_idx) in enumerate(zip(frame_numbers, timeseries_idx)):
        # Column 0-2: Video frames
        for vid_idx in range(num_videos):
            # Get frame range
            start_frame = max(0, frame_num - average_range)
            end_frame = min(total_frames[vid_idx]-1, frame_num + average_range)
            frame_range = list(range(start_frame, end_frame + 1))
            
            # Read and average frames
            frames = []
            for f in frame_range:
                caps[vid_idx].set(cv2.CAP_PROP_POS_FRAMES, f)
                ret, frame = caps[vid_idx].read()
                if ret:
                    frames.append(frame)
            
            # Plot video frame
            ax = axes[row_idx, vid_idx]
            if frames:
                avg_frame = cv2.cvtColor(np.mean(frames, axis=0).astype(np.uint8), cv2.COLOR_BGR2RGB)
                ax.imshow(avg_frame)
            ax.set_title(f"Video {vid_idx+1}\nFrames {start_frame}-{end_frame}\n{title} {row_idx}")
            ax.axis('off')
        
        # Column 3: Time series plot
        ts_ax = axes[row_idx, 3]
        
        # Determine time series window (ts_idx ± timeseries_range)
        ts_start = max(0, ts_idx - timeseries_range)
        ts_end = min(len(dEye_trial)-1, ts_idx + timeseries_range)
        
        # Create x-axis values centered at the event
        x_values = np.arange(ts_start, ts_end+1) - ts_idx
        
        # Plot time series
        ts_ax.plot(x_values, dEye_trial[ts_start:ts_end+1], label='Eye Movement', color='purple')
        ts_ax.plot(x_values, dHeadTrial[ts_start:ts_end+1], label='Head Movement', color='blue')
        ts_ax.plot(x_values, dGazeTrial[ts_start:ts_end+1], label='Gaze', color='green',alpha = .50)
        
        # Mark event time
        ts_ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        ts_ax.set_title(f"Time Series\nFrames {ts_start}-{ts_end}\n{title} {row_idx}")
        ts_ax.set_xlabel("Seconds relative to event")
        ts_ax.legend()
        ts_ax.grid(True)
        ts_ax.xaxis.set_major_formatter(ticker.FuncFormatter(frames_to_seconds))
        ts_ax.set_ylim(-1000,1000)
    
    # Release video captures
    for cap in caps:
        cap.release()
    
    plt.tight_layout()
    return fig, axes

def find_event_frames(video_timestamps, event_timestamps):
    """
    Find which events occurred during the video and their corresponding video frames.
    Returns (None, None) if no events are found within the video duration.
    
    Args:
        video_timestamps (array-like): Monotonically increasing array of timestamps for each video frame.
        event_timestamps (array-like): Array of timestamps when events occurred.
        
    Returns:
        tuple: (event_indices, frame_indices) or (None, None) if no events found
               - event_indices: Indices of the original event array that fall within video duration
               - frame_indices: Corresponding video frame indices for these events
    """
    video_ts = np.asarray(video_timestamps)
    event_ts = np.asarray(event_timestamps)
    
    if len(video_ts) == 0 or len(event_ts) == 0:
        return None, None
    
    # Find which events are within video duration and their original indices
    valid_mask = (event_ts >= video_ts[0]) & (event_ts <= video_ts[-1])
    event_indices = np.where(valid_mask)[0]
    valid_events = event_ts[valid_mask]
    
    # Return None if no valid events found
    if len(valid_events) == 0:
        return None, None
    
    # Find corresponding frame indices
    frame_indices = np.searchsorted(video_ts, valid_events, side='right') - 1
    frame_indices = np.clip(frame_indices, 0, len(video_ts) - 1)
    
    return event_indices.tolist(), frame_indices.tolist()

def find_closest(array, target):
    """
    Find the element in an array that is closest to the target float value.
    
    Parameters:
    array (list): List of numbers to search through
    target (float): The target value to find the closest element to
    
    Returns:
    The element in the array closest to the target value
    """
    if  array.size == 0:  # Handle empty array case
        raise ValueError("Input array must not be empty")
    
    # Initialize with first element
    closest = array[0]
    min_diff = abs(target - closest)
    
    for element in array[1:]:
        current_diff = abs(target - element)
        if current_diff < min_diff:
            min_diff = current_diff
            closest = element
    
    return closest,np.argwhere(array==closest)[0][0]

def frames_to_seconds(x, pos,fps = 60):
    return f"{x / fps:.2f}"

def create_gaze_pdf_report(df, imu_data, output_filename='gaze_report.pdf', output_dir='.'):
    """
    Create a PDF report with gaze analysis plots.
    
    Parameters:
    -----------
    df : DataFrame
        Input data containing trial information
    imu_data : dict
        IMU data dictionary with keys: ['dEye_dps', 'dHead', 'dGaze', 'eyeT', 'Openephys_time']
    output_filename : str, optional
        Name of the output PDF file (default: 'gaze_report.pdf')
    output_dir : str, optional
        Directory where the PDF will be saved (default: current directory)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    
    # Define gaze colormap (head-eye dominance)
    bounds = [-1, -0.5, -0.1, 0.1, 0.5, 1]
    colors = [
        (0, 0, 1, 1),        # Dark blue
        (0.5, 0.5, 1, 1),    # Lighter blue
        (0.4, 0.8, 0.8, 1),   # Light grey
        (1, 0.5, 0.5, 1),    # Light red
        (1, 0, 0, 1)         # Dark red
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)


    # Define speed colormap (0-50)
    speed_bounds = [0, 12.5, 25, 37.5, 50]
    speed_colors = [
        [0.8, 0.9, 0.7, 1],    # Light green
        [0.5, 0.8, 0.3, 1],     # Medium green
        [0.7, 0.4, 0.8, 1],     # Lavender
        [0.5, 0.1, 0.5, 1]      # Dark purple
    ]
    speed_cmap = ListedColormap(speed_colors)
    speed_norm = BoundaryNorm(speed_bounds, speed_cmap.N)

    
    with PdfPages(output_path) as pdf:
        rows_per_page = 9
        total_pages = int(np.ceil(len(df) / rows_per_page))
        
        for page_num in range(total_pages):
            fig, axes = plt.subplots(3, 3, figsize=(15, 15))
            axes = axes.flatten()
            
            start_idx = page_num * rows_per_page
            end_idx = min((page_num + 1) * rows_per_page, len(df))
            page_rows = df.iloc[start_idx:end_idx]
            
            for i, (ax, (_, row)) in enumerate(zip(axes, page_rows.iterrows())):
                try:
                    # Process IMU data for this trial
                    trial_imu_corrected = row.ts_trial_timestamps - imu_data['Openephys_time'][0]
                    
                    # Find closest timestamps
                    start_timestamp, start_index = find_closest(imu_data['eyeT'].flatten()[:-1], trial_imu_corrected[0])
                    end_timestamp, end_index = find_closest(imu_data['eyeT'].flatten()[:-1], trial_imu_corrected[-1])
                    
                    # Extract trial data
                    dEye_dps_trial = imu_data['dEye_dps'][start_index:end_index]
                    dHead_trial = imu_data['dHead'][start_index:end_index]
                    dGaze_trial = imu_data['dGaze'][start_index:end_index]
                    eyeT_trial = imu_data['eyeT'].flatten()[:-1][start_index:end_index]
                    
                    # Classify gaze movements
                    gaze_results = classify_gaze_movements(dHead_trial, dGaze_trial, eyeT_trial)
                    gazeL_trial_idx = gaze_results[4]
                    gazeR_trial_idx = gaze_results[5]
                    
                    # Compute onset ratios and colors
                    l_gaze_ratio = compute_onset_ratios(dEye_dps_trial, dHead_trial, gazeL_trial_idx)
                    r_gaze_ratio = compute_onset_ratios(dEye_dps_trial, dHead_trial, gazeR_trial_idx)
                    l_gaze_c = cmap(norm(l_gaze_ratio))
                    r_gaze_c = cmap(norm(r_gaze_ratio))


                    speed_colors = speed_cmap(speed_norm(row.ts_speed))
                    speed_colors = np.concatenate((speed_colors,speed_colors[0].reshape(1, -1)), axis=0)
                    
                    # Plot arena and obstacle
                    plot_arena(row.to_frame().T, ax)
                    plot_orginal_obstacle(row.to_frame().T, ax, row.obstacle_cluster)

                    # Plot head trajectory
                    ax.scatter(row.ts_head_cen_x_cm, row.ts_head_cen_y_cm,c =speed_colors ,alpha=.3)
                    
                    # Plot gaze points
                    ax.scatter(row.ts_head_cen_x_cm[gazeL_trial_idx],
                              row.ts_head_cen_y_cm[gazeL_trial_idx],
                              c=l_gaze_c, s=50, marker='*')
                    ax.scatter(row.ts_head_cen_x_cm[gazeR_trial_idx],
                              row.ts_head_cen_y_cm[gazeR_trial_idx],
                              c=r_gaze_c, s=50, marker='o')
                    
                    
                    
                    # Add trial title
                    trial_id = getattr(row, 'trial_id', f'Trial {start_idx + i + 1}')
                    ax.set_title(trial_id, fontsize=10)
                    ax.axis('off')
                    
                except Exception as e:
                    print(f"Error processing row {start_idx + i}: {str(e)}")
                    ax.axis('off')
                    ax.text(0.5, 0.5, f"Error\n{str(e)}", ha='center', va='center', color='red')
            
            # Add colorbar to the figure
            cbar_ax = fig.add_axes([0.25, 1, 0.5, 0.02])
            sm = ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_ticks(bounds)
            cbar.set_label('Movement Dominance\n(Blue = Head | Red = Eye)', fontsize=10)
            
            # Add speed colorbar (bottom) if speed data provided
            speed_cax = fig.add_axes([0.25, 0, 0.5, 0.02])
            speed_sm = ScalarMappable(cmap=speed_cmap, norm=speed_norm)
            speed_sm.set_array([])
            speed_cbar = fig.colorbar(speed_sm, cax=speed_cax, orientation='horizontal')
            speed_cbar.set_label('Speed (0-50)', fontsize=10)
            speed_cbar.set_ticks(speed_bounds)
            #speed_cbar.set_ticklabels(['0-12.5', '12.5-25', '25-37.5', '37.5-50'])
            
            # Add legend
            legend_elements = [
                Line2D([0], [0], marker='*', color='w', label='Left Gaze',
                      markerfacecolor='black', markersize=10),
                Line2D([0], [0], marker='o', color='w', label='Right Gaze',
                      markerfacecolor='black', markersize=10)
            ]
            fig.legend(handles=legend_elements, loc='lower right', 
                      bbox_to_anchor=(0.9, 0.05), frameon=True)
            
            # Adjust layout and save page
            fig.tight_layout(rect=[0, 0, 0.9, 1])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"PDF report successfully saved to: {output_path}")

def process_and_save_gaze_histogram(df, imu_dict, output_dir, var,topT,filename='gaze_distance_ratio_histogram.npz'):
    """
    Process gaze data and save distance-ratio 2D histogram as NPZ file.
    
    Args:
        df: DataFrame containing trial data with 'trial_timestamps' and 'distance_from_edge' columns
        imu_dict: Dictionary containing IMU data with required fields
        output_dir: Directory where to save the NPZ file
        filename: Name of the output file (default: 'gaze_distance_ratio_histogram.npz')
    """
    # Flatten and process the timestamps and distances
    topT = flatten_list_of_arrays(df[topT].to_list()) - imu_dict['Openephys_time'][0]
    dist = flatten_list_of_arrays(df[var].to_list())
    dist = dist[np.where(dist > 0)[0]]
    topT = topT[np.where(dist > 0)[0]]

    # Find event frames for left and right gaze
    event_indices_gazeL, frame_indices_gazeL = find_event_frames(topT, imu_dict['gazeL_event'])
    event_indices_gazeR, frame_indices_gazeR = find_event_frames(topT, imu_dict['gazeR_event'])

    # Combine left and right gaze indices
    gaze_inds_imu_time_trial = np.hstack([event_indices_gazeL, event_indices_gazeR])
    gaze_inds_trial = np.hstack([frame_indices_gazeL, frame_indices_gazeR])
    gaze_inds = np.hstack([imu_dict['gazeL_idx'], imu_dict['gazeR_idx']])

    # Compute ratios during task
    ratio_during_task = compute_onset_ratios(
        imu_dict['dEye_dps'], 
        imu_dict['dHead'], 
        gaze_inds[gaze_inds_imu_time_trial]
    )

    # Remove NaN values
    bad_indices = np.isnan(ratio_during_task)
    good_indices = ~bad_indices

    # Compute 2D histogram
    hist, xedges, yedges = np.histogram2d(
        dist[gaze_inds_trial[good_indices]],
        ratio_during_task[good_indices]
    )

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results as NPZ file
    output_path = os.path.join(output_dir, filename)
    np.savez(
        output_path,
        hist=hist,
        xedges=xedges,
        yedges=yedges,
        dist_values=dist[gaze_inds_trial[good_indices]],
        ratio_values=ratio_during_task[good_indices]
    )
    
    return output_path