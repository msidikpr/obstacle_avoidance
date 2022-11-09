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
import matplotlib.colors as mcolors
import os, fnmatch

#import sys3
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