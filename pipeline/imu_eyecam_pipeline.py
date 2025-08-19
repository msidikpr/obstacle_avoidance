import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')



import warnings

from pipeline.imu_eyecam_worldcam_preprocess import IMU_eyecam_world_ephys_Processing


# metadata json path

metadata_path = r"\\goeppert\Vol2\mike\metadata\gaze_shift\oa_cam_050725_8_9.json"



#C:\Users\nlab\Desktop\mike_bonsai\recordings\metadata
# task name
# 'oa' for object avoidance
# 'gd' for gap detection

task_name = 'oa'

session = IMU_eyecam_world_ephys_Processing(metadata_path) 
#session.preprocess_eyecam()
#session.preprocess_topdown()
session.preprocess_world()
#session.process_gazeshift()