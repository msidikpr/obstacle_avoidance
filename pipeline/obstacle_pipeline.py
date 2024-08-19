import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')



import warnings

from obstacle_avoidance_processing import AvoidanceProcessing


# metadata json path

metadata_path = r"D:\obstacle_avoidance\metadata\J701_704\J704_oadark.json"


#C:\Users\nlab\Desktop\mike_bonsai\recordings\metadata
# task name
# 'oa' for object avoidance
# 'gd' for gap detection

task_name = 'oa'

session = AvoidanceProcessing(metadata_path, task=task_name) 
session.change_dlc_project(r"D:\obstacle_avoidance\deeplabcut\obstacle_obstacle_avoidance_070124-Mike-2024-07-01\config.yaml")
session.dlc_project
session.preprocess()
session.process()

