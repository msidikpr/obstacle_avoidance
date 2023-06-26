import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
import warnings
warnings.filterwarnings('ignore')


from obstacle_avoidance_processing import AvoidanceProcessing


# metadata json path
metadata_path = r"D:\obstacle_avoidance\metadata\compiled_data\G8CK1_G8CK_oa_061623.json"


#C:\Users\nlab\Desktop\mike_bonsai\recordings\metadata
# task name
# 'oa' for object avoidance
# 'gd' for gap detection
task_name = 'non_obstalce'
#task_name = 'oa'

session = AvoidanceProcessing(metadata_path, task=task_name) 
session.change_dlc_project(r"D:\obstacle_avoidance\deeplabcut\no_obstacle_041723-Mike-2023-04-18\config.yaml")
session.dlc_project
session.preprocess()
session.process()
