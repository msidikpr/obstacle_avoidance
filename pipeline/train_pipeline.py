import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
import warnings
warnings.filterwarnings('ignore')


from obstacle_avoidance_processing import AvoidanceProcessing


# metadata json path
metadata_path = r"D:\obstacle_avoidance\metadata\SFN\G8CK_train_test.json"



#C:\Users\nlab\Desktop\mike_bonsai\recordings\metadata
# task name
# 'oa' for object avoidance
# 'gd' for gap detection
task_name = 'non_obstalce'
#task_name = 'oa'

session = AvoidanceProcessing(metadata_path, task=task_name) 
session.change_dlc_project(r"D:\obstacle_avoidance\deeplabcut\training_tracking_081424-Mike-2024-08-14\config.yaml")
session.dlc_project
session.preprocess()
session.process()
