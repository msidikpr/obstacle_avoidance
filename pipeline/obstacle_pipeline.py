import sys
sys.path.insert(0, 'C:/Users/Niell Lab/Documents/GitHub/obstacle_avoidance')

warnings.filterwarnings('ignore')


from obstacle_avoidance_processing import AvoidanceProcessing


# metadata json path
metadata_path = r'T:\Mike\metadata\G8CK1\G8CK1_52623_train_051123.json'


#C:\Users\nlab\Desktop\mike_bonsai\recordings\metadata
# task name
# 'oa' for object avoidance
# 'gd' for gap detection

task_name = 'oa'

session = AvoidanceProcessing(metadata_path, task=task_name) 
session.change_dlc_project(r"E:\other_deeplabcut_projects\project_name-Mike-2023-04-28")
session.dlc_project
session.preprocess()
session.process()

