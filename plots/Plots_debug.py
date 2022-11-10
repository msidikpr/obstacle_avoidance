import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
from plots import plot_oa
session = plot_oa('D:/obstacle_avoidance/metadata/J19RT.json')
session.gather_session_df()
session.cluster()
session.plot_trace_cluster_single_animal('D:obstacle_avoidance/figures/tracking_oa','J19RT' )