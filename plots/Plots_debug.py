import sys
sys.path.insert(0, 'C:/Users/nlab/Documents/GitHub/obstacle_avoidance')
from plots import plot_oa
session = plot_oa('D:/obstacle_avoidance/metadata/J20LT.json')
session.gather_session_df()
session.process_df()
session.plot_headangle('D:obstacle_avoidance/figures/tracking_oa','J20LT_headangle_112123')
session.intersect_histogram('D:obstacle_avoidance/figures/tracking_oa','J20LT_head_intersect_hist_112123','obstacle_intersect_nose_y')

#session.plot_trace_cluster_single_animal('D:obstacle_avoidance/figures/tracking_oa','J20LT' )