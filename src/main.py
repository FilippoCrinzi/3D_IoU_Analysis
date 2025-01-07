import intersection_over_union as iou

trajectories = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat'
point_cloud_time1 = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle_time_1_downsampled_gridAverage_step0.60.csv'
path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'
num_points = 100

iou = iou.iou_for_frame(path_mercedes, trajectories,point_cloud_time1, 0, num_points)



