import intersection_over_union as iou

path_point_cloud = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_COMfixed_vehicle_time_1_downsampled_gridAverage_step0.60.csv'
point_cloud_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/'
trajectories = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories'
                '/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat')
path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'
num_points = 3000


iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'ellipsoid')