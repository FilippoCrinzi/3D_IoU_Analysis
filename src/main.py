import intersection_over_union as iou

path_point_cloud = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_COMfixed_vehicle_time_7_downsampled_gridAverage_step0.60.csv'
point_cloud_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/'
trajectories = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories'
                '/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat')
path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'
num_points = 3000


iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'bbox')
#iou.align_iou_for_frame2(path_mercedes, trajectories, point_cloud_dir, 32, num_points, True)
#iou.iou_for_frame(path_mercedes, trajectories, point_cloud_dir, 5, num_points, True)
#iou.iou_ellipsoid_for_frame(path_mercedes, trajectories, point_cloud_dir, 41, num_points, True)
#iou.sg.visualize_compare(point_cloud_dir, path_mercedes, trajectories, 7, 'ellipsoid')
#iou.align_iou_ellipsoid_for_frame2(path_mercedes, trajectories, point_cloud_dir, 49, num_points, True)
#iou.normal_iou(path_mercedes, trajectories, point_cloud_dir, num_points)
#iou.align_iou(path_mercedes, trajectories, point_cloud_dir, num_points)
#iou.sg.show_point_cloud(path_point_cloud)
#iou.sg.show_ellipsoid(path_point_cloud)