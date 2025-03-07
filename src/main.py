import intersection_over_union as iou

path_point_cloud = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_COMfixed_vehicle_time_23_downsampled_gridAverage_step0.60.csv'
point_cloud_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/'
trajectories = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories'
                '/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat')
path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'
num_points = 3000
path_image = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle_time_23.csv'

# iou.compare_iou(path_mercedes, trajectories, point_cloud_dir, num_points, 'bbox')
# iou.align_iou_for_frame2(path_mercedes, trajectories, point_cloud_dir, 32, num_points, True)
# iou.iou_for_frame(path_mercedes, trajectories, point_cloud_dir, 48, num_points, True)

# iou.iou_ellipsoid_for_frame(path_mercedes, trajectories, point_cloud_dir, 23, num_points, True)

# iou.sg.visualize_compare(point_cloud_dir, path_mercedes, trajectories, 23, 'bbox')
# iou.align_iou_ellipsoid_for_frame2(path_mercedes, trajectories, point_cloud_dir, 49, num_points, True)

# iou.normal_iou(path_mercedes, trajectories, point_cloud_dir, num_points)
# iou.align_iou(path_mercedes, trajectories, point_cloud_dir, num_points)

# iou.sg.show_point_cloud(path_image)
# iou.sg.show_ellipsoid(path_point_cloud)

# iou.sg.generate_superquadric_from_point_cloud(path_point_cloud, 0.6, 0.6)
# iou.iou_superquadric_for_frame(path_mercedes, trajectories, point_cloud_dir, 33, num_points, [1.5,2], True)

# iou.superquadric_iou_all_frames(path_mercedes, trajectories, point_cloud_dir, num_points, [0.8, 0.7])
iou.superquadric_iou_all_combination(path_mercedes, trajectories, point_cloud_dir, num_points)