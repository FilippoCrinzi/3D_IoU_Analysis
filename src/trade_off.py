import intersection_over_union as iou
import trajectory_loader as tl
import time


def trade_off_align(path_obj, path_point_cloud):
    num_point = [500, 1000, 5000, 7500, 10000, 20000]
    values_iou = []
    points = []
    times = []
    for num_points in num_point:
        start_time = time.time()
        iou_value = iou.align_iou_for_frame(path_obj, path_point_cloud, num_points, False)
        end_time = time.time()
        times.append(end_time - start_time)
        values_iou.append(iou_value)
        points.append(num_points)
        print(f"Num Points: {num_points} --> IoU: {iou_value} --> Time: {end_time - start_time}")
    tl.plot_trade_off_results(points, values_iou, times)


if __name__ == "__main__":
    point_cloud_time1 = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds'
                         '/PointCloud_traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle_time_43_downsampled_gridAverage_step0.60.csv')
    path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'

    trade_off_align(path_mercedes, point_cloud_time1)
