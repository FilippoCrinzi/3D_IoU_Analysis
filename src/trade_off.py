import intersection_over_union as iou
import utils
import time


def trade_off(path_obj, path_point_cloud_dir, frame, path_trajectories):
    num_point = [100, 250, 500, 1000, 2500, 5000, 10000, 15000]
    values_iou = []
    values_align_iou = []
    points = []
    times = []
    times_align = []
    for num_points in num_point:
        start_time = time.time()
        iou_value = iou.iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, False)
        end_time = time.time()
        t = end_time - start_time
        times.append(t)
        values_iou.append(iou_value)
        start_time = time.time()
        iou_align_value = iou.align_iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, False)
        end_time = time.time()
        values_align_iou.append(iou_align_value)
        t_align = end_time - start_time
        times_align.append(t_align)
        points.append(num_points)
        print(f"Num Points: {num_points} --> IoU: {iou_value} --> Time: {t}")
        print(f"Num Points: {num_points} --> Align IoU: {iou_align_value} --> Time: {t_align}")

    utils.plot_trade_off_results(points, values_iou, values_align_iou, times, times_align)


if __name__ == "__main__":
    point_cloud_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/'
    path_mercedes = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/models/MercedesGLS580.obj'
    trajectories = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories'
                    '/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat')

    trade_off(path_mercedes, point_cloud_dir, 29, trajectories)
