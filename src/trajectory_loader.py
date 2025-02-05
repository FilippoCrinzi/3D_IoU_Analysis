import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import shape_generator as sg
from utils import extract_time_number
import math


def get_vehicle_position(mat_file=None, frame=None):
    mat = sio.loadmat(mat_file)
    traj_gt_vehicle = mat['Traj_gt_vehicle']
    vehicle_position = traj_gt_vehicle[frame]
    return vehicle_position


def plot_iou_results(frames, iou_values, num_points=None):
    if num_points is not None:
        label_text = f'IoU per Frame (#Points:{num_points})'
    else:
        label_text = 'IoU per Frame'
    plt.plot(frames, iou_values, 'o-', color='blue', label=label_text)
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title('Andamento dei Valori di IoU per Frame', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'Intersection over Union': iou_values,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_graph.png', dpi=400)


def plot_trade_off_results(num_points, iou_values, times):
    plt.figure(figsize=(10, 9))
    plt.plot(num_points, iou_values, 'o-', color='green', label='IoU per #Points')
    plt.xlabel('Numero di Punti', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title('Trade-Off', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    plt.show()
    results_df = pd.DataFrame({
        'Points': num_points,
        'Intersection over Union': iou_values,
        'Times': times,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_results.csv', index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_graph.png', dpi=400)


#test per confrontare la posizione della bounding box con la posizione del veicolo
def test_trajectories(path_trajectories=None, path_point_cloud_dir=None):
    point_cloud_files = sorted(
        [f for f in os.listdir(path_point_cloud_dir) if f.endswith('.csv')], key=extract_time_number
    )
    for k, file_name in enumerate(point_cloud_files, start=0):
        file_path = os.path.join(path_point_cloud_dir, file_name)
        bbox = sg.create_bounding_box(file_path)
        position = get_vehicle_position(path_trajectories, k)
        x, y, _, _, heading = position
        center_bbox = bbox.get_center()
        difference = (center_bbox[0] - x, center_bbox[1] - y)
        module = math.sqrt(difference[0] ** 2 + difference[1] ** 2)
        print(f"Frame: {k} --> Difference: {difference}, Module: {module:.4f}")



if __name__ == "__main__":
    point_cloud_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/point_clouds/'
    trajectories = ('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/data/trajectories'
                    '/traj_argo_50_AV_MercedesGLS580_scans50_s7_h2_5_10_v3_vehicle.mat')
    test_trajectories(trajectories, point_cloud_dir)
