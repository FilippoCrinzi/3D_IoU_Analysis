import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_vehicle_position(mat_file=None, frame=None):
    mat = sio.loadmat(mat_file)
    traj_gt_vehicle = mat['Traj_gt_vehicle']
    vehicle_position = traj_gt_vehicle[frame]
    return vehicle_position


def plot_iou_results(frames, iou_values_bbox, iou_values_ellipsoid, num_points):
    plt.plot(frames, iou_values_bbox, 'o-', color='blue', label='IoU per Frame (Bounding Box)')
    plt.plot(frames, iou_values_ellipsoid, 'o-', color='red', label='IoU per Frame (Ellipsoid)')
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title(f'Andamento dei Valori di IoU per Frame (#Points:{num_points})', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'Intersection over Union bbox': iou_values_bbox,
        'Intersection over Union ellipsoid': iou_values_ellipsoid,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_graph.png', dpi=500)


def plot_iou_align_results(frames, iou_values_bbox, iou_values_ellipsoid, num_points):
    plt.plot(frames, iou_values_bbox, 'o-', color='blue', label='IoU per Frame (Bounding Box)')
    plt.plot(frames, iou_values_ellipsoid, 'o-', color='red', label='IoU per Frame (Ellipsoid)')
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title(f'Andamento dei Valori di IoU per Frame (Forme allineate #Points:{num_points})', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'Intersection over Union bbox': iou_values_bbox,
        'Intersection over Union ellipsoid': iou_values_ellipsoid,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_align_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/IoU_align_graph.png', dpi=500)


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


def plot_compare_results(frames, iou_values, iou_align_values, num_points, shape):
    plt.plot(frames, iou_values, 'o-', color='blue', label='IoU per Frame')
    plt.plot(frames, iou_align_values, 'o-', color='green', label='Align IoU per Frame')
    plt.xlabel('Frame (Tempo)', fontsize=12)
    plt.ylabel('IoU (Intersection over Union)', fontsize=12)
    plt.legend()
    plt.title(f'Andamento dei Valori di IoU per Frame ({shape} #Points:{num_points})', fontsize=14)
    plt.grid(True)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.tight_layout()
    results_df = pd.DataFrame({
        'Frame': frames,
        'IoU': iou_values,
        'Align_IoU': iou_align_values,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/compare_results.csv',
                      index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/compare_graph.png', dpi=500)