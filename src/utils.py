import re
import scipy.io as sio
import os
import open3d as o3d
import numpy as np
import intersection_over_union as iou
import matplotlib.pyplot as plt
import pandas as pd


def get_vehicle_position(mat_file=None, frame=None):
    mat = sio.loadmat(mat_file)
    traj_gt_vehicle = mat['Traj_gt_vehicle']
    vehicle_position = traj_gt_vehicle[frame]
    return vehicle_position


def extract_time_number(filename):
    match = re.search(r'vehicle_time_(\d+)', filename)
    return int(match.group(1)) if match else float('inf')


def read_file_mat(file_path):
    try:
        # Carica il file .mat
        mat_data = sio.loadmat(file_path)

        # Stampa le chiavi e i contenuti
        print(f"Contenuto del file '{file_path}':\n")
        for key, value in mat_data.items():
            if key.startswith('__'):  # Ignora meta-informazioni
                continue
            print(f"Variabile: {key}")
            print(f"Tipo: {type(value)}")
            print(f"Dimensioni: {getattr(value, 'shape', 'N/A')}")
            print(f"Contenuto: {value}\n")
    except Exception as e:
        print(f"Errore nella lettura del file .mat: {e}")


def sort_directory_files(path_point_cloud_dir):
    point_cloud_files = sorted(
        [f for f in os.listdir(path_point_cloud_dir) if f.endswith('.csv')], key=extract_time_number
    )
    return point_cloud_files


def compute_height_from_bottom(mesh):
    iou.sg.align_mesh_mercedes_to_origin(mesh)
    center = mesh.get_center()
    vertices = np.asarray(mesh.vertices)
    min_z = np.min(vertices[:, 2])
    distance = center[2] - min_z
    bottom_point = np.array([center[0], center[1], min_z])

    # Crea punti visibili
    center_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    center_sphere.paint_uniform_color([1, 0, 0])  # Rosso per il centro
    center_sphere.translate(center)

    bottom_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    bottom_sphere.paint_uniform_color([0, 0, 1])  # Blu per il punto più basso
    bottom_sphere.translate(bottom_point)

    # Mostra tutto
    o3d.visualization.draw_geometries([mesh, center_sphere, bottom_sphere])

    return distance


def plot_iou_results(frames, iou_values_bbox, iou_values_ellipsoid, num_points):
    plt.plot(frames, iou_values_bbox, 'o-', color='blue', label='IoU per Frame (Bounding Box)', markersize=3.5, linewidth=0.8)
    plt.plot(frames, iou_values_ellipsoid, 'o-', color='red', label='IoU per Frame (Ellipsoid)', markersize=3.5, linewidth=0.8)
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
    plt.plot(frames, iou_values_bbox, 'o-', color='blue', label='IoU per Frame (Bounding Box)', markersize=3.5, linewidth=0.8)
    plt.plot(frames, iou_values_ellipsoid, 'o-', color='red', label='IoU per Frame (Ellipsoid)', markersize=3.5, linewidth=0.8)
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


def plot_trade_off_results(num_points, iou_values, iou_align_values, times, times_align):
    plt.figure(figsize=(10, 9))
    plt.plot(num_points, iou_values, 'o-', color='green', label='IoU per #Points', markersize=3.5, linewidth=1.1)
    plt.plot(num_points, iou_align_values, 'o-', color='blue', label='Align IoU per #Points', markersize=3.5, linewidth=1.1)
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
        'IoU': iou_values,
        'Times': times,
        'Align IoU': iou_align_values,
        'Times Align': times_align,
    })
    results_df.to_csv('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off_results.csv', index=False)
    plt.savefig('/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/trade_off.png', dpi=400)


def plot_compare_results(frames, iou_values, iou_align_values, num_points, shape):
    plt.plot(frames, iou_values, 'o-', color='blue', label='IoU per Frame', markersize=3.5, linewidth=0.8)
    plt.plot(frames, iou_align_values, 'o-', color='green', label='Align IoU per Frame', markersize=3.5, linewidth=0.8)
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


def sign_power(base, exp):
    """ Implementa x^e mantenendo il segno di x per supportare e < 1 """
    return np.sign(base) * (np.abs(base) ** exp)


def compute_T_matrix(yaw, pitch, roll):
    """ Calcola la matrice di rotazione T a partire dai valori di yaw, pitch e roll """
    T = np.zeros((3, 3))
    T[0, 0] = np.cos(pitch) * np.cos(roll)
    T[0, 1] = np.cos(pitch) * np.sin(roll)
    T[0, 2] = -np.sin(pitch)
    T[1, 0] = np.sin(yaw) * np.sin(pitch) * np.cos(roll) - np.cos(yaw) * np.sin(roll)
    T[1, 1] = np.sin(yaw) * np.sin(pitch) * np.sin(roll) + np.cos(yaw) * np.cos(roll)
    T[1, 2] = np.sin(yaw) * np.cos(pitch)
    T[2, 0] = np.cos(yaw) * np.sin(pitch) * np.cos(roll) + np.sin(yaw) * np.sin(roll)
    T[2, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.sin(yaw) * np.cos(roll)
    T[2, 2] = np.cos(yaw) * np.cos(pitch)
    return T
