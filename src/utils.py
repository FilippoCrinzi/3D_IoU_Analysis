import re
import scipy.io as sio
import os
import open3d as o3d
import numpy as np


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
