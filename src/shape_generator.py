import open3d as o3d
import pandas as pd
import numpy as np


def create_bounding_box(csv_file=None):
    df = pd.read_csv(csv_file)
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obbox = pcd.get_oriented_bounding_box()
    obbox.color = (1, 0, 0)

    return obbox

def show_point_cloud(csv_file=None):
    df = pd.read_csv(csv_file)
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    bbox = create_bounding_box(csv_file)
    o3d.visualization.draw_geometries([pcd, bbox])

# Function to compute the region where shoot the points
def compute_bounding_box_union(mesh, bbo):
    mesh_bbox = mesh.get_axis_aligned_bounding_box()
    mesh_min = mesh_bbox.get_min_bound()
    mesh_max = mesh_bbox.get_max_bound()

    bbo_bbox = bbo.get_axis_aligned_bounding_box()
    bbo_min = bbo_bbox.get_min_bound()
    bbo_max = bbo_bbox.get_max_bound()

    union_min = np.minimum(mesh_min, bbo_min)
    union_max = np.maximum(mesh_max, bbo_max)
    union_bbox = o3d.geometry.AxisAlignedBoundingBox(union_min, union_max)
    union_bbox.color = (0, 1, 0)

    return union_bbox

def align_bbox_to_origin(bbox):
    translation = -bbox.get_center()
    bbox.translate(translation)
    direction = bbox.R[:, 0]
    target_direction = np.array([1, 0, 0])
    axis = np.cross(direction, target_direction)
    angle = np.arccos(np.dot(direction, target_direction) / (np.linalg.norm(direction) * np.linalg.norm(target_direction)))
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    bbox.rotate(rotation_matrix, center=(0, 0, 0))

def align_mesh_to_origin(mesh):
    translation = -mesh.get_center()
    mesh.translate(translation)
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi, 0])
    mesh.rotate(rotation_matrix, center=mesh.get_center())


