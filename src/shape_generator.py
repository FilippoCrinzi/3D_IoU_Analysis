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
    obb = pcd.get_oriented_bounding_box()
    obb.color = (1, 0, 0)

    return obb


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
