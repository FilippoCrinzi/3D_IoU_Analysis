import numpy as np
import open3d as o3d
import trimesh


def generate_random_points_bbox(bbox, num_points):
    min_bound, max_bound = bbox
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)
    points = np.random.uniform(low=min_bound, high=max_bound, size=(num_points, 3))
    return points

def is_point_inside_mesh(mesh, point):
    triangle_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    mesh_id = scene.add_triangles(triangle_mesh)
    origin = np.array(point)
    direction = np.array([1.0, 0.0, 0.0])

    rays = o3d.core.Tensor([np.concatenate([origin, direction])], dtype=o3d.core.Dtype.Float32)
    ans = scene.cast_rays(rays)

    t_hit = ans['t_hit'].numpy()
    valid_intersections = t_hit < float('inf')
    num_intersections = valid_intersections.sum()
    return num_intersections % 2 == 1

def is_point_inside_bbox(bbox, point):
    point = np.array(point)
    # Trasform the point in the local coordinate system of the bounding box
    point_local = np.linalg.inv(bbox.R) @ (point - bbox.center)
    # Define the min and max bounds of the bounding box
    min_bound = bbox.extent * -0.5
    max_bound = bbox.extent * 0.5
    return np.all(point_local >= min_bound) and np.all(point_local <= max_bound)