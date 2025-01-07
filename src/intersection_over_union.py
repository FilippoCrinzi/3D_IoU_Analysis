import trajectory_loader as tl
import open3d as o3d
import numpy as np
import shape_generator as sg
import point_sampling as ps


def iou_mesh_bounding_box(mesh, bbox, points):
    intersection = 0
    union = 0
    for point in points:
        inside_mesh = ps.is_point_inside_mesh(mesh, point)

        inside_bbox = ps.is_point_inside_bbox(bbox, point)

        if inside_mesh and inside_bbox:
            intersection += 1
        if inside_mesh or inside_bbox:
            union += 1
    return intersection / union

def iou_for_bounding_box(bbox1, bbox2, points):
    intersection = 0
    union = 0
    for point in points:
        inside_bbox1 = ps.is_point_inside_bbox(bbox1, point)
        inside_bbox2 = ps.is_point_inside_bbox(bbox2, point)

        if inside_bbox1 and inside_bbox2:
            intersection += 1
        if inside_bbox1 or inside_bbox2:
            union += 1
    return intersection / union

def iou_for_mesh(mesh1, mesh2, points):
    intersection = 0
    union = 0
    for point in points:
        inside_mesh1 = ps.is_point_inside_mesh(mesh1, point)
        inside_mesh2 = ps.is_point_inside_mesh(mesh2, point)

        if inside_mesh1 and inside_mesh2:
            intersection += 1
        if inside_mesh1 or inside_mesh2:
            union += 1
    return intersection / union


def iou_for_frame(path_obj, path_trajectories, path_point_cloud, frame, num_points):
    mesh = o3d.io.read_triangle_mesh(path_obj)
    bbox = sg.create_bounding_box(path_point_cloud)
    position = tl.get_vehicle_position(path_trajectories, frame)  # ricavo la posizione della mercedes al frame dato
    x, y, _, _, heading = position
    z = 0.90 # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])# ruoto e traslo la mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    translation = np.array([x, y, z]) - mesh.get_center()
    mesh.translate(translation)
    union_bbox = sg.compute_bounding_box_union(mesh, bbox)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound,max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    hull, _ = mesh.compute_convex_hull()
    iou = iou_mesh_bounding_box(hull, bbox, random_points)
    print("iou frame1 -->", iou)
    o3d.visualization.draw_geometries([hull, bbox, union_bbox, point_cloud_random])
    return iou

def align_iou_for_frame(path_obj, path_point_cloud, num_points):
    mesh = o3d.io.read_triangle_mesh(path_obj)
    bbox = sg.create_bounding_box(path_point_cloud)
    translation = bbox.center - mesh.get_center()


    mesh.rotate(bbox.R, center=mesh.get_center())
    mesh.translate(translation)

    union_bbox = sg.compute_bounding_box_union(mesh, bbox)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound,max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    hull, _ = mesh.compute_convex_hull()
    iou = iou_mesh_bounding_box(hull, bbox, random_points)
    print("iou frame1 -->", iou)

    o3d.visualization.draw_geometries([hull, bbox, union_bbox, point_cloud_random])
    return iou