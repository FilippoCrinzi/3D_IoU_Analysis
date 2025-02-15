import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import shape_generator as sg
import point_sampling as ps
import utils
import trajectory_loader as tl


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


def iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    bbox = sg.create_bounding_box(path_point_cloud)
    center_bbox = bbox.get_center()
    position = tl.get_vehicle_position(path_trajectories, frame - 1)  # ricavo la posizione della mercedes al frame dato
    x, y, _, _, heading = position
    z = 1  # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
    pi_mezzi = np.radians(90)
    point = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    point.translate([x, y, z])
    point.paint_uniform_color([1, 0, 0])
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    translation = np.array([x, y, z]) - mesh.get_center()
    mesh.translate(translation)
    union_bbox = sg.compute_bounding_box_union(mesh, bbox)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    hull, _ = mesh.compute_convex_hull()
    iou = iou_mesh_bounding_box(hull, bbox, random_points)
    if visualize:
        print("bbox center -->", center_bbox)
        print("mesh center -->", mesh.get_center())
        print(f"iou frame {frame}:", iou)
        o3d.visualization.draw_geometries([mesh, bbox, union_bbox, point_cloud_random])
    return iou


def iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    ellipsoid = sg.create_ellipsoid_from_point_cloud(path_point_cloud)
    position = tl.get_vehicle_position(path_trajectories, frame - 1)  # ricavo la posizione della mercedes al frame dato
    x, y, _, _, heading = position
    z = 1.03  # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    translation = np.array([x, y, z]) - mesh.get_center()
    mesh.translate(translation)
    union_bbox = sg.compute_bounding_box_union_mesh(mesh, ellipsoid)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(ellipsoid)
    wireframe.paint_uniform_color([1, 0, 0])
    hull, _ = mesh.compute_convex_hull()
    iou = iou_for_mesh(hull, ellipsoid, random_points)
    if visualize:
        coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        print("iou frame -->", iou)
        o3d.visualization.draw_geometries([mesh, wireframe, union_bbox, point_cloud_random, coordinates])
    return iou


def align_iou_for_frame(path_obj, path_point_cloud_dir, frame, num_points, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    #print(point_cloud_files[frame - 1])
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    bbox = sg.create_bounding_box(path_point_cloud)

    sg.align_mesh_mercedes_to_origin(mesh)
    sg.align_bbox_to_origin(bbox)

    hull, _ = mesh.compute_convex_hull()
    union_bbox = sg.compute_bounding_box_union(mesh, bbox)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    iou = iou_mesh_bounding_box(hull, bbox, random_points)
    if visualize:
        coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        print(f"iou frame {frame}:", iou)
        o3d.visualization.draw_geometries([mesh, bbox, union_bbox, coordinates, point_cloud_random])
    return iou


def align_iou_ellipsoid_for_frame(path_obj, path_point_cloud_dir, frame, num_points,  visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    #print(point_cloud_files[frame - 1])
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    ellipsoid = sg.create_ellipsoid_from_point_cloud(path_point_cloud)
    bbox = sg.create_bounding_box(path_point_cloud)
    sg.align_bbox_to_origin(bbox)
    sg.align_mesh_mercedes_to_origin(mesh)
    sg.align_ellipsoid_to_origin(ellipsoid)
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(ellipsoid)
    wireframe.paint_uniform_color([1, 0, 0])
    hull, _ = mesh.compute_convex_hull()
    union_bbox = sg.compute_bounding_box_union_mesh(mesh, ellipsoid)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    iou = iou_for_mesh(hull, ellipsoid, random_points)
    if visualize:
        print("iou ellipsoid-->", iou)
        coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([mesh, wireframe, union_bbox, coordinates, point_cloud_random])
    return iou


def normal_iou(path_obj, path_trajectories, path_point_cloud_dir, num_points):
    num_files = len([f for f in os.listdir(path_point_cloud_dir) if os.path.isfile(os.path.join(path_point_cloud_dir, f))])
    values_bbox = []
    values_ellipsoid = []
    values_superellisoid = []
    frame = []
    for k in range(1, num_files):
        print(f"Frame: {k}")
        iou_bbox = iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
        print("IoU bbox:", iou_bbox)
        iou_ellipsoid = iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
        print("IoU ellipsoid:", iou_ellipsoid)
        values_bbox.append(iou_bbox)
        values_ellipsoid.append(iou_ellipsoid)
        frame.append(k)
    tl.plot_iou_results(frame, values_bbox, values_ellipsoid, num_points)
    plt.show()


def align_iou(path_obj, path_point_cloud_dir, num_points):
    num_files = len([f for f in os.listdir(path_point_cloud_dir) if os.path.isfile(os.path.join(path_point_cloud_dir, f))])
    values_bbox = []
    values_ellipsoid = []
    values_superellisoid = []
    frame = []
    for k in range(1, num_files):
        print(f"Frame: {k}")
        iou_bbox = align_iou_for_frame(path_obj, path_point_cloud_dir, k, num_points, False)
        print("IoU bbox:", iou_bbox)
        iou_ellipsoid = align_iou_ellipsoid_for_frame(path_obj, path_point_cloud_dir, k, num_points, False)
        print("IoU ellipsoid:", iou_ellipsoid)
        values_bbox.append(iou_bbox)
        values_ellipsoid.append(iou_ellipsoid)
        frame.append(k)
    tl.plot_iou_align_results(frame, values_bbox, values_ellipsoid, num_points)
    plt.show()


def compare_iou(path_obj, path_trajectories, path_point_cloud_dir, num_points, shape):
    num_files = len([f for f in os.listdir(path_point_cloud_dir) if os.path.isfile(os.path.join(path_point_cloud_dir, f))])
    iou_values = []
    iou_align_values = []
    frame = []
    if shape == 'bbox':
        for k in range(1, num_files):
            print(f"Frame: {k}")
            iou_bbox = iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
            print("IoU:", iou_bbox)
            iou_align = align_iou_for_frame(path_obj, path_point_cloud_dir, k, num_points, False)
            print("IoU align:", iou_align)
            iou_align_values.append(iou_align)
            iou_values.append(iou_bbox)
            frame.append(k)
        tl.plot_compare_results(frame, iou_values, iou_align_values, num_points, shape)
        plt.show()

    elif shape == 'ellipsoid':
        for k in range(1, num_files):
            print(f"Frame: {k}")
            iou = iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
            print("IoU:", iou)
            iou_align = align_iou_ellipsoid_for_frame(path_obj, path_point_cloud_dir, k, num_points, False)
            print("IoU align:", iou_align)
            iou_align_values.append(iou_align)
            iou_values.append(iou)
            frame.append(k)
        tl.plot_compare_results(frame, iou_values, iou_align_values, num_points, shape)
        plt.show()
    else:
        print("Shape not recognized")

