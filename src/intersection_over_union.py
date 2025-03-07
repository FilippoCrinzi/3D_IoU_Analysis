import csv

import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import shape_generator as sg
import point_sampling as ps
import utils
import time


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
    points_mesh1 = 0
    points_mesh2 = 0
    bho = []
    for point in points:
        inside_mesh1 = ps.is_point_inside_mesh(mesh1, point)
        inside_mesh2 = ps.is_point_inside_mesh(mesh2, point)
        if inside_mesh1:
            points_mesh1 += 1
        if inside_mesh2:
            points_mesh2 += 1
            bho.append(1)
        else:
            bho.append(0)
        if inside_mesh1 and inside_mesh2:
            intersection += 1
        if inside_mesh1 or inside_mesh2:
            union += 1
    print("Points mesh 1:", points_mesh1)
    print("Points mesh 2:", points_mesh2)
    print("Intersection:", intersection)
    print("Union:", union)
    return intersection / union


def iou_for_superquadric(mesh_veicle, centroid, rotation, semi_axes, epsilon, points):
    intersection = 0
    union = 0
    points_superquadric = 0
    points_mesh = 0
    bho = []
    point_colors = []
    for point in points:
        inside_superquadric = ps.is_point_inside_superquadric(point, centroid, rotation, semi_axes, epsilon)
        inside_mesh = ps.is_point_inside_mesh(mesh_veicle, point)
        if inside_superquadric:
            points_superquadric += 1
            bho.append(1)
        else:
            bho.append(0)
        if inside_mesh:
            points_mesh += 1
        if inside_superquadric and inside_mesh:
            intersection += 1
        if inside_superquadric or inside_mesh:
            union += 1
        point_colors.append([0, 1, 0] if inside_superquadric else [1, 0, 0])
    # print("Points superquadric:", points_superquadric)
    # print("Points mesh:", points_mesh)
    # print("Intersection:", intersection)
    # print("Union:", union)
    return intersection / union

########################################################################################################################


def iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    bbox = sg.create_bounding_box(path_point_cloud)
    center_bbox = bbox.get_center()
    position = utils.get_vehicle_position(path_trajectories, frame - 1)  # ricavo la posizione della mercedes al frame dato
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
    union_bbox = sg.union_bbox_mesh(mesh, bbox)
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
    position = (utils.get_vehicle_position(path_trajectories, frame - 1))  # ricavo la posizione della mercedes al frame dato
    x, y, _, _, heading = position
    z = 1.03  # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    translation = np.array([x, y, z]) - mesh.get_center()
    mesh.translate(translation)
    union_bbox = sg.union_mesh(mesh, ellipsoid)
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
        o3d.visualization.draw_geometries([mesh, wireframe, union_bbox, point_cloud_random])
    return iou


def iou_superquadric_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, epsilon, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh_veicle = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    mesh, wireframe, centroid, rotation, semi_axis, vectors = sg.generate_superquadric_from_point_cloud(path_point_cloud, epsilon[0], epsilon[1])
    position = (utils.get_vehicle_position(path_trajectories, frame - 1))
    x, y, _, _, heading = position
    z = 1.03
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh_veicle.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh_veicle.rotate(rotation_matrix, center=mesh_veicle.get_center())
    translation = np.array([x, y, z]) - mesh_veicle.get_center()
    mesh_veicle.translate(translation)
    union_bbox = sg.union_mesh(mesh, mesh_veicle)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    hull, _ = mesh_veicle.compute_convex_hull()
    iou = iou_for_superquadric(hull, centroid, rotation.T, semi_axis, epsilon, random_points)
    # point_cloud_random.colors = o3d.utility.Vector3dVector(point_colors)
    if visualize:
        coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        print("iou frame -->", iou)
        wireframe.paint_uniform_color([0, 0.39, 0])
        o3d.visualization.draw_geometries([wireframe, mesh_veicle, union_bbox, point_cloud_random])
    return iou


########################################################################################################################


def align_iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    bbox = sg.create_bounding_box(path_point_cloud)
    position = utils.get_vehicle_position(path_trajectories, frame - 1)  # ricavo la posizione della mercedes al frame dato
    x, y, _, _, heading = position
    z = 1.03  # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
    pi_mezzi = np.radians(90)
    point = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    point.translate([x, y, z])
    point.paint_uniform_color([1, 0, 0])
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    translation = np.array([x, y, z]) - mesh.get_center()
    mesh.translate(translation)
    sg.align_bbox_to_mesh(x, y, z, heading, bbox)
    union_bbox = sg.union_bbox_mesh(mesh, bbox)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    hull, _ = mesh.compute_convex_hull()
    iou = iou_mesh_bounding_box(hull, bbox, random_points)
    if visualize:
        print(f"iou frame {frame}:", iou)
        coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([mesh, bbox, union_bbox, point_cloud_random])
    return iou


def align_iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    ellipsoid = sg.create_ellipsoid_from_point_cloud(path_point_cloud)
    position = utils.get_vehicle_position(path_trajectories, frame - 1)  # ricavo la posizione della mercedes al frame dato
    x, y, _, _, heading = position
    z = 1.03  # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh.rotate(rotation_matrix, center=mesh.get_center())
    translation = np.array([x, y, z]) - mesh.get_center()
    mesh.translate(translation)
    sg.align_ellipsoid_to_mesh(x, y, z, heading, ellipsoid)
    union_bbox = sg.union_mesh(mesh, ellipsoid)
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
        o3d.visualization.draw_geometries([mesh, wireframe, union_bbox, point_cloud_random])
    return iou


def align_iou_superquadric_for_frame(path_obj, path_trajectories, path_point_cloud_dir, frame, num_points, epsilon, visualize):
    point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
    mesh_veicle = o3d.io.read_triangle_mesh(path_obj)
    file_name = point_cloud_files[frame - 1]
    path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
    position = (utils.get_vehicle_position(path_trajectories, frame - 1))
    x, y, _, _, heading = position
    z = 1.03
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh_veicle.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
    mesh_veicle.rotate(rotation_matrix, center=mesh_veicle.get_center())
    translation = np.array([x, y, z]) - mesh_veicle.get_center()
    mesh_veicle.translate(translation)
    mesh, wireframe, centroid, rotation, semi_axis, vectors = sg.generate_superquadric_from_point_cloud(
        path_point_cloud, epsilon[0], epsilon[1])
    rotation_superquadric = np.array([
        [np.cos(-heading), -np.sin(-heading), 0],
        [np.sin(-heading), np.cos(-heading), 0],
        [0, 0, 1]
    ])
    vectors = sg.transform_superquadric(vectors[0], vectors[1], vectors[2], [x, y, z], rotation_superquadric)
    mesh_superquadric, wireframe = sg.create_mesh_wireframe(vectors[0], vectors[1], vectors[2])
    union_bbox = sg.union_mesh(mesh_superquadric, mesh_veicle)
    min_bound = union_bbox.get_min_bound()
    max_bound = union_bbox.get_max_bound()
    random_points = ps.generate_random_points_bbox((min_bound, max_bound), num_points)
    point_cloud_random = o3d.geometry.PointCloud()
    point_cloud_random.points = o3d.utility.Vector3dVector(random_points)
    hull, _ = mesh_veicle.compute_convex_hull()
    iou = iou_for_superquadric(hull, [x, y, z], rotation_superquadric, semi_axis, epsilon, random_points)
    if visualize:
        coordinates = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        print("iou frame -->", iou)
        o3d.visualization.draw_geometries([hull, wireframe, union_bbox, point_cloud_random])
    return iou


########################################################################################################################


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
    (utils.plot_iou_results(frame, values_bbox, values_ellipsoid, num_points))
    plt.show()


def align_iou(path_obj, path_trajectories, path_point_cloud_dir, num_points):
    num_files = len([f for f in os.listdir(path_point_cloud_dir) if os.path.isfile(os.path.join(path_point_cloud_dir, f))])
    values_bbox = []
    values_ellipsoid = []
    values_superellisoid = []
    frame = []
    for k in range(1, num_files):
        print(f"Frame: {k}")
        iou_bbox = align_iou_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
        print("IoU bbox:", iou_bbox)
        iou_ellipsoid = align_iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
        print("IoU ellipsoid:", iou_ellipsoid)
        values_bbox.append(iou_bbox)
        values_ellipsoid.append(iou_ellipsoid)
        frame.append(k)
    utils.plot_iou_align_results(frame, values_bbox, values_ellipsoid, num_points)
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
        utils.plot_compare_results(frame, iou_values, iou_align_values, num_points, shape)
        plt.show()

    elif shape == 'ellipsoid':
        for k in range(1, num_files):
            print(f"Frame: {k}")
            iou = iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
            print("IoU:", iou)
            iou_align = align_iou_ellipsoid_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, False)
            print("IoU align:", iou_align)
            iou_align_values.append(iou_align)
            iou_values.append(iou)
            frame.append(k)
        utils.plot_compare_results(frame, iou_values, iou_align_values, num_points, shape)
        plt.show()
    else:
        print("Shape not recognized")


def superquadric_iou_all_frames(path_obj, path_trajectories, path_point_cloud_dir, num_points, epsilon):
    num_files = len(
        [f for f in os.listdir(path_point_cloud_dir) if os.path.isfile(os.path.join(path_point_cloud_dir, f))])
    iou_values = []
    iou_align_values = []
    frame = []
    start_time = time.time()
    for k in range(1, num_files):
        print(f"Frame: {k}")
        iou = iou_superquadric_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, epsilon, False)
        print("IoU:", iou)
        # align_iou = align_iou_superquadric_for_frame(path_obj, path_trajectories, path_point_cloud_dir, k, num_points, epsilon, False)
        # print("IoU align:", align_iou)
        iou_values.append(iou)
        #iou_align_values.append(align_iou)
        frame.append(k)
    end_time = time.time()
    print("Total time:", end_time - start_time)
    return frame, iou_values


def superquadric_iou_all_combination(path_obj, path_trajectories, path_point_cloud_dir, num_points):
    epsilon1_values = np.arange(0.2, 0.3, 0.2)
    epsilon2_values = np.arange(0.2, 0.3, 0.2)
    results = []
    output_dir = '/Users/filippocrinzi/Documents/UNIFI/Tesi/3DIoUAnalysis/results/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, 'epsilon_analysis.csv')

    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['epsilon1', 'epsilon2'] + [f'frame{frame}' for frame in range(1, 51)])
        for epsilon1 in epsilon1_values:
            for epsilon2 in epsilon2_values:
                frame, iou_values = superquadric_iou_all_frames(path_obj, path_trajectories, path_point_cloud_dir,
                                                                num_points, (epsilon1, epsilon2))
                row = [epsilon1, epsilon2] + iou_values
                results.append(row)
                writer.writerow(row)
    return np.array(results)