import open3d as o3d
import pandas as pd
import numpy as np
import os
import utils
from math import sqrt


def create_bounding_box(csv_file=None):
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
def union_bbox_mesh(mesh, bbo):
    mesh_bbox = mesh.get_axis_aligned_bounding_box()
    mesh_min = mesh_bbox.get_min_bound()
    mesh_max = mesh_bbox.get_max_bound()

    bbo_bbox = bbo.get_axis_aligned_bounding_box()
    bbo_min = bbo_bbox.get_min_bound()
    bbo_max = bbo_bbox.get_max_bound()

    union_min = np.minimum(mesh_min, bbo_min)
    union_max = np.maximum(mesh_max, bbo_max)
    union_bbox = o3d.geometry.AxisAlignedBoundingBox(union_min, union_max)
    union_bbox.color = (0, 0, 1)

    return union_bbox


def union_mesh(mesh1, mesh2):
    bbox1 = mesh1.get_axis_aligned_bounding_box()
    bbox2 = mesh2.get_axis_aligned_bounding_box()
    min_bound = np.minimum(bbox1.get_min_bound(), bbox2.get_min_bound())
    max_bound = np.maximum(bbox1.get_max_bound(), bbox2.get_max_bound())
    union_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    union_bbox.color = (0, 1, 0)
    return union_bbox


def align_bbox_to_origin(bbox):
    translation = -bbox.get_center()
    bbox.translate(translation)

    #Allinea la bounding box con l'asse X
    direction = bbox.R[:, 0]
    target_direction = np.array([1, 0, 0])
    axis = np.cross(direction, target_direction)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(direction, target_direction) / (np.linalg.norm(direction) * np.linalg.norm(target_direction)))
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        bbox.rotate(rotation_matrix, center=(0, 0, 0))

    #Allinea la base della bounding box al piano XY
    z_direction = bbox.R[:, 2]
    target_z = np.array([0, 0, 1])
    axis = np.cross(z_direction, target_z)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_direction, target_z) / (np.linalg.norm(z_direction) * np.linalg.norm(target_z)))
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        bbox.rotate(rotation_matrix, center=(0, 0, 0))


def align_mesh_mercedes_to_origin(mesh):
    translation = np.array([0, 0, 0])-mesh.get_center()
    mesh.translate(translation)
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi, 0])
    mesh.rotate(rotation_matrix, center=mesh.get_center())


def create_ellipsoid_from_point_cloud(csv_file=None):
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
    points = points[~np.isnan(points).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    centroid = np.mean(points, axis=0)
    cov_matrix = np.cov(points.T)  # rappresenta la variazione di ogni variabile rispetto alle altre (inclusa se stessa)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    radii = np.sqrt(eigenvalues) * 2
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    sphere.compute_vertex_normals()
    scaling_matrix = np.diag(np.append(radii, 1))
    sphere.transform(scaling_matrix)
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = eigenvectors
    sphere.transform(rotation_matrix)
    sphere.translate(centroid)
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wireframe.paint_uniform_color([0.6, 0.6, 0.6])
    # o3d.visualization.draw_geometries([pcd, wireframe])
    return sphere


def show_ellipsoid(csv_file=None):
    df = pd.read_csv(csv_file)
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    ellipsoid = create_ellipsoid_from_point_cloud(csv_file)
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(ellipsoid)
    wireframe.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([wireframe, pcd])


def align_vectors(v_from, v_to):
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    axis = np.cross(v_from, v_to)
    if np.linalg.norm(axis) < 1e-6:
        return np.eye(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(v_from, v_to), -1.0, 1.0))
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


def align_ellipsoid_to_origin(ellipsoid):
    translation = -ellipsoid.get_center()
    ellipsoid.translate(translation)
    points = np.asarray(ellipsoid.vertices)
    covariance_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(-eigenvalues)
    primary_axis = eigenvectors[:, sorted_indices[0]]
    secondary_axis = eigenvectors[:, sorted_indices[1]]
    third_axis = eigenvectors[:, sorted_indices[2]]
    target_x = np.array([1, 0, 0])
    rotation_matrix_x = align_vectors(primary_axis, target_x)
    ellipsoid.rotate(rotation_matrix_x, center=(0, 0, 0))
    new_secondary_axis = rotation_matrix_x @ secondary_axis
    target_y = np.array([0, 1, 0])
    rotation_matrix_y = align_vectors(new_secondary_axis, target_y)
    ellipsoid.rotate(rotation_matrix_y, center=(0, 0, 0))


def align_bbox_to_mesh(x, y, z, heading, bbox):
    traslation = np.array([x, y, z]) - bbox.get_center()
    bbox.translate(traslation)
    # Allinea la bounding box con l'asse X
    direction = bbox.R[:, 0]
    target_direction = np.array([1, 0, 0])
    axis = np.cross(direction, target_direction)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(
            np.dot(direction, target_direction) / (np.linalg.norm(direction) * np.linalg.norm(target_direction)))
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        bbox.rotate(rotation_matrix, center=(x, y, z))

    # Allinea la base della bounding box al piano XY
    z_direction = bbox.R[:, 2]
    target_z = np.array([0, 0, 1])
    axis = np.cross(z_direction, target_z)
    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(z_direction, target_z) / (np.linalg.norm(z_direction) * np.linalg.norm(target_z)))
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        bbox.rotate(rotation_matrix, center=(x, y, z))
    rotation_matrix = bbox.get_rotation_matrix_from_xyz([0, 0, heading])
    bbox.rotate(rotation_matrix, center=(x, y, z))


def align_ellipsoid_to_mesh(x, y, z, heading, ellipsoid):
    translation =  np.array([x, y, z]) - ellipsoid.get_center()
    ellipsoid.translate(translation)
    points = np.asarray(ellipsoid.vertices)
    covariance_matrix = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(-eigenvalues)
    primary_axis = eigenvectors[:, sorted_indices[0]]
    secondary_axis = eigenvectors[:, sorted_indices[1]]
    third_axis = eigenvectors[:, sorted_indices[2]]
    target_x = np.array([1, 0, 0])
    rotation_matrix_x = align_vectors(primary_axis, target_x)
    ellipsoid.rotate(rotation_matrix_x, center=(x, y, z))
    new_secondary_axis = rotation_matrix_x @ secondary_axis
    target_y = np.array([0, 1, 0])
    rotation_matrix_y = align_vectors(new_secondary_axis, target_y)
    ellipsoid.rotate(rotation_matrix_y, center=(x, y, z))
    rotation_matrix = ellipsoid.get_rotation_matrix_from_xyz([0, 0, heading])
    ellipsoid.rotate(rotation_matrix, center=(x, y, z))


def visualize_compare(path_point_cloud_dir, path_obj, path_trajectories, frame, shape, epsilon):
    if shape == 'bbox':
        point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
        mesh = o3d.io.read_triangle_mesh(path_obj)
        file_name = point_cloud_files[frame - 1]
        path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
        bbox = create_bounding_box(path_point_cloud)
        aligned_bbox= create_bounding_box(path_point_cloud)
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
        align_bbox_to_mesh(x, y, z, heading, aligned_bbox)
        aligned_bbox.color = (0, 0, 1)
        o3d.visualization.draw_geometries([bbox, aligned_bbox, mesh])
    elif shape == 'ellipsoid':
        point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
        mesh = o3d.io.read_triangle_mesh(path_obj)
        file_name = point_cloud_files[frame - 1]
        path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
        ellipsoid = create_ellipsoid_from_point_cloud(path_point_cloud)
        ellipsoid_aligned = create_ellipsoid_from_point_cloud(path_point_cloud)
        position = utils.get_vehicle_position(path_trajectories, frame - 1)  # ricavo la posizione della mercedes al frame dato
        x, y, _, _, heading = position
        z = 1  # definisco z come la metà dell'altezza della macchina senno il centro della mesh sarebbe a terra
        pi_mezzi = np.radians(90)
        rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])  # ruoto e traslo la mesh
        mesh.rotate(rotation_matrix, center=mesh.get_center())
        translation = np.array([x, y, z]) - mesh.get_center()
        mesh.translate(translation)
        align_ellipsoid_to_mesh(x, y, z, heading, ellipsoid_aligned)
        wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(ellipsoid)
        wireframe.paint_uniform_color([1, 0.3, 0.3])
        wireframe_aligned = o3d.geometry.LineSet.create_from_triangle_mesh(ellipsoid_aligned)
        wireframe_aligned.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries([wireframe, wireframe_aligned, mesh])
    elif shape == 'superquadric':
        point_cloud_files = utils.sort_directory_files(path_point_cloud_dir)
        mesh_veicle = o3d.io.read_triangle_mesh(path_obj)
        file_name = point_cloud_files[frame - 1]
        path_point_cloud = os.path.join(path_point_cloud_dir, file_name)
        position = (utils.get_vehicle_position(path_trajectories, frame - 1))
        x, y, _, _, heading = position
        z = 1.03
        pi_mezzi = np.radians(90)
        rotation_matrix = mesh_veicle.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi + heading, 0])
        mesh_veicle.rotate(rotation_matrix, center=mesh_veicle.get_center())
        translation = np.array([x, y, z]) - mesh_veicle.get_center()
        mesh_veicle.translate(translation)
        mesh, wireframe,_,_,_,_ = generate_superquadric_from_point_cloud(path_point_cloud, epsilon[0], epsilon[1])
        mesh_aligned, wireframe_aligned, _, _, _, vectors = generate_superquadric_from_point_cloud(path_point_cloud, epsilon[0], epsilon[1])
        rotation_superquadric = np.array([
            [np.cos(-heading), -np.sin(-heading), 0],
            [np.sin(-heading), np.cos(-heading), 0],
            [0, 0, 1]
        ])
        vectors = transform_superquadric(vectors[0], vectors[1], vectors[2], [x, y, z], rotation_superquadric)
        mesh_superquadric, wireframe_aligned = create_mesh_wireframe(vectors[0], vectors[1], vectors[2])
        wireframe_aligned.paint_uniform_color([0, 1, 0])
        wireframe.paint_uniform_color([1, 0.3, 0.3])
        o3d.visualization.draw_geometries([wireframe_aligned, wireframe, mesh_veicle])
    else:
        print('Shape non riconosciuta')

#Superquadriche


def generate_superquadric(a, b, c, epsilon1, epsilon2):
    num_points = 100
    theta = np.linspace(-np.pi / 2, np.pi / 2, num_points)  # Parametro verticale
    phi = np.linspace(0, 2 * np.pi, num_points)  # Parametro orizzontale
    theta, phi = np.meshgrid(theta, phi)

    x = a * utils.sign_power(np.cos(theta), epsilon1) * utils.sign_power(np.cos(phi), epsilon2)
    y = b * utils.sign_power(np.cos(theta), epsilon1) * utils.sign_power(np.sin(phi), epsilon2)
    z = c * utils.sign_power(np.sin(theta), epsilon1)

    return x, y, z  # Restituisce i tre array separati


def transform_superquadric(x, y, z, translation, rotation):
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T
    rotated_points = np.dot(points, rotation)
    rotated_x = rotated_points[:, 0].reshape(x.shape)
    rotated_y = rotated_points[:, 1].reshape(y.shape)
    rotated_z = rotated_points[:, 2].reshape(z.shape)
    rotated_x += translation[0]
    rotated_y += translation[1]
    rotated_z += translation[2]
    return rotated_x, rotated_y, rotated_z


def create_mesh_wireframe(x, y, z):
    """ Crea una mesh e un wireframe dai punti generati """
    points = np.vstack((x.ravel(), y.ravel(), z.ravel())).T  # Converte in lista di punti

    # Creazione di facce della mesh usando l'ordinamento dei punti
    num_points_x = x.shape[0]
    num_points_y = x.shape[1]
    faces = []

    for i in range(num_points_x - 1):
        for j in range(num_points_y - 1):
            idx1 = i * num_points_y + j
            idx2 = i * num_points_y + (j + 1)
            idx3 = (i + 1) * num_points_y + j
            idx4 = (i + 1) * num_points_y + (j + 1)

            faces.append([idx1, idx2, idx3])  # Primo triangolo
            faces.append([idx2, idx4, idx3])  # Secondo triangolo

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # Creazione del wireframe
    edges = []
    for i in range(num_points_x - 1):
        for j in range(num_points_y - 1):
            idx1 = i * num_points_y + j
            idx2 = i * num_points_y + (j + 1)
            idx3 = (i + 1) * num_points_y + j
            idx4 = (i + 1) * num_points_y + (j + 1)

            edges.extend([[idx1, idx2], [idx1, idx3], [idx2, idx4], [idx3, idx4]])

    wireframe = o3d.geometry.LineSet()
    wireframe.points = o3d.utility.Vector3dVector(points)
    wireframe.lines = o3d.utility.Vector2iVector(edges)
    wireframe.paint_uniform_color([1, 0, 0])

    return mesh, wireframe


def calculate_semi_axes(points, rotation_matrix):
    # Step 1: Centrare i punti (traslazione)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Step 2: Ruotare i punti usando la matrice di rotazione
    rotated_points = np.dot(centered_points, rotation_matrix.T)  # Ruotiamo i punti

    # Step 3: Calcolare la distanza massima lungo ogni asse principale
    semi_axes = np.max(np.abs(rotated_points), axis=0)  # Troviamo il massimo lungo ogni asse

    return semi_axes


def compute_matrix_M(points):
    # Calcola il centroide (media delle coordinate)
    centroid = np.mean(points, axis=0)
    x0, y0, z0 = centroid

    # Inizializza la matrice M come una matrice 3x3 di zeri
    M = np.zeros((3, 3))

    # Numero di punti
    N = points.shape[0]

    # Calcola la matrice M
    for i in range(N):
        xi, yi, zi = points[i]
        dx = xi - x0
        dy = yi - y0
        dz = zi - z0

        # Aggiorna la matrice M con i contributi di ciascun punto
        M[0, 0] += dy**2 + dz**2
        M[0, 1] += -dy * dx
        M[0, 2] += -dz * dx
        M[1, 0] += -dx * dy
        M[1, 1] += dx**2 + dz**2
        M[1, 2] += -dz * dy
        M[2, 0] += -dx * dz
        M[2, 1] += -dy * dz
        M[2, 2] += dx**2 + dy**2

    # Normalizza la matrice dividendo per il numero di punti
    M = M / N

    return M


def generate_superquadric_from_point_cloud(csv_file, e1, e2):
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
    points = points[~np.isnan(points).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    centroid = np.mean(points, axis=0)
    M = compute_matrix_M(points)
    eigenvalues, eigenvectors = np.linalg.eigh(M)
    a, b, c = calculate_semi_axes(points, eigenvectors.T)
    x, y, z = generate_superquadric(a, b, c, e1, e2)
    rotated_x, rotated_y, rotated_z = transform_superquadric(x, y, z, centroid, eigenvectors.T)
    mesh, wireframe = create_mesh_wireframe(rotated_x, rotated_y, rotated_z)
    # o3d.visualization.draw_geometries([pcd, wireframe,mesh])
    return mesh, wireframe, centroid, eigenvectors, [a, b, c], [x, y, z]

