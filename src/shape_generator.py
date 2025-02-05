import open3d as o3d
import pandas as pd
import numpy as np


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
    union_bbox.color = (0, 0, 1)

    return union_bbox


def compute_bounding_box_union_mesh(mesh1, mesh2):
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
    translation = -mesh.get_center()
    mesh.translate(translation)
    pi_mezzi = np.radians(90)
    rotation_matrix = mesh.get_rotation_matrix_from_xyz([pi_mezzi, pi_mezzi, 0])
    mesh.rotate(rotation_matrix, center=mesh.get_center())


#prova funzioni per creare ellissoide
def create_ellipsoid_from_bounding_box(csv_file=None):
    bbox = create_bounding_box(csv_file)
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    center = bbox.get_center()
    extents = bbox.extent
    a, b, c = extents / 2
    # Crea una sfera unitaria
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=50)

    # Scala la sfera per trasformarla in un ellissoide
    sphere.scale(1.0, center=(0, 0, 0))  # Reset dello scaling
    scaling_matrix = np.diag([a, b, c, 1])  # Matrice di scaling
    sphere.transform(scaling_matrix)  # Scala la sfera

    # Trasla l'ellissoide al centro della bounding box
    sphere.translate(center)

    # Liscia la mesh e aggiungi normali
    sphere.paint_uniform_color(np.array([0, 0, 1], dtype=np.float64))
    sphere.compute_vertex_normals()
    wireframe = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])
    #o3d.visualization.draw_geometries([pcd, wireframe, bbox])
    return sphere


def create_ellipsoid_from_point_cloud(csv_file=None):
    bbox = create_bounding_box(csv_file)
    column_names = ['x', 'y', 'z']
    df = pd.read_csv(csv_file, header=None, names=column_names)
    points = df[['x', 'y', 'z']].to_numpy()
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
    wireframe.paint_uniform_color([0.5, 0.5, 0.5])
    #o3d.visualization.draw_geometries([pcd, wireframe, bbox])
    return sphere


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