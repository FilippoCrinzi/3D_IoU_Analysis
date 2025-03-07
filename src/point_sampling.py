import numpy as np
import open3d as o3d
import utils
import shape_generator as sg
from scipy.spatial.transform import Rotation as R


def generate_random_points_bbox(bbox, num_points):
    min_bound, max_bound = bbox
    min_bound = np.array(min_bound)
    max_bound = np.array(max_bound)
    points = np.random.uniform(low=min_bound, high=max_bound, size=(num_points, 3))
    return points


def is_point_inside_mesh(mesh, point):
    triangle_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(triangle_mesh)
    origin = np.array(point)
    # Direzione arbitraria
    direction = np.array([1.0, 0.0, 0.0])
    num_intersections = 0
    # Piccolo passo per evitare auto-intersezioni
    margin = 1e-6
    while True:
        rays = o3d.core.Tensor([np.concatenate([origin, direction])], dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)
        time_hit = ans['t_hit'].numpy()[0]
        if np.isinf(time_hit):
            break
        num_intersections += 1
        intersection_point = origin + time_hit * direction
        origin = intersection_point + margin * direction
    return num_intersections % 2 == 1


def is_point_inside_bbox(bbox, point):
    point = np.array(point)
    # Trasform the point in the local coordinate system of the bounding box
    point_local = np.linalg.inv(bbox.R) @ (point - bbox.center)
    # Define the min and max bounds of the bounding box
    min_bound = bbox.extent * -0.5
    max_bound = bbox.extent * 0.5
    return np.all(point_local >= min_bound) and np.all(point_local <= max_bound)


def is_point_inside_superquadric(point, centroid, rotation, semi_axes, epsilon):
    r = R.from_matrix(rotation)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    T = utils.compute_T_matrix(yaw, -pitch, -roll)
    x, y, z = point
    x0, y0, z0 = centroid
    ax, ay, az = semi_axes
    Xx = T[0, 0] * x0 + T[0, 1] * y0 + T[0, 2] * z0
    Xy = T[1, 0] * x0 + T[1, 1] * y0 + T[1, 2] * z0
    Xz = T[2, 0] * x0 + T[2, 1] * y0 + T[2, 2] * z0
    f1 = (T[0, 0] * x + T[0, 1] * y + T[0, 2] * z - Xx) / ax
    f2 = (T[1, 0] * x + T[1, 1] * y + T[1, 2] * z - Xy) / ay
    f3 = (T[2, 0] * x + T[2, 1] * y + T[2, 2] * z - Xz) / az
    F = (abs(f1) ** (2 / epsilon[1]) + abs(f2) ** (2 / epsilon[1])) ** (epsilon[1] / epsilon[0]) + abs(f3) ** (2 / epsilon[0])
    return F <= 0.98


def test_is_point_inside_mesh():
    mesh = o3d.geometry.TriangleMesh.create_box(width=2, height=2, depth=2)
    mesh.translate([-1, -1, -1])
    inside_point = np.array([0, 0, 0])
    outside_point = np.array([-2, 0.5, 0])
    surface_point = np.array([1, 0, 0])

    assert is_point_inside_mesh(mesh, inside_point) == True, "Errore: il punto interno è stato considerato fuori"
    assert is_point_inside_mesh(mesh, outside_point) == False, "Errore: il punto esterno è stato considerato dentro"

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=30)
    sphere.translate([0, 0, 0])
    assert is_point_inside_mesh(sphere, inside_point) == True, "Errore: il punto (0, 0, 0) è stato considerato fuori"
    assert is_point_inside_mesh(sphere, outside_point) == False, "Errore: il punto esterno è stato considerato dentro"

    print("Tutti i test sono passati!")


def test_is_point_inside_bbox():
    bbox = o3d.geometry.OrientedBoundingBox(center=[0, 0, 0], R=np.eye(3), extent=[2, 2, 2])
    points = np.array([
        [0, 0, 0],   # Dentro
        [1, 1, 1],   # Dentro
        [2, 2, 2],   # Fuori
        [-1, -1, -1], # Dentro
        [3, 0, 0]    # Fuori
    ])
    point_colors = []
    visualization_points = o3d.geometry.PointCloud()
    visualization_points.points = o3d.utility.Vector3dVector(points)
    for i, point in enumerate(points):
        inside = is_point_inside_bbox(bbox, point)
        print(f"Punto {point}: {'Dentro' if inside else 'Fuori'}")
        point_colors.append([0, 1, 0] if inside else [1, 0, 0])  # Verde se dentro, rosso se fuori
    visualization_points.colors = o3d.utility.Vector3dVector(point_colors)
    #o3d.visualization.draw_geometries([bbox, visualization_points])


def test_is_point_inside_superquadric():
    centroid = [0, 0, 0]
    # rotation = np.eye(3)
    rotation = np.array([
        [0.86436122, 0.50254732, 0.01805216],
        [-0.50140626, 0.86402564, -0.04529301],
        [-0.03835941, 0.03009806, 0.99881062]
    ])
    semi_axes = [3, 2, 1]
    epsilon = [0.9, 0.8]
    x, y, z = sg.generate_superquadric(semi_axes[0], semi_axes[1], semi_axes[2], epsilon[0], epsilon[1])
    x2, y2, z2 = sg.transform_superquadric(x, y, z, centroid, rotation)
    mesh, wireframe = sg.create_mesh_wireframe(x2, y2, z2)
    points = np.array([
        [0, 0, 1],  # Fuori
        [-3, 0, 0],
        [3, 0, 0], [0, 2, 0], [0, 0, 1], [-3, 0, 0], [0, -2, 0], [0, 0, -1],
        [2.9, 0.1, 0], [0.1, 1.9, 0], [0.1, 0.1, 0.9], [-2.9, -0.1, 0], [-0.1, -1.9, 0], [-0.1, -0.1, -0.9],

        # Punti all'interno
        [1, 1, 0], [-1, -1, 0], [0.5, 0.5, 0.5], [-0.5, -0.5, -0.5], [1.5, 0.5, 0], [-1.5, -0.5, 0],
        [0, 1, 0.5], [0, -1, -0.5], [0.5, 0, 1], [-0.5, 0, -1],

        # Punti all'esterno e lontani
        [4, 0, 0], [0, 3, 0], [0, 0, 2], [-4, 0, 0], [0, -3, 0], [0, 0, -2],
        [5, 5, 5], [-5, -5, -5], [6, 0, 0], [0, 6, 0], [0, 0, 3], [-6, 0, 0], [0, -6, 0], [0, 0, -3],

        # Punti nel centro
        [0, 0, 0], [0.1, 0.1, 0.1], [-0.1, -0.1, -0.1], [0.2, 0.2, 0.2], [-0.2, -0.2, -0.2],

        # Altri punti casuali vicino e lontano
        [2, 1, 0.5], [-2, -1, -0.5], [3.1, 0.2, 0.1], [-3.1, -0.2, -0.1],
        [1.8, 1.2, 0.8], [-1.8, -1.2, -0.8], [2.5, 0.5, 0.5], [-2.5, -0.5, -0.5],

        # Punti casuali generati
        *np.random.uniform(-4, 4, (60, 3))

    ])
    point_colors = []
    visualization_points = o3d.geometry.PointCloud()
    visualization_points.points = o3d.utility.Vector3dVector(points)
    for i, point in enumerate(points):
        inside = is_point_inside_superquadric(point, centroid, rotation, semi_axes, epsilon)
        mesh_inside = is_point_inside_mesh(mesh, point)
        print(f"Punto {point}: {'Dentro' if inside else 'Fuori'}")
        print(f"Punto {point}: {'Dentro mesh' if mesh_inside else 'Fuori mesh'}")
        point_colors.append([0, 1, 0] if inside else [1, 0, 0])  # Verde se dentro, rosso se fuori
    visualization_points.colors = o3d.utility.Vector3dVector(point_colors)
    wireframe.paint_uniform_color([0.7, 0.7, 0.7])
    print(rotation)
    o3d.visualization.draw_geometries([visualization_points, wireframe])


if __name__ == "__main__":
    test_is_point_inside_mesh()
    test_is_point_inside_bbox()
    test_is_point_inside_superquadric()
