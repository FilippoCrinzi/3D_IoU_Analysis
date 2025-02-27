import numpy as np
import open3d as o3d


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
    o3d.visualization.draw_geometries([bbox, visualization_points])


if __name__ == "__main__":
    test_is_point_inside_mesh()
    test_is_point_inside_bbox()
