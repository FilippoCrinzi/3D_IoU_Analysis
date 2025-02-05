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
    scene.add_triangles(triangle_mesh)

    origin = np.array(point)
    direction = np.array([1.0, 0.0, 0.0])  # Direzione arbitraria lungo X
    num_intersections = 0
    epsilon = 1e-6  # Piccolo passo per evitare auto-intersezioni

    while True:
        rays = o3d.core.Tensor([np.concatenate([origin, direction])], dtype=o3d.core.Dtype.Float32)
        ans = scene.cast_rays(rays)

        t_hit = ans['t_hit'].numpy()[0]
        if np.isinf(t_hit):
            break

        num_intersections += 1
        intersection_point = origin + t_hit * direction
        origin = intersection_point + epsilon * direction

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


if __name__ == "__main__":
    test_is_point_inside_mesh()
