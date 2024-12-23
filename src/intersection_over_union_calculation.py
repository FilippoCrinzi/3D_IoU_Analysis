import open3d as o3d
import numpy as np
import point_sampling as ps


def iou_for_bounding_box(mesh, bbox, points):
    intersection = 0
    union = 0
    for point in points:
        inside_mesh = ps.is_point_inside_mesh(mesh, point)

        inside_bbox = ps.is_point_inside_bbox(bbox, point)

        if inside_mesh and inside_bbox:
            intersection += 1
        if inside_mesh or inside_bbox:
            union += 1
    return intersection, union
