"""
Copyright 2019-2020 Selameab (https://github.com/Selameab)

Primary Author: Hamed H. Aghdam
"""

import numpy as np
from utils.point_cloud import compute_mask_accurate
import random
from core.transforms_3D import transform, rot_y_matrix, translation_matrix
from core.boxes import translate_box_3D, get_corners_3D


def per_box_dropout(dropout_ratio=0.1):
    def _per_box_dropout(velo, boxes):
        pts, ref = velo
        ids_to_remove = []
        for box in boxes:
            ids = list(np.where(compute_mask_accurate(pts, box))[0])
            if len(ids) > 10:
                random.shuffle(ids)
                ids_to_remove += ids[:int(len(ids) * dropout_ratio)]
        pts = np.delete(pts, ids_to_remove, axis=1)
        ref = np.delete(ref, ids_to_remove, axis=1)
        return (pts, ref), boxes

    return _per_box_dropout


# trans_range - Tuple of tuple
def per_box_rotation_translation(rot_range, trans_range):
    def _per_box_rotation_translation(velo, boxes):
        pts, ref = velo
        for box in boxes:
            # Create random translation and rotation matrices
            alpha = np.random.uniform(rot_range[0], rot_range[1])
            R = rot_y_matrix(alpha)
            trans_x = np.random.uniform(trans_range[0][0], trans_range[0][1])
            trans_y = np.random.uniform(trans_range[1][0], trans_range[1][1])
            trans_z = np.random.uniform(trans_range[2][0], trans_range[2][1])

            ids = list(np.where(compute_mask_accurate(pts, box))[0])

            center = np.array([[box.x, box.y, box.z]]).T
            pts[:, ids] = transform(R, pts[:, ids] - center) + center + np.array([[trans_x, trans_y, trans_z]]).T

            # Transform box
            box = translate_box_3D(box, trans_x, trans_y, trans_z)
            box.yaw += alpha

        return (pts, ref), boxes

    return _per_box_rotation_translation


def flip_along_y():
    def _flip_along_y(velo, boxes):
        pts, ref = velo
        pts[0] = -pts[0]

        for b in boxes:
            b.x *= -1
            if 0 <= b.yaw < np.pi / 2:
                b.yaw = np.pi - b.yaw
            elif -np.pi / 2 <= b.yaw < 0:
                b.yaw = -(b.yaw + np.pi)
            elif np.pi / 2 <= b.yaw < np.pi:
                b.yaw = np.pi - b.yaw
            elif -np.pi <= b.yaw < -np.pi / 2:
                b.yaw = -(b.yaw + np.pi)

        return (pts, ref), boxes

    return _flip_along_y


def global_scale(scale_range):
    def _global_scale(velo, boxes):
        pts, ref = velo
        s = np.random.uniform(scale_range[0], scale_range[1])
        pts = pts * s
        for b in boxes:
            b.x *= s
            b.y *= s
            b.z *= s
            b.w *= s
            b.l *= s
            b.h *= s
        return (pts, ref), boxes

    return _global_scale


def global_rot(rot_range):
    def _global_rot(velo, boxes):
        pts, ref = velo
        alpha = np.random.uniform(rot_range[0], rot_range[1])
        R = rot_y_matrix(alpha)
        pts = transform(R, pts)
        for box in boxes:
            box.x, box.y, box.z = transform(R, np.array([[box.x, box.y, box.z]]).T)[:, 0]
            box.yaw += alpha

        return (pts, ref), boxes

    return _global_rot


def global_trans(trans_range):
    def _global_trans(velo, boxes):
        pts, ref = velo
        trans_x = np.random.uniform(trans_range[0][0], trans_range[0][1])
        trans_y = np.random.uniform(trans_range[1][0], trans_range[1][1])
        trans_z = np.random.uniform(trans_range[2][0], trans_range[2][1])

        T = translation_matrix(trans_x, trans_y, trans_z)
        pts = transform(T, pts)
        for box in boxes:
            box.x, box.y, box.z = transform(T, np.array([[box.x, box.y, box.z]]).T)[:, 0]

        return (pts, ref), boxes

    return _global_trans


def main():
    from .reader import Reader, CARS_ONLY
    from . import visualizers as kv
    from .kitti_utils import count_points_accurate

    reader = Reader(CARS_ONLY)

    t = '000032'

    # Drop pts
    velo, boxes = reader.get_velo(t), reader.get_boxes_3D(t)
    [print(count_points_accurate(velo[0], b)) for b in boxes]
    print("-" * 50)
    velo, boxes = per_box_dropout(dropout_ratio=0.1)(velo, boxes)
    [print(count_points_accurate(velo[0], b)) for b in boxes]
    kv.open3d(velo[0], gt_boxes=boxes)

    # Per box rotation and translation
    velo, boxes = reader.get_velo(t), reader.get_boxes_3D(t)
    velo, boxes = per_box_rotation_translation(rot_range=(-1.57, 1.57), trans_range=((-2, 2), (-0.1, 0.1), (-2, 2)))(velo, boxes)
    kv.open3d(velo[0], gt_boxes=boxes)

    # Flip
    velo, boxes = reader.get_velo(t), reader.get_boxes_3D(t)
    velo, boxes = flip_along_y()(velo, boxes)
    kv.open3d(velo[0], gt_boxes=boxes)

    # Scale
    velo, boxes = reader.get_velo(t), reader.get_boxes_3D(t)
    velo, boxes = global_scale((0.9, 1.0))(velo, boxes)
    kv.open3d(velo[0], gt_boxes=boxes)

    # Rot
    velo, boxes = reader.get_velo(t), reader.get_boxes_3D(t)
    velo, boxes = global_rot((0.05, 0.05))(velo, boxes)
    kv.open3d(velo[0], gt_boxes=boxes)

    # Trans
    velo, boxes = reader.get_velo(t), reader.get_boxes_3D(t)
    velo, boxes = global_trans(((-2, 2), (-0.5, 0.5), (-3, 3)))(velo, boxes)
    kv.open3d(velo[0], gt_boxes=boxes)
