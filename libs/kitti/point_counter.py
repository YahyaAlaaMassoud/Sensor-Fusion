import numpy as np
from .core import Box3D


# Uses the smallest non-oriented box to segment
def compute_mask_estimate(pts, box_3D):
    mins = np.min(box_3D.get_corners(), axis=0)
    maxes = np.max(box_3D.get_corners(), axis=0)
    mask = (mins[0] <= pts[:, 0]) & (pts[:, 0] <= maxes[0]) & \
           (mins[1] <= pts[:, 1]) & (pts[:, 1] <= maxes[1]) & \
           (mins[2] <= pts[:, 2]) & (pts[:, 2] <= maxes[2])
    return mask


# Tests if each point is in box
def compute_mask_accurate(pts, box_3D):
    v0P = pts - box_3D.get_corners()[0]
    v01 = box_3D.get_corners()[1] - box_3D.get_corners()[0]
    v03 = box_3D.get_corners()[3] - box_3D.get_corners()[0]
    v04 = box_3D.get_corners()[4] - box_3D.get_corners()[0]

    p1 = np.dot(v0P, v01)
    p3 = np.dot(v0P, v03)
    p4 = np.dot(v0P, v04)

    mask = (0 <= p1) & (p1 <= np.dot(v01, v01)) & \
           (0 <= p3) & (p3 <= np.dot(v03, v03)) & \
           (0 <= p4) & (p4 <= np.dot(v04, v04))
    return mask


# Slightly expands box to account for labeling noise
def compute_mask_expanded(pts, box_3D):
    box_3D = Box3D(h=box_3D.h + 0.1, w=box_3D.w + 0.2, l=box_3D.l + 0.2,
                   x=box_3D.x, y=box_3D.y, z=box_3D.z,
                   yaw=box_3D.yaw)
    return compute_mask_accurate(pts, box_3D)


def count_points_estimate(pts, box_3D):
    return np.sum(compute_mask_estimate(pts, box_3D))


def count_points_accurate(pts, box_3D):
    return np.sum(compute_mask_accurate(pts, box_3D))


def count_points_expanded(pts, box_3D):
    return np.sum(compute_mask_expanded(pts, box_3D))


def count_points_column(pts, box_3D):
    box_3D = Box3D(h=20, w=box_3D.w + 0.2, l=box_3D.l + 0.2,
                   x=box_3D.x, y=10, z=box_3D.z,
                   yaw=box_3D.yaw)
    return np.sum(compute_mask_accurate(pts, box_3D))
