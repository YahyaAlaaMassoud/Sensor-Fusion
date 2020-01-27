import numpy as np
from .core import Box2D, project
from .reader import IMG_HEIGHT, IMG_WIDTH


# Checks if center of image lies on the image plane
def box_in_image_plane(box, proj_matrix):
    projected_pt = project([[box.x, box.y, box.z]], proj_matrix)
    return ((projected_pt[:, 0] >= 0) & (projected_pt[:, 0] <= IMG_WIDTH) &
            (projected_pt[:, 1] >= 0) & (projected_pt[:, 1] <= IMG_HEIGHT))[0]


# Projects 3D boxes to image plane
def project_3D_box_to_image_plane(boxes_3D, proj_matrix):
    boxes_2D = []
    for box_3D in boxes_3D:
        projected_corners = project(box_3D.corners, proj_matrix)
        x1, y1 = np.min(projected_corners, axis=0)
        x2, y2 = np.max(projected_corners, axis=0)
        boxes_2D += [Box2D((x1, y1, x2, y2), mode=Box2D.CORNER_CORNER, cls=box_3D.cls)]
    return boxes_2D

def box_filter(pts, box_lim, decorations=None):
    x_range, y_range, z_range = box_lim
    mask = ((pts[0] >= x_range[0]) & (pts[0] <= x_range[1]) &
            (pts[1] >= y_range[0]) & (pts[1] <= y_range[1]) &
            (pts[2] >= z_range[0]) & (pts[2] <= z_range[1]))
    pts = pts[:, mask]
    return pts if decorations is None else pts, decorations[:, mask]


# img_size = (img_height, img_width)
def fov_filter(pts, P, img_size, decorations=None):
    pts_projected = project(P, pts)
    mask = ((pts_projected[0] >= 0) & (pts_projected[0] <= img_size[1]) &
            (pts_projected[1] >= 0) & (pts_projected[1] <= img_size[0]))
    pts = pts[:, mask]
    return pts if decorations is None else pts, decorations[:, mask]
