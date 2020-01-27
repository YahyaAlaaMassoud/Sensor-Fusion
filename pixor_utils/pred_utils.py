
import numpy as np

from libs.kitti.core import Box2D, project

def boxes_to_pred_str(boxes_3D, P):
    lines = []
    for box_3D in boxes_3D:
        box_2D = project_box_3D(box_3D, P)
        lines += [
            f"{box_3D.cls} -1 -1 0 {box_2D.x1} {box_2D.y1} {box_2D.x2} {box_2D.y2} {box_3D.h} {box_3D.w} {box_3D.l} {box_3D.x} {box_3D.y} {box_3D.z} {box_3D.yaw} {box_3D.confidence}\n"
        ]
    return lines

def project_box_3D(box, P):
    corners = project(P, box.get_corners())
    x1, y1 = np.min(corners, axis=1)
    x2, y2 = np.max(corners, axis=1)
    return Box2D((x1, y1, x2, y2), mode=Box2D.CORNER_CORNER)
