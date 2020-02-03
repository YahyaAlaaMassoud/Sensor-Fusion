import numpy as np

from core.transforms_3D import rot_y_matrix, translation_matrix, transform


class Box2D:
    # Modes
    CORNER_CORNER = 0
    CORNER_DIM = 1
    CENTER_DIM = 2

    def __init__(self, values, mode, cls=None, confidence=None, text=None):
        self.cls = cls
        self.confidence = confidence
        self.text = text
        if mode == Box2D.CORNER_CORNER:
            self.x1, self.y1, self.x2, self.y2 = values
            self.cx, self.cy, self.w, self.h = (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2, self.x2 - self.x1, self.y2 - self.y1
        elif mode == Box2D.CENTER_DIM:
            self.cx, self.cy, self.w, self.h = values
            self.x1, self.y1, self.x2, self.y2 = self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2
        elif mode == Box2D.CORNER_DIM:
            self.x1, self.y1, self.w, self.h = values
            self.cx, self.cy, self.x2, self.y2 = self.x1 + self.w / 2, self.y1 + self.h / 2, self.x1 + self.w, self.y1 + self.h

    def corner_corner(self):
        return self.x1, self.y1, self.x2, self.y2

    def center_dim(self):
        return self.cx, self.cy, self.w, self.h

    def __str__(self):
        return "Center: (%.3f, %.3f)   H,W:  (%.3f, %.3f)" % (self.cx, self.cy, self.h, self.w)


class Box3D:
    def __init__(self, h, w, l, x, y, z, yaw, cls=None, confidence=None, text=None):
        # Copy params
        self.h, self.w, self.l = h, w, l
        self.x, self.y, self.z = x, y, z
        self.yaw = yaw
        self.cls = cls
        self.confidence = confidence
        self.text = text

        self.__corners = None
        self.__arrow_pts = None

    def get_corners(self):
        if self.__corners is None:
            # Compute corners
            self.__corners = np.array([[-self.l / 2, self.l / 2, self.l / 2, -self.l / 2, -self.l / 2, self.l / 2, self.l / 2, -self.l / 2],
                                       [0, 0, 0, 0, -self.h, -self.h, -self.h, -self.h],
                                       [-self.w / 2, -self.w / 2, self.w / 2, self.w / 2, -self.w / 2, -self.w / 2, self.w / 2, self.w / 2]], dtype=np.float32)
            H = np.dot(translation_matrix(self.x, self.y, self.z), rot_y_matrix(self.yaw))
            self.__corners = transform(H, self.__corners)
        return self.__corners

    def get_arrow_pts(self):
        if self.__arrow_pts is None:
            # Compute arrow from center pointing forward
            self.__arrow_pts = np.array([[0, (self.l / 2) + 1],
                                         [-self.h / 2, -self.h / 2],
                                         [0, 0]])
            H = np.dot(translation_matrix(self.x, self.y, self.z), rot_y_matrix(self.yaw))
            self.__arrow_pts = transform(H, self.__arrow_pts)
        return self.__arrow_pts

    def __str__(self):
        return "Center: (%.3f, %.3f, %.3f)   HWL: (%.3f, %.3f, %.3f)  YAW: %.3f" % (self.x, self.y, self.z, self.h, self.w, self.l, self.yaw)

def get_corners_3D(box):
    corners = np.array([[-box.l / 2, box.l / 2, box.l / 2, -box.l / 2, -box.l / 2, box.l / 2, box.l / 2, -box.l / 2],
                        [0, 0, 0, 0, -box.h, -box.h, -box.h, -box.h],
                        [-box.w / 2, -box.w / 2, box.w / 2, box.w / 2, -box.w / 2, -box.w / 2, box.w / 2, box.w / 2]], dtype=np.float32)
    H = np.dot(translation_matrix(box.x, box.y, box.z), rot_y_matrix(box.yaw))
    return transform(H, corners)

def translate_box_3D(box, x, y, z):
    box.x += x
    box.y += y
    box.z += z
    return box
