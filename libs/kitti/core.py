import numpy as np

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

def rot_y_matrix(angle):
    return np.array([[np.math.cos(angle), 0, np.math.sin(angle), 0],
                     [0, 1, 0, 0],
                     [-np.math.sin(angle), 0, np.math.cos(angle), 0],
                     [0, 0, 0, 1]], dtype=np.float32)

def translation_matrix(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=np.float32)


def C2H(pts):
    return np.insert(pts, 3, values=1, axis=0)


def H2C(pts):
    return pts[:3, :] / pts[3:, :]


def transform(H, pts):
    return H2C(np.dot(H, C2H(pts)))


def project(P, pts):
    pts = transform(P, pts)
    pts = pts[:2, :] / pts[2, :]
    return pts
