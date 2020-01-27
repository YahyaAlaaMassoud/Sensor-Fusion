import numpy as np


def C2H(pts):
    return np.insert(pts, 2, values=1, axis=0)


def H2C(pts):
    return pts[:2, :] / pts[2:, :]


def scale_matrix(dx, dy):
    return np.array([[dx, 0, 0],
                     [0, dy, 0],
                     [0, 0, 1]], dtype=np.float32)


def transform(H, pts):
    return H2C(np.dot(H, C2H(pts)))
