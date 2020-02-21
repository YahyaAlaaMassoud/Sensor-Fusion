
import os
import numpy as np
import deepdish as dd

from viz import open3d, bev, imshow
from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
from core.boxes import Box3D
from data_utils.augmentation import PointCloudAugmenter
from pprint import pprint
from operator import itemgetter
from shapely.geometry import Polygon

'''
    https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
    https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    https://www.cs.hmc.edu/ACM/lectures/intersections.html
    https://docs.yoyogames.com/source/dadiospice/002_reference/movement%20and%20collisions/collisions/rectangle_in_triangle.html
    https://math.stackexchange.com/questions/3210769/how-do-i-check-if-a-3d-point-is-between-2-other-3d-points
'''

DS_DIR = '/home/salam/datasets/KITTI/training'
reader = KITTI(DS_DIR, CARS_ONLY)

ids = reader.get_ids('train')

def check_colinear(a, b, c):
    ab = b - a
    ac = c - a
    cross = np.cross(ab, ac)
    if cross.all() < 1e-12:
        # print(cross)
        # print(a, b, c)
        return True
    return False

def get_angle_between_two_vectors(src, p2, p3):
    a = src - p2
    b = src - p3
    dot = np.dot(a, b)
    mag_a = np.linalg.norm(a)
    mag_b = np.linalg.norm(b)
    angle = np.arccos(dot / (mag_a * mag_b))
    return angle

def intersectLines( pt1, pt2, ptA, ptB ): 
    """ this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    """

    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB
    dx = xB - x;  dy = yB - y

    # we need to find the (typically unique) values of r and s
    # that will satisfy
    #
    # (x1, y1) + r(dx1, dy1) = (x, y) + s(dx, dy)
    #
    # which is the same as
    #
    #    [ dx1  -dx ][ r ] = [ x-x1 ]
    #    [ dy1  -dy ][ s ] = [ y-y1 ]
    #
    # whose solution is
    #
    #    [ r ] = _1_  [  -dy   dx ] [ x-x1 ]
    #    [ s ] = DET  [ -dy1  dx1 ] [ y-y1 ]
    #
    # where DET = (-dx1 * dy + dy1 * dx)
    #
    # if DET is too small, they're parallel
    #
    DET = (-dx1 * dy + dy1 * dx)

    if np.fabs(DET) < DET_TOLERANCE: return (0,0,0,0,0)

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    return ( xi, yi, 1, r, s )

def point_in_triangle(p, p0, p1, p2):
    s = p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * p[0] + (p0[0] - p2[0]) * p[1]
    t = p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * p[0] + (p1[0] - p0[0]) * p[1]

    if ((s < 0) is not (t < 0)):
        return False

    A = -p1[1] * p2[0] + p0[1] * (p2[0] - p1[0]) + p0[0] * \
        (p1[1] - p2[1]) + p1[0] * p2[1]

    if A < 0:
        return s <= 0 and s + t >= A
    return s >= 0 and s + t <= A

def is_box_valid(limit, box_bev_pts):
    (x1, y1), (x2, y2) = limit
    limit_p = Polygon([(0, 0), (x1, y1), (x2, y2)])
    box_p   = Polygon([(box_bev_pts[0][0], box_bev_pts[0][1]),
                        (box_bev_pts[1][0], box_bev_pts[1][1]),
                        (box_bev_pts[2][0], box_bev_pts[2][1]),
                        (box_bev_pts[3][0], box_bev_pts[3][1])])
    if limit_p.intersection(box_p).area > 0:
        return False
    return True

t = '000013'#ids[4]
boxes = reader.get_boxes_3D(t)
pts, ref = reader.get_velo(t, use_fov_filter=False)

# open3d(pts, boxes)
# imshow(bev(pts, pred_boxes=boxes, title="GT"))

lidar_src = (0, 0)
border_1  = (70, 40), (70, -40)
border_2  = np.array([ 40,  0]), np.array([ 40, 70])
border_3  = np.array([-40,  0]), np.array([-40, 70])

circles = []
d1_clr, d2_clr = (255, 0, 0), (0, 255, 0)
box_limits = []
all_limits = []
rand_num   = 30
random_cars_dir = 'data_utils/aug_utils/annotations/cars/'
random_cars = os.listdir(random_cars_dir)
random_cars = np.random.choice(random_cars, size=rand_num)

sampled_cars = []
for random_car in random_cars:
    sampled_car = dd.io.load(os.path.join(random_cars_dir, random_car))
    sampled_pts = sampled_car['pts'].T
    sampled_box = Box3D(h=sampled_car['box_h'], 
                        w=sampled_car['box_w'],
                        l=sampled_car['box_l'],
                        x=sampled_car['box_x'],
                        y=sampled_car['box_y'],
                        z=sampled_car['box_z'],
                        yaw=sampled_car['box_yaw'],
                        cls=sampled_car['box_cls'])
    data = {
        'pts': sampled_pts,
        'num_pts': sampled_pts.shape[1],
        'box': sampled_box,
        'bev_corners': sampled_box.get_bev_box().T,
        'bev_center': sampled_box.get_bev_center(),
    }
    sampled_cars.append(data)
    
sampled_cars = sorted(sampled_cars, key=itemgetter('num_pts'), reverse=False)

def get_box_limit(box):
    diag_1, diag_2 = box.get_bev_diags()
    angle_diag_1 = get_angle_between_two_vectors(lidar_src, diag_1[0], diag_1[1])
    angle_diag_2 = get_angle_between_two_vectors(lidar_src, diag_2[0], diag_2[1])
    
    if angle_diag_1 > angle_diag_2:
        ref_diag = diag_1
    else:
        ref_diag = diag_2
        
    # for pt in ref_diag:
    #     circles.append(((pt[0], pt[1]), d1_clr))
    # c = box.get_bev_center()
    # circles.append((c, d1_clr))
    
    inter1_x, inter1_y, _, _, _ = intersectLines(border_1[0], border_1[1], lidar_src, (ref_diag[0][0], ref_diag[0][1]))
    inter2_x, inter2_y, _, _, _ = intersectLines(border_1[0], border_1[1], lidar_src, (ref_diag[1][0], ref_diag[1][1]))
        
    all_limits.append(np.array([[inter1_y, 0, inter1_x],
                       [0,0,0]]))
    
    all_limits.append(np.array([[inter2_y, 0, inter2_x],
                       [0,0,0]]))
    
    return (inter1_x, inter1_y), (inter2_x, inter2_y)

for i, box in enumerate(boxes):
    box = boxes[i]
    (x1, y1), (x2, y2) = get_box_limit(box)
    box_limits.append(((x1, y1), (x2, y2)))
    

cnt     = 0
new_pts = []
for car in sampled_cars:
    valid = True
    for limit in box_limits:
        (x1, y1), (x2, y2) = limit
        
        box_pts = car['bev_corners']
        
        limit_p = Polygon([(0,0), (x1, y1), (x2, y2)])
        box_p   = Polygon([(box_pts[0][0], box_pts[0][1]),
                           (box_pts[1][0], box_pts[1][1]),
                           (box_pts[2][0], box_pts[2][1]),
                           (box_pts[3][0], box_pts[3][1])])
        if limit_p.intersection(box_p).area > 0:
            valid = False
        # print('res', valid)
        # print('---')
        del limit_p, box_p
    
    if valid:
        new_pts.append(car['pts'])
        boxes.append(car['box'])
        (x1, y1), (x2, y2) = get_box_limit(car['box'])
        box_limits.append(((x1, y1), (x2, y2)))
    else:
        cnt += 1

print('sampled {0} out of {1}'.format(rand_num-cnt, rand_num))


ss = None
for new_sample in new_pts:
    new_sample_pts = new_sample.T
    # pts = np.concatenate((pts, new_sample), axis=1)
    print(new_sample_pts.shape)
    new_ids = []
    for i in range(new_sample_pts.shape[0]):
        q = pts.T - new_sample_pts[i,:]
        s = np.reshape(-1. * new_sample_pts[i,:], (1,3))
        # print(q.shape, s.shape)
        cc = np.cross(s, q)
        # print(cc.shape)
        inds = np.where(cc == 0)[0]
        # print(len(inds), inds)
        new_ids.extend(inds)
print(len(new_ids))
ss = pts.T[new_ids]

open3d(ss.T, boxes, limits=all_limits)
# # open3d(sampled_pts2, boxes)
# imshow(bev(pts, pred_boxes=boxes, title="GT", circles=circles))
