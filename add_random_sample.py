
import os
import numpy as np
import deepdish as dd

from viz import open3d, bev, imshow
from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
from core.boxes import Box3D
from data_utils.augmentation import PointCloudAugmenter
from pprint import pprint
from operator import itemgetter

'''
    https://stackoverflow.com/questions/21037241/how-to-determine-a-point-is-inside-or-outside-a-cube
    https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    https://stackoverflow.com/questions/1211212/how-to-calculate-an-angle-from-three-points
    https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
    https://www.cs.hmc.edu/ACM/lectures/intersections.html
    https://docs.yoyogames.com/source/dadiospice/002_reference/movement%20and%20collisions/collisions/rectangle_in_triangle.html
'''

DS_DIR = '/home/salam/datasets/KITTI/training'
reader = KITTI(DS_DIR, CARS_ONLY)

ids = reader.get_ids('train')

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

t = '000010'#ids[4]
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

random_cars_dir = 'data_utils/aug_utils/annotations/cars/'
random_cars = os.listdir(random_cars_dir)
random_cars = np.random.choice(random_cars, size=20)

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
    
sampled_cars = sorted(sampled_cars, key=itemgetter('num_pts'), reverse=True)

def get_box_limit(box, name=None):
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
    
    # print((ref_diag[0][0], ref_diag[0][1]))
    # print((ref_diag[1][0], ref_diag[1][1]))
    inter1_x, inter1_y, _, _, _ = intersectLines(border_1[0], border_1[1], lidar_src, (ref_diag[0][0], ref_diag[0][1]))
    inter2_x, inter2_y, _, _, _ = intersectLines(border_1[0], border_1[1], lidar_src, (ref_diag[1][0], ref_diag[1][1]))
    # print(inter1_x, inter1_y)
    # print(inter2_x, inter2_y)
    
    if name:
        print(name)
        
    all_limits.append(np.array([[inter1_y, 0, inter1_x],
                       [0,0,0]]))
    
    all_limits.append(np.array([[inter2_y, 0, inter2_x],
                       [0,0,0]]))
    
    return (inter1_x, inter1_y), (inter2_x, inter2_y)

for i, box in enumerate(boxes):
    
    box = boxes[i]
    
    (x1, y1), (x2, y2) = get_box_limit(box)
    
    box_limits.append(((x1, y1), (x2, y2)))

cnt = 0
for car in sampled_cars:
    valid = True
    for limit in box_limits:
        (x1, y1), (x2, y2) = limit
        limit = np.array([[x1, y1],
                          [x2, y2]])
        for pt in car['bev_corners']:
            is_found = point_in_triangle(pt, lidar_src, limit[0], limit[1])
            if is_found:
                valid = False
        if point_in_triangle(car['bev_center'], lidar_src, limit[0], limit[1]):
            valid = False
    
    if valid:
        pts = np.concatenate((pts, car['pts']), axis=1)
        boxes.append(car['box'])
        print(len(all_limits))
        (x1, y1), (x2, y2) = get_box_limit(box, 'new')
        print(len(all_limits))
        box_limits.append(((x1, y1), (x2, y2)))
    else:
        cnt += 1

print('la2 {0}'.format(cnt))
# print(len(all_limits))


open3d(pts, boxes, limits=all_limits)
# # open3d(sampled_pts2, boxes)
# imshow(bev(pts, pred_boxes=boxes, title="GT", circles=circles))
