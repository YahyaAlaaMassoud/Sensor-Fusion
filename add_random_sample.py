
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

def line(p1, p2):
    A = p1[1] - p2[1]
    B = p2[0] - p1[0]
    C = p1[0] * p2[1] - p2[0] * p1[1]
    return A, B, -C

def intersection(L1, L2):
    D  = L1[0] * L2[1] - L1[1] * L2[0]
    Dx = L1[2] * L2[1] - L1[1] * L2[2]
    Dy = L1[0] * L2[2] - L1[2] * L2[0]
    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

def find_intersection(a, b, c, d):
    l1, l2 = line(a, b), line(c, d)
    inter = intersection(l1, l2)
    return inter


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

open3d(pts, boxes)
# imshow(bev(pts, pred_boxes=boxes, title="GT"))

lidar_src = np.array([0, 0])
border_1  = np.array([ 40, 70]), np.array([-40, 70])
border_2  = np.array([ 40,  0]), np.array([ 40, 70])
border_3  = np.array([-40,  0]), np.array([-40, 70])

circles = []
d1_clr, d2_clr = (255, 0, 0), (0, 255, 0)
box_limits = []
all_limits = []

random_cars_dir = 'data_utils/aug_utils/annotations/cars/'
random_cars = os.listdir(random_cars_dir)
random_cars = np.random.choice(random_cars, size=30)

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
    
    inter_11 = find_intersection(lidar_src, ref_diag[0], border_1[0], border_1[1])
    inter_12 = find_intersection(lidar_src, ref_diag[1], border_1[0], border_1[1])
    
    inter_21 = find_intersection(lidar_src, ref_diag[0], border_2[0], border_2[1])
    inter_22 = find_intersection(lidar_src, ref_diag[1], border_2[0], border_2[1])
    
    inter_31 = find_intersection(lidar_src, ref_diag[0], border_3[0], border_3[1])
    inter_32 = find_intersection(lidar_src, ref_diag[1], border_3[0], border_3[1])
    
    return ref_diag#(inter_11, inter_12), (inter_21, inter_22), (inter_31, inter_32)

for i, box in enumerate(boxes):
    
    box = boxes[i]
    
    # (inter_11, inter_12), (inter_21, inter_22), (inter_31, inter_32) = get_box_limit(box)
    diag = get_box_limit(box)
    
    # box_limits.append(((inter_11, inter_12), (inter_21, inter_22), (inter_31, inter_32)))
    box_limits.append(diag)
    
    all_limits.append([[diag[0][1], 0, diag[0][0]],
                       [0,0,0]])
    
    all_limits.append([[diag[1][1], 0, diag[1][0]],
                       [0,0,0]])
            
    # all_limits.append([[inter_12[0], 0, inter_12[1]],
    #                    [0,0,0]])

    # all_limits.append([[inter_21[1], 0, inter_21[0]],
    #                    [inter_22[1], 0, inter_22[0]]])
    
    # all_limits.append([[inter_31[1], 0, inter_31[0]],
    #                    [inter_32[1], 0, inter_32[0]]])


cnt = 0
for car in sampled_cars:
    valid = True
    for limit in box_limits:
        # limit1, limit2, limit3 = limit
        # limits = [limit1, limit2, limit3]
    # for lim in limits:
        for pt in car['bev_corners']:
            is_found = point_in_triangle(pt, lidar_src, limit[0], limit[1])
            if is_found:
                valid = False
        if point_in_triangle(car['bev_center'], lidar_src, limit[0], limit[1]):
            valid = False
    
    if valid:
        pts = np.concatenate((pts, car['pts']), axis=1)
        boxes.append(car['box'])
        # (inter_11, inter_12), (inter_21, inter_22), (inter_31, inter_32) = get_box_limit(car['box'])
        # box_limits.append(((inter_11, inter_12), (inter_21, inter_22), (inter_31, inter_32)))
        diag = get_box_limit(box)
        box_limits.append(diag)
    else:
        cnt += 1

print('la2 {0}'.format(cnt))
print(len(all_limits))

open3d(pts, boxes, limits=None)
# # open3d(sampled_pts2, boxes)
# # imshow(bev(pts, pred_boxes=boxes, title="GT", circles=circles))
