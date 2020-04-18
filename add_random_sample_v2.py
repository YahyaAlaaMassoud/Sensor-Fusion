
import os
import numpy as np
import deepdish as dd

from viz import open3d, bev, imshow
from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
from core.boxes import Box3D, translate_box_3D
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
    https://math.stackexchange.com/questions/274712/calculate-on-which-side-of-a-straight-line-is-a-given-point-located
    https://forums.anandtech.com/threads/given-a-plane-and-a-point-how-can-i-determine-which-side-of-the-plane-the-point-is-on.162930/
    https://math.stackexchange.com/questions/214187/point-on-the-left-or-right-side-of-a-plane-in-3d-space
    https://stackoverflow.com/questions/15688232/check-which-side-of-a-plane-points-are-on
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

def check_point_side_3d(a, b, c, pts):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    b_ = b - a
    c_ = c - a
    n    = np.cross(b_, c_).T
    side = np.dot(n, pts)
    d = np.where(side < 0, -1, 1)
    return d

def check_point_side_2d(ax, ay, bx, by, pts_x, pts_y):
    d = (pts_x - ax) * (by - ay) - (pts_y - ay) * (bx - ax)
    d = np.where(d < 0, -1, 1)
    return d

for t in ['000010', '000011', '000012', '000013', '000014']:
    # t = '000010'#ids[4]
    boxes = reader.get_boxes_3D(t)
    if len(boxes) <= 0:
        print(len(boxes))
        continue
    pts, ref = reader.get_velo(t, use_fov_filter=False)

    ys = []
    for box in boxes:
        ys.append(box.y)

    ys = sorted(ys, reverse=False)
    median_y = np.median(ys)
    mean_y   = np.mean(ys)
    # print(ys)
    # print(median_y)
    # print(mean_y)

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
    np.random.seed(0)
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
            'frame_id': sampled_car['frame_id'], 
            'pts': sampled_pts,
            'num_pts': sampled_pts.shape[1],
            'box': sampled_box,
            'bev_corners': sampled_box.get_bev_box().T,
            'bev_center': sampled_box.get_bev_center(),
        }
        sampled_cars.append(data)
        
    sampled_cars = sorted(sampled_cars, key=itemgetter('num_pts'), reverse=False)

    def get_box_limit(box, name=None):
        diag_1, diag_2 = box.get_bev_diags()
        diag_3d_1, diag_3d_2 = box.get_3d_diag()
        
        angle_diag_1 = get_angle_between_two_vectors(lidar_src, diag_1[0], diag_1[1])
        angle_diag_2 = get_angle_between_two_vectors(lidar_src, diag_2[0], diag_2[1])
        
        if angle_diag_1 > angle_diag_2:
            ref_diag = diag_1
            diag_3d  = diag_3d_1
        else:
            ref_diag = diag_2
            diag_3d = diag_3d_2

        inter1_x, inter1_y, _, _, _ = intersectLines(border_1[0], border_1[1], lidar_src, (ref_diag[0][0], ref_diag[0][1]))
        inter2_x, inter2_y, _, _, _ = intersectLines(border_1[0], border_1[1], lidar_src, (ref_diag[1][0], ref_diag[1][1]))
            
        # pprint(diag_3d)
        # pprint(ref_diag)
        # print(diag_3d[0][0][-1])
        # print(diag_3d[1][0][-1])
        min_z = diag_3d[0][0][-1]
        max_z = diag_3d[1][0][-1]

        if name is not None:
            all_limits.append(np.array([[inter1_y, 0, inter1_x],
                                        [0,0,0]]))
            
            all_limits.append(np.array([[inter2_y, 0, inter2_x],
                                        [0,0,0]]))

            all_limits.append(np.array([[ref_diag[0][1], min_z, ref_diag[0][0]],
                                        [0,0,0]]))
            
            all_limits.append(np.array([[ref_diag[1][1], min_z, ref_diag[1][0]],
                                        [0,0,0]]))

        
        return (inter1_x, inter1_y), (inter2_x, inter2_y), max_z, min_z, ref_diag

    for i, box in enumerate(boxes):
        box = boxes[i]
        (x1, y1), (x2, y2), max_z, min_z, ref_diag = get_box_limit(box)
        box_limits.append(((x1, y1), (x2, y2), max_z, min_z, ref_diag))
        

    cnt     = 0
    new_pts = []
    valid_cars = []
    valid_cars_limit = {}
    new_boxes = []
    for car in sampled_cars:
        valid = True
        valid_cars_limit[car['frame_id']] = []
        for limit in box_limits:
            (x1, y1), (x2, y2), _, _, _ = limit
            
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
            y_diff = median_y - car['box'].y
            # print('median', median_y)
            # # y_diff = mean_y - car['box'].y
            # print('mean  ', mean_y)
            # print('real y', car['box'].y)
            # print('y_diff', y_diff)
            # print('------')
            # print('old y', car['box'].y)
            car_pts, car_box, _ = PointCloudAugmenter.rotate_translate(rotation=0, translation=[[0, y_diff, 0]])(gt_boxes_3d=[car['box']], pts=car['pts'].T)
            car_box = car_box[0]
            car_box = Box3D(h=car_box.h, w=car_box.w, l=car_box.l, x=car_box.x, y=car_box.y, z=car_box.z, yaw=car_box.yaw, cls=car_box.cls)
            # print('new y', car_box.y)
            # print('-----')
            # boxes.append(car_box)
            car['box'] = car_box
            car['pts'] = car_pts.T
            (x1, y1), (x2, y2), max_z, min_z, ref_diag = get_box_limit(car_box, 'new')
            box_limits.append(((x1, y1), (x2, y2), max_z, min_z, ref_diag))
            valid_cars.append(car)
            valid_cars_limit[car['frame_id']].append(((x1, y1), (x2, y2), max_z, min_z, ref_diag))
            new_boxes.append(car_box)
        else:
            cnt += 1

    print('sampled {0} out of {1}'.format(rand_num-cnt, rand_num))

    # open3d(pts, gt_boxes=boxes, sampled_boxes=new_boxes, limits=[])

    to_delete_pts = None
    final_boxes = []
    for i, car in enumerate(valid_cars):
        for limit in valid_cars_limit[car['frame_id']]:
            (y1, x1), (y2, x2), max_z, min_z, ref_diag = limit
            # print((x1, y1), (x2, y2), max_z, min_z, ref_diag)
            side1_a, side1_b = (0, 0), (x2, y2) # Left
            side2_a, side2_b = (0, 0), (x1, y1) # Right
            # pprint(car['box'].get_corners().T)
            side4_a, side4_b, side4_c = (0, 0, 0), (ref_diag[0][0], min_z, ref_diag[0][1]), (ref_diag[1][0], min_z, ref_diag[1][1]) # Bottom diag
            side5_a, side5_b, side5_c = (0, 0, 0), (ref_diag[0][0], max_z, ref_diag[0][1]), (ref_diag[1][0], max_z, ref_diag[1][1]) # top diag

            d1 = check_point_side_2d(side1_a[1], side1_a[0], side1_b[1], side1_b[0], pts[2,:], pts[0,:]) # left
            d2 = check_point_side_2d(side2_a[1], side2_a[0], side2_b[1], side2_b[0], pts[2,:], pts[0,:]) # right
            d  = check_point_side_3d(side4_a, side4_b, side4_c, pts[(2, 1, 0), :]) # up (bottom diag)
            dt = check_point_side_3d(side5_a, side5_b, side5_c, pts[(2, 1, 0), :]) # up (top diag)
            ds = check_point_side_2d(ref_diag[0][0], ref_diag[0][1], ref_diag[1][0], ref_diag[1][1], pts[2,:], pts[0,:]) # up front/back
            inds_d1 = np.where((d1 ==  1) & (d2 == -1) & (d ==  1))[0]
            inds_d2 = np.where((d1 == -1) & (d2 ==  1) & (d == -1))[0]
            # print(inds_d1.shape, inds_d2.shape)
            if to_delete_pts is None and len(inds_d1) is not 0:
                to_delete_pts = pts[:, inds_d1]
                cur_del_ids   = np.where((d1 == 1) & (d2 == -1) & (d ==  1) & (ds == -1))[0]
                cur_del = pts[:, cur_del_ids]
            elif to_delete_pts is None and len(inds_d2) is not 0:
                to_delete_pts = pts[:, inds_d2]
                cur_del_ids   = np.where((d1 == -1) & (d2 == 1) & (d == -1) & (ds == 1))[0]
                cur_del = pts[:, cur_del_ids]
            elif to_delete_pts is not None and len(inds_d1) is not 0:
                to_delete_pts = np.concatenate((to_delete_pts, pts[:, inds_d1]), axis=1)
                cur_del_ids   = np.where((d1 == 1) & (d2 == -1) & (d == 1) & (ds == -1))[0]
                cur_del = pts[:, cur_del_ids]
            elif to_delete_pts is not None and len(inds_d2) is not 0:
                to_delete_pts = np.concatenate((to_delete_pts, pts[:, inds_d2]), axis=1)
                cur_del_ids   = np.where((d1 == -1) & (d2 == 1) & (d == -1) & (ds == 1))[0]
                cur_del = pts[:, cur_del_ids]
            
            keep_ids1 = np.where((d1 == 1) & (d2 == -1) & (dt == 1) & (ds == 1))[0]
            keep_ids2 = np.where((d1 == -1) & (d2 == 1) & (dt == -1) & (ds == -1))[0]
            # print(len(keep_ids1), len(keep_ids2))
            pts_keep = pts[:, keep_ids1]
            pts_keep = np.concatenate((pts_keep, pts[:, keep_ids2]), axis=1)

            bev_box = car['box'].get_bev_box().T
            l1 = check_point_side_2d(bev_box[0][0], bev_box[0][1], bev_box[1][0], bev_box[1][1], pts_keep[2,:], pts_keep[0,:])
            l2 = check_point_side_2d(bev_box[1][0], bev_box[1][1], bev_box[2][0], bev_box[2][1], pts_keep[2,:], pts_keep[0,:])
            l3 = check_point_side_2d(bev_box[2][0], bev_box[2][1], bev_box[3][0], bev_box[3][1], pts_keep[2,:], pts_keep[0,:])
            l4 = check_point_side_2d(bev_box[3][0], bev_box[3][1], bev_box[0][0], bev_box[0][1], pts_keep[2,:], pts_keep[0,:])
            dt2 = check_point_side_3d(side5_a, side5_b, side5_c, pts_keep[(2, 1, 0), :])
            rem_top = np.where((l1 == 1) & (l3 == 1) & (l2 == 1) & (l4 == 1) & (dt2 == 1))[0]

            pts_keep = np.delete(pts_keep, rem_top, axis=1)


            if cur_del is not None:
                # print(cur_del.shape)
                if cur_del.shape[1] <= 150:
                    pts = np.delete(pts, inds_d2, axis=1)
                    pts = np.delete(pts, inds_d1, axis=1)
                    final_boxes.append(car['box'])
                    new_pts.append(car['pts'])
            
        
            pts = np.concatenate((pts, pts_keep), axis=1)


            # open3d(pts, gt_boxes=boxes, sampled_boxes=final_boxes, limits=[])

            # d = check_point_side_3d(side4_b, side4_c, pts)
            # inds_d = np.where(d == -1)[0]
            # to_delete_pts3d = pts[:, inds_d]
            # print(inds_d.shape)

    for new_sample in new_pts:
        new_sample_pts = new_sample.T
        pts = np.concatenate((pts, new_sample), axis=1)

    print('got     {0} out of {1}'.format(len(final_boxes), rand_num))
    print('----------------------')

    # open3d(to_delete_pts, gt_boxes=boxes, sampled_boxes=final_boxes, limits=all_limits)
    open3d(pts, gt_boxes=boxes, sampled_boxes=final_boxes, limits=all_limits)