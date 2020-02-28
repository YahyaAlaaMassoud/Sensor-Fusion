
import os
import numpy as np
import deepdish as dd
import timeit

# from viz import open3d, bev, imshow
from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
from core.boxes import Box3D, translate_box_3D
from data_utils.augmentation import PointCloudAugmenter
from pprint import pprint
from operator import itemgetter
from shapely.geometry import Polygon

all_limits = []

def get_angle_between_vectors(src, p1, p2):
    src, p1, p2 = np.array(src), np.array(p1), np.array(p2)
    a = src - p1
    b = src - p2
    dot = np.dot(a, b)
    mag_a, mag_b = np.linalg.norm(a), np.linalg.norm(b)
    angle = np.arccos(dot / (mag_a * mag_b))
    return angle

def find_intersection_point(pt1, pt2, ptA, ptB):
    """ 
        this returns the intersection of Line(pt1,pt2) and Line(ptA,ptB)
        
        returns a tuple: (xi, yi, valid, r, s), where
        (xi, yi) is the intersection
        r is the scalar multiple such that (xi,yi) = pt1 + r*(pt2-pt1)
        s is the scalar multiple such that (xi,yi) = pt1 + s*(ptB-ptA)
            valid == 0 if there are 0 or inf. intersections (invalid)
            valid == 1 if it has a unique intersection ON the segment    
    """
    DET_TOLERANCE = 0.00000001

    # the first line is pt1 + r*(pt2-pt1)
    # in component form:
    x1, y1 = pt1;   x2, y2 = pt2
    dx1 = x2 - x1;  dy1 = y2 - y1

    # the second line is ptA + s*(ptB-ptA)
    x, y = ptA;   xB, yB = ptB
    dx = xB - x;  dy = yB - y

    DET = (-dx1 * dy + dy1 * dx)

    if np.fabs(DET) < DET_TOLERANCE: 
        return 0, 0, 0, 0, 0

    # now, the determinant should be OK
    DETinv = 1.0/DET

    # find the scalar amount along the "self" segment
    r = DETinv * (-dy  * (x-x1) +  dx * (y-y1))

    # find the scalar amount along the input line
    s = DETinv * (-dy1 * (x-x1) + dx1 * (y-y1))

    # return the average of the two descriptions
    xi = (x1 + r*dx1 + x + s*dx)/2.0
    yi = (y1 + r*dy1 + y + s*dy)/2.0
    return xi, yi, 1, r, s

def check_point_side_3d(a, b, c, pts):
    a, b, c = np.asarray(a), np.asarray(b), np.asarray(c)
    b_   = b - a
    c_   = c - a
    n    = np.cross(b_, c_).T
    side = np.dot(n, pts)
    d = np.where(side < 0, -1, 1)
    return d

def check_point_side_2d(ax, ay, bx, by, pts_x, pts_y):
    d = (pts_x - ax) * (by - ay) - (pts_y - ay) * (bx - ax)
    d = np.where(d < 0, -1, 1)
    return d


def add_random_sample(num_samples=30, sort_desc=False, filter_wall_thresh=150, random_samples_dir='data_utils/aug_utils/annotations/cars/'):

    random_samples     = os.listdir(random_samples_dir)
    random_samples     = np.random.choice(random_samples, size=num_samples)

    samples            = []

    for random_sample in random_samples:
        sample     = dd.io.load(os.path.join(random_samples_dir, random_sample))
        sample_pts = sample['pts'].T
        sample_box = Box3D(h=sample['box_h'], 
                           w=sample['box_w'],
                           l=sample['box_l'],
                           x=sample['box_x'],
                           y=sample['box_y'],
                           z=sample['box_z'],
                           yaw=sample['box_yaw'],
                           cls=sample['box_cls'])
        
        samples.append({
            'frame_id': sample['frame_id'],
            'pts':      sample_pts,
            'num_pts':  sample_pts.shape[1],
            'box':      sample_box,
        })

    samples = sorted(samples, 
                     key=itemgetter('num_pts'),
                     reverse=sort_desc)

    lidar_src    = (0, 0)
    lidar_src_3d = (0, 0, 0)
    border_start = (70, 40)
    border_end   = (70, -40)

    def _get_box_limits(box, name=None):
        diag1, diag2       = box.get_bev_diags()
        diag1_3d, diag2_3d = box.get_3d_diag()

        angle1 = get_angle_between_vectors(lidar_src, diag1[0], diag1[1])
        angle2 = get_angle_between_vectors(lidar_src, diag2[0], diag2[1])

        if angle1 > angle2:
            ref_diag = diag1
            diag_3d  = diag1_3d
        else:
            ref_diag = diag2
            diag_3d  = diag2_3d

        inter1_x, inter1_y, _, _, _ = find_intersection_point(border_start,
                                                              border_end,
                                                              lidar_src,
                                                              (ref_diag[0][0], ref_diag[0][1]))

        inter2_x, inter2_y, _, _, _ = find_intersection_point(border_start,
                                                              border_end,
                                                              lidar_src,
                                                              (ref_diag[1][0], ref_diag[1][1]))

        min_h = diag_3d[0][0][-1]
        max_h = diag_3d[1][0][-1]

        # if name is not None:
        #     all_limits.append(np.array([[inter1_y, 0, inter1_x],
        #                                 [0,0,0]]))
            
        #     all_limits.append(np.array([[inter2_y, 0, inter2_x],
        #                                 [0,0,0]]))

            # all_limits.append(np.array([[ref_diag[0][1], min_z, ref_diag[0][0]],
            #                             [0,0,0]]))
            
            # all_limits.append(np.array([[ref_diag[1][1], min_z, ref_diag[1][0]],
            #                             [0,0,0]]))

        return {
            'inter1':   (inter1_x, inter1_y),
            'inter2':   (inter2_x, inter2_y),
            'ref_diag': ref_diag,
            'max_h':    max_h,
            'min_h':    min_h,
        }

    def _add_random_sample(gt_boxes, pts, ref=None):

        if pts.shape[0] != 3:
            pts = pts.T

        ys = sorted([box.y for box in gt_boxes], reverse=False)
        median_y = np.median(ys)
        
        box_limits = []

        for box in gt_boxes:
            limit_dict = _get_box_limits(box)
            box_limits.append(limit_dict)

        valid_samples        = []
        valid_samples_limits = {}
        for sample in samples:
            valid = True
            valid_samples_limits[sample['frame_id']] = []

            for limit in box_limits:
                (x1, y1) = limit['inter1']
                (x2, y2) = limit['inter2']
                box_bev  = sample['box'].get_bev_box().T

                limit_p = Polygon([lidar_src,
                                   (x1, y1),
                                   (x2, y2)])
                box_p   = Polygon([(box_bev[0][0], box_bev[0][1]),
                                   (box_bev[1][0], box_bev[1][1]),
                                   (box_bev[2][0], box_bev[2][1]),
                                   (box_bev[3][0], box_bev[3][1])])

                if limit_p.intersection(box_p).area > 0:
                    valid = False

                del limit_p, box_p
            
            if valid:
                y_diff = median_y - sample['box'].y

                new_pts, new_boxes, _ = PointCloudAugmenter.rotate_translate(rotation=0, translation=[[0, y_diff, 0]])(gt_boxes_3d=[sample['box']], pts=sample['pts'].T)
                new_box = new_boxes[0]
                new_box = Box3D(h=new_box.h, 
                                w=new_box.w,
                                l=new_box.l,
                                x=new_box.x,
                                y=new_box.y,
                                z=new_box.z,
                                yaw=new_box.yaw,
                                cls=new_box.cls)
                
                sample['box'] = new_box
                sample['pts'] = new_pts.T

                new_limit = _get_box_limits(sample['box'])
                box_limits.append(new_limit)
                valid_samples_limits[sample['frame_id']].append(new_limit)
                valid_samples.append(sample)
        
        new_pts       = []
        final_samples = []
        for sample in valid_samples:
            for limit in valid_samples_limits[sample['frame_id']]:
                (y1, x1) = limit['inter1']
                (y2, x2) = limit['inter2']
                max_h, min_h = limit['max_h'], limit['min_h']
                ref_diag = limit['ref_diag']

                side1_a, side1_b = lidar_src, (x2, y2) # left  of frustum (line)
                side2_a, side2_b = lidar_src, (x1, y1) # right of frustum (line)
                
                side4_a, side4_b, side4_c = lidar_src_3d, (ref_diag[0][0], min_h, ref_diag[0][1]), (ref_diag[1][0], min_h, ref_diag[1][1]) # bottom diag (plane)
                side5_a, side5_b, side5_c = lidar_src_3d, (ref_diag[0][0], max_h, ref_diag[0][1]), (ref_diag[1][0], max_h, ref_diag[1][1]) # top    diag (plane)

                # check the side of all points using the left line of the frustum
                d_l = check_point_side_2d(side1_a[1], side1_a[0], side1_b[1], side1_b[0], pts[2,:], pts[0,:])
                # check the side of all points using the right line of the frustum
                d_r = check_point_side_2d(side2_a[1], side2_a[0], side2_b[1], side2_b[0], pts[2,:], pts[0,:]) 
                # check the side of all points using the plane from lidar source to the coordinates of the bottom diagonal
                d_b = check_point_side_3d(side4_a, side4_b, side4_c, pts[(2, 1, 0), :])
                # check the side of all points using the plane from lidar source to the coordinates of the top diagonal
                d_t = check_point_side_3d(side5_a, side5_b, side5_c, pts[(2, 1, 0), :])
                # check the side of all points using a line between the diagonal coordinates (find pts in_front/behind the car using its diagonal as reference)
                d_d = check_point_side_2d(ref_diag[0][0], ref_diag[0][1], ref_diag[1][0], ref_diag[1][1], pts[2,:], pts[0,:])

                inds_d1 = np.where((d_l ==  1) & (d_r == -1) & (d_b ==  1))[0]
                inds_d2 = np.where((d_l == -1) & (d_r ==  1) & (d_b == -1))[0]

                # cur_del are the points that are going to be deleted from the original PC to add the new sample
                if len(inds_d1) is not 0:
                    cur_del_ids = np.where((d_l == 1) & (d_r == -1) & (d_b == 1) & (d_d == -1))[0]
                    cur_del     = pts[:, cur_del_ids]
                elif len(inds_d2) is not 0:
                    cur_del_ids = np.where((d_l == -1) & (d_r == 1) & (d_b == -1) & (d_d == 1))[0]
                    cur_del     = pts[:, cur_del_ids]

                # keep the points that are above the top diagonals of the new sampled box (possible buildings behind the box)
                keep_ids1 = np.where((d_l == 1) & (d_r == -1) & (d_t == 1) & (d_d == 1))[0]
                keep_ids2 = np.where((d_l == -1) & (d_r == 1) & (d_t == -1) & (d_d == -1))[0]
                
                pts_keep = None
                if len(keep_ids1) is not 0:
                    pts_keep = pts[:, keep_ids1]
                    pts_keep = np.concatenate((pts_keep, pts[:, keep_ids2]), axis=1)
                elif len(keep_ids2) is not 0:
                    pts_keep = pts[:, keep_ids2]

                if pts_keep is not None:
                    # get the points that are above the car (on top) to be removed 
                    bev_box = sample['box'].get_bev_box().T
                    l1 = check_point_side_2d(bev_box[0][0], bev_box[0][1], bev_box[1][0], bev_box[1][1], pts_keep[2,:], pts_keep[0,:])
                    l2 = check_point_side_2d(bev_box[1][0], bev_box[1][1], bev_box[2][0], bev_box[2][1], pts_keep[2,:], pts_keep[0,:])
                    l3 = check_point_side_2d(bev_box[2][0], bev_box[2][1], bev_box[3][0], bev_box[3][1], pts_keep[2,:], pts_keep[0,:])
                    l4 = check_point_side_2d(bev_box[3][0], bev_box[3][1], bev_box[0][0], bev_box[0][1], pts_keep[2,:], pts_keep[0,:])

                    d_t2 = check_point_side_3d(side5_a, side5_b, side5_c, pts_keep[(2, 1, 0), :])
                    rem_top = np.where((l1 == 1) & (l3 == 1) & (l2 == 1) & (l4 == 1) & (d_t2 == 1))[0]

                    if len(rem_top) > 0:
                        mask = np.ones((pts_keep.shape[1]), dtype=bool)
                        mask[rem_top] = False
                        pts_keep = pts_keep[:,mask]
                    
                    if cur_del is not None:
                        if cur_del.shape[1] <= filter_wall_thresh:
                            mask = np.ones((pts.shape[1]), dtype=bool)
                            mask[inds_d1] = False
                            mask[inds_d2] = False
                            pts = pts[:,mask]
                            final_samples.append(sample['box'])
                            pts = np.concatenate((pts, sample['pts']), axis=1)
                            pts = np.concatenate((pts, pts_keep), axis=1)
                            # _get_box_limits(sample['box'], '')

        # print('added {0} new boxes in the scene'.format(len(final_samples)))
        
        gt_boxes.extend(final_samples)
        return pts, gt_boxes, ref

    return _add_random_sample

# DS_DIR = '/home/salam/datasets/KITTI/training'
# reader = KITTI(DS_DIR, CARS_ONLY)

# ids = reader.get_ids('train')

# for t in ids[40:100]:
#     # t = '000012'
#     gt_boxes_3d = reader.get_boxes_3D(t)
#     org_pts, _  = reader.get_velo(t, use_fov_filter=False)

#     if len(gt_boxes_3d) <= 1:
#         continue

#     np.random.seed(0)
#     pts, new_boxes, _ = add_random_sample(num_samples=40, sort_desc=True)(gt_boxes_3d, org_pts)
#     print('-----------------')


#     # open3d(org_pts, gt_boxes=gt_boxes, sampled_boxes=new_boxes, limits=all_limits)

#     # open3d(org_pts, gt_boxes_3d, limits=[])
#     open3d(pts, gt_boxes=gt_boxes_3d, sampled_boxes=new_boxes, limits=all_limits)
#     all_limits = []

#     # imshow(bev(org_pts, gt_boxes_3d))
#     # imshow(bev(pts, new_boxes))
