
import numpy as np
import random

class PointCloudAugmenter:
    @staticmethod
    def rot_matrix_3d(th_x=0, th_y=0, th_z=0, direction='cw'):
        if direction not in ['cw', 'ccw']:
            raise ValueError()

        if direction == 'cw':
            rot_x = np.asarray([[1, 0, 0], [0, np.cos(th_x), -np.sin(th_x)], [0, np.sin(th_x), np.cos(th_x)]])
            rot_y = np.asarray([[np.cos(th_y), 0, np.sin(th_y)], [0, 1, 0], [-np.sin(th_y), 0, np.cos(th_y)]])
            rot_z = np.asarray([[np.cos(th_z), -np.sin(th_z), 0], [np.sin(th_z), np.cos(th_z), 0], [0, 0, 1]])
        elif direction == 'ccw':
            rot_x = np.asarray([[1, 0, 0], [0, np.cos(th_x), np.sin(th_x)], [0, -np.sin(th_x), np.cos(th_x)]])
            rot_y = np.asarray([[np.cos(th_y), 0, -np.sin(th_y)], [0, 1, 0], [np.sin(th_y), 0, np.cos(th_y)]])
            rot_z = np.asarray([[np.cos(th_z), np.sin(th_z), 0], [-np.sin(th_z), np.cos(th_z), 0], [0, 0, 1]])
        else:
            raise ValueError()

        return rot_x, rot_y, rot_z

    @staticmethod
    def validate_bb_yaws(gt_boxes_3d):
        for bb in gt_boxes_3d:
            if not -np.pi <= bb.yaw <= np.pi:
                raise ValueError('Invalid Yaw!')

    @staticmethod
    def filter_boxes(gt_boxes_3d, pts, reflectance, min_num_points=5):
        list_bb = []
        for bb in gt_boxes_3d:
            inds = PointCloudAugmenter.find_containing_points(bb, pts)

            if len(inds) >= min_num_points:
                list_bb.append(bb)
                
        return pts, list_bb, reflectance

    @staticmethod
    def find_point_side(pts_x, pts_y, point_a, point_b):
        d = (pts_x - point_a[0, 0]) * (point_b[0, 1] - point_a[0, 1]) - (pts_y - point_a[0, 1]) * (point_b[0, 0] - point_a[0, 0])
        d = np.where(d < 0, -1, 1)
        return d

    @staticmethod
    def cut_flip_stitch():
        
        def __cut_flip_stitch(gt_boxes_3d, pts, reflectance):
            if len(gt_boxes_3d) < 2:
                return pts, gt_boxes_3d, reflectance

            point_a = np.asarray([[0, 0]])
            angle = 70*np.pi/180
            r = 70

            list_valid_theta = []
            dic_theta_box_map = {}
            for theta in np.linspace(-angle, angle, 70):
                point_b = np.asarray([[r * np.cos(theta), r * np.sin(theta)]])
                list_box_side_label_pos = []
                list_box_side_label_neg = []
                count_pos = 0
                count_neg = 0
                for i, box in enumerate(gt_boxes_3d):
                    corners = box.get_bev_box()
                    d = PointCloudAugmenter.find_point_side(corners[:, 0], corners[:, 1], point_a, point_b)

                    if abs(d.sum()) == corners.shape[0]:
                        if d[0] < 0:
                            list_box_side_label_neg.append(box)
                            count_neg += 1
                        else:
                            list_box_side_label_pos.append(box)
                            count_pos += 1
                    else:
                        break

                if count_pos + count_neg == len(gt_boxes_3d):
                    if count_neg > 0 and count_pos > 1:
                        list_valid_theta.append(theta)
                        dic_theta_box_map[theta] = (list_box_side_label_pos, list_box_side_label_neg)
            if len(list_valid_theta) > 0:
                theta = random.sample(list_valid_theta, 1)[0]
                list_box_side_label_pos, list_box_side_label_neg = dic_theta_box_map[theta]
                point_b = np.asarray([[r * np.cos(theta), r * np.sin(theta)]])

                d = PointCloudAugmenter.find_point_side(pts[:, 2], pts[:, 0], point_a, point_b)
                if random.uniform(0, 1) < 0.5:
                    pts_keep = pts[d == 1, :].copy()
                    reflectance_keep = reflectance[d == 1, ...].copy()
                    box_3d_flipped = [box.clone() for box in list_box_side_label_pos]
                    box_3d_keep = list_box_side_label_pos
                else:
                    pts_keep = pts[d == -1, :].copy()
                    reflectance_keep = reflectance[d == -1, ...].copy()
                    box_3d_flipped = [box.clone() for box in list_box_side_label_neg]
                    box_3d_keep = list_box_side_label_neg

                pts_flipped, box_3d_flipped, _ = PointCloudAugmenter.flip_along_x()(box_3d_flipped, pts_keep.copy(), reflectance)
                pts_flipped, box_3d_flipped, _ = PointCloudAugmenter.rotate_translate(rotation=2*theta, translation=0)(box_3d_flipped, pts_flipped, reflectance)
                reflectance_flipped = reflectance_keep.copy()

                for box in box_3d_flipped:
                    corners = box.get_bev_box()
                    center = np.mean(corners, 0)

                    if 0 <= center[0] <= 70.4 and -40 <= center[1] <= 40:
                        box_3d_keep.append(box)

                # box_3d_keep.extend(box_3d_flipped)
                pts_res = np.concatenate([pts_keep, pts_flipped], 0)
                reflectance_res = np.concatenate([reflectance_keep, reflectance_flipped], 0)

                return pts_res, box_3d_keep, reflectance_res
            else:
                return pts, gt_boxes_3d, reflectance
            
        return __cut_flip_stitch

    @staticmethod
    def keep_valid_data(gt_boxes_3d, pts_res, reflectance_res):
        ind = np.where((pts_res[:, 0] >= -40) & (pts_res[:, 0] <= 40) & (pts_res[:, 2] >= 0) & (pts_res[:, 2] <= 70))[0]
        pts_res = pts_res[ind, :]
        reflectance_res = reflectance_res[ind, ...]

        list_final_bb = []
        for bb in gt_boxes_3d:
            if (-40 <= bb.x <= 40) and (-1 <= bb.y <= 2.5) and (0 <= bb.z <= 70):
                list_final_bb.append(bb)

        return pts_res, list_final_bb, reflectance_res

    @staticmethod
    def global_background_dropout(dropout_ratio=0.1):
        
        def __global_background_dropout(gt_boxes_3d, pts, reflectance):
            inds = []
            for bb in gt_boxes_3d:
                inds.extend(PointCloudAugmenter.find_containing_points(bb, pts))

            if pts.shape[0] > 10000:
                inds_to_keep = list(set(range(pts.shape[0])).difference(set(inds)))

                random.shuffle(inds_to_keep)
                inds_to_keep = inds_to_keep[:int(len(inds_to_keep) * (1-dropout_ratio))]
                inds_to_keep.extend(inds)

                pts = pts[inds_to_keep, ...]
                reflectance = reflectance[inds_to_keep, ...]

            return pts, gt_boxes_3d, reflectance
        
        return __global_background_dropout

    @staticmethod
    def per_box_dropout(dropout_ratio=0.1):
        
        def __per_box_dropout(gt_boxes_3d, pts, reflectance):
            for bb in gt_boxes_3d:
                inds = PointCloudAugmenter.find_containing_points(bb, pts)

                if len(inds) > 15:
                    random.shuffle(inds)
                    inds_to_remove = inds[:int(len(inds)*dropout_ratio)]

                    inds_to_keep = list(range(pts.shape[0]))
                    inds_to_keep = list(set(inds_to_keep).difference(set(inds_to_remove)))

                    pts = pts[inds_to_keep, ...]
                    reflectance = reflectance[inds_to_keep, ...]

            return pts, gt_boxes_3d, reflectance
        
        return __per_box_dropout

    @staticmethod
    def per_box_rotation_translation(rotation_range, translation_range):
        
        def __per_box_rotation_translation(gt_boxes_3d, pts, reflectance):
            for bb in gt_boxes_3d:
                inds = PointCloudAugmenter.find_containing_points(bb, pts)

                xyz = pts[inds, :3]
                bb_center = [[bb.x, bb.y, bb.z]]
                xyz = xyz - bb_center

                r = np.random.uniform(-rotation_range, rotation_range)
                t = np.asarray([[np.random.randn() * translation_range, 0, np.random.randn() * translation_range]])
                rot = PointCloudAugmenter.rot_matrix_3d(0, r, 0)
                rot = np.dot(rot[0], np.dot(rot[1], rot[2]))
                xyz = np.dot(rot, xyz.T).T + t + bb_center

                bb_xyz = t
                bb.x += bb_xyz[0, 0]
                bb.y += bb_xyz[0, 1]
                bb.z += bb_xyz[0, 2]
                bb.yaw += r
                PointCloudAugmenter.correct_box_rotation(bb)
                # bb.compute_corners()

                pts[inds, :3] = xyz[...]

            return pts, gt_boxes_3d, reflectance
        
        return __per_box_rotation_translation

    @staticmethod
    def find_containing_points(bb, pts):
        v1 = bb.get_corners().T[1, :] - bb.get_corners().T[0, :]
        v2 = bb.get_corners().T[3, :] - bb.get_corners().T[0, :]
        v3 = bb.get_corners().T[4, :] - bb.get_corners().T[0, :]
        q = pts[:, :3] - bb.get_corners().T[0:1, :]
        p1 = np.sum(q * v1[None, :], axis=1)
        p2 = np.sum(q * v2[None, :], axis=1)
        p3 = np.sum(q * v3[None, :], axis=1)
        inds = np.where((p1 >= 0) & (p1 <= np.dot(v1, v1)) &
                        (p2 >= 0) & (p2 <= np.dot(v2, v2)) &
                        (p3 >= 0) & (p3 <= np.dot(v3, v3)))[0]
        return inds

    @staticmethod
    def flip_along_x():
        
        def __flip_along_x(gt_boxes_3d, pts, reflectance):
            pts[:, 0] = -pts[:, 0]
            for bb in gt_boxes_3d:
                bb.x *= -1
                if 0 <= bb.yaw < np.pi / 2:
                    bb.yaw = np.pi - bb.yaw
                elif -np.pi / 2 <= bb.yaw < 0:
                    bb.yaw = -(bb.yaw + np.pi)
                elif np.pi / 2 <= bb.yaw < np.pi:
                    bb.yaw = np.pi - bb.yaw
                elif -np.pi <= bb.yaw < -np.pi / 2:
                    bb.yaw = -(bb.yaw + np.pi)
                PointCloudAugmenter.correct_box_rotation(bb)
                # bb.compute_corners()
            return pts, gt_boxes_3d, reflectance
        
        return __flip_along_x

    @staticmethod
    def correct_box_rotation(bb):
        if not -np.pi <= bb.yaw <= np.pi:
            if bb.yaw < -np.pi:
                bb.yaw = 2*np.pi+bb.yaw
            elif bb.yaw > np.pi:
                bb.yaw = -2*np.pi + bb.yaw
            assert -np.pi <= bb.yaw <= np.pi

    @staticmethod
    def rotate_translate(rotation_range=np.pi / 20, translation_range=0.25, rotation=None, translation=None):
        
        def __rotate_translate(gt_boxes_3d, pts, reflectance):
            u = np.random.uniform
            r = u(-rotation_range, rotation_range)
            # t = [[np.random.randn()*translation_range, 0, np.random.randn()*translation_range]]
            t = [[u(-translation_range, translation_range), 0, u(-translation_range, translation_range)]]
            if rotation is not None:
                r = rotation
            if translation is not None:
                t = translation

            rot = PointCloudAugmenter.rot_matrix_3d(0, r, 0)
            rot = np.dot(rot[0], np.dot(rot[1], rot[2]))
            pts[:, :3] = np.dot(rot, pts[:, :3].T).T + t
            for bb in gt_boxes_3d:
                xyz = np.dot(rot, np.asarray([[bb.x, bb.y, bb.z]]).T).T + t
                bb.x = xyz[0, 0]
                bb.y = xyz[0, 1]
                bb.z = xyz[0, 2]
                bb.yaw += r
                PointCloudAugmenter.correct_box_rotation(bb)
                # bb.compute_corners()
            return pts, gt_boxes_3d, reflectance
        
        return __rotate_translate

    @staticmethod
    def scale():
        
        def __scale(gt_boxes_3d, pts, reflectance):
            s = np.random.uniform(0.95, 1.05)
            pts[:, :3] = pts[:, :3] * s
            for bb in gt_boxes_3d:
                bb.x *= s
                bb.y *= s
                bb.z *= s
                bb.w *= s
                bb.l *= s
                bb.h *= s
                # bb.compute_corners()
            return pts, gt_boxes_3d, reflectance
        
        return __scale