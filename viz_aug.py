
import os
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt

# from viz import open3d, bev, imshow
from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
from core.boxes import Box3D
from core.kitti import box_filter 
from data_utils.augmentation import PointCloudAugmenter
from encoding_utils.pointcloud_encoder import OccupancyCuboidKITTI
from pixor_targets import PIXORTargets
from data_utils.training_gen import Training_Generator_Thread
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
    https://math.stackexchange.com/questions/208577/find-if-three-points-in-3-dimensional-space-are-collinear
'''

DS_DIR = '/home/salam/datasets/KITTI/training'
reader = KITTI(DS_DIR, CARS_ONLY)

ids = reader.get_ids('train')


# pts, boxes, _ = PointCloudAugmenter.per_box_rotation_translation(rotation_range=np.pi / 15., translation_range=0.25)(boxes, pts)
# pts, boxes, _ = PointCloudAugmenter.flip_along_x()(boxes, pts)
# pts, boxes, _ = PointCloudAugmenter.rotate_translate(rotation=25. * np.pi / 180., translation=0)(boxes, pts)
# pts, boxes, _ = PointCloudAugmenter.scale()(boxes, pts)
# pts, boxes, _ = PointCloudAugmenter.rotate_translate(rotation=0, translation=0.6)(boxes, pts)
# pts, boxes, _ = PointCloudAugmenter.global_background_dropout(0.1)(boxes, pts)
# print(pts.shape)
# pts, boxes, _ = PointCloudAugmenter.keep_valid_data(boxes, pts)
from viz import open3d

for i in range(50):
    t = ids[i]
    boxes = reader.get_boxes_3D(t)
    print('then',len(boxes))
    pts, ref = reader.get_velo(t, use_fov_filter=False)
    # print('after get_velo', pts.shape)
    # print('after preparing for augs', pts.shape)
    s = Training_Generator_Thread(queue=[], max_queue_size=1, 
                                                reader=None, frame_ids=[],
                                                pc_encoder=None, target_encoder=None, 
                                                batch_size=1)
    # if len(boxes) > 1:
        # print('before rand sam', pts.shape)
    pts, boxes, _ = s.add_random_sample(boxes, pts)
    print('now',len(boxes))
    print('---------')
    # open3d(pts, boxes)
    # pts, _ , boxes = s.rand_aug(pts=pts, gt_boxes_3D=boxes, aug_prob=1.0)
    pts, _ , boxes = s.sequence_aug(pts=pts, gt_boxes_3D=boxes, aug_prob=1.0)
    print(len(boxes))
    open3d(pts.T, boxes)
    # print(pts.shape)
    # print('after finishing augs', pts.shape)
    # open3d(pts.T, boxes)
    # pc_encoder = OccupancyCuboidKITTI(x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=2.5, df=0.1, densify=True)

    target_encoder = PIXORTargets(shape=(200, 175), stats=dd.io.load('kitti_stats/stats.h5'),
                                               P_WIDTH=70, P_HEIGHT=80, P_DEPTH=3.5)

    # encoded_pc = pc_encoder.encode(pts)
    # plt.imshow(np.sum(encoded_pc, axis=2) > 0)
    # plt.axis('off')
    # plt.savefig('test_utils/test_encoded_pc1.png', bbox_inches='tight', pad_inches=0)

    # target = np.squeeze(target_encoder.encode(boxes))
    # plt.imshow(target[...,0])
    # plt.axis('off')
    # # plt.savefig('test_utils/test_encoded_target1.png', bbox_inches='tight', pad_inches=0)
    # plt.show()
    # plt.clf()

    # open3d(pts.T, boxes)
