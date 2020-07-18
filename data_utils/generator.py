
"""
Copyright 2019-2020 Selameab (https://github.com/Selameab)
"""

import numpy as np
import tensorflow as tf
import threading
import random

from .augmentation import PointCloudAugmenter
from .add_rand_sample import add_random_sample


def encode_batch(encoder, batch):
    if encoder is not None:
        return encoder.encode_batch(batch)
    else:
        return np.array(batch)


def KITTIGen(reader, frame_ids, batch_size, pc_encoder=None, target_encoder=None):
    reader = reader
    frame_ids = frame_ids
    random.shuffle(frame_ids)

    batch_size = batch_size
    pc_encoder = pc_encoder
    target_encoder = target_encoder

    add_random_sample_gen = add_random_sample(num_samples=100, sort=False, filter_wall_thresh=300)
    augmentations_arr = [
                        PointCloudAugmenter.per_box_dropout(0.1), 
                        PointCloudAugmenter.flip_along_x(),
                        PointCloudAugmenter.per_box_rotation_translation(rotation_range=np.pi / 15., translation_range=0.1),
                        PointCloudAugmenter.scale(), 
                        PointCloudAugmenter.rotate_translate(rotation_range=20. * np.pi / 180., translation_range=0),
                        PointCloudAugmenter.rotate_translate(rotation_range=0, translation_range=0.75),
                        PointCloudAugmenter.global_background_dropout(0.1),
                    ]

    augmentations_dict = {
            'per_box_dropout': PointCloudAugmenter.per_box_dropout(0.1),
            'flip_along_x': PointCloudAugmenter.flip_along_x(),
            'per_box_rot_trans': PointCloudAugmenter.per_box_rotation_translation(rotation_range=np.pi / 15., translation_range=0.15),
            'scale': PointCloudAugmenter.scale(),
            'rotate': PointCloudAugmenter.rotate_translate(rotation_range=45. * np.pi / 180., translation_range=0),
            'translate': PointCloudAugmenter.rotate_translate(rotation_range=0, translation_range=0.75),
            'global_dropout': PointCloudAugmenter.global_background_dropout(0.1),
        }

    def sequence_aug(pts, gt_boxes_3D, ref=None, aug_prob=0.85):
        if pts.shape[1] != 3:
            pts = pts.T
        seq = [
                ('per_box_rot_trans', 1.), 
                ('flip_along_x', 0.5),
                ('rotate', 1.),
                ('scale', 1.),
                ('translate', 1.),
                ('per_box_dropout', 0.25),
                ('global_dropout', 0.05), # per_box
        ]

        if np.random.uniform() < aug_prob:
            for aug, prob in seq:
                if np.random.uniform() < prob:
                    pts, gt_boxes_3D, _ = augmentations_dict[aug](gt_boxes_3D, pts)
            pts, gt_boxes_3D, _ = PointCloudAugmenter.keep_valid_data(gt_boxes_3D, pts)

        return pts, ref, gt_boxes_3D
    
    def rand_aug(pts, gt_boxes_3D, ref=None, aug_prob=0.5):
        if pts.shape[1] != 3:
            pts = pts.T
        
        if np.random.uniform() < aug_prob:
            number_of_augs = 2#np.random.randint(low=1, high=len(augmentations_arr))
            augs_ids       = np.random.choice(len(augmentations_arr), 
                                                size=number_of_augs, 
                                                replace=False)
            for id in augs_ids:
                pts, gt_boxes_3D, _ = augmentations_arr[id](gt_boxes_3D, pts)
                # print(self.augmentations_arr[id].__name__)
            # pts, gt_boxes_3D, _ = augmentations_arr[number_of_augs](gt_boxes_3D, pts)
            pts, gt_boxes_3D, _ = PointCloudAugmenter.keep_valid_data(gt_boxes_3D, pts)
        return pts, ref, gt_boxes_3D
    

    def get_ids(batch_id):
        return frame_ids[batch_id * batch_size: (batch_id + 1) * batch_size]


    def get_batch(batch_id):
        selected_ids = get_ids(batch_id)
        velo_batch, img_batch, depth_batch, intensity_batch, height_batch, boxes_3D_batch = [], [], [], [], [], []
        for i in range(len(selected_ids)):
            # Load data
            pts, ref = reader.get_velo(selected_ids[i], workspace_lim=((-35, 35), (-1, 3), (0, 70)), use_fov_filter=True)  # Load velo
            gt_boxes_3D = reader.get_boxes_3D(selected_ids[i])

            # Augment
            if len(gt_boxes_3D) > 0:
                if np.random.uniform() < 1.:
                # print('before', len(gt_boxes_3D))
                # print('before rand sam', pts.shape)
                    pts, gt_boxes_3D, _ = add_random_sample_gen(gt_boxes_3D, pts)
                # print(selected_frame_ids)
                # print('after rand sam', pts.shape)
                # print('after', len(gt_boxes_3D))

            pts, _, gt_boxes_3D = sequence_aug(pts, gt_boxes_3D, aug_prob=0.6)
            # # # print('before aug', pts.shape)
            # pts, _, gt_boxes_3D = rand_aug(pts, gt_boxes_3D, aug_prob=0.5)

            if pts.shape[1] != 3:
                pts = pts.T

            # Add to batch
            velo_batch += [pts]
            boxes_3D_batch += [gt_boxes_3D]

            img_batch += [reader.get_image(selected_ids[i])]
            _, _, P2 = reader.get_calib(selected_ids[i])
            depth_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='depth')]
            intensity_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='intensity')]
            height_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='height')]

        return encode_batch(pc_encoder, velo_batch), \
               encode_batch(None, depth_batch), \
               encode_batch(None, intensity_batch), \
               encode_batch(None, height_batch), \
               encode_batch(target_encoder, boxes_3D_batch)
    
    data_len = int(np.ceil(len(frame_ids) / batch_size))

    for i in range(data_len):
        yield get_batch(i)

class Generator(tf.keras.utils.Sequence):
    def __init__(self, reader, frame_ids, batch_size, pc_encoder=None, target_encoder=None):
        self.reader = reader
        self.frame_ids = frame_ids

        self.batch_size = batch_size
        self.pc_encoder = pc_encoder
        self.target_encoder = target_encoder

        self.add_random_sample = add_random_sample(num_samples=35, sort_desc=True, filter_wall_thresh=200)

        self.augmentations_arr = [
                        PointCloudAugmenter.per_box_dropout(0.1), 
                        PointCloudAugmenter.flip_along_x(),
                        PointCloudAugmenter.per_box_rotation_translation(rotation_range=np.pi / 15., translation_range=0.1),
                        PointCloudAugmenter.scale(), 
                        PointCloudAugmenter.rotate_translate(rotation_range=20. * np.pi / 180., translation_range=0),
                        # PointCloudAugmenter.rotate_translate(rotation_range=0, translation_range=0.75),
                        PointCloudAugmenter.global_background_dropout(0.1),
                        # PointCloudAugmenter.cut_flip_stitch(),
                        ]
                        
        self.lock = threading.Lock()

    def rand_aug(self, pts, gt_boxes_3D, ref=None, aug_prob=0.5):
        if pts.shape[1] != 3:
            pts = pts.T
        
        if np.random.uniform() < aug_prob:
            number_of_augs = np.random.randint(low=1, high=len(self.augmentations_arr))
            # augs_ids       = np.random.choice(len(self.augmentations_arr), 
            #                                     size=number_of_augs, 
            #                                     replace=False)
            # for id in augs_ids:
            #     pts, gt_boxes_3D, _ = self.augmentations_arr[id](gt_boxes_3D, pts)
                # print(self.augmentations_arr[id].__name__)
            pts, gt_boxes_3D, _ = self.augmentations_arr[number_of_augs](gt_boxes_3D, pts)
            pts, gt_boxes_3D, _ = PointCloudAugmenter.keep_valid_data(gt_boxes_3D, pts)
        return pts, ref, gt_boxes_3D

    def _get_pc(self, selected_ids):
        pc_batch = []
        for i in range(len(selected_ids)):
            pts, _ = self.reader.get_velo(selected_frame_ids[i], use_fov_filter=False)  # Load velo
            pc_batch.append(pts)

        return pc_batch

    def get_ids(self, batch_id):
        return self.frame_ids[batch_id * self.batch_size: (batch_id + 1) * self.batch_size]

    def __len__(self):
        return int(np.ceil(len(self.frame_ids) / self.batch_size))

    def __getitem__(self, batch_id):
        selected_ids = self.get_ids(batch_id)
        velo_batch, boxes_3D_batch = [], []
        for i in range(len(selected_ids)):
            # Load data
            pts, _ = self.reader.get_velo(selected_ids[i], use_fov_filter=False)  # Load velo
            gt_boxes_3D = self.reader.get_boxes_3D(selected_ids[i])

            # Augment
            if len(gt_boxes_3D) > 1:
                if np.random.uniform() < 0.9:
                    # print('before', len(gt_boxes_3D))
                    # print('before rand sam', pts.shape)
                    pts, gt_boxes_3D, _ = self.add_random_sample(gt_boxes_3D, pts)
                    # print(selected_frame_ids)
                    # print('after rand sam', pts.shape)
                    # print('after', len(gt_boxes_3D))

            # pts, _, gt_boxes_3D = self.sequence_aug(pts, gt_boxes_3D, aug_prob=0.5)
            # # print('before aug', pts.shape)
            pts, _, gt_boxes_3D = self.rand_aug(pts, gt_boxes_3D, aug_prob=0.5)

            if pts.shape[1] != 3:
                pts = pts.T

            # Add to batch
            velo_batch += [pts]
            boxes_3D_batch += [gt_boxes_3D]

        return encode_batch(self.pc_encoder, velo_batch), encode_batch(self.target_encoder, boxes_3D_batch)
