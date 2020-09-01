
"""
Copyright 2019-2020 Selameab (https://github.com/Selameab)
"""

import numpy as np
import tensorflow as tf
import threading
import random
import timeit
# import knn
import cv2
import copy

from core.transforms_3D import project
from .augmentation import PointCloudAugmenter, ImageAugmenter
from .add_rand_sample import add_random_sample


def encode_batch(encoder, batch):
    if encoder is not None:
        return encoder.encode_batch(batch)
    else:
        return np.array(batch)


def KITTIGen(reader, frame_ids, batch_size, pc_encoder=None, target_encoder=None, aug=True):
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

    pc_aug_dict = {
        'per_box_dropout': PointCloudAugmenter.per_box_dropout(0.1),
        'flip_along_x': PointCloudAugmenter.flip_along_x(),
        'scale': PointCloudAugmenter.scale(),
        'rotate': PointCloudAugmenter.rotate_translate(rotation_range=45. * np.pi / 180., translation_range=0),
        'translate': PointCloudAugmenter.rotate_translate(rotation_range=0, translation_range=5.),
        'global_dropout': PointCloudAugmenter.global_background_dropout(0.1),
    }

    pc_only_aug_dict = {
        'flip_along_x': PointCloudAugmenter.flip_along_x_pts_only(),
        'scale': PointCloudAugmenter.scale_pts_only(),
        'translate': PointCloudAugmenter.rotate_translate_pts_only(),
    }

    img_aug_dict = {
        'flip_along_x': ImageAugmenter.flip_along_x(),
        'scale': ImageAugmenter.scale(),
        'translate': ImageAugmenter.translate(),
    }

    def aug_flip_along_x(data_dict):
        aug_pc = copy.copy(data_dict['pc'])
        aug_gt_boxes_3D = copy.copy(data_dict['boxes'])
        aug_pc, aug_gt_boxes_3D, _ = pc_aug_dict['flip_along_x'](aug_gt_boxes_3D, aug_pc)
        aug_world_pts2x = pc_only_aug_dict['flip_along_x'](data_dict['world_pts2x'])
        aug_world_pts4x = pc_only_aug_dict['flip_along_x'](data_dict['world_pts4x'])
        aug_world_pts8x = pc_only_aug_dict['flip_along_x'](data_dict['world_pts8x'])
        aug_nearest2x = pc_only_aug_dict['flip_along_x'](data_dict['nearest2x'])
        aug_nearest4x = pc_only_aug_dict['flip_along_x'](data_dict['nearest4x'])
        aug_nearest8x = pc_only_aug_dict['flip_along_x'](data_dict['nearest8x'])
        aug_img = img_aug_dict['flip_along_x'](data_dict['img'])
        aug_depth_map = img_aug_dict['flip_along_x'](data_dict['depth_map'])
        aug_intensity_map = img_aug_dict['flip_along_x'](data_dict['intensity_map'])
        aug_height_map = img_aug_dict['flip_along_x'](data_dict['height_map'])
        
        assert aug_img.shape == data_dict['img'].shape
        assert aug_depth_map.shape == data_dict['depth_map'].shape
        assert aug_intensity_map.shape == data_dict['intensity_map'].shape
        assert aug_height_map.shape == data_dict['height_map'].shape
        
        return {
            'boxes': aug_gt_boxes_3D,
            'pc': aug_pc,
            'img': aug_img,
            'depth_map': aug_depth_map,
            'intensity_map': aug_intensity_map,
            'height_map': aug_height_map,
            'world_pts2x': aug_world_pts2x.T,
            'world_pts4x': aug_world_pts4x.T,
            'world_pts8x': aug_world_pts8x.T,
            'nearest2x': aug_nearest2x.T,
            'nearest4x': aug_nearest4x.T,
            'nearest8x': aug_nearest8x.T,
        }

    def aug_per_box_dropout(gt_boxes_3d, pc, img, depth_map, intensity_map, height_map):
        aug_pc = copy.copy(pc)
        aug_gt_boxes_3D = copy.copy(gt_boxes_3d)
        aug_pc, aug_gt_boxes_3D, _ = pc_aug_dict['per_box_dropout'](aug_gt_boxes_3D, aug_pc)
        aug_img = img
        aug_depth_map = depth_map
        aug_intensity_map = intensity_map
        aug_height_map = height_map

        assert aug_img.shape == img.shape
        assert aug_depth_map.shape == depth_map.shape
        assert aug_intensity_map.shape == intensity_map.shape
        assert aug_height_map.shape == height_map.shape

        return aug_gt_boxes_3D, aug_pc, aug_img, aug_depth_map, aug_intensity_map, aug_height_map
    
    def aug_global_dropout(gt_boxes_3d, pc, img, depth_map, intensity_map, height_map):
        aug_pc = copy.copy(pc)
        aug_gt_boxes_3D = copy.copy(gt_boxes_3d)
        aug_pc, aug_gt_boxes_3D, _ = pc_aug_dict['global_dropout'](aug_gt_boxes_3D, aug_pc)
        aug_img = img
        aug_depth_map = depth_map
        aug_intensity_map = intensity_map
        aug_height_map = height_map

        assert aug_img.shape == img.shape
        assert aug_depth_map.shape == depth_map.shape
        assert aug_intensity_map.shape == intensity_map.shape
        assert aug_height_map.shape == height_map.shape

        return aug_gt_boxes_3D, aug_pc, aug_img, aug_depth_map, aug_intensity_map, aug_height_map

    def aug_translate(data_dict):
        aug_pc = copy.copy(data_dict['pc'])
        aug_gt_boxes_3D = copy.copy(data_dict['boxes'])
        aug_pc, aug_gt_boxes_3D, _, aug_dict = pc_aug_dict['translate'](aug_gt_boxes_3D, aug_pc)
        aug_pc, aug_gt_boxes_3D, _ = PointCloudAugmenter.keep_valid_data(aug_gt_boxes_3D, aug_pc)
        aug_world_pts2x = pc_only_aug_dict['translate'](data_dict['world_pts2x'], aug_dict['t'])
        aug_world_pts4x = pc_only_aug_dict['translate'](data_dict['world_pts4x'], aug_dict['t'])
        aug_world_pts8x = pc_only_aug_dict['translate'](data_dict['world_pts8x'], aug_dict['t'])
        aug_nearest2x = pc_only_aug_dict['translate'](data_dict['nearest2x'], aug_dict['t'])
        aug_nearest4x = pc_only_aug_dict['translate'](data_dict['nearest4x'], aug_dict['t'])
        aug_nearest8x = pc_only_aug_dict['translate'](data_dict['nearest8x'], aug_dict['t'])
        t = np.squeeze(aug_dict['t']).tolist()
        # print(t)
        t = np.float32([[1, 0, np.ceil(t[0] * 1242/140)],
                        [0, 1, np.ceil(t[1] * 375/10)]])#8.87142857143)]])
        
        aug_img = img_aug_dict['translate'](data_dict['img'], t)
        aug_depth_map = img_aug_dict['translate'](data_dict['depth_map'], t)
        aug_intensity_map = img_aug_dict['translate'](data_dict['intensity_map'], t)
        aug_height_map = img_aug_dict['translate'](data_dict['height_map'], t)
        
        assert aug_img.shape == data_dict['img'].shape
        assert aug_depth_map.shape == data_dict['depth_map'].shape
        assert aug_intensity_map.shape == data_dict['intensity_map'].shape
        assert aug_height_map.shape == data_dict['height_map'].shape
        
    #     return aug_gt_boxes_3D, aug_pc, aug_img, aug_depth_map, aug_intensity_map, aug_height_map
        return {
            'boxes': aug_gt_boxes_3D,
            'pc': aug_pc,
            'img': aug_img,
            'depth_map': aug_depth_map,
            'intensity_map': aug_intensity_map,
            'height_map': aug_height_map,
            'world_pts2x': aug_world_pts2x.T,
            'world_pts4x': aug_world_pts4x.T,
            'world_pts8x': aug_world_pts8x.T,
            'nearest2x': aug_nearest2x.T,
            'nearest4x': aug_nearest4x.T,
            'nearest8x': aug_nearest8x.T,
        }
        
    def aug_scale(data_dict):
        aug_pc = copy.copy(data_dict['pc'])
        aug_gt_boxes_3D = copy.copy(data_dict['boxes'])
        aug_pc, aug_gt_boxes_3D, _, aug_dict = pc_aug_dict['scale'](aug_gt_boxes_3D, aug_pc)
        aug_pc, aug_gt_boxes_3D, _ = PointCloudAugmenter.keep_valid_data(aug_gt_boxes_3D, aug_pc)
        aug_world_pts2x = pc_only_aug_dict['scale'](data_dict['world_pts2x'], aug_dict['s'])
        aug_world_pts4x = pc_only_aug_dict['scale'](data_dict['world_pts4x'], aug_dict['s'])
        aug_world_pts8x = pc_only_aug_dict['scale'](data_dict['world_pts8x'], aug_dict['s'])
        aug_nearest2x = pc_only_aug_dict['scale'](data_dict['nearest2x'], aug_dict['s'])
        aug_nearest4x = pc_only_aug_dict['scale'](data_dict['nearest4x'], aug_dict['s'])
        aug_nearest8x = pc_only_aug_dict['scale'](data_dict['nearest8x'], aug_dict['s'])
        # print(aug_dict['s'])
        
        aug_img = img_aug_dict['scale'](data_dict['img'], aug_dict['s'])
        aug_depth_map = img_aug_dict['scale'](data_dict['depth_map'], aug_dict['s'])
        aug_intensity_map = img_aug_dict['scale'](data_dict['intensity_map'], aug_dict['s'])
        aug_height_map = img_aug_dict['scale'](data_dict['height_map'], aug_dict['s'])
        
        assert aug_img.shape == data_dict['img'].shape
        assert aug_depth_map.shape == data_dict['depth_map'].shape
        assert aug_intensity_map.shape == data_dict['intensity_map'].shape
        assert aug_height_map.shape == data_dict['height_map'].shape
            
    #     return aug_gt_boxes_3D, aug_pc, aug_img, aug_depth_map, aug_intensity_map, aug_height_map
        return {
            'boxes': aug_gt_boxes_3D,
            'pc': aug_pc,
            'img': aug_img,
            'depth_map': aug_depth_map,
            'intensity_map': aug_intensity_map,
            'height_map': aug_height_map,
            'world_pts2x': aug_world_pts2x.T,
            'world_pts4x': aug_world_pts4x.T,
            'world_pts8x': aug_world_pts8x.T,
            'nearest2x': aug_nearest2x.T,
            'nearest4x': aug_nearest4x.T,
            'nearest8x': aug_nearest8x.T,
        }

    def apply_aug(pc, gt_boxes_3D, img, depth_map, intensity_map, height_map, #target_map_2d, 
                  world_pts2x, world_pts4x, world_pts8x, nearest2x, nearest4x, nearest8x):
        if pc.shape[1] != 3:
            pc = pc.T

        aug_out = {
            'boxes': gt_boxes_3D,
            'pc': pc,
            'img': img,
            'depth_map': depth_map,
            'intensity_map': intensity_map,
            'height_map': height_map,
            'world_pts2x': world_pts2x,
            'world_pts4x': world_pts4x,
            'world_pts8x': world_pts8x,
            'nearest2x': nearest2x,
            'nearest4x': nearest4x,
            'nearest8x': nearest8x,
        }
        
        if np.random.uniform() < 0.5 and aug:
            # print('I am augmenting (flip)')
            aug_out = aug_flip_along_x(aug_out)  

        #     aug_gt_boxes_3D, aug_pc, aug_img, aug_depth_map, aug_intensity_map, aug_height_map = aug_per_box_dropout(aug_gt_boxes_3D, 
        #                                                                                                           aug_pc, 
        #                                                                                                           aug_img, 
        #                                                                                                           aug_depth_map,
        #                                                                                                           aug_intensity_map,
        #                                                                                                           aug_height_map)   
        # if np.random.uniform() < 0.25 and aug:
        #     aug_gt_boxes_3D, aug_pc, aug_img, aug_depth_map, aug_intensity_map, aug_height_map = aug_global_dropout(aug_gt_boxes_3D, 
        #                                                                                                           aug_pc, 
        #                                                                                                           aug_img, 
        #                                                                                                           aug_depth_map,
        #                                                                                                           aug_intensity_map,
        #                                                                                                           aug_height_map)                                                                                                                         

        if np.random.uniform() < 0.85 and aug:
            # print('I am augmenting (scale + translate)')
            aug_out = aug_translate(aug_out)
            aug_out = aug_scale(aug_out)

        return aug_out
    

    def get_ids(batch_id):
        return frame_ids[batch_id * batch_size: (batch_id + 1) * batch_size]

    
    def bev2world(idx, jdx, bev_width, bev_length, world_width, world_length):
        disc_factor_w, disc_factor_l = world_width / bev_width, world_length / bev_length
        return np.array([idx * disc_factor_w - world_width / 2., 0.5, (bev_length - jdx) * disc_factor_l])

    
    def get_world_pts(pt_cloud, bev_width, bev_length, image_downsampling_factor, P2, parts=4):
        # https://www.sicara.ai/blog/2017-07-05-fast-custom-knn-sklearn-cython
        if pt_cloud.shape[0] != 3:
            pt_cloud = pt_cloud.T
        world_pts = []
        # one time for each dim
        for i in range(bev_length):
            for j in range(bev_width):
                world_pts.append(bev2world(j, i, bev_width, bev_length, 80, 70))
        all_inds = []
        for i in range(parts):
            cur_part = np.array(world_pts[i * len(world_pts) // parts:i * len(world_pts) // parts + len(world_pts) // parts]).T
            _, inds = knn.knn(cur_part.astype(np.float32),
                            pt_cloud.astype(np.float32),
                            1)
            inds = np.squeeze(inds) - 1
            all_inds = all_inds + inds.tolist()
        world_pts = np.array(world_pts).T
        nearest = pt_cloud[:,all_inds]
        geo_feature = nearest - world_pts
        nearest_projected = project(P2, nearest).astype(np.int32).T // image_downsampling_factor
        return nearest_projected.reshape((bev_length, bev_width, 2)), geo_feature.reshape((bev_length, bev_width, 3))


    def get_batch(batch_id):
        selected_ids = get_ids(batch_id)
        velo_batch, img_batch, depth_batch, intensity_batch, height_batch, boxes_3D_batch, target_map_batch = [], [], [], [], [], [], []
        mappings_2x, mappings_4x, mappings_8x = [], [], []
        geos_2x, geos_4x, geos_8x = [], [], []
        for i in range(len(selected_ids)):
            # Load data
            pts, ref = reader.get_velo(selected_ids[i], workspace_lim=((-35, 35), (-1, 3), (0, 70)), use_fov_filter=True)  # Load velo
            gt_boxes_3D = reader.get_boxes_3D(selected_ids[i])
            gt_boxes_2D = reader.get_boxes_2D(selected_ids[i])

            # add random boxes
            # if len(gt_boxes_3D) > 0:
            #     if np.random.uniform() < 1.:
            #         pts, gt_boxes_3D, _ = add_random_sample_gen(gt_boxes_3D, pts)

            if pts.shape[1] != 3:
                pts = pts.T

            # Add to batch
            # velo_batch += [pts]
            # boxes_3D_batch += [gt_boxes_3D]

            # img = reader.get_image(selected_ids[i])
            # _, _, P2 = reader.get_calib(selected_ids[i])

            # start = timeit.default_timer()
            # depth_map = reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='depth')
            # print('depth_map in:', timeit.default_timer() - start)

            # start = timeit.default_timer()
            # intensity_map = reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='intensity')
            # print('intensity_map in:', timeit.default_timer() - start)

            # start = timeit.default_timer()
            # height_map = reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='height')
            # print('height_map in:', timeit.default_timer() - start)

            img, depth_map, intensity_map, height_map = reader.get_rangeview_preprocessing(selected_ids[i])

            target_map = np.zeros((94, 311), dtype=np.uint8)
            for box in gt_boxes_2D:
                r = np.ceil(((box.w / 4) * (box.h / 4)) / (target_map.shape[0] * target_map.shape[1]) * 100)
                cv2.circle(target_map, (int(box.cx / 4), int(box.cy / 4)), int((box.h / 4) / 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
            target_map = np.expand_dims(target_map, -1).astype(np.float32) / 255.

            world_pts2x, world_pts4x, world_pts8x, nearest2x, nearest4x, nearest8x = reader.get_contfuse_nearest(selected_ids[i])

            aug_out = apply_aug(pts, 
                                gt_boxes_3D,
                                img,
                                depth_map,
                                intensity_map, 
                                height_map,
                                # target_map,
                                world_pts2x,
                                world_pts4x,
                                world_pts8x,
                                nearest2x,
                                nearest4x,
                                nearest8x)

            velo_batch += [aug_out['pc']]
            boxes_3D_batch += [aug_out['boxes']]
            img_batch += [aug_out['img']]
            depth_batch += [aug_out['depth_map']]
            intensity_batch += [aug_out['intensity_map']]
            height_batch += [aug_out['height_map']]
            # target_map_batch += [aug_target_2d]
            mapping_2x, geo_2x = reader.compute_contfuse_mapping(selected_ids[i], aug_out['world_pts2x'], aug_out['nearest2x'], 256, 224, 2)
            mapping_4x, geo_4x = reader.compute_contfuse_mapping(selected_ids[i], aug_out['world_pts4x'], aug_out['nearest4x'], 128, 112, 4)
            mapping_8x, geo_8x = reader.compute_contfuse_mapping(selected_ids[i], aug_out['world_pts8x'], aug_out['nearest8x'], 64, 56, 8)

            # bev_length, bev_width = 448, 512
            # mapping_2x, mapping_4x, mapping_8x, geo_2x, geo_4x, geo_8x = reader.get_contfuse_preprocessing(selected_ids[i])
            # mapping_2x[:,:,(0,1)] = mapping_2x[:,:,(1,0)]
            # mapping_4x[:,:,(0,1)] = mapping_4x[:,:,(1,0)]
            # mapping_8x[:,:,(0,1)] = mapping_8x[:,:,(1,0)]
            mappings_2x += [mapping_2x]
            mappings_4x += [mapping_4x]
            mappings_8x += [mapping_8x]
            geos_2x += [geo_2x]
            geos_4x += [geo_4x]
            geos_8x += [geo_8x]
            # mapping_2x, geo_2x = get_world_pts(pts, bev_width // 2, bev_length // 2, 4, P2)
            # mapping_4x, geo_4x = get_world_pts(pts, bev_width // 4, bev_length // 4, 4, P2)
            # mapping_8x, geo_8x = get_world_pts(pts, bev_width // 8, bev_length // 8, 4, P2)

        # start = timeit.default_timer()
        return encode_batch(pc_encoder, velo_batch), \
               encode_batch(None, img_batch), \
               encode_batch(None, depth_batch), \
               encode_batch(None, intensity_batch), \
               encode_batch(None, height_batch), \
               encode_batch(None, mappings_2x), \
               encode_batch(None, mappings_4x), \
               encode_batch(None, mappings_8x), \
               encode_batch(None, geos_2x), \
               encode_batch(None, geos_4x), \
               encode_batch(None, geos_8x), \
               encode_batch(target_encoder, boxes_3D_batch), \
               selected_ids
        # print('encoding took {}'.format(timeit.default_timer() - start))
    
    data_len = int(np.ceil(len(frame_ids) / batch_size))

    for i in range(data_len):
        start = timeit.default_timer()
        yield get_batch(i)
        # print('batch took {}'.format(timeit.default_timer() - start))



def KITTIValGen(reader, frame_ids, batch_size, pc_encoder=None, target_encoder=None):
    reader = reader
    frame_ids = frame_ids
    # random.shuffle(frame_ids)

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
        mappings_2x, mappings_4x, mappings_8x = [], [], []
        geos_2x, geos_4x, geos_8x = [], [], []
        for i in range(len(selected_ids)):
            # Load data
            pts, ref = reader.get_velo(selected_ids[i], workspace_lim=((-35, 35), (-1, 3), (0, 70)), use_fov_filter=True)  # Load velo
            gt_boxes_3D = reader.get_boxes_3D(selected_ids[i])

            # add random boxes
            # if len(gt_boxes_3D) > 0:
            #     if np.random.uniform() < 1.:
            #         pts, gt_boxes_3D, _ = add_random_sample_gen(gt_boxes_3D, pts)

            # pts, _, gt_boxes_3D = sequence_aug(pts, gt_boxes_3D, aug_prob=0.6)
            # # # print('before aug', pts.shape)
            # pts, _, gt_boxes_3D = rand_aug(pts, gt_boxes_3D, aug_prob=0.5)

            if pts.shape[1] != 3:
                pts = pts.T

            # Add to batch
            velo_batch += [pts]
            boxes_3D_batch += [gt_boxes_3D]

            # img_batch += [reader.get_image(selected_ids[i])]
            # _, _, P2 = reader.get_calib(selected_ids[i])
            # depth_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='depth')]
            # intensity_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='intensity')]
            # height_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='height')]
            img, depth_map, intensity_map, height_map = reader.get_rangeview_preprocessing(selected_ids[i])

            img_batch += [img]
            depth_batch += [depth_map]
            intensity_batch += [intensity_map]
            height_batch += [height_map]

            # bev_length, bev_width = 448, 512
            mapping_2x, mapping_4x, mapping_8x, geo_2x, geo_4x, geo_8x = reader.get_contfuse_preprocessing(selected_ids[i])
            # mapping_2x[:,:,(0,1)] = mapping_2x[:,:,(1,0)]
            # mapping_4x[:,:,(0,1)] = mapping_4x[:,:,(1,0)]
            # mapping_8x[:,:,(0,1)] = mapping_8x[:,:,(1,0)]
            mappings_2x += [mapping_2x]
            mappings_4x += [mapping_4x]
            mappings_8x += [mapping_8x]
            geos_2x += [geo_2x]
            geos_4x += [geo_4x]
            geos_8x += [geo_8x]

        # start = timeit.default_timer()
        return encode_batch(pc_encoder, velo_batch), \
               encode_batch(None, img_batch), \
               encode_batch(None, depth_batch), \
               encode_batch(None, intensity_batch), \
               encode_batch(None, height_batch), \
               encode_batch(None, mappings_2x), \
               encode_batch(None, mappings_4x), \
               encode_batch(None, mappings_8x), \
               encode_batch(None, geos_2x), \
               encode_batch(None, geos_4x), \
               encode_batch(None, geos_8x), \
               selected_ids
        # print('encoding took {}'.format(timeit.default_timer() - start))
    
    data_len = int(np.ceil(len(frame_ids) / batch_size))

    for i in range(data_len):
        start = timeit.default_timer()
        yield get_batch(i)
        # print('batch took {}'.format(timeit.default_timer() - start))


def KITTICarRecognition2DGen(reader, frame_ids, batch_size):
    reader = reader
    frame_ids = frame_ids
    random.shuffle(frame_ids)

    batch_size = batch_size


    def get_ids(batch_id):
        return frame_ids[batch_id * batch_size: (batch_id + 1) * batch_size]

    
    def get_batch(batch_id):
        selected_ids = get_ids(batch_id)
        img_batch, depth_batch, intensity_batch, height_batch = [], [], [], []
        target_high_batch, target_mid_batch, target_low_batch = [], [], []
        for i in range(len(selected_ids)):
            # Load data
            pts, ref = reader.get_velo(selected_ids[i], workspace_lim=((-35, 35), (-1, 3), (0, 70)), use_fov_filter=True)  # Load velo
            gt_boxes_2D = reader.get_boxes_2D(selected_ids[i])

            img_batch += [reader.get_image(selected_ids[i])]
            _, _, P2 = reader.get_calib(selected_ids[i])
            depth_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='depth')]
            intensity_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='intensity')]
            height_batch += [reader.get_range_view(img=None, pts=pts, ref=ref, P2=P2, gt_boxes=None, pred_boxes=None, out_type='height')]
            
            target_map = np.zeros((94, 311), dtype=np.uint8)
            for box in gt_boxes_2D:
                r = np.ceil(((box.w / 4) * (box.h / 4)) / (target_map.shape[0] * target_map.shape[1]) * 100)
                cv2.circle(target_map, (int(box.cx / 4), int(box.cy / 4)), int((box.h / 4) / 2), (255, 255, 255), -1, lineType=cv2.LINE_AA)
            target_map = np.expand_dims(target_map, -1).astype(np.float32) / 255.

            target_high_batch += [np.expand_dims(cv2.resize(target_map, (621, 188)), -1)]
            target_mid_batch += [target_map]            
            target_low_batch += [np.expand_dims(cv2.resize(target_map, (156, 47)), -1)]

        # start = timeit.default_timer()
        return encode_batch(None, img_batch), \
               encode_batch(None, depth_batch), \
               encode_batch(None, intensity_batch), \
               encode_batch(None, height_batch), \
               encode_batch(None, target_high_batch), \
               encode_batch(None, target_mid_batch), \
               encode_batch(None, target_low_batch)
        # print('encoding took {}'.format(timeit.default_timer() - start))
    
    data_len = int(np.ceil(len(frame_ids) / batch_size))

    for i in range(data_len):
        start = timeit.default_timer()
        yield get_batch(i)
        # print('batch took {}'.format(timeit.default_timer() - start))



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
