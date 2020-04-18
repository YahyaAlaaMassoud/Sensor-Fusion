
import threading
import numpy as np
import time
import timeit
import random

from .augmentation import PointCloudAugmenter
from .add_rand_sample import add_random_sample
from core.kitti import box_filter 
from queue import Queue

import matplotlib.pyplot as plt

class Training_Generator_Thread(threading.Thread):
    def __init__(self, queue, max_queue_size, reader, frame_ids, batch_size, pc_encoder, target_encoder, verbose=False):
        threading.Thread.__init__(self)
        self.daemon = True

        self.queue = Queue()
        self.max_queue_size = max_queue_size

        self.reader = reader
        self.frame_ids = frame_ids
        random.shuffle(self.frame_ids)
        self.pc_encoder, self.target_encoder = pc_encoder, target_encoder
        self.batch_size = batch_size

        self.batch_id = 0
        self.stop_flag = False
        self.batch_count = int(np.ceil(len(self.frame_ids) / self.batch_size))
        
        self.add_random_sample = add_random_sample(num_samples=35, sort_desc=True, filter_wall_thresh=200)
        self.augmentations_dict = {
            'per_box_dropout': PointCloudAugmenter.per_box_dropout(0.1),
            'flip_along_x': PointCloudAugmenter.flip_along_x(),
            'per_box_rot_trans': PointCloudAugmenter.per_box_rotation_translation(rotation_range=np.pi / 15., translation_range=0.15),
            'scale': PointCloudAugmenter.scale(),
            'rotate': PointCloudAugmenter.rotate_translate(rotation_range=20. * np.pi / 180., translation_range=0),
            # 'translate': PointCloudAugmenter.rotate_translate(rotation_range=0, translation_range=0.75),
            'global_dropout': PointCloudAugmenter.global_background_dropout(0.1),
        }
        
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

        self.verbose = verbose
        
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

    def sequence_aug(self, pts, gt_boxes_3D, ref=None, aug_prob=0.85):
        if pts.shape[1] != 3:
            pts = pts.T
        seq = [
                ('per_box_rot_trans', 1.), 
                ('flip_along_x', 0.5),
                # ('rotate', 1.),
                ('scale', 1.),
                # ('translate', 1.),
                ('global_dropout', 0.1), # per_box
        ]

        if np.random.uniform() < aug_prob:
            for aug, prob in seq:
                if np.random.uniform() < prob:
                    pts, gt_boxes_3D, _ = self.augmentations_dict[aug](gt_boxes_3D, pts)
            pts, gt_boxes_3D, _ = PointCloudAugmenter.keep_valid_data(gt_boxes_3D, pts)

        return pts, ref, gt_boxes_3D

    def add_batch(self):
        # Create a batch using current batch_id
        selected_frame_ids = self.frame_ids[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]
        # start = timeit.default_timer()
        batch = {'frame_ids': selected_frame_ids,
                 'encoded_pcs': np.zeros(shape=(len(selected_frame_ids),) + self.pc_encoder.get_output_shape(), dtype=np.float32),
                 'encoded_targets': np.zeros(shape=(len(selected_frame_ids),) + self.target_encoder.get_output_shape(), dtype=np.float32)}

        start = timeit.default_timer()
        for i in range(len(selected_frame_ids)):
            # Input
            pts, _ = self.reader.get_velo(selected_frame_ids[i], use_fov_filter=False)  # Load velo
            # Output
            gt_boxes_3D = self.reader.get_boxes_3D(selected_frame_ids[i])

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
            # # print('after aug', pts.shape)

            if pts.shape[1] != 3:
                pts = pts.T

            batch['encoded_pcs'][i]     = self.pc_encoder.encode(pts)
            batch['encoded_targets'][i] = self.target_encoder.encode(gt_boxes_3D)

        # end = timeit.default_timer()
        # print('adding batch took {0}', end-start)
        self.queue.put(batch)
        self.batch_id += 1

    def run(self):
        if self.verbose:
            print('Training Generator Thread started...')
            
        while (not self.stop_flag) and self.batch_id < self.batch_count:
            if self.queue.qsize() < self.max_queue_size:
                self.add_batch()
            else:                
                time.sleep(1.)  # Slow down if input pipeline is faster than network

        if self.verbose:
            print('Training Generator Thread terminated...')


class TrainingGenerator:
    def __init__(self, reader, frame_ids, batch_size, pc_encoder, target_encoder, n_threads, max_queue_size):
        np.random.shuffle(frame_ids)
        self.frame_ids = frame_ids

        # Create queue
        self.queue = Queue()

        # Create threads
        self.threads = []
        self.batch_count = 0
        for frame_ids_subset in np.array_split(frame_ids, n_threads):
            thread = Training_Generator_Thread(queue=self.queue, max_queue_size=max_queue_size, 
                                               reader=reader, frame_ids=frame_ids_subset,
                                               pc_encoder=pc_encoder, target_encoder=target_encoder, 
                                               batch_size=batch_size)
            self.batch_count += thread.batch_count
            self.threads += [thread]
        
        print(len(self.threads))
        
    def get_batch(self):
        for _ in range(100):
            for thread in self.threads:
                if thread.queue.qsize() is not 0:
                    try:
                        batch = thread.queue.get(timeout=150)
                        batch['queue_size'] = thread.queue.qsize()
                        return batch
                    except:
                        continue
            time.sleep(5.)
        print('all Qs are empty')

    def start(self):
        for thread in self.threads:
            print('thread started')
            thread.start()

    def stop(self):
        for thread in self.threads:
            thread.stop_flag = True
            thread.queue = None
            thread = None
        self.threads = None