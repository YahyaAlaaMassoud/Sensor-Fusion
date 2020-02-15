
import threading
import numpy as np
import time
import timeit
import random

from .augmentation import PointCloudAugmenter
from core.kitti import box_filter 
from queue import Queue

class Training_Generator_Thread(threading.Thread):
    def __init__(self, queue, max_queue_size, reader, frame_ids, batch_size, pc_encoder, target_encoder, verbose=False):
        super(Training_Generator_Thread, self).__init__()
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
        
        self.augmentations = [
                                PointCloudAugmenter.per_box_dropout(0.1), 
                                PointCloudAugmenter.flip_along_x(),
                                PointCloudAugmenter.per_box_rotation_translation(rotation_range=(-1.57, 1.57), translation_range=((-2, 2), (-0.1, 0.1), (-2, 2))),
                                PointCloudAugmenter.scale(), 
                                PointCloudAugmenter.rotate_translate(),
                                PointCloudAugmenter.global_background_dropout(0.1),
                                PointCloudAugmenter.cut_flip_stitch(),
                             ]
        
        self.verbose = verbose

    def add_batch(self):
        # Create a batch using current batch_id
        selected_frame_ids = self.frame_ids[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]
        # start = timeit.default_timer()
        batch = {'frame_ids': selected_frame_ids,
                 'encoded_pcs': np.zeros(shape=(len(selected_frame_ids),) + self.pc_encoder.get_output_shape(), dtype=np.float32),
                 'encoded_targets': np.zeros(shape=(len(selected_frame_ids),) + self.target_encoder.get_output_shape(), dtype=np.float32)}

        for i in range(len(selected_frame_ids)):
            # Input
            pts, reflectance = self.reader.get_velo(selected_frame_ids[i], use_fov_filter=False)  # Load velo
            # Output
            gt_boxes_3D = self.reader.get_boxes_3D(selected_frame_ids[i])
            pts, gt_boxes_3D, reflectance = PointCloudAugmenter.filter_boxes(gt_boxes_3d=gt_boxes_3D, 
                                                                             pts=pts,
                                                                             reflectance=reflectance, 
                                                                             min_num_points=5)
            
            if self.augmentations is not None:
                aug_fn = np.random.choice(self.augmentations)
                if aug_fn is not None:
                    # print(aug_fn.__name__)
                    pts, gt_boxes_3D, reflectance = aug_fn(gt_boxes_3D, pts.T, reflectance.T)
                    pts = box_filter(pts.T, ((-40, 40), (-1, 2.5), (0, 70)))
            
            batch['encoded_pcs'][i] = self.pc_encoder.encode(pts.T, reflectance)
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

    def get_batch(self):
        for _ in range(20):
            for thread in self.threads:
                if thread.queue.qsize() is not 0:
                    batch = thread.queue.get(timeout=50)
                    batch['queue_size'] = thread.queue.qsize()
                    return batch
            time.sleep(3.)
        print('all Qs are empty')

    def start(self):
        for thread in self.threads:
            thread.start()

    def stop(self):
        for thead in self.threads:
            thead.stop_flag = True