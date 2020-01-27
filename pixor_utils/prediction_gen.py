
import threading
from queue import Queue
import numpy as np
import time

class Prediction_Generator_Thread(threading.Thread):
    def __init__(self, queue, max_queue_size, reader, frame_ids, batch_size, pc_encoder, verbose=False):
        super(Prediction_Generator_Thread, self).__init__()
        self.daemon = True

        self.queue = Queue()
        self.max_queue_size = max_queue_size

        self.reader = reader
        self.frame_ids = frame_ids
        self.pc_encoder = pc_encoder
        self.batch_size = batch_size

        self.batch_id = 0
        self.stop_flag = False
        self.batch_count = int(np.ceil(len(self.frame_ids) / self.batch_size))
        
        self.verbose = verbose

    def add_batch(self):
        # Create a batch using current batch_id
        selected_frame_ids = self.frame_ids[self.batch_id * self.batch_size: (self.batch_id + 1) * self.batch_size]

        batch = {'frame_ids': selected_frame_ids,
                 'encoded_pcs': np.zeros(shape=(len(selected_frame_ids),) + self.pc_encoder.get_output_shape(), dtype=np.float32),
                 'pts': []}

        for i in range(len(selected_frame_ids)):
            # Input
            pts, reflectance = self.reader.get_velo(selected_frame_ids[i], use_fov_filter=False)  # Load velo
            batch['encoded_pcs'][i] = self.pc_encoder.encode(pts.T, reflectance)
            batch['pts'] += [pts]

        self.queue.put(batch)
        self.batch_id += 1

    def run(self):
        if self.verbose:
            print('Prediction Generator Thread started...')
            
        while (not self.stop_flag) and self.batch_id < self.batch_count:
            if self.queue.qsize() < self.max_queue_size:
                self.add_batch()
            else:                
                time.sleep(1.)  # Slow down if input pipeline is faster than network

        if self.verbose:
            print('Prediction Generator Thread terminated...')


class PredictionGenerator:
    def __init__(self, reader, frame_ids, batch_size, pc_encoder, n_threads, max_queue_size):
        self.frame_ids = frame_ids

        # Create queue
        self.queue = Queue()

        # Create threads
        self.threads = []
        self.batch_count = 0
        for frame_ids_subset in np.array_split(frame_ids, n_threads):
            thread = Prediction_Generator_Thread(queue=self.queue, max_queue_size=max_queue_size, 
                                               reader=reader, frame_ids=frame_ids_subset,
                                               pc_encoder=pc_encoder, batch_size=batch_size)
            self.batch_count += thread.batch_count
            self.threads += [thread]

    def get_batch(self):
        for _ in range(20):
            for thread in self.threads:
                if thread.queue.qsize() is not 0:
                    batch = thread.queue.get(timeout=50)
                    batch['queue_size'] = thread.queue.qsize()
                    return batch
            time.sleep(1.)
        print('all Qs are empty')

    def start(self):
        for thread in self.threads:
            thread.start()

    def stop(self):
        for thead in self.threads:
            thead.stop_flag = True