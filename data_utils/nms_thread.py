
import threading
import numpy as np
import time
import timeit
import random
import os

from pixor_utils.post_processing import nms_bev
from pixor_utils.pred_utils import boxes_to_pred_str
from queue import Queue

class NMSThread(threading.Thread):
    def __init__(self, reader, target_encoder, preds, exp_path, done):
        super(NMSThread, self).__init__()
        self.daemon = True

        self.preds = preds
        self.exp_path = exp_path
        self.reader = reader
        self.target_encoder = target_encoder
        self.done = done

        self.stop_flag = False

        self.nms = nms_bev('iou', 0.1, max_boxes=50, axis_aligned=False)

    def apply_nms(self):
        # print('I began applying NMS!')
        for pred in self.preds:
            if self.stop_flag:
                break
            frame, outmap = pred['frame'], pred['outmap']
            if not os.path.exists(os.path.join(self.exp_path, frame + '.txt')):
                decoded_boxes = self.target_encoder.decode(np.squeeze(outmap), 0.05)
                filtered_boxes = self.nms(decoded_boxes)
                lines = boxes_to_pred_str(filtered_boxes, self.reader.get_calib(frame)[2])
                with open(os.path.join(self.exp_path, frame + '.txt'), 'w') as txt:
                    if len(filtered_boxes) > 0:
                        txt.writelines(lines)
                self.done.append(frame)
            # print('I have finished frame #', frame)
        self.stop_flag = True
        # print('I am done!')
    
    def run(self):
        self.apply_nms()


class NMSGenerator:
    def __init__(self, reader, target_encoder, preds, exp_path, n_threads):
        # Create threads
        self.threads = []
        self.done = []
        for preds_subset in np.array_split(preds, n_threads):
            thread = NMSThread(reader=reader, target_encoder=target_encoder, preds=preds_subset, exp_path=exp_path, done=self.done)
            self.threads.append(thread)
    
    def start(self):
        for thread in self.threads:
            thread.start()

    def stop(self):
        for thead in self.threads:
            thead.stop_flag = True

# from pprint import pprint
# import deepdish as dd

# pprint(dd.io.load('kitti_stats/stats.h5'))