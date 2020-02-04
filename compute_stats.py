
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import deepdish as dd
import pprint

from datasets.kitti import KITTI
from datasets.kitti import ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY

from pixor_targets import PIXORTargets

DS_DIR = os.path.expanduser('/home/salam/datasets/KITTI/training')

kitti = KITTI(DS_DIR, CARS_ONLY)

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

target_encoder = PIXORTargets(shape=(200, 175), P_WIDTH=70, P_HEIGHT=80, P_DEPTH=3.5)

sin, cos = [], []
yaw = []
w, l, h  = [], [], []
alt = []
x_off, y_off, z_off = [], [], []

i = 0
for id in train_ids:
    boxes = kitti.get_boxes_3D(id)
    for box in boxes:
        sin.append(np.sin(box.yaw))
        cos.append(np.cos(box.yaw))
        w.append(np.log(box.w))
        l.append(np.log(box.l))
        h.append(np.log(box.h))
        alt.append(box.y)
        
        x, z = target_encoder.generate_offset_stats(box)
        
        x_off += x
        z_off += z
        
    if i % 200 == 0:
        print('finished {0}'.format(i))
    i += 1
    
stats = {
    'mean': {
        'cos': np.mean(cos),
        'sin': np.mean(sin),
        'log_w': np.mean(w),
        'log_l': np.mean(l),
        'log_h': np.mean(h),
        'alt': np.mean(alt),
        'dx': np.mean(x_off),
        'dz': np.mean(z_off),
    },
    'std': {
        'cos': np.std(cos),
        'sin': np.std(sin),
        'log_w': np.std(w),
        'log_l': np.std(l),
        'log_h': np.std(h),
        'alt': np.std(alt),
        'dx': np.std(x_off),
        'dz': np.std(z_off),
    },
}

np.save('kitti_stats/cos.npy', cos)
np.save('kitti_stats/sin.npy', sin)
np.save('kitti_stats/log_w.npy', w)
np.save('kitti_stats/log_l.npy', l)
np.save('kitti_stats/log_h.npy', h)
np.save('kitti_stats/alt.npy', alt)
np.save('kitti_stats/dx.npy', x_off)
np.save('kitti_stats/dz.npy', z_off)

dd.io.save('kitti_stats/stats.h5', stats)

