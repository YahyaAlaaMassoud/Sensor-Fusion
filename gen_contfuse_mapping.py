
from config import configs

import os
import io
import matplotlib
import timeit
import random
import pprint
import knn
import cv2
import deepdish as dd
import numpy as np
import matplotlib.pyplot as plt

from numba import njit, prange, jit
from datetime import datetime
from skimage.transform import resize
from scipy.interpolate import griddata
from core.kitti import KITTI
from core.transforms_3D import transform, project
from pixor_utils.model_utils import load_model, save_model
from data_utils.training_gen import TrainingGenerator
from data_utils.generator import Generator, KITTIGen
from tt import bev
from pixor_targets import PIXORTargets
from pixor_utils.post_processing import nms_bev
from test_utils.unittest import test_pc_encoder, test_target_encoder
from encoding_utils.pointcloud_encoder import OccupancyCuboidKITTI
from encoding_utils.voxelizer import BEVVoxelizer

DS_DIR = os.path.expanduser(configs['dataset_path'])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
import tensorflow.keras.backend as K

device_name = tf.test.gpu_device_name()
os.system('clear')
print('Conncted to Device:', device_name)

# Point Cloud Encoder
INPUT_SHAPE = configs['input_shape']

# Training
BATCH_SIZE = configs['hyperparams']['batch_size']
LEARNING_RATE = configs['hyperparams']['lr']
EPOCHS = configs['hyperparams']['epochs']
NUM_THREADS = configs['hyperparams']['num_threads']
MAX_Q_SIZE = configs['hyperparams']['max_q_size']

kitti = KITTI(DS_DIR, configs['training_target'])

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

def bev2world(idx, jdx, bev_width, bev_length, world_width, world_length):
    disc_factor_w, disc_factor_l = world_width / bev_width, world_length / bev_length
    x = idx * disc_factor_w - world_width / 2.
    z = (bev_length - jdx) * disc_factor_l
    return np.array([x, 0.5, z])

def get_world_pts(pt_cloud, bev_width, bev_length, image_downsampling_factor, P2, parts=4):
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

for i, id in enumerate(train_ids + val_ids):
    pc, _ = kitti.get_velo(id, workspace_lim=((-40, 40), (-1, 3), (0, 70)), use_fov_filter=True)
    _, _, P2 = kitti.get_calib(id)
#     img = kitti.get_image(id)
#     img = cv2.resize(img, (311, 94))
#     print('img.shape', img.shape)
    bev_shape = (448, 512)
    mapping2x, geo_feat2x = get_world_pts(pc, bev_shape[1] // 2, bev_shape[0] // 2, 2, P2)
    mapping2x[:,:,(0,1)] = mapping2x[:,:,(1,0)]

    mapping4x, geo_feat4x = get_world_pts(pc, bev_shape[1] // 4, bev_shape[0] // 4, 4, P2)
    mapping4x[:,:,(0,1)] = mapping4x[:,:,(1,0)]

    mapping8x, geo_feat8x = get_world_pts(pc, bev_shape[1] // 8, bev_shape[0] // 8, 8, P2)
    mapping8x[:,:,(0,1)] = mapping8x[:,:,(1,0)]
#     print(mapping2x.shape, geo_feat2x.shape)
#     print(mapping4x.shape, geo_feat4x.shape)
#     print(mapping8x.shape, geo_feat8x.shape)
    np.savez_compressed('/comm_dat/DATA/KITTI/contfuse_preprocess/{}'.format(id), 
                        mapping2x=mapping2x,
                        mapping4x=mapping4x,
                        mapping8x=mapping8x,
                        geo_feat2x=geo_feat2x,
                        geo_feat4x=geo_feat4x,
                        geo_feat8x=geo_feat8x,)
    print(i, id)