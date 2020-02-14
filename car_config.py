
import os
import numpy as np
import deepdish as dd

from datasets.kitti import ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY, SMALL_OBJECTS
from models.pixor_det import create_pixor_det, BiFPN
from training_utils.losses import total_loss, smooth_l1_loss, binary_focal_loss, absolute_diff_loss
from training_utils.metrics import objectness_metric, regression_metric
from pixor_targets import PIXORTargets
from encoding_utils.pointcloud_encoder import OccupancyCuboidKITTI
from tensorflow.keras.optimizers import Adam, Adamax, SGD, RMSprop

configs = {
    'dataset_path': '/home/salam/datasets/KITTI/training', # absolute path
    'gpu_id': '0', # zero-indexed (1-car | 0-ped) (7-car | 0-ped)
    'phy_width': 70,
    'phy_height': 80,
    'phy_depth': 3.5,
    'input_shape': (800, 700, 35),
    'target_encoder': PIXORTargets,
    'pc_encoder': OccupancyCuboidKITTI,
    'target_shape': (200, 175),
    'target_means': np.array([-6.4501972e-03, 
                              2.1161055e-02, 
                              -2.8581850e-02, 
                              -9.2645199e-04, 
                              5.5109012e-01, 
                              1.4958924e+00]),
    'target_stds': np.array([0.4555288, 
                             0.8899461, 
                             1.4819515, 
                             0.76802343, 
                             0.16860875, 
                             0.33837482]),
    'mean_height': 1.52,
    'mean_altitude': 1.71,
    'stats': dd.io.load('kitti_stats/stats.h5'),
    'ckpts_dir': 'car_bifpn_v2_concat_aug_abs',
    'training_target': CARS_ONLY, # one of the enums {ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLIST_ONLY}
    'model_fn': create_pixor_det,
    'losses': {
        'output_map': total_loss(alpha=0.25, gamma=2.0, reg_loss_name='abs', reg_channels=8, weight=1.0, subsampling_flag=False),
    },
    'metrics': {
        'output_map': [objectness_metric(alpha=0.25, gamma=2.0, subsampling_flag=False),
                       regression_metric(reg_loss_name='abs', reg_channels=8, weight=1.0, subsampling_flag=False)],
    },
    'callbacks': [],
    'optimizer': Adam,
    'experiment_name': '/car_bifpn_v2_concat_aug_abs',
    'start_epoch': 0,
    'use_pretrained': False,
    'last_ckpt_json': 'outputs/car_bifpn_v2_concat_aug_abs/car_bifpn_v2_concat_aug_abs_epoch_29.json', # absolute path
    'last_ckpt_h5': 'outputs/car_bifpn_v2_concat_aug_abs/car_bifpn_v2_concat_aug_abs_epoch_29.h5', # absolute path
    'custom_objects': {'BiFPN': BiFPN}, # Dict as {'BiFPN': BiFPN}
    'current_file_name': __file__,
    'warmup': True,
    'warmup_min': 0.0001,
    'warmup_max': 0.001,
    'warmup_epochs': 3,
    
    'hyperparams': {
        'batch_size': 2,
        'lr': 0.0001,
        'epochs': 1000,
        'num_threads': 6,
        'max_q_size': 8,
        'validate_every': 1000,
        'ckpt_every': 100,
        'n_val_samples': 300,
        'map_every': 5000,
    },
    
    'notes': 'w/ augmentations | gamma=2.0 | point cloud encoder (points under points) | 3 bifpns | 104 channels bifpns | out1 concatenated with out of bifpns'
}