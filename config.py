
import os
import numpy as np
import deepdish as dd

from datasets.kitti import ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY, SMALL_OBJECTS
from models.pixor_det import create_pixor_det, BiFPN
from models.det_exp import create_new_det
from models.small_pixor_det import create_small_pixor_det
from models.small_det import create_small_det
from models.pixor_pp import create_pixor_pp
from models.pixor_pp_obj import create_pixor_pp_obj
from models.initializers import PriorProbability
from models.efficient_det import create_efficient_det
from models.sensor_fusion_net import create_sensor_fusion_net
from models.small_efficient_det import create_small_efficient_det
from models.activations import Swish
from models.blocks import GroupNormalization
from training_utils.losses import smooth_l1_loss, focal, abs_loss
from training_utils.metrics import reg_metric
from training_utils.lr_schedulers import CosineDecayRestarts, CosineDecay, const_lr, NoisyLinearCosineDecay
from pixor_targets import PIXORTargets
from encoding_utils.pointcloud_encoder import OccupancyCuboidKITTI
from tensorflow.keras.optimizers import Adam, Adamax, SGD, RMSprop, Adadelta
from tensorflow.keras.layers import Activation


configs = {
    'dataset_path': '/home/salam/datasets/KITTI/training', # absolute path
    'gpu_id': '0', # zero-indexed (1-car | 0-ped) (7-car | 0-ped)
    'phy_width': 70,
    'phy_height': 80,
    'phy_depth': 4,
    'input_shape': (800, 700, 10),
    'target_encoder': PIXORTargets(shape=(200, 175), stats=dd.io.load('kitti_stats/stats.h5'),
                                    P_WIDTH=70, P_HEIGHT=80, P_DEPTH=4, subsampling_factor=(0.8, 1.2)),
    'pc_encoder': OccupancyCuboidKITTI(x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=3, df=[0.1, 0.1, 0.4], densify=False),
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
    'ckpts_dir': 'sensor_fusion_1',
    'training_target': CARS_ONLY, # one of the enums {ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLIST_ONLY}
    'model_fn': create_sensor_fusion_net,
    'losses': {
        'obj_map': focal(alpha=0.75, gamma=1., subsampling_flag=True, data_format='channels_last'),
        'geo_map': smooth_l1_loss(sigma=3., reg_channels=11, data_format='channels_last'),
    },
    'metrics': {
        'geo_map': [
                    reg_metric(0, 11),
                    reg_metric(1, 11),
                    reg_metric(2, 11),
                    reg_metric(3, 11),
                    reg_metric(4, 11),
                    reg_metric(5, 11),
                    reg_metric(6, 11),
                    reg_metric(7, 11),
                    reg_metric(8, 11),
                    reg_metric(9, 11),
                    reg_metric(10, 11),
                   ],
    },
    'callbacks': [],
    'optimizer': Adam,
    'experiment_name': '/sensor_fusion_1',
    'start_epoch': 0,
    'use_pretrained': False,
    'last_ckpt_json': 'outputs/car_efficient_det/car_efficient_det_epoch_59.json', # absolute path
    'last_ckpt_h5': 'outputs/car_efficient_det/car_efficient_det_epoch_59.h5', # absolute path
    'custom_objects': {'BiFPN': BiFPN, 'PriorProbability': PriorProbability, 'Swish': Swish, 'GroupNormalization': GroupNormalization}, # Dict as {'BiFPN': BiFPN}
    'current_file_name': __file__,
    'stable': 0.0001,
    'schedule1': const_lr(1e-3), # CosineDecay(initial_learning_rate=1e-8, decay_steps=1800, alpha=1e6),
    'schedule2': const_lr(1e-4), #CosineDecay(initial_learning_rate=1e-3, decay_steps=20000, alpha=0.1),
    'warmup_steps': 1856 * 1,
    'data_format': 'channels_last',
    'vis_every': 30, # 12691

    'hyperparams': {
        'batch_size': 2,
        'lr': 0.1,
        'epochs': 100000,
        'num_threads': 4,
        'max_q_size': 2,
        'validate_every': 1000,
        'ckpt_every': 10000,
        'n_val_samples': 300,
        'map_every': 5000,
    },

    'chosen_ids': [
        '000040',
        '000042',
        '000047',
        '000048',
        '000050',
        '000052',
        '000053',
        '000058',
        '000059',
        '000061',
        '000062',
        '000063',
        '000065',
        '000066',
        '000076',
    ],

    'notes': '''
        without random sampling augmentation
        with 0.5 change random augmentation selected
        weighted BiFPN
    '''
}