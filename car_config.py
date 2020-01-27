
import numpy as np

from datasets.kitti import ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY
from models.pixor_pp import create_pixor_pp
from models.pixor_det import create_pixor_det, BiFPN
from pixor_utils.losses import pixor_loss, binary_focal_loss_metric, smooth_L1_metric
from pixor_targets import PIXORTargets
from pixor_utils.pointcloud_encoder import OccupancyCuboid
from tensorflow.keras.optimizers import Adam, Adamax

configs = {
    'dataset_path': '/home/salam/datasets/KITTI/training', # absolute path
    'gpu_id': '0', # zero-indexed (1-car | 0-ped) (7-car | 0-ped)
    'phy_width': 70,
    'phy_height': 80,
    'phy_depth': 3.5,
    'input_shape': (800, 700, 35),
    'target_encoder': PIXORTargets,
    'pc_encoder': OccupancyCuboid,
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
    'ckpts_dir': 'ckpts_biFPN_car',
    'training_target': CARS_ONLY, # on of the enums {ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLIST_ONLY}
    'model_fn': create_pixor_det,
    'losses': {
        'output_map': pixor_loss,
    },
    'metrics': {
        'output_map': [smooth_L1_metric, binary_focal_loss_metric],
    },
    'optimizer': Adam,
    'experiment_name': '/pixor_bifpn_car_',
    'start_epoch': 4,
    'use_pretrained': True,
    'last_ckpt_json': 'outputs/ckpts_biFPN_car/pixor_bifpn_car__epoch_3.json', # absolute path
    'last_ckpt_h5': 'outputs/ckpts_biFPN_car/pixor_bifpn_car__epoch_3.h5', # absolute path
    'custom_objects': {'BiFPN': BiFPN}, # Dict as {'BiFPN': BiFPN}
    
    'hyperparams': {
        'batch_size': 1,
        'lr': 0.0001,
        'epochs': 1000,
        'num_threads': 6,
        'max_q_size': 8,
        'validate_every': 1,
        'ckpt_every': 1,
        'n_val_samples': 300,
    }
}