
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import deepdish as dd

DS_DIR = os.path.expanduser('/home/salam/datasets/KITTI/training')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.optimizers import Adam
# import tensorflow as tf

from core.boxes import Box2D, Box3D
from datasets.kitti import KITTI, ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY

from pixor_utils.training_gen import TrainingGenerator
from pixor_utils.prediction_gen import PredictionGenerator
from pixor_utils.post_processing import nms_bev
from pixor_utils.losses import pixor_loss, binary_focal_loss_metric, smooth_L1_metric
from pixor_utils.params import params
from pixor_utils.model_utils import save_model, load_model
from pixor_utils.pointcloud_encoder import OccupancyCuboid
from pixor_utils.pred_utils import boxes_to_pred_str
from get_pred import generate_preds

from pixor_targets import PIXORTargets

from models.meta_learning import create_model
from models.pixor_det import BiFPN

from test_utils.unittest import test_pc_encoder, test_target_encoder

# print(tf.test.gpu_device_name())

# Physical Space
P_WIDTH, P_HEIGHT, P_DEPTH = 70, 80, 3.5

# Point Cloud Encoder
INPUT_SHAPE = 800, 700, 35

# Target Encoder
TARGET_SHAPE = (200, 175)
TARGET_MEANS = np.array([-6.4501972e-03, 2.1161055e-02, -2.8581850e-02, -9.2645199e-04, 5.5109012e-01, 1.4958924e+00])
TARGET_STDS = np.array([0.4555288, 0.8899461, 1.4819515, 0.76802343, 0.16860875, 0.33837482])
MEAN_HEIGHT, MEAN_ALTITUDE = 1.52, 1.71

# Training
BATCH_SIZE = params['batch_size']
LEARNING_RATE = params['learning_rate']
KERNEL_REGULARIZER = None
EPOCHS = params['epochs']
NUM_THREADS = params['n_threads']
MAX_Q_SIZE = params['max_queue_size']

kitti = KITTI(DS_DIR, CARS_ONLY)

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

pc_encoder     = OccupancyCuboid(shape=INPUT_SHAPE, P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)
target_encoder = PIXORTargets(shape=TARGET_SHAPE,
                               stats=dd.io.load('kitti_stats/stats.h5'),
                               P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)

#--------------------------------------#
#-----------RUN UNIT TESTS-------------#
#--------------------------------------#
os.system('clear')
rand_idx = np.random.randint(0, len(train_ids))
'''
  IDS that suck:
    - 001690
    - 002333
'''
test_id  = train_ids[rand_idx]
print(test_id)
pts, _ = kitti.get_velo(test_id, use_fov_filter=False)
test_pc_encoder(pc_encoder, pts.T)
boxes = kitti.get_boxes_3D(test_id)
test_target_encoder(target_encoder, boxes)


for i in [10]:
  epoch = i
  chkpt_dir  = 'outputs/ckpts_ped_3bifpn_head3Conv_concat_aug_abs/'
  exp_id     = 'pixor_pp_3bifpn_aug_abs_concat_ped'
  chkpt_json = chkpt_dir + 'pixor_3bifpn_head3Conv_ped_concat_aug_abs_epoch_{0}.json'.format(epoch)
  chkpt_h5   = chkpt_dir + 'pixor_3bifpn_head3Conv_ped_concat_aug_abs_epoch_{0}.h5'.format(epoch)
  exp_name   = exp_id + '_{0}/data/'.format(epoch)

  # # Create dirs
  # # EXP_DIR = os.path.join(exp_name)

  # # print("Creating directory: " + EXP_DIR)
  # # os.makedirs(EXP_DIR, exist_ok=True)

  trained_model = load_model(chkpt_json, chkpt_h5, {'BiFPN': BiFPN})
  optimizer = Adam(lr=LEARNING_RATE)
  losses = {
              'output_map': pixor_loss
            }
  metrics = {
              'output_map': [smooth_L1_metric, binary_focal_loss_metric]
            }

  for layer in trained_model.layers:
      layer.trainable = False

  trained_model.compile(optimizer=optimizer,
                        loss=losses,
                        metrics=metrics)

  trained_model.summary()

  generate_preds(trained_model, kitti, pc_encoder, target_encoder, val_ids, epoch, chkpt_dir, exp_id)
