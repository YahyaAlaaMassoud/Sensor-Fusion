
from pixor_utils.comet_ml import CometMLLogger

from config import configs
# from ped_config import configs

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import random
import pprint

from datetime import datetime
from datasets.kitti import KITTI
from pixor_utils.model_utils import load_model, save_model
from data_utils.training_gen import TrainingGenerator

from test_utils.unittest import test_pc_encoder, test_target_encoder

DS_DIR = os.path.expanduser(configs['dataset_path'])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configs["gpu_id"]


import tensorflow as tf
import tensorflow.keras.backend as K

device_name = tf.test.gpu_device_name()
os.system('clear')
print('Conncted to Device:', device_name)

# Physical Space
P_WIDTH, P_HEIGHT, P_DEPTH = configs['phy_width'], configs['phy_height'], configs['phy_depth']#70, 80, 3.5
print('P_WIDTH:', P_WIDTH, '| P_HEIGHT:', P_HEIGHT, '| P_DEPTH:', P_DEPTH)
# Point Cloud Encoder
INPUT_SHAPE = configs['input_shape']

# Target Encoder
TARGET_SHAPE = configs['target_shape']

# Training
BATCH_SIZE = configs['hyperparams']['batch_size']
LEARNING_RATE = configs['hyperparams']['lr']
EPOCHS = configs['hyperparams']['epochs']
NUM_THREADS = configs['hyperparams']['num_threads']
MAX_Q_SIZE = configs['hyperparams']['max_q_size']

# Create dirs
OUTPUTS_DIR = os.path.join('outputs')
CKPTS_DIR   = os.path.join(OUTPUTS_DIR, configs['ckpts_dir'])

for d in [OUTPUTS_DIR, CKPTS_DIR]:
    print("Creating directory: " + d)
    os.makedirs(d, exist_ok=True)

kitti = KITTI(DS_DIR, configs['training_target'])

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

pc_encoder = configs['pc_encoder'](x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=2.5, df=0.1, densify=True)

target_encoder = configs['target_encoder'](shape=TARGET_SHAPE, stats=configs['stats'],
                                           P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)

pprint.pprint(configs['stats'])

#--------------------------------------#
#-----------RUN UNIT TESTS-------------#
#--------------------------------------#
os.system('clear')
rand_idx = np.random.randint(0, len(train_ids))
test_id  = train_ids[rand_idx]
pts, _ = kitti.get_velo(test_id, use_fov_filter=False)
test_pc_encoder(pc_encoder, pts.T)
boxes = kitti.get_boxes_3D(test_id)
test_target_encoder(target_encoder, boxes)
#--------------------------------------#
#-----------RUN UNIT TESTS-------------#
#--------------------------------------#

if configs['use_pretrained'] is False:
    model = configs['model_fn']()
else:
    model = load_model(configs['last_ckpt_json'], configs['last_ckpt_h5'], configs['custom_objects'])
    
optimizer = configs['optimizer'](lr=LEARNING_RATE)
losses = configs['losses']
metrics = configs['metrics']

model.compile(optimizer=optimizer,
              loss=losses,
              metrics=metrics)

model.summary()

os.system('clear')
print('Everything is ready:')
print('Unit tests passes [OK]')
print('Model create/loaded [OK]')
print('Model compiled')
print('Training will start... NOW!')

os.system('cp {0} {1}'.format(configs['current_file_name'], CKPTS_DIR + '/'))

experiment = CometMLLogger()

val_steps   = 0
train_steps = 0
lr_steps    = 0
cur_lr = configs['warmup_min']

for cur_epoch in range(1, EPOCHS):
    with experiment.experiment.train():
        train_gen = TrainingGenerator(reader=kitti, frame_ids=train_ids, batch_size=BATCH_SIZE,
                                      pc_encoder=pc_encoder, target_encoder=target_encoder, 
                                      n_threads=NUM_THREADS, max_queue_size=MAX_Q_SIZE)

        train_gen.start()
        start = timeit.default_timer()
        
        progress = train_gen.batch_count // 10
        cur_progress = progress

        print('Total number of batches is ->',train_gen.batch_count)
        
        for batch_id in range(train_gen.batch_count):
            # Fetch batch
            batch = train_gen.get_batch()
            encoded_pcs, encoded_targets = batch['encoded_pcs'], batch['encoded_targets']
            
            if configs['warmup']:
                if train_steps is not 0 and train_steps % (train_gen.batch_count // 4) == 0:
                    if cur_lr == configs['warmup_max']:
                        print('changing to 0.0001')
                        cur_lr = configs['warmup_min']
                    else:
                        print('changing to 0.001')
                        cur_lr = configs['warmup_max']
                    if cur_epoch > configs['warmup_epochs']:
                        cur_lr = configs['warmup_min']
                K.set_value(model.optimizer.lr, cur_lr)

            # start = timeit.default_timer()
            metrics = model.train_on_batch(x=encoded_pcs, y=encoded_targets, reset_metrics=False)
            # print('train_on_batch took {0}'.format(timeit.default_timer() - start))
            
            for metric_name, metric_val in zip(model.metrics_names, metrics):
                experiment.log_metric(metric_name, metric_val, train_steps)

            # experiment.log_metric('lr', model.optimizer.lr.numpy(), train_steps)
            train_steps += 1

            if batch_id > cur_progress:
                print('Processed {0} train samples in {1}'.format(cur_progress, (timeit.default_timer() - start) // 60))
                cur_progress += progress

        train_gen.stop()
        
        os.system('clear')
        print('Finished training the {0}-th epoch in {1}'.format(cur_epoch + 1 + configs['start_epoch'], (timeit.default_timer() - start) // 60))
        
    if cur_epoch % configs['hyperparams']['ckpt_every'] == 0 and cur_epoch is not 0:
        print('Saving current checkpoint...')
        save_model(model, CKPTS_DIR, configs['experiment_name'], cur_epoch + configs['start_epoch'])