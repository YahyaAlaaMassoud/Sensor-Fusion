
from pixor_utils.comet_ml import CometMLLogger

from car_config import configs
# from ped_config import configs

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import timeit
import random

from datasets.kitti import KITTI
from pixor_utils.model_utils import load_model, save_model
from pixor_utils.training_gen import TrainingGenerator
from pixor_utils.prediction_gen import PredictionGenerator

from test_utils.unittest import test_pc_encoder, test_target_encoder

DS_DIR = os.path.expanduser(configs['dataset_path'])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configs["gpu_id"]

np.random.seed(0)

import tensorflow as tf

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
TARGET_MEANS = configs['target_means']#np.array([-6.4501972e-03, 2.1161055e-02, -2.8581850e-02, -9.2645199e-04, 5.5109012e-01, 1.4958924e+00])
TARGET_STDS = configs['target_stds']#np.array([0.4555288, 0.8899461, 1.4819515, 0.76802343, 0.16860875, 0.33837482])
MEAN_HEIGHT, MEAN_ALTITUDE = configs['mean_height'], configs['mean_altitude']#1.52, 1.71

# Training
BATCH_SIZE = configs['hyperparams']['batch_size']
LEARNING_RATE = configs['hyperparams']['lr']
EPOCHS = configs['hyperparams']['epochs']
NUM_THREADS = configs['hyperparams']['num_threads']
MAX_Q_SIZE = configs['hyperparams']['max_q_size']

# Create dirs
OUTPUTS_DIR = os.path.join('outputs')
CKPTS_DIR = os.path.join(OUTPUTS_DIR, configs['ckpts_dir'])

for d in [OUTPUTS_DIR, CKPTS_DIR]:
    print("Creating directory: " + d)
    os.makedirs(d, exist_ok=True)

kitti = KITTI(DS_DIR, configs['training_target'])

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

pc_encoder = configs['pc_encoder'](shape=INPUT_SHAPE, P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)
target_encoder = configs['target_encoder'](shape=TARGET_SHAPE,
                                            target_means=TARGET_MEANS, target_stds=TARGET_STDS,
                                            mean_height=MEAN_HEIGHT, mean_altitude=MEAN_ALTITUDE,
                                            P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)

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

experiment = CometMLLogger()

val_steps = 0
train_steps = 0

for cur_epoch in range(EPOCHS):
    with experiment.experiment.train():
        train_gen = TrainingGenerator(reader=kitti, frame_ids=train_ids[:100], batch_size=BATCH_SIZE,
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

            total_loss, geo_loss, obj_loss = model.train_on_batch(x=encoded_pcs, y=encoded_targets)

            experiment.log_metric('total_loss', total_loss, train_steps)
            experiment.log_metric('obj_loss', obj_loss, train_steps)
            experiment.log_metric('geo_loss', geo_loss, train_steps)

            train_steps += 1

            del encoded_pcs, encoded_targets

            if batch_id > cur_progress:
                print('Processed {0} train samples in {1}'.format(cur_progress, (timeit.default_timer() - start) // 60))
                cur_progress += progress

        train_gen.stop()
        
        os.system('clear')
        print('Finished training the {0}-th epoch in {1}'.format(cur_epoch + 1 + configs['start_epoch'], (timeit.default_timer() - start) // 60))
        
    if cur_epoch % configs['hyperparams']['validate_every'] == 0 and cur_epoch is not 0:
        with experiment.experiment.validate():
            val_samples = configs['hyperparams']['n_val_samples']
            val_samples = np.random.choice(val_ids, val_samples, replace=False)
            val_gen = TrainingGenerator(reader=kitti, frame_ids=val_samples, batch_size=BATCH_SIZE,
                                        pc_encoder=pc_encoder, target_encoder=target_encoder,
                                        n_threads=NUM_THREADS, max_queue_size=MAX_Q_SIZE)
            
            val_gen.start()
            start = timeit.default_timer()
        
            progress = train_gen.batch_count // 10
            cur_progress = progress
            
            for batch_id in range(val_gen.batch_count):
                # Fetch batch
                batch = val_gen.get_batch()
                encoded_pcs, encoded_targets = batch['encoded_pcs'], batch['encoded_targets']
                predictions_map = model.predict_on_batch(x=encoded_pcs)
                
                total_loss = configs['losses']['output_map'](encoded_targets, predictions_map)
                obj_loss   = configs['metrics']['output_map'][1](encoded_targets, predictions_map)
                geo_loss   = configs['metrics']['output_map'][0](encoded_targets, predictions_map)

                experiment.log_metric('total_loss', total_loss, val_steps)
                experiment.log_metric('obj_loss', obj_loss, val_steps)
                experiment.log_metric('geo_loss', geo_loss, val_steps)
                
                val_steps += 1

                del encoded_pcs, encoded_targets

                if batch_id > cur_progress:
                    print('Processed {0} train samples in {1}'.format(cur_progress, (timeit.default_timer() - start) // 60))
                    cur_progress += progress

            val_gen.stop()
            
            os.system('clear')
            print('Finished validating the {0}-th epoch in {1}'.format(cur_epoch + 1 + configs['start_epoch'], (timeit.default_timer() - start) // 60))
            
    if cur_epoch % configs['hyperparams']['ckpt_every'] == 0 and cur_epoch is not 0:
        print('Saving current checkpoint...')
        save_model(model, CKPTS_DIR, configs['experiment_name'], cur_epoch + configs['start_epoch'])