
from pixor_utils.comet_ml import CometMLLogger

from config import configs
# from ped_config import configs

import os
import numpy as np
import matplotlib
import timeit
import random
import pprint

matplotlib.use('pdf')

import matplotlib.pyplot as plt

from datetime import datetime
from core.kitti import KITTI
from pixor_utils.model_utils import load_model, save_model
from data_utils.training_gen import TrainingGenerator

from test_utils.unittest import test_pc_encoder, test_target_encoder

DS_DIR = os.path.expanduser(configs['dataset_path'])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = configs["gpu_id"]
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

pc_encoder = configs['pc_encoder']

target_encoder = configs['target_encoder']

pprint.pprint(configs['stats'])

# #--------------------------------------#
# #-----------RUN UNIT TESTS-------------#
# #--------------------------------------#
# os.system('clear')
# rand_idx = np.random.randint(0, len(train_ids))
# test_id  = train_ids[rand_idx]
# pts, _ = kitti.get_velo(test_id, use_fov_filter=False)
# test_pc_encoder(pc_encoder, pts.T)
# boxes = kitti.get_boxes_3D(test_id)
# test_target_encoder(target_encoder, boxes)
# #--------------------------------------#
# #-----------RUN UNIT TESTS-------------#
# #--------------------------------------#

logdir = "{0}/logs/".format(CKPTS_DIR) + datetime.now().strftime("%m-%d-%H-%M")
scalar_logdir = logdir + "/scalars/"
file_writer = tf.summary.create_file_writer(scalar_logdir)
file_writer.set_as_default()

imgs_logdir = logdir + "/images/"
img_writer = tf.summary.create_file_writer(imgs_logdir)

hist_logdir = logdir + "/histograms/"
hist_writer = tf.summary.create_file_writer(hist_logdir)

if configs['use_pretrained'] is False:
    model = configs['model_fn'](data_format=configs['data_format'])
else:
    model = load_model(configs['last_ckpt_json'], configs['last_ckpt_h5'], configs['custom_objects'])
    
optimizer = configs['optimizer'](lr=LEARNING_RATE)
losses = configs['losses']
metrics = configs['metrics']

model.compile(
                optimizer=optimizer,
                loss=losses,
                metrics=metrics
             )

model.summary()

# @tf.function
# def traceme(x):
#     return model(x)

# tf.summary.trace_on(
#     graph=True,
#     profiler=True,
# )
# traceme(tf.zeros((1, configs['input_shape'][0], configs['input_shape'][1], configs['input_shape'][2])))
# with file_writer.as_default():
#     tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)

os.system('clear')
print('Everything is ready:')
print('Unit tests passes [OK]')
print('Model create/loaded [OK]')
print('Model compiled')
print('Training will start... NOW!')

os.system('cp {0} {1}'.format(configs['current_file_name'], CKPTS_DIR + '/'))

val_steps   = 0
train_steps = 0
lr_steps    = 0
cur_lr = 0
sched2 = False

# train_ids = configs['chosen_ids']
for cur_epoch in range(1, EPOCHS):
    train_gen = TrainingGenerator(reader=kitti, frame_ids=train_ids, batch_size=BATCH_SIZE,
                                    pc_encoder=pc_encoder, target_encoder=target_encoder, 
                                    n_threads=NUM_THREADS, max_queue_size=MAX_Q_SIZE)

    train_gen.start()
    start = timeit.default_timer()
    
    progress = train_gen.batch_count // 10
    cur_progress = progress

    print('Total number of batches is ->', train_gen.batch_count)

    # # batches = []
    # # for _ in range(train_gen.batch_count):
    # #     batches.append(train_gen.get_batch())

    # # batch = train_gen.get_batch()
    # # encoded_pcs, encoded_targets = batch['encoded_pcs'], batch['encoded_targets']

    # # while True:
    # #     for batch in batches:
    for batch_id in range(train_gen.batch_count):
    #     # Fetch batch
        batch = train_gen.get_batch()
        batch_ids = batch['frame_ids']
        encoded_pcs, encoded_targets = batch['encoded_pcs'], batch['encoded_targets']

        if lr_steps < configs['warmup_steps'] and sched2 is False:
            if configs['schedule1'] is not None:
                cur_lr = configs['schedule1'](lr_steps)
        else:
            if sched2 is False:
                lr_steps = 0
                sched2 = True
            if configs['schedule2'] is not None:
                cur_lr = configs['schedule2'](lr_steps)

        K.set_value(model.optimizer.lr, cur_lr)
        with file_writer.as_default():
            tf.summary.scalar('learning rate', data=cur_lr, step=train_steps)

        # start = datetime.now()
        metrics = model.fit(x=encoded_pcs, y=[encoded_targets[:,:,:,0:1], encoded_targets], epochs=1, verbose=False)
        # print((datetime.now() - start).total_seconds())

        for metric_name, metric_val in metrics.history.items():
            with file_writer.as_default():
                tf.summary.scalar(metric_name, data=metric_val[0], step=train_steps)

        # for i, id in enumerate(batch_ids):
        #     outmap = model(inputs=encoded_pcs, training=False)
        #     obj = outmap[i].numpy()
        #     tar = encoded_targets[i:i+1,:,:,0:1]
        #     with img_writer.as_default():
        #         out = np.concatenate((tar, obj), axis=0)
        #         tf.summary.image('{}'.format(id), out, step=train_steps)

        # if train_steps % configs['vis_every'] is 0 and train_steps is not 0:
        #     for var in model.trainable_variables:
        #         with hist_writer.as_default():
        #             tf.summary.histogram(var.name, var, step=train_steps)

        train_steps += 1
        lr_steps += 1

        if train_steps % configs['vis_every'] is 0 and train_steps is not 0:
            for chosen_id in configs['chosen_ids']:
                pts, _ = kitti.get_velo(chosen_id, use_fov_filter=False)
                boxes  = kitti.get_boxes_3D(chosen_id)

                if pts.shape[1] != 3:
                    pts = pts.T

                pc  = np.expand_dims(pc_encoder.encode(pts), axis=0)
                tar = np.expand_dims(np.squeeze(target_encoder.encode(boxes)[:,:,0:1]), axis=0)
                tar = np.expand_dims(tar, axis=-1)

                outmap = model(inputs=pc, training=False)
                obj = outmap[0].numpy()
                with img_writer.as_default():
                    out = np.concatenate((tar, obj), axis=0)
                    tf.summary.image('{}_tar'.format(chosen_id), out, step=train_steps)

                    pc = np.expand_dims((np.sum(np.squeeze(pc), axis=2) > 0).astype(np.float32), axis=-1)
                    pc = np.expand_dims(pc, axis=0)
                    tf.summary.image('{}_pc'.format(chosen_id), pc, step=train_steps)
                
                del pts, boxes, pc, tar, outmap, obj

        del batch
        # if batch_id > cur_progress:
        #     print('Processed {0} train samples in {1}'.format(cur_progress, (timeit.default_timer() - start) // 60))
        #     cur_progress += progress

    train_gen.stop()
    del train_gen

    # os.system('clear')
    print('Finished training the {0}-th epoch in {1}'.format(cur_epoch + 1 + configs['start_epoch'], (timeit.default_timer() - start) // 60))


    if cur_epoch % configs['hyperparams']['ckpt_every'] == 0 and cur_epoch is not 0:
        print('Saving current checkpoint...')
        save_model(model, CKPTS_DIR, configs['experiment_name'], cur_epoch + configs['start_epoch'])