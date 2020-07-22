
from config import configs
# from ped_config import configs

import os
import io
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
from data_utils.generator import Generator, KITTIGen
from tt import bev
from pixor_utils.post_processing import nms_bev
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


logdir = "{0}/logs/".format(CKPTS_DIR) + datetime.now().strftime("%m-%d-%H-%M")
scalar_logdir = logdir + "/scalars/"
file_writer = tf.summary.create_file_writer(scalar_logdir)
file_writer.set_as_default()

imgs_logdir = logdir + "/images/"
img_writer = tf.summary.create_file_writer(imgs_logdir)

hist_logdir = logdir + "/histograms/"
hist_writer = tf.summary.create_file_writer(hist_logdir)

if configs['use_pretrained'] is False:
    model = configs['model_fn']()
else:
    model = load_model(configs['last_ckpt_json'], configs['last_ckpt_h5'], configs['custom_objects'])
    
optimizer = configs['optimizer'](lr=LEARNING_RATE)
losses = configs['losses']
metrics = configs['metrics']

model.summary()

os.system('cp {0} {1}'.format(configs['current_file_name'], CKPTS_DIR + '/'))

def compute_loss(model, x, y, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout, BatchNorm, etc.).
    obj, geo = model(x, training=training)
    
    obj_loss = losses['obj_map'](y[0], obj) * 1.0
    geo_loss = losses['geo_map'](y[1], geo) * 3.0
    
    return [obj_loss, geo_loss]

def compute_metric(model, x, y, training):
    metric_values = {}

    obj, geo = model(x, training=training)
    for key, metric_list in metrics.items():
        if key == 'geo_map':
            for metric_fn in metric_list:
                metric_values[metric_fn.__name__] = metric_fn(y[1], geo)
        elif key == 'obj_map':
            for metric_fn in metric_list:
                metric_values[metric_fn.__name__] = metric_fn(y[0], obj)

    for key, loss_fn in losses.items():
        if key == 'obj_map':
            metric_values[loss_fn.__name__] = loss_fn(y[0], obj)
        else:
            metric_values[loss_fn.__name__] = loss_fn(y[1], geo)
    
    return metric_values

def compute_grads(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = compute_loss(model, inputs, targets, training=True)
    grads = tape.gradient(loss_value, model.trainable_variables)
    grads = [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(model.trainable_variables, grads)]
    return loss_value, grads

train_steps = 0
lr_steps = 0
sched2 = False
cur_lr = 0
nms  = nms_bev('dist', -2.5, max_boxes=10)

print(configs['experiment_name'])

# from data_utils.add_rand_sample import add_random_sample
# add_random_sample_gen = add_random_sample(num_samples=40, sort_desc=False, filter_wall_thresh=200)

# for t in train_ids:
#     pts, _ = kitti.get_velo(t, use_fov_filter=False)
#     boxes  = kitti.get_boxes_3D(t)

#     # print('pts.shape', pts.shape)
#     # print('len(boxes)', len(boxes))
#     if len(boxes) > 1:
#         pts, boxes, _ = add_random_sample_gen(boxes, pts)
#     else:
#         continue

#     if pts.shape[1] != 3:
#         pts = pts.T

#     pc  = np.expand_dims(pc_encoder.encode(pts), axis=0)
#     tar = target_encoder.encode(boxes)
#     obj_map, obj_mask = tar[0][...,0], tar[0][...,1]
#     print('obj_map.shape', obj_map.shape)
#     obj_map  = np.expand_dims(np.expand_dims(np.squeeze(obj_map), axis=0), axis=-1)
#     obj_mask = np.expand_dims(np.expand_dims(np.squeeze(obj_mask), axis=0), axis=-1)
#     with img_writer.as_default():
#         tf.summary.image('{}_obj_map'.format(t), obj_map, step=train_steps)

#     cnvs = bev(pts=pts.T, gt_boxes=boxes)
#     plt.imshow(cnvs)
#     plt.axis('off')
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
#     buf.seek(0)
#     plot = tf.image.decode_png(buf.getvalue(), channels=4)
#     plot = tf.expand_dims(plot, 0)

#     with img_writer.as_default():
#         tf.summary.image('{}_plot_{}'.format(t, len(kitti.get_boxes_3D(t))), plot, step=train_steps)

# train_ids = configs['chosen_ids']
for cur_epoch in range(1, EPOCHS):
    start = timeit.default_timer()

    train_gen = KITTIGen(kitti, configs['chosen_ids'], BATCH_SIZE, pc_encoder=pc_encoder, target_encoder=target_encoder)
    # train_gen = KITTIGen(kitti, val_ids, BATCH_SIZE, pc_encoder=pc_encoder, target_encoder=target_encoder)

    data_len = int(np.ceil(len(train_ids) / BATCH_SIZE))
    print('training on {} batches'.format(data_len))
    print(configs['experiment_name'])

    for batch in train_gen:
        pc, rgb_img, depth_map, intensity_map, height_map, (obj, geo) = batch
        print('bev.shape          ', pc.shape)
        print('rgb_img.shape      ', rgb_img.shape)
        print('depth_map.shape    ', depth_map.shape)
        print('intensity_map.shape', intensity_map.shape)
        print('height_map.shape   ', height_map.shape)
        print('obj.shape          ', obj.shape)
        print('geo.shape          ', geo.shape)
        print('------------------------')

        if lr_steps < configs['warmup_steps'] and sched2 is False:
            if configs['schedule1'] is not None:
                cur_lr = configs['schedule1'](lr_steps)
        else:
            if sched2 is False:
                lr_steps = 0
                sched2 = True
            if configs['schedule2'] is not None:
                cur_lr = configs['schedule2'](lr_steps)

        K.set_value(optimizer.lr, cur_lr)
        with file_writer.as_default():
            tf.summary.scalar('learning rate', data=cur_lr, step=train_steps)
        
        # start = datetime.now()
        loss_value, grads = compute_grads(model, [pc, rgb_img, depth_map, intensity_map, height_map], (obj, geo))

        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        # print((datetime.now() - start).total_seconds())

        cur_metrics = compute_metric(model, [pc, rgb_img, depth_map, intensity_map, height_map], (obj, geo), True)   

        for metric_name, metric_val in cur_metrics.items():
            with file_writer.as_default():
                tf.summary.scalar(metric_name, data=metric_val, step=train_steps)

        if train_steps % configs['vis_every'] is 0 and train_steps is not 0:
            for var in model.trainable_variables:
                with hist_writer.as_default():
                    tf.summary.histogram(var.name, var, step=train_steps)

        if train_steps % configs['vis_every'] is 0 and train_steps is not 0:
            vis_gen = KITTIGen(kitti, configs['chosen_ids'], 1, pc_encoder=pc_encoder, target_encoder=target_encoder)
            ii = 0
            for batch in vis_gen:
                pc, rgb_img, depth_map, intensity_map, height_map, (obj, geo) = batch
                
                obj_map, obj_mask = obj[...,0], obj[...,1]
                obj_map  = np.expand_dims(np.expand_dims(np.squeeze(obj_map), axis=0), axis=-1)
                obj_mask = np.expand_dims(np.expand_dims(np.squeeze(obj_mask), axis=0), axis=-1)
                with img_writer.as_default():
                    tf.summary.image('{}_obj_map_{}'.format(configs['chosen_ids'][ii], obj_map.shape), obj_map, step=train_steps)
                    tf.summary.image('{}_obj_mask_{}'.format(configs['chosen_ids'][ii], obj_mask.shape), obj_mask, step=train_steps)

                    tf.summary.image('{}_rgb_img_{}'.format(configs['chosen_ids'][ii], rgb_img.shape), rgb_img, step=train_steps)
                    tf.summary.image('{}_depth_{}'.format(configs['chosen_ids'][ii], depth_map.shape), depth_map, step=train_steps)
                    # tf.summary.image('{}_intensity_{}'.format(configs['chosen_ids'][ii], intensity_map.shape), intensity_map, step=train_steps)
                    # tf.summary.image('{}_height_{}'.format(configs['chosen_ids'][ii], height_map.shape), height_map, step=train_steps)

                outmap = model(inputs=[pc, rgb_img, depth_map, intensity_map, height_map], training=False)
                obj = outmap[0].numpy()
                geo = outmap[1].numpy()

                outmap = np.squeeze(np.concatenate((obj, geo), axis=-1))
                decoded_boxes = target_encoder.decode(outmap, 0.25)
                filtered_boxes = nms(decoded_boxes)
                pts, _ = kitti.get_velo(configs['chosen_ids'][ii], use_fov_filter=False)
                boxes  = kitti.get_boxes_3D(configs['chosen_ids'][ii])
                cnvs = bev(pts=pts.T, gt_boxes=boxes, pred_boxes=filtered_boxes)
                cnvs = np.rot90(cnvs)
                plt.imshow(cnvs)
                plt.axis('off')
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                buf.seek(0)
                plot = tf.image.decode_png(buf.getvalue(), channels=4)
                plot = tf.expand_dims(plot, 0)

                with img_writer.as_default():
                    tf.summary.image('{}_plot_{}'.format(configs['chosen_ids'][ii], plot.shape), plot, step=train_steps)
                    tf.summary.image('{}_obj_pred_{}'.format(configs['chosen_ids'][ii], obj.shape), obj, step=train_steps)

                # del pts, boxes, pc, tar, outmap, obj

                ii += 1

        train_steps += 1
        lr_steps += 1

    print('Finished training the {0}-th epoch in {1}, @ {2}'.format(cur_epoch + 1 + configs['start_epoch'], (timeit.default_timer() - start) // 60, datetime.now()))

    if cur_epoch % configs['hyperparams']['ckpt_every'] == 0 and cur_epoch is not 0:
        print('Saving current checkpoint...')
        save_model(model, CKPTS_DIR, configs['experiment_name'], cur_epoch + configs['start_epoch'])