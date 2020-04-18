
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
import os
import pprint
import deepdish as dd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import gc
import tensorflow as tf
import datetime

from timeit import default_timer as timer
from numba import cuda
from pixor_utils.model_utils import load_model
from training_utils.losses import focal, smooth_l1_loss
# from training_utils.metrics import objectness_metric, regression_metric
from models.pixor_det import BiFPN
from models.initializers import PriorProbability
from datasets.kitti import KITTI, ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY, SMALL_OBJECTS
from encoding_utils.pointcloud_encoder import OccupancyCuboidKITTI
from pixor_targets import PIXORTargets
from data_utils.prediction_gen import PredictionGenerator
from data_utils.nms_thread import NMSGenerator
from pixor_utils.post_processing import nms_bev
from pixor_utils.pred_utils import boxes_to_pred_str
from test_utils.unittest import test_pc_encoder, test_target_encoder

from tensorflow.keras.optimizers import Adam

chkpts_dir = 'outputs'
DS_DIR = os.path.expanduser('/home/salam/datasets/KITTI/training')

def generate_preds(model, kitti_reader, pc_encoder, target_encoder, frame_ids, epoch, ckpts_dir, exp_id, n_threads=6, max_queue_size=8, split='val'):
    
    exp_name = '/{0}-{1}-split-epoch-{2}'.format(exp_id, split, epoch)
    exp_path = ckpts_dir + exp_name + '/data/'
    
    os.makedirs(exp_path, exist_ok=True)
    
    val_gen = PredictionGenerator(reader=kitti_reader, frame_ids=frame_ids, batch_size=1,
                                  pc_encoder=pc_encoder, n_threads=n_threads, max_queue_size=max_queue_size)
    val_gen.start()
    
    for batch_id in range(val_gen.batch_count):
        batch = val_gen.get_batch()
        frames, encoded_pcs = batch['frame_ids'], batch['encoded_pcs']
        outmap = np.squeeze(model.predict_on_batch(encoded_pcs).numpy())
        decoded_boxes = target_encoder.decode(np.squeeze(outmap), 0.05)
        decoded_boxes = nms_bev('iou', 0.1, max_boxes=55, axis_aligned=False)(decoded_boxes)
        lines = boxes_to_pred_str(decoded_boxes, kitti_reader.get_calib(frames[0])[2])
        with open(os.path.join(exp_path, frames[0] + '.txt'), 'w') as txt:
            if len(decoded_boxes) > 0:
                txt.writelines(lines)
        if batch_id is not 0 and batch_id % 2 is 0:
            print('Finished predicting {0} frames...'.format(batch_id))
        
    val_gen.stop()
    
    os.system('cp -r {0} {1}'.format(os.getcwd() + '/' + ckpts_dir + exp_name, '/home/yahyaalaa/Yahya/kitti_dev/eval_kitti/build/results/'))
    os.system('cd eval_kitti/build/ && ./evaluate_object {0} {1}'.format(exp_name[1:], split))

def generate_preds_v2(model, kitti_reader, pc_encoder, target_encoder, frame_ids, epoch, ckpts_dir, exp_id, n_threads=8, max_queue_size=16, split='val'):
    
    exp_name = '/{0}-{1}-split-epoch-{2}'.format(exp_id, split, epoch)
    exp_path = ckpts_dir + exp_name + '/data/'
    
    os.makedirs(exp_path, exist_ok=True)
    
    batch_size = 2
    val_gen = PredictionGenerator(reader=kitti_reader, frame_ids=frame_ids, batch_size=batch_size,
                                  pc_encoder=pc_encoder, n_threads=n_threads, max_queue_size=max_queue_size)
    val_gen.start()
    
    pred_boxes = []
    finished = 0

    for batch_id in range(val_gen.batch_count):
        batch = val_gen.get_batch()
        frames, encoded_pcs = batch['frame_ids'], batch['encoded_pcs']
        for i in range(encoded_pcs.shape[0]):
            start = datetime.datetime.now()
            outmap = model.predict_on_batch(np.expand_dims(encoded_pcs[i], axis=0))
            end = datetime.datetime.now()
            print(int((end-start).total_seconds() * 1000))
            # print(outmap.shape)
            obj = outmap[0]#.numpy()
            geo = outmap[1]#.numpy()
            outmap = np.concatenate((obj, geo), axis=-1)
            print('outmap.shape', outmap.shape)
            pred_boxes.append({
                'frame': frames[i],
                'outmap': outmap,
            })
        finished += encoded_pcs.shape[0]
        print('finished predicting {0} examples'.format(finished))

    val_gen.stop()

    del model
    tf.keras.backend.clear_session()
    gc.collect()
    cuda.get_current_device().reset()
    print('deleted model')

    nms_threads = 8
    gen = NMSGenerator(reader=kitti_reader, target_encoder=target_encoder, preds=pred_boxes, exp_path=exp_path, n_threads=nms_threads)
    gen.start()
    new_size = 0
    while True:
        # if len(gen.done) != new_size:
        #     new_size = len(gen.done)
        #     print('Done with ', new_size, 'frames')
        if len(gen.done) == nms_threads:
            break
    gen.stop()

    os.system('cp -r {0} {1}'.format(os.getcwd() + '/' + ckpts_dir + exp_name, '/home/yahyaalaa/Yahya/kitti_dev/eval_kitti/build/results/'))
    os.system('cd eval_kitti/build/ && ./evaluate_object {0} {1}'.format(exp_name[1:], split))

def show_dirs(dirs):
    print('Which checkpoint do you want to generate predictions for?')
    for i, dir in enumerate(dirs):
        print(i + 1, '-', dir)
        
def customInput(in_type, mssg, err_mssg=None, fn=None, fn_val=None):
    while True:
        if fn is not None:
            if fn_val is not None:
                fn(fn_val)
            else:
                fn()
        try:
            value = in_type(input(mssg))
            return value
        except ValueError:
            os.system('clear')
            print(err_mssg)

def main():
    os.system('clear')
    chkpts = os.listdir(chkpts_dir)
    
    chosen_chkpt = ''
    while True:
        chkpt_id = customInput(in_type=int,
                               mssg='Enter a number between 1 and {0}: '.format(len(chkpts)),
                               err_mssg='Make sure the number is between 1 and {0}!\n'.format(len(chkpts)), 
                               fn=show_dirs, 
                               fn_val=chkpts)
        os.system('clear')        
        if chkpt_id >= 1 and chkpt_id <= len(chkpts):
            chosen_chkpt = chkpts[chkpt_id - 1]
            break
    
    chkpt_dir = os.path.join(chkpts_dir, chosen_chkpt)
    exp_id    = 'preds_' + chosen_chkpt
    
    numberOfPreds = customInput(in_type=int,
                                mssg='Please enter the number of checkpoints you want to test: ')
    
    epochList = []
    for i in range(numberOfPreds):
        epoch = customInput(in_type=int,
                            mssg='{0} - Enter epoch number: '.format(i + 1),
                            err_mssg='Please nter an integer number!')
        epochList.append(epoch)
        
    os.system('clear')
    all_chkpt_files = os.listdir(chkpt_dir)
    chkpts_map = {}
    for epoch in epochList:
        chkpts_map[epoch] = {}
        for chkpt_file in all_chkpt_files:
            if '_{0}.'.format(epoch) in chkpt_file:
                ext = chkpt_file.split('.')[-1]
                print('Checkpoint for epoch {0} - {1}: OK'.format(epoch, ext))
                chkpts_map[epoch][ext] = os.path.join(chkpt_dir, chkpt_file)
    
    val = False
    train = False
    while True:
        split_ids = customInput(in_type=int,
                                mssg='0 -> evaluate on train set\n1 -> evaluate on validation set\n',
                                err_mssg='Please enter valid input!')
        # os.system('clear')
        if split_ids == 0:
            train = True
            break
        elif split_ids == 1:
            val = True
            break
        
    class_dict = {
        'car': CARS_ONLY,
        'ped': PEDESTRIANS_ONLY,
        'cycl': CYCLISTS_ONLY,
        'vec': ALL_VEHICLES,
        'sml': SMALL_OBJECTS,
    }
    target_class = CARS_ONLY
    while True:
        chosen_class = customInput(in_type=str,
                                   mssg="car  -> evaluate on cars only\n" + \
                                        "ped  -> evaluate on pedestrians only\n" + \
                                        "vec  -> evaluate on all vehicle types (cars, vans, trucks)\n" + \
                                        "sml  -> evaluate on all small objects (pedestrians, cyclists)\n" + \
                                        "cycl -> evaluate on cyclists only\n",
                                   err_mssg='Please enter valid input!')
        # os.system('clear')    
        if chosen_class in ['car', 'ped', 'cycl', 'vec', 'sml']:
            target_class = class_dict[chosen_class]
            break
    
    #--------------------------------------#
    #----------------KITTI-----------------#
    #--------------------------------------#
    # Physical Space
    P_WIDTH, P_HEIGHT, P_DEPTH = 70, 80, 3.5

    # Point Cloud Encoder
    INPUT_SHAPE = 800, 700, 35

    # Target Encoder
    TARGET_SHAPE = (200, 175)
    
    kitti = KITTI(DS_DIR, target_class)

    train_ids = kitti.get_ids('train')
    val_ids = kitti.get_ids('val')

    pc_encoder     = OccupancyCuboidKITTI(x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=3, df=[0.1, 0.1, 0.2], densify=True)
    target_encoder = PIXORTargets(shape=TARGET_SHAPE,
                                  stats=dd.io.load('kitti_stats/stats.h5'), subsampling_factor=(0.8, 1.2),
                                  P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)
    #--------------------------------------#
    #----------------KITTI-----------------#
    #--------------------------------------#

    #--------------------------------------#
    #-----------RUN UNIT TESTS-------------#
    #--------------------------------------#
    # os.system('clear')
    # rand_idx = np.random.randint(0, len(train_ids))
    # test_id  = train_ids[rand_idx]
    # pts, _ = kitti.get_velo(test_id, use_fov_filter=False)
    # test_pc_encoder(pc_encoder, pts.T)
    # boxes = kitti.get_boxes_3D(test_id)
    # test_target_encoder(target_encoder, boxes)
    #--------------------------------------#
    #-----------RUN UNIT TESTS-------------#
    #--------------------------------------#

    for key, val in chkpts_map.items():
        if len(val) > 0:
            model = load_model(val['json'], val['h5'], {'BiFPN': BiFPN, 'PriorProbability': PriorProbability})
            # optimizer = Adam(lr=0.0001)
            # losses = {
            #         # 'obj_map': focal(alpha=0.75, gamma=1., subsampling_flag=False, data_format='channels_last'),
            #         # 'geo_map': smooth_l1_loss(sigma=3., reg_channels=8, data_format='channels_last'),
            #         }
            # metrics = {
            #         }

            for layer in model.layers:
                layer.trainable = False

            # model.compile(optimizer=optimizer,
            #                 loss=losses,
            #                 metrics=metrics)

            model.summary()

            if train:
                ids = train_ids
                split = 'train'
            elif val:
                ids = val_ids
                split = 'val'
                
            generate_preds_v2(model=model, 
                           kitti_reader=kitti, 
                           pc_encoder=pc_encoder, 
                           target_encoder=target_encoder, 
                           frame_ids=ids, 
                           epoch=key, 
                           ckpts_dir=chkpt_dir, 
                           exp_id=exp_id, 
                           split=split)

if __name__ == "__main__":
    '''
        Each checkpoint should be saved within 2 files (H5 & JSON)
        Each checkpoint name should have the same style: experimentName_epochNumber.ext
    '''
    main()