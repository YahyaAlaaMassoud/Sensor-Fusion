
import os
import numpy as np

from pixor_utils.prediction_gen import PredictionGenerator
from pixor_utils.post_processing import nms_bev
from pixor_utils.pred_utils import boxes_to_pred_str

def generate_preds(model, kitti_reader, pc_encoder, target_encoder, frame_ids, epoch, ckpts_dir, exp_id, n_threads=6, max_queue_size=8):
    
    exp_name = '/{0}-epoch-{1}'.format(exp_id, epoch)
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
        decoded_boxes = nms_bev(decoded_boxes, iou_thresh=0.1, max_boxes=500, min_hit=0, axis_aligned=False)
        lines = boxes_to_pred_str(decoded_boxes, kitti_reader.get_calib(frames[0])[2])
        with open(os.path.join(exp_path, frames[0] + '.txt'), 'w') as txt:
            if len(decoded_boxes) > 0:
                txt.writelines(lines)
        if batch_id is not 0 and batch_id % 2 is 0:
            print('Finished predicting {0} frames...'.format(batch_id))
        
    val_gen.stop()
    
    os.system('cp -r {0} {1}'.format(os.getcwd() + '/' + ckpts_dir + exp_name, '/home/yahyaalaa/Yahya/kitti_dev/eval_kitti/build/results/'))
    os.system('cd eval_kitti/build/ && ./evaluate_object {0} val'.format(exp_name[1:]))