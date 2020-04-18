
import os 
import deepdish as dd

from training_gen import TrainingGenerator
from encoding_utils.pointcloud_encoder import OccupancyCuboidKITTI
from pixor_targets_new import PixorTargets3D
from datasets.kitti import KITTI, CARS_ONLY

DS_DIR = os.path.expanduser('/home/salam/datasets/KITTI/training')

kitti = KITTI(DS_DIR, CARS_ONLY)

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

pc_encoder = OccupancyCuboidKITTI(x_min=0, x_max=70, y_min=-40, y_max=40, z_min=-1, z_max=2.5, df=0.2, densify=False)

# Physical Space
P_WIDTH, P_HEIGHT, P_DEPTH = 70, 80, 3.5
# Point Cloud Encoder
INPUT_SHAPE = (800, 700, 35)
# Target Encoder
TARGET_SHAPE = (200, 175)
# Stats
stats = dd.io.load('kitti_stats/stats.h5')

target_encoder = PixorTargets3D(shape=TARGET_SHAPE, stats=stats, P_WIDTH=P_WIDTH, P_HEIGHT=P_HEIGHT, P_DEPTH=P_DEPTH)

train_gen = TrainingGenerator(kitti, train_ids, 2, pc_encoder, target_encoder, 1, 20)

train_gen.start()

for batch_id in range(train_gen.batch_count):
    # Fetch batch
    batch = train_gen.get_batch()
    encoded_pcs, encoded_targets = batch['encoded_pcs'], batch['encoded_targets']
    
    print(encoded_pcs.shape, encoded_targets.shape)

train_gen.stop()
