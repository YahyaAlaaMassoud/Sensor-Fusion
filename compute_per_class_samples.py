

import deepdish as dd

from core.kitti import KITTI, ALL_OBJECTS, CARS_ONLY
from data_utils.augmentation import PointCloudAugmenter

DS_DIR = '/home/salam/datasets/KITTI/training'
reader = KITTI(DS_DIR, CARS_ONLY)

ids = reader.get_ids('train')

for id in ids:
    boxes = reader.get_boxes_3D(id)
    pts, _ = reader.get_velo(id, use_fov_filter=False)
    
    for i, box in enumerate(boxes):
        pts_inside_ids = PointCloudAugmenter.find_containing_points(box, pts.T)
        if len(pts_inside_ids) > 5:
            pts_inside = pts[:, pts_inside_ids].T
            
            data = {
                'frame_id': id,
                'num_points': len(pts_inside_ids),
                'pts': pts_inside,
                'box_x': box.x,
                'box_y': box.y,
                'box_z': box.z,
                'box_w': box.w,
                'box_l': box.l,
                'box_h': box.h,
                'box_yaw': box.yaw,
                'box_cls': box.cls,
            }
            
            dd.io.save('data_utils/aug_utils/annotations/cars/{0}-{1}.h5'.format(id, i), data)