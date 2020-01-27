
import os
import io
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import timeit

DS_DIR = os.path.expanduser('/home/salam/datasets/KITTI/training')

from core.boxes import Box2D, Box3D
from core.kitti import KITTI, ALL_VEHICLES, CARS_ONLY, PEDESTRIANS_ONLY, CYCLISTS_ONLY, ALL_OBJECTS
from core.tf_records import create_tf_example

import tensorflow as tf

from object_detection.utils import dataset_util

CLASSES = {'Car': ['Car'], 'Pedestrian': ['Pedestrian'],  'Cyclist': ['Cyclist']}
CLASS_TO_ID = {'Car': 1, 
               'Pedestrian': 2, 
               'Cyclist': 3}

kitti = KITTI(DS_DIR, CLASSES)

train_ids = kitti.get_ids('train')
val_ids = kitti.get_ids('val')
micro_ids = kitti.get_ids('micro')

# id = '004139'
# img = kitti.get_image(id)
# print(img.shape)

def tf_records_generator(train=True):
    flags = tf.app.flags
    if train:
        flags.DEFINE_string('output_path', 'tf_records/train.tfrecord', '')
    else:
        flags.DEFINE_string('output_path', 'tf_records/val.tfrecord', '')
    FLAGS = flags.FLAGS

    writer = tf.io.TFRecordWriter(FLAGS.output_path)

    i = 0
    ids = train_ids if train == True else val_ids
    for id in ids:
        img = kitti.get_image(id)
        bboxes = kitti.get_boxes_2D(id)
        h, w = img.shape[0], img.shape[1]
        xmins, ymins, xmaxs, ymaxs = [], [], [], []
        classes_text, classes = [], []

        for box in bboxes:
            xmins.append(box.x1 / w)
            ymins.append(box.y1 / h)
            xmaxs.append(box.x2 / w)
            ymaxs.append(box.y2 / h)
            classes_text.append(str.encode(box.cls))
            classes.append(CLASS_TO_ID[box.cls])

        example = create_tf_example({
            'img': img,
            'w': w,
            'h': h,
            'xmins': xmins,
            'xmaxs': xmaxs,
            'ymins': ymins,
            'ymaxs': ymaxs,
            'classes_text': classes_text,
            'classes': classes
        })

        writer.write(example.SerializeToString())
        i += 1

        if i % 100 == 0:
            print('finished {0} examples...'.format(i))

    writer.close()


# tf_records_generator(train=False)