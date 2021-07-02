```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from PIL import Image
from encoding_utils.voxelizer import BEVVoxelizer
import sf_demo_utils as sf
```


```python
seq_num = '0084'

seq_dir = '/home/yahyaalaa/Yahya/KITTI_full_sequences/2011_09_26_drive_{}_sync/2011_09_26_drive_{}_sync'
seq_dir = seq_dir.format(seq_num, seq_num)

dirs = sf.define_data_dir(seq_dir)

dataset_ids = sf.get_dataset_ids(seq_dir, dirs['rgb_dir'])
```


```python
voxelizer = sf.get_preprocess_fn()
target_encoder, nms = sf.get_postprocess_fn()
```


```python
sensor_fusion_model = sf.get_sensor_fusion_model(trainable=False);
```

    WARNING:tensorflow:Layer Downsample_Block1_0 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    WARNING:tensorflow:Layer Downsample_Block1_1 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    1st RV Block (None, 188, 621, 32)
    2nd RV Block (None, 94, 311, 64)
    3d RV Block (None, 47, 156, 128)
    Concat RV Block (None, 94, 311, 192)
    WARNING:tensorflow:Layer concatenate is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    WARNING:tensorflow:Layer concatenate_1 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    WARNING:tensorflow:Layer concatenate_2 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    
    WARNING:tensorflow:Layer BEV_Block1_0 is casting an input tensor from dtype float64 to the layer's dtype of float32, which is new behavior in TensorFlow 2.  The layer has dtype float32 because it's dtype defaults to floatx.
    
    If you intended to run this layer in float32, you can safely ignore this warning. If in doubt, this warning is likely only an issue if you are porting a TensorFlow 1.X model to TensorFlow 2.
    
    To change all layers to have dtype float64 by default, call `tf.keras.backend.set_floatx('float64')`. To change just this layer, pass dtype='float64' to the layer constructor. If you are the author of this layer, you can disable autocasting by passing autocast=False to the base Layer constructor.
    



```python
test_id = 3
cur_timestamp, seq_dir = dataset_ids[test_id][0], dataset_ids[test_id][1]

calib_data = sf.get_calib_data(data_dirs=dirs)
input_data = sf.get_input_data(ts=cur_timestamp,
                               calib_data=calib_data,
                               data_dirs=dirs,
                               voxelizer=voxelizer)
```


```python
img = sf.get_image_from_data(input_data)
bev = sf.get_bev_from_data(input_data)

plt.figure(figsize = (15, 15))
plt.imshow(np.squeeze(bev))

plt.figure(figsize = (15, 15))
plt.imshow(np.squeeze(img))
```


```python
boxes = sf.inference(sensor_fusion_model, input_data, target_encoder, nms)
```


```python
rv = sf.range_view(np.squeeze(img), calib_data['P2'], boxes)

plt.figure(figsize = (15, 15))
plt.imshow(np.squeeze(img))

plt.figure(figsize = (15, 15))
plt.imshow(np.squeeze(rv))
```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).





    <matplotlib.image.AxesImage at 0x7f8db1d573c8>




![png](output_7_2.png)



![png](output_7_3.png)

