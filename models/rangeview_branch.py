
import tensorflow as tf

from tensorflow.keras.layers import Average, Add, MaxPool2D, Lambda, Concatenate

from .blocks import conv_block, create_input_layer, deep_fuse_layer, create_res_conv_block, transform_rangeview_to_bev

def create_range_view_branch(input_shapes=[(375, 1242, 3), 
                                           (375, 1242, 1), 
                                           (375, 1242, 1), 
                                           (375, 1242, 1),
                                           (224, 256 , 2),
                                           (112, 128 , 2),
                                           (56 , 64  , 2),
                                           (224, 256 , 3),
                                           (112, 128 , 3),
                                           (56 , 64  , 3),], 
                             input_names=['rgb_img_input', 
                                          'depth_map_input', 
                                          'intensity_map_input', 
                                          'height_map_input',
                                          'mapping_2x',
                                          'mapping_4x',
                                          'mapping_8x',
                                          'geo_2x',
                                          'geo_4x',
                                          'geo_8x',],
                             input_types=['float32',
                                          'float32',
                                          'float32',
                                          'float32',
                                          'int32',
                                          'int32',
                                          'int32',
                                          'float32',
                                          'float32',
                                          'float32',]):
  init_inputs = []
  inputs_dict = {}
  for name, shape, dtype in zip(input_names, input_shapes, input_types):
    inp = create_input_layer(input_shape=shape, name=name, dtype=dtype)
    init_inputs.append(inp)  
    inputs_dict[name] = inp
  
  conv_inputs = []
  for input in init_inputs:
    if 'mapping' not in input.name and 'geo' not in input.name:
      conv_inputs.append(conv_block(input, 32, 7, 2)) # /2

  x = Average()(conv_inputs)
  l1 = x

  for i in range(4): # same res
    x = create_res_conv_block(x, 32, 3)
  l2 = x
  
  for i in range(4): # /4
    x = create_res_conv_block(x, 64, 3, i==0)
  l3 = x

  for i in range(6): # /8
    x = create_res_conv_block(x, 128, 3, i==0)
  l4 = x 

  out_map_sz = (112, 128)
  TOP_DOWN_PYRAMID_SIZE = 96

  L2 = tf.image.resize(l2, (l3.shape[1], l3.shape[2]), 'nearest')
  L2 = conv_block(L2, TOP_DOWN_PYRAMID_SIZE, 3, 1)
  L2 = conv_block(L2, TOP_DOWN_PYRAMID_SIZE, 1, 1)

  L3 = conv_block(l3, TOP_DOWN_PYRAMID_SIZE, 1, 1)

  L4 = tf.image.resize(l4, (l3.shape[1], l3.shape[2]), 'nearest')
  L4 = conv_block(L4, TOP_DOWN_PYRAMID_SIZE, 3, 1)
  L4 = conv_block(L4, TOP_DOWN_PYRAMID_SIZE, 1, 1)

  RV_OUT = Add()([L2, L3, L4])

  RV2BEV_2x = Lambda(tf.gather_nd, arguments={'indices': inputs_dict['mapping_2x'], 'batch_dims': 1})(RV_OUT)
  RV2BEV_2x = Concatenate()([RV2BEV_2x, inputs_dict['geo_2x']]) # add geometric feature
  RV2BEV_2x = transform_rangeview_to_bev(RV2BEV_2x, 128)

  RV2BEV_4x = Lambda(tf.gather_nd, arguments={'indices': inputs_dict['mapping_4x'], 'batch_dims': 1})(RV_OUT)
  RV2BEV_4x = Concatenate()([RV2BEV_4x, inputs_dict['geo_4x']]) # add geometric feature
  RV2BEV_4x = transform_rangeview_to_bev(RV2BEV_4x, 192)

  RV2BEV_8x = Lambda(tf.gather_nd, arguments={'indices': inputs_dict['mapping_8x'], 'batch_dims': 1})(RV_OUT)
  RV2BEV_8x = Concatenate()([RV2BEV_8x, inputs_dict['geo_8x']]) # add geometric feature
  RV2BEV_8x = transform_rangeview_to_bev(RV2BEV_8x, 256)

  # print(RV2BEV_2x.shape, RV2BEV_4x.shape, RV2BEV_8x.shape)
  
  return {
      'inputs': init_inputs,
      'outputs': [
        RV2BEV_2x,
        RV2BEV_4x,
        RV2BEV_8x
      ],
  }