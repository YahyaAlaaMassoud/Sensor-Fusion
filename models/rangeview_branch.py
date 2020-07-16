
from tensorflow.keras.layers import Average

from .blocks import conv_block, create_input_layer, deep_fuse_layer

def create_range_view_branch(input_shape=(375, 1242, 3), 
                             input_names=['rgb_img_input', 'depth_map_input', 'intensity_map_input', 'height_map_input']):
  init_inputs = []
  for name in input_names:
    init_inputs.append(create_input_layer(input_shape=input_shape, name=name))
  
  conv_inputs = []
  for input in init_inputs:
    conv_inputs.append(conv_block(input, 12, 7, 2))

  inputs = deep_fuse_layer(conv_inputs, 24, 3, 1)      # 2
  avg_l1 = Average()(inputs)
  inputs = deep_fuse_layer(inputs, 24, 3, 1)      # 3
  avg_l2 = Average()(inputs)
  inputs = deep_fuse_layer(inputs, 48, 3, 2)      # 4
  avg_l3 = Average()(inputs)
  inputs = deep_fuse_layer(inputs, 48, 3, 1)      # 5
  avg_l4 = Average()(inputs)
  inputs = deep_fuse_layer(inputs, 92, 3, 2)      # 6
  avg_l5 = Average()(inputs)

  return {
      'inputs': init_inputs,
      'outputs': [
        avg_l2,          
        avg_l3,
        avg_l4,
        avg_l5,
      ],
  }