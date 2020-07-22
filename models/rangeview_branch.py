
from tensorflow.keras.layers import Average

from .blocks import conv_block, create_input_layer, deep_fuse_layer, create_res_conv_block

def create_range_view_branch(input_shape=(375, 1242, 3), 
                             input_names=['rgb_img_input', 'depth_map_input', 'intensity_map_input', 'height_map_input']):
  init_inputs = []
  for name in input_names:
    if name == 'rgb_img_input':
      init_inputs.append(create_input_layer(input_shape=(375, 1242, 3), name=name))  
    else:
      init_inputs.append(create_input_layer(input_shape=(375, 1242, 1), name=name))
  
  conv_inputs = []
  for input in init_inputs:
    conv_inputs.append(conv_block(input, 24, 7, 2)) # /2

  x = Average()(conv_inputs)
  l1 = x

  for i in range(3): # same res
    x = create_res_conv_block(x, 24, 3)
  l2 = x
  
  for i in range(4): # same res
    x = create_res_conv_block(x, 48, 3, i==0)
  l3 = x

  for i in range(4): # same res
    x = create_res_conv_block(x, 96, 3, i==0)
  l4 = x 
  
  # inputs = deep_fuse_layer(conv_inputs, 24, 3, False)      # 2
  # avg_l1 = Average()(inputs)
  # inputs = deep_fuse_layer(inputs, 24, 3, False)      # 3
  # avg_l2 = Average()(inputs)
  # inputs = deep_fuse_layer(inputs, 48, 3, True)      # 4 /2
  # avg_l3 = Average()(inputs)
  # inputs = deep_fuse_layer(inputs, 48, 3, False)      # 5
  # avg_l4 = Average()(inputs)
  # inputs = deep_fuse_layer(inputs, 92, 3, True)      # 6
  # avg_l5 = Average()(inputs)

  return {
      'inputs': init_inputs,
      'outputs': [
        l1,          
        l2,
        l3,
        l4,
      ],
  }