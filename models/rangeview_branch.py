
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
    conv_inputs.append(conv_block(input, 32, 7, 2)) # /2

  x = Average()(conv_inputs)
  l1 = x

  for i in range(4): # same res
    x = create_res_conv_block(x, 32, 3)
  l2 = x
  
  for i in range(4): # same res
    x = create_res_conv_block(x, 64, 3, i==0)
  l3 = x

  for i in range(6): # same res
    x = create_res_conv_block(x, 128, 3, i==0)
  l4 = x 
  
  return {
      'inputs': init_inputs,
      'outputs': [
        l2,
        l3,
        l4,
      ],
  }