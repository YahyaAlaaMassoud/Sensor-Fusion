
import tensorflow as tf

from tensorflow.keras.layers import Conv2D, SeparableConv2D, ReLU, BatchNormalization, Add

def create_conv_block(input_tensor, filters, kernel_size, num_layers):
  x = input_tensor
  for i in range(num_layers):
    with tf.name_scope("ConvBlock"):
      x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
  return x

def create_sep_conv_block(input_tensor, filters, kernel_size, num_layers):
  x = input_tensor
  for _ in range(num_layers):
    with tf.name_scope("SepConvBlock"):
      x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
      x = BatchNormalization()(x)
      x = ReLU()(x)
  return x 

def create_res_conv_block(input_tensor, filters, kernel_size, num_layers):
  last_input = input_tensor
  x = input_tensor
  for _ in range(num_layers):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  return Add()([x, last_input])

def create_res_sep_conv_block(input_tensor, filters, kernel_size, num_layers):
  last_input = input_tensor
  x = input_tensor
  for _ in range(num_layers):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  return Add()([x, last_input])