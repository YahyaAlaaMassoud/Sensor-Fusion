
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D, SeparableConv2D, ReLU, BatchNormalization, Add, DepthwiseConv2D, Activation, Dense, \
                                       GlobalAveragePooling2D, Reshape, multiply
from tensorflow.keras.initializers import RandomNormal

def squeeze_excite_block(input_tensor, ratio=4):
  init = input_tensor
  channel_axis = 1 if K.image_data_format() == "channels_first" else -1
  filters = input_tensor.shape[channel_axis]
  se_shape = (1, 1, filters)

  se = GlobalAveragePooling2D()(init)
  se = Reshape(se_shape)(se)
  se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
  se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

  if K.image_data_format() == 'channels_first':
      se = Permute((3, 1, 2))(se)

  x = multiply([init, se])
  return x

def inverted_res_block(input_tensor, filters, kernel_size, expansion_factor, width_mul, strides, res=False, act='relu', max_relu_val=None, excite=False, kr=None):
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
  # Depth
  tchannel = K.int_shape(input_tensor)[channel_axis] * expansion_factor
  # Width
  cchannel = int(filters * width_mul)

  x = conv_block(input_tensor=input_tensor, filters=tchannel, kernel_size=(1, 1), strides=(1, 1), max_relu_val=max_relu_val, act=act, BN=True)

  x = DepthwiseConv2D(kernel_size, strides=(strides, strides), depth_multiplier=1, padding='same', kernel_regularizer=kr)(x)
  x = BatchNormalization(axis=channel_axis)(x)
  if act == 'relu':
    x = ReLU(max_value=max_relu_val)(x)

  x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=kr)(x)
  x = BatchNormalization(axis=channel_axis)(x)

  if excite:
    x = squeeze_excite_block(x)

  if res:
    x = Add()([x, input_tensor])

  return x

def conv_block(input_tensor, filters, kernel_size, strides, BN=True, data_format='channels_last', max_relu_val=None, act='relu', kr=None):
  options = {
    'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
    'bias_initializer': 'zeros',
    'kernel_regularizer': kr,
  }
  x = input_tensor
  with tf.name_scope("ConvBlock"):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=not BN, data_format=data_format, **options)(x)
    if BN is True:
      if data_format == 'channels_first':
        x = BatchNormalization(axis=1)(x)
      else:
        x = BatchNormalization(axis=-1)(x)
    if act == 'relu':
      x = ReLU(max_value=max_relu_val)(x)
  return x

def create_sep_conv_block(input_tensor, filters, kernel_size, BN=True, data_format='channels_last', max_relu_val=None):
  x = input_tensor
  with tf.name_scope("SepConvBlock"):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, data_format=data_format)(x)
    if BN is True:
      if data_format == 'channels_first':
        x = BatchNormalization(axis=1)(x)
      else:
        x = BatchNormalization(axis=-1)(x)
    x = ReLU(max_value=max_relu_val)(x)
  return x 

def create_depthwise_conv_block(input_tensor, kernel_size, strides, BN=True, data_format='channels_last', max_relu_val=None):
  x = input_tensor
  with tf.name_scope("DepthwiseConvBlock"):
    x = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same', use_bias=not BN, data_format=data_format)(x)
    if BN is True:
      if data_format == 'channels_first':
        x = BatchNormalization(axis=1, momentum=0.999)(x)
      else:
        x = BatchNormalization(axis=-1, momentum=0.999)(x)
    x = ReLU(max_value=max_relu_val)(x)
  return x 

def create_res_conv_block(input_tensor, filters, kernel_size, BN=True, data_format='channels_last', max_relu_val=None):
  last_input = input_tensor
  x = input_tensor
  x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, data_format=data_format)(x)
  if BN is True:
      if data_format == 'channels_first':
        x = BatchNormalization(axis=1)(x)
      else:
        x = BatchNormalization(axis=-1)(x)
  x = ReLU(max_value=max_relu_val)(x)
  return Add()([x, last_input])

def create_res_sep_conv_block(input_tensor, filters, kernel_size, BN=True, data_format='channels_last', max_relu_val=None):
  last_input = input_tensor
  x = input_tensor
  x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, data_format=data_format)(x)
  if BN is True:
    if data_format == 'channels_first':
      x = BatchNormalization(axis=1)(x)
    else:
      x = BatchNormalization(axis=-1)(x)
  x = ReLU(max_value=max_relu_val)(x)
  return Add()([x, last_input])

def class_head(input_tensor, num_classes, depth, filters, data_format='channels_last', max_relu_val=None):
  options = {
    'kernel_size': 3,
    'strides': 1,
    'padding': 'same',
    'data_format': data_format,
    'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
  }

  x = input_tensor
  for _ in range(depth):
    x = Conv2D(
      filters=filters,
      bias_initializer='zeros',
      **options
    )(x)
    x = ReLU(max_value=max_relu_val)(x)

  output = Conv2D(
    filters=num_classes,
    bias_initializer='zeros',
    name='obj_map',
    activation='sigmoid',
    **options
  )(x)

  return output

def reg_head(input_tensor, num_classes, depth, filters, data_format='channels_last', max_relu_val=None):
  options = {
    'kernel_size': 3,
    'strides': 1,
    'padding': 'same',
    'kernel_initializer': RandomNormal(mean=0.0, stddev=0.01, seed=None),
    'bias_initializer': 'zeros',
    'data_format': data_format,
  }

  x = input_tensor
  for _ in range(depth):
    x = Conv2D(
      filters=filters,
      **options
    )(x)
    x = ReLU(max_value=max_relu_val)(x)
  
  output = Conv2D(
    filters=num_classes,
    name='geo_map',
    **options
  )(x)

  return output