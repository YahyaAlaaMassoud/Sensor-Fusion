
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add, Average, Lambda, UpSampling2D, DepthwiseConv2D, ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from .layers import BiFPN

RESIZE_METHOD = 'nearest'

def get_new_shape(fmap, resize_factor):
  # print('old shape', (fmap.shape[1], fmap.shape[2]))
  if fmap.shape[1] % 5 is not 0:
    new_w = (fmap.shape[1] * resize_factor + 1) 
  else:
    new_w = (fmap.shape[1] * resize_factor)
  if fmap.shape[2] % 5 is not 0:
    new_h = (fmap.shape[2] * resize_factor + 1)
  else:
    new_h = (fmap.shape[2] * resize_factor)
  # print('new shape', new_w, new_h)
  return (new_w, new_h)

def build_BiFPN_v1(fmap1, fmap2, fmap3, filters=192):
  with tf.name_scope("BiFPN_Layer"):

    fmap3_resized = tf.image.resize(fmap3, size=get_new_shape(fmap3, 2), method=RESIZE_METHOD)#tf.image.resize_with_pad(fmap3, target_height=fmap3_shape[1], target_width=fmap3_shape[0], method=RESIZE_METHOD) #
    fmap3_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3)(fmap3_resized)

    fmap2_inter_out = BiFPN()([fmap3_resized, fmap2])
    fmap2_inter_out = create_sep_conv_block(fmap2_inter_out, filters=filters, kernel_size=3, num_layers=1)
    fmap2_inter_out_resized = tf.image.resize(fmap2_inter_out, size=get_new_shape(fmap2_inter_out, 2), method=RESIZE_METHOD)#tf.image.resize_with_pad(fmap2_inter_out, target_height=fmap2_inter_out_shape[1], target_width=fmap2_inter_out_shape[0], method=RESIZE_METHOD) #
    fmap2_inter_out_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3)(fmap2_inter_out_resized)
    
    fmap1_out = BiFPN()([fmap2_inter_out_resized, fmap1])
    fmap1_out = create_sep_conv_block(fmap1_out, filters=filters, kernel_size=3, num_layers=1)
    fmap1_out_resized = AveragePooling2D()(fmap1_out)
    
    fmap2_out = BiFPN()([fmap1_out_resized, fmap2_inter_out, fmap2])
    fmap2_out = create_sep_conv_block(fmap2_out, filters=filters, kernel_size=3, num_layers=1)
    fmap2_out_resized = AveragePooling2D()(fmap2_out)

    fmap3_out = BiFPN()([fmap2_out_resized, fmap3])
    fmap3_out = create_sep_conv_block(fmap3_out, filters=filters, kernel_size=3, num_layers=1)
    
    return fmap1_out, fmap2_out, fmap3_out

def build_BiFPN_full_res(f1, f2, f3, f4, filters, data_format, max_relu_val=None):

  def _sep_conv(x):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, data_format=data_format)(x)
    if BN is True:
      x = BatchNormalization()(x)
    x = ReLU(max_value=max_relu_val)(x)
    return x

  with tf.name_scope("BiFPN_Layer"):

    f1_resized = f1
    f2_inter   = BiFPN()([f2, f1_resized])
    f2_inter   = create_depthwise_conv_block(f2_inter, strides=1, kernel_size=3, BN=True, data_format=data_format)
    
    f2_inter_resized = MaxPooling2D()(f2_inter)
    f3_inter         = BiFPN()([f3, f2_inter_resized])
    f3_inter         = create_depthwise_conv_block(f3_inter, strides=1, kernel_size=3, BN=True, data_format=data_format)

    f3_inter_resized = MaxPooling2D()(f3_inter)
    f4_out           = BiFPN()([f4, f3_inter_resized])
    f4_out           = create_depthwise_conv_block(f4_out, strides=1, kernel_size=3, BN=True, data_format=data_format)

    f4_out_resized = tf.image.resize(f4_out, size=get_new_shape(f4_out, 2), method=RESIZE_METHOD)
    f4_out_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3, data_format=data_format)(f4_out_resized)
    f3_out         = BiFPN()([f3, f3_inter, f4_out_resized])
    f3_out         = create_depthwise_conv_block(f3_out, strides=1, kernel_size=3, BN=True, data_format=data_format)

    f3_out_resized = tf.image.resize(f3_out, size=get_new_shape(f3_out, 2), method=RESIZE_METHOD)
    f3_out_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3, data_format=data_format)(f3_out_resized)
    f2_out         = BiFPN()([f2, f2_inter, f3_out_resized])
    f2_out         = create_depthwise_conv_block(f2_out, strides=1, kernel_size=3, BN=True, data_format=data_format)

    f2_out_resized = f2_out#tf.image.resize(f2_out, size=get_new_shape(f2_out, 2), method=RESIZE_METHOD)
    f2_out_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3, data_format=data_format)(f2_out_resized)
    f1_out         = BiFPN()([f1, f2_out_resized])
    f1_out         = create_depthwise_conv_block(f1_out, strides=1, kernel_size=3, BN=True, data_format=data_format)

    return f1_out, f2_out, f3_out, f4_out

  
def build_BiFPN_v2(f1, f2, f3, f4, filters=192, data_format='channels_last', max_relu_val=None, BN=False, kernel_size=3):

  def _sep_conv(x):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, data_format=data_format)(x)
    if BN is True:
      x = BatchNormalization()(x)
    x = ReLU(max_value=max_relu_val)(x)
    return x

  with tf.name_scope("BiFPN_Layer"):

    f2_inter = Add()([f1, f2])
    f2_inter = _sep_conv(f2_inter)

    f3_inter = Add()([f3, f2_inter])
    f3_inter = _sep_conv(f3_inter)

    f4_out = Add()([f4, f3_inter])
    f4_out = _sep_conv(f4_out)

    f3_out = Add()([f3, f3_inter, f4_out])
    f3_out = _sep_conv(f3_out)

    f2_out = Add()([f2, f2_inter, f3_out])
    f2_out = _sep_conv(f2_out)

    f1_out = Add()([f1, f2_out])
    f1_out = _sep_conv(f1_out)

    return f1_out, f2_out, f3_out, f4_out


def create_small_pixor_det(input_shape=(400, 350, 17), kr=None, ki='he_normal', data_format=None):
    def _cbr(t, filters, name, max_pool=False, batch_norm=True, kernel_size=3, max_relu_val=None):
        with tf.name_scope(name):
            if max_pool:
                t = MaxPooling2D((2, 2), strides=2)(t)
            t = Conv2D(filters, kernel_size=kernel_size, activation=None, padding='same', use_bias=not batch_norm, kernel_regularizer=kr, kernel_initializer=ki)(t)
            if batch_norm:
                t = BatchNormalization()(t)
            t = ReLU(max_value=max_relu_val)(t)
        return t

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Block 1
    with tf.name_scope('Block_1'):
        x = _cbr(x, 32, "C1")
        x = _cbr(x, 32, "C2")
        b1_out = x
        x = _cbr(x, 64, "MaxPool", max_pool=True)

    # Block 2
    with tf.name_scope('Block_2'):
        x = _cbr(x, 64, "C1")
        b2_out = x
        x = _cbr(x, 128, "MaxPool", max_pool=True)

    # Block 3
    with tf.name_scope('Block_3'):
        for i in range(1, 3):
            x = _cbr(x, 128, "C{}".format(i))
        b3_out = x
        x = _cbr(x, 256, "MaxPool", max_pool=True)

    # Block 4
    with tf.name_scope('Block_4'):
        for i in range(1, 3):
            x = _cbr(x, 256, "C{}".format(i))
        b4_out = x

    bifpn_filters = 256

    # Aggregate feature maps
    with tf.name_scope('Concatenation'):
      # Downsample block 1 twice
      b1_out = AveragePooling2D((2, 2), strides=2)(b1_out)
      b1_out = _cbr(b1_out, bifpn_filters, '', kernel_size=1)

      # Downsample block 2
      b2_out = _cbr(b2_out, bifpn_filters, '', kernel_size=1)

      b3_out = tf.image.resize(b3_out, size=b2_out.shape[1:3])
      b3_out = _cbr(b3_out, bifpn_filters, '', kernel_size=1)

      # Upsample block 4
      b4_out = tf.image.resize(b4_out, size=b3_out.shape[1:3])
      b4_out = _cbr(b4_out, bifpn_filters, '', kernel_size=3)
      b4_out = tf.image.resize(b4_out, size=b3_out.shape[1:3])
      b4_out = _cbr(b4_out, bifpn_filters, '', kernel_size=1)

      b1_out, b2_out, b3_out, b4_out = build_BiFPN_v2(b1_out, b2_out, b3_out, b4_out, max_relu_val=6, filters=bifpn_filters)
      b1_out, b2_out, b3_out, b4_out = build_BiFPN_v2(b1_out, b2_out, b3_out, b4_out, max_relu_val=6, filters=bifpn_filters)

      x = Concatenate(axis=-1, name='Concat')([b1_out, b2_out, b3_out, b4_out])
      # x = Add()([b1_out, b2_out, b3_out, b4_out])

    # Head
    with tf.name_scope('Head'):
      b1 = _cbr(x, bifpn_filters//2, "B1", max_relu_val=6)
      b1 = _cbr(b1, bifpn_filters//2, "B1", max_relu_val=6)

      b2 = _cbr(x, bifpn_filters//2, "B2", max_relu_val=6)
      b2 = _cbr(b2, bifpn_filters//2, "B2", max_relu_val=6)
      # for i in range(2, 5):
      #     x = _cbr(x, bifpn_filters, "C{}".format(i), batch_norm=False, max_relu_val=None)

      obj_map = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='obj_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(b1)
      geo_map = Conv2D(11, kernel_size=3, padding='same', activation=None, name='geo_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(b2)

    return Model(inputs=input_tensor, outputs=[obj_map, geo_map])


# model = create_pixor_det()
# model.summary()
