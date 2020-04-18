
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

class BiFPN(tf.keras.layers.Layer):

  def __init__(self, **kwargs):
    if 'eps' in kwargs:
      del kwargs['eps']
    super(BiFPN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.w = self.add_weight(name='w',
                             shape=(len(input_shape),),
                             initializer=tf.keras.initializers.constant(1 / len(input_shape)), #'uniform',
                             trainable=True)
    self.eps = 0.0001
    super(BiFPN, self).build(input_shape)

  def call(self, x):
    assert isinstance(x, list)

    w = ReLU()(self.w)
    # w = self.w
    x = tf.reduce_sum([w[i] * x[i] for i in range(len(x))], axis=0)
    x = x / (tf.reduce_sum(w) + self.eps)
    
    return x

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super(BiFPN, self).get_config()
    return config

def _build_BiFPN_full_res(f1, f2, f3, filters=192, kernel_size=3, BN=False, max_relu_val=None, weighted=False):

    def _sep_conv(x):
        x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-4, l2=5e-4))(x)
        if BN is True:
            x = BatchNormalization()(x)
        x = ReLU(max_value=max_relu_val)(x)
        return x

    fusion_fn = Add
    if weighted:
        fusion_fn = BiFPN
    print(fusion_fn.__name__)

    f1_resized = AveragePooling2D()(f1)
    f2_inter   = fusion_fn()([f2, f1_resized])
    f2_inter   = _sep_conv(f2_inter)

    f2_inter_resized = AveragePooling2D()(f2_inter)
    f3_out           = fusion_fn()([f2_inter_resized, f3])
    f3_out           = _sep_conv(f3_out)

    f3_out_resized = UpSampling2D()(f3_out)
    f3_out_resized = ZeroPadding2D(((0,0), (0,1)))(f3_out_resized)
    f2_out         = fusion_fn()([f2, f2_inter, f3_out_resized])
    f2_out           = _sep_conv(f2_out)

    f2_out_resized = UpSampling2D()(f2_out)
    f1_out         = fusion_fn()([f2_out_resized, f1])
    f1_out           = _sep_conv(f1_out)

    return f1_out, f2_out, f3_out


def create_small_det(input_shape=(800, 700, 17), kr=tf.keras.regularizers.l1_l2(l1=5e-4, l2=5e-4), ki='he_normal', data_format=None):
    
    def _cbr(t, filters, name, max_pool=False, batch_norm=True, kernel_size=3, max_relu_val=None, strides=1):
        with tf.name_scope(name):
            if max_pool:
                t = MaxPooling2D((2, 2), strides=2)(t)
            t = Conv2D(filters, kernel_size=kernel_size, strides=strides, activation=None, padding='same', use_bias=not batch_norm, kernel_regularizer=kr, kernel_initializer=ki)(t)
            if batch_norm:
                t = BatchNormalization()(t)
            t = ReLU(max_value=max_relu_val)(t)
        return t

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    # Block 1
    with tf.name_scope('Block_1'):
        last_in = x
        x = _cbr(x, 32, "C1", kernel_size=7, strides=2) # 800

        for _ in range(4):
            x = _cbr(x, 64, "C2") # 400

        b1_out = x
        x = _cbr(x, 128, "MaxPool", max_pool=True) # 200

    # Block 2
    with tf.name_scope('Block_2'):
        last_in = x
        for i in range(6):
            x = _cbr(x, 128, "C{}".format(i))
        b2_out = x
        x = Add()([last_in, x])
        x = _cbr(x, 256, "MaxPool", max_pool=True) # 100

    # Block 3
    with tf.name_scope('Block_3'):
        for i in range(6):
            x = _cbr(x, 256, "C{}".format(i))
        b3_out = x

    bifpn_filters = 144

    # Aggregate feature maps
    with tf.name_scope('Concatenation'):
        b1_out = _cbr(b1_out, bifpn_filters, '', kernel_size=1)
        b2_out = _cbr(b2_out, bifpn_filters, '', kernel_size=1)
        b3_out = _cbr(b3_out, bifpn_filters, '', kernel_size=1)
        
        b1_out, b2_out, b3_out = _build_BiFPN_full_res(b1_out, b2_out, b3_out, filters=bifpn_filters, max_relu_val=None, weighted=False)

        b1_out = AveragePooling2D()(b1_out)
        b1_out = _cbr(b1_out, bifpn_filters, '')

        b3_out = UpSampling2D()(b3_out)
        b3_out = ZeroPadding2D(((0,0), (0,1)))(b3_out)
        b3_out = _cbr(b3_out, bifpn_filters, '')

        x = Add()([b1_out, b2_out, b3_out])

    # Head
    with tf.name_scope('Head'):
      b1 = _cbr(x, bifpn_filters, "B1", max_relu_val=None, batch_norm=True)
      b1 = _cbr(b1, bifpn_filters, "B1", max_relu_val=None, batch_norm=False)
      b1 = _cbr(b1, bifpn_filters, "B1", max_relu_val=None, batch_norm=False)

      b2 = _cbr(x, bifpn_filters, "B2", max_relu_val=None, batch_norm=True)
      b2 = _cbr(b2, bifpn_filters, "B2", max_relu_val=None, batch_norm=False)
      b2 = _cbr(b2, bifpn_filters, "B2", max_relu_val=None, batch_norm=False)

      obj_map = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='obj_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(b1)
      geo_map = Conv2D(11, kernel_size=3, padding='same', activation=None, name='geo_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(b2)

    return Model(inputs=input_tensor, outputs=[obj_map, geo_map])


# model = create_small_det()
# model.summary()
# for layer in model.layers:
#     print(layer.name)