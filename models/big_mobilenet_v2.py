
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add, Average, Lambda, UpSampling2D, DepthwiseConv2D, ZeroPadding2D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from .blocks import conv_block, inverted_res_block, squeeze_excite_block
from .layers import BiFPN

RESIZE_METHOD = 'nearest'


def _build_BiFPN_full_res(f1, f2, f3, filters=192, kernel_size=3, BN=False, max_relu_val=None, weighted=False, excite=False):

    def _sep_conv(x):
        x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=not BN, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-4, l2=5e-4))(x)
        if BN is True:
            x = BatchNormalization()(x)
        x = ReLU(max_value=max_relu_val)(x)
        if excite:
            x = squeeze_excite_block(x)
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


def create_big_mobilenet_v2(input_shape=(800, 700, 20), kr=tf.keras.regularizers.l2(1e-4), ki='he_normal', data_format=None):

    input_tensor = Input(shape=input_shape)
    x = input_tensor

    ef = 3

    # Block 1
    with tf.name_scope('Block_1'):
        x = conv_block(input_tensor=x, filters=20, kernel_size=3, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')
        x = inverted_res_block(input_tensor=x, filters=20, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=False, act='relu', excite=True)
        for _ in range(1):
            x = inverted_res_block(input_tensor=x, filters=20, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=True, act='relu', excite=True)
        b1_out = x

    # Block 2
    with tf.name_scope('Block_2'):
        x = MaxPooling2D()(x)
        x = inverted_res_block(input_tensor=x, filters=40, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=False, act='relu', excite=True)
        for _ in range(2):
            x = inverted_res_block(input_tensor=x, filters=40, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=True, act='relu', excite=True)
        b2_out = x

    # Block 2
    with tf.name_scope('Block_2'):
        x = MaxPooling2D()(x)
        x = inverted_res_block(input_tensor=x, filters=60, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=False, act='relu', excite=True)
        for _ in range(3):
            x = inverted_res_block(input_tensor=x, filters=60, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=True, act='relu', excite=True)
        b3_out = x

    with tf.name_scope('Block_2'):
        x = MaxPooling2D()(x)
        x = inverted_res_block(input_tensor=x, filters=80, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=False, act='relu', excite=True)
        for _ in range(3):
            x = inverted_res_block(input_tensor=x, filters=80, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=True, act='relu', excite=True)
        b4_out = x

    # Aggregate feature maps
    with tf.name_scope('Concatenation'):
        bifpn_filters = 40

        b2_out = conv_block(input_tensor=b2_out, filters=bifpn_filters, kernel_size=1, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')

        b3_out = conv_block(input_tensor=b3_out, filters=bifpn_filters, kernel_size=1, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')

        b4_out = conv_block(input_tensor=b4_out, filters=bifpn_filters, kernel_size=1, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')

        # print('b1_out.shape', b1_out.shape)
        # print('b2_out.shape', b2_out.shape)
        # print('b3_out.shape', b3_out.shape)

        b2_out, b3_out, b4_out = _build_BiFPN_full_res(b2_out, b3_out, b4_out, filters=bifpn_filters, max_relu_val=None, weighted=False, excite=True)
        
        b1_out = AveragePooling2D()(b1_out)
        b1_out = AveragePooling2D()(b1_out)

        b2_out = AveragePooling2D()(b2_out)
        # b2_out = conv_block(input_tensor=b2_out, filters=bifpn_filters, kernel_size=3, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')
        # b2_out = squeeze_excite_block(b2_out)

        # b3_out = conv_block(input_tensor=b3_out, filters=bifpn_filters, kernel_size=3, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')
        # b3_out = squeeze_excite_block(b3_out)

        b4_out = tf.image.resize(b4_out, size=b3_out.shape[1:3])
        # b4_out = conv_block(input_tensor=b4_out, filters=bifpn_filters, kernel_size=3, strides=1, BN=True, data_format='channels_last', max_relu_val=None, act='relu')
        # b4_out = squeeze_excite_block(b4_out)

        # x = Add()([b2_out, b3_out, b4_out])
        x = Concatenate(axis=-1)([b1_out, b2_out, b3_out, b4_out])

    # Head
    with tf.name_scope('Head'):
        ef = 2
        x = inverted_res_block(input_tensor=x, filters=bifpn_filters, kernel_size=3, expansion_factor=ef, width_mul=1, strides=1, res=False, act='relu', excite=True)
        for _ in range(4):
            x = conv_block(input_tensor=x, filters=bifpn_filters, kernel_size=3, strides=1, BN=False, data_format='channels_last', max_relu_val=None, act='relu')
            x = squeeze_excite_block(x)

        obj_map = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='obj_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(x)
        geo_map = Conv2D(11, kernel_size=3, padding='same', activation=None, name='geo_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(x)

    return Model(inputs=input_tensor, outputs=[obj_map, geo_map])


# model = create_big_mobilenet_v2()
# model.summary()
