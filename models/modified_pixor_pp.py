
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import Sequence
import tensorflow.keras.backend as K

def create_modified_pixor_pp_v1(input_shape=(800, 700, 35), 
                                kernel_regularizer=None,
                                downsample_factor=4):
    '''
        This architecture key differences:
            1- Using separable convolutions in some deep layers
            2- Uses large kernel sizes on the first blocks, and larger on deeper blocks
            3- Uses 1x1 conolution in the output layer
            4- Using strided convolution instead of maxpooling to downsize the spatial dimensions
            
        # Params: 3,969,031
    '''
    K.clear_session()
    
    KERNEL_REG       = kernel_regularizer
    KERNEL_SIZE      = {
                         "Block1": 7,
                         "Block2": 5,
                         "Block3": 3,
                         "Block4": 3,
                         "Header": 3,
                         "Out": 1
                       }
    PADDING          = 'same'
    FILTERS          = 256
    MAXPOOL_SIZE     = 2
    MAXPOOL_STRIDES  = None
    DECONV_STRIDES   = 2
    CLASS_CHANNELS   = 1
    REGRESS_CHANNELS = 6
    
    inp = Input(shape=input_shape)
    x   = inp
    
    with tf.name_scope("Backbone"):
        with tf.name_scope("Block1"):
            x = Conv2D(filters=FILTERS // 16, kernel_size=KERNEL_SIZE['Block1'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Block1'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Block1'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block1_out = x
            block1_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block1_out)
            block1_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block1_out)
#             print("Block 1 output shape: " + str(block1_out.shape))
        
        with tf.name_scope("Block2"):
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Block2'], padding=PADDING, strides=2, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block2_out = x
            block2_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block2_out)
#             print("Block 2 output shape: " + str(block2_out.shape))
            
        with tf.name_scope("Block3"):
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block3'], padding=PADDING, strides=2, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block3_out = x
#             print("Block 3 output shape: " + str(block3_out.shape))
            
        with tf.name_scope("Block4"):
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, strides=2, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = SeparableConv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block4_out = x
            block4_out = tf.image.resize(block4_out, size=(input_shape[0] // downsample_factor, 
                                                           input_shape[1] // downsample_factor))
#             print("Block 4 output shape: " + str(block4_out.shape))
            
    concat = Concatenate(axis=-1)([block1_out, block2_out, block3_out, block4_out])
    
    with tf.name_scope("Header"):
        x = concat
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE['Header'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE['Header'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(filters=FILTERS, kernel_size=KERNEL_SIZE['Header'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SeparableConv2D(filters=FILTERS, kernel_size=KERNEL_SIZE['Header'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE['Header'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        
    with tf.name_scope("Out"):
        objectness_map = Conv2D(CLASS_CHANNELS, kernel_size=KERNEL_SIZE['Out'], padding=PADDING, kernel_regularizer=KERNEL_REG, activation='sigmoid', name='objectness_map')(x)
        geometric_map  = Conv2D(REGRESS_CHANNELS, kernel_size=KERNEL_SIZE['Out'], padding=PADDING, kernel_regularizer=KERNEL_REG, name='geometric_map')(x)
        
    output_map = Concatenate(axis=-1, name='output_map')([objectness_map, geometric_map])
    model = Model(inp, output_map)
    return model

def create_bottleneck_layer(prev_layer,
                            out_channels,
                            bottleneck_channels,
                            kernel_size):
    bottleneck = Conv2D(filters=bottleneck_channels,
                        kernel_size=1,
                        padding='same')(prev_layer)
    bottleneck = BatchNormalization()(bottleneck)
    bottleneck = ReLU()(bottleneck)
    
    conv       = Conv2D(filters=bottleneck_channels,
                        kernel_size=kernel_size,
                        padding='same')(bottleneck)
    conv       = BatchNormalization()(conv)
    conv       = ReLU()(conv)
    
    out        = Conv2D(filters=out_channels,
                        kernel_size=1,
                        padding='same')(conv)
    out        = BatchNormalization()(out)
    out        = ReLU()(out)
    add        = Add()([out, prev_layer])
    return add

def create_modified_pixor_pp_v2(input_shape=(800, 700, 35), 
                                kernel_regularizer=None,
                                downsample_factor=4):
    '''
        This architecture key differences:
            1- Using Bottle neck layers to reduce the number of parameters (Block 4 / Header)
            2- Using strided convolution instead of maxpooling to downsize the spatial dimensions
            
        # Params: 4,202,567
    '''
    K.clear_session()
    
    KERNEL_REG         = kernel_regularizer
    KERNEL_SIZE        = 3
    PADDING            = 'same'
    FILTERS            = 256
    BOTTLENECK_FILTERS = 92
    MAXPOOL_SIZE       = 2
    MAXPOOL_STRIDES    = None
    DECONV_STRIDES     = 2
    CLASS_CHANNELS     = 1
    REGRESS_CHANNELS   = 6
    
    inp = Input(shape=input_shape)
    x   = inp
    
    with tf.name_scope("Backbone"):
        with tf.name_scope("Block1"):
            x = Conv2D(filters=FILTERS // 16, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block1_out = x
            block1_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block1_out)
            block1_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block1_out)
        
        with tf.name_scope("Block2"):
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE, padding=PADDING, strides=2, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block2_out = x
            block2_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block2_out)
            
        with tf.name_scope("Block3"):
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE, padding=PADDING, strides=2, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block3_out = x
            
        with tf.name_scope("Block4"):
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE, padding=PADDING, strides=2, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            
            x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
            x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
            x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
            x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
            x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
            block4_out = x
            block4_out = tf.image.resize(block4_out, size=(input_shape[0] // downsample_factor, 
                                                           input_shape[1] // downsample_factor))
            
    concat = Concatenate(axis=-1)([block1_out, block2_out, block3_out, block4_out])
    
    with tf.name_scope("Header"):
        x = concat
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
        x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
        x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
        x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
        x = create_bottleneck_layer(x, FILTERS, BOTTLENECK_FILTERS, KERNEL_SIZE)
        
    with tf.name_scope("Out"):
        objectness_map = Conv2D(CLASS_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG, activation='sigmoid', name='objectness_map')(x)
        geometric_map  = Conv2D(REGRESS_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG, name='geometric_map')(x)
        
    output_map = Concatenate(axis=-1, name='output_map')([objectness_map, geometric_map])
    model = Model(inp, output_map)
    return model

# model = create_modified_pixor_pp_v1(input_shape=(1000, 1000, 35))
# model = create_modified_pixor_pp_v2(input_shape=(1000, 1000, 35))
# model.summary()
