import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import Sequence
import tensorflow.keras.backend as K

def create_pixor_pp(input_shape=(800, 700, 35), downsample_factor=4, kernel_regularizer=None):
    K.clear_session()
    
    KERNEL_REG       = kernel_regularizer
    KERNEL_SIZE      = 3
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
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block1_out = x
            block1_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block1_out)
            block1_out = AveragePooling2D(pool_size=MAXPOOL_SIZE)(block1_out)
#             print("Block 1 output shape: " + str(block1_out.shape))
        
        with tf.name_scope("Block2"):
            x = MaxPooling2D(pool_size=MAXPOOL_SIZE)(x)
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
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
#             print("Block 2 output shape: " + str(block2_out.shape))
            
        with tf.name_scope("Block3"):
            x = MaxPooling2D(pool_size=MAXPOOL_SIZE)(x)
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
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
#             print("Block 3 output shape: " + str(block3_out.shape))
            
        with tf.name_scope("Block4"):
            x = MaxPooling2D(pool_size=MAXPOOL_SIZE)(x)
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            block4_out = x
            block4_out = tf.image.resize(block4_out, size=(input_shape[0] // downsample_factor, 
                                                           input_shape[1] // downsample_factor))
#             print("Block 4 output shape: " + str(block4_out.shape))
            
    concat = Concatenate(axis=-1)([block1_out, block2_out, block3_out, block4_out])
    
    with tf.name_scope("Header"):
        x = concat
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = ReLU()(x)
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = ReLU()(x)
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = ReLU()(x)
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = ReLU()(x)
        x = Conv2D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
        x = ReLU()(x)
        
        objectness_map = Conv2D(CLASS_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG, activation='sigmoid', name='objectness_map')(x)
        geometric_map  = Conv2D(REGRESS_CHANNELS, kernel_size=KERNEL_SIZE, padding=PADDING, kernel_regularizer=KERNEL_REG, name='geometric_map')(x)
        
    output_map = Concatenate(axis=-1, name='output_map')([objectness_map, geometric_map])
    model = Model(inp, output_map)
    return model

# model = create_pixor_pp((1000, 1000, 35))
# model.summary()