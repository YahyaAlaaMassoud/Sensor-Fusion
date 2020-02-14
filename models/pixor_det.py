
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add, Average, Lambda, UpSampling2D, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .blocks import create_conv_block, create_sep_conv_block, create_res_conv_block, create_res_sep_conv_block
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
  
def build_BiFPN_v2(fmap1, fmap2, fmap3, fmap4, filters=192):
  with tf.name_scope("BiFPN_Layer"):
    
    fmap3_inter = BiFPN()([fmap4, fmap3])
    fmap3_inter = create_sep_conv_block(fmap3_inter, filters=filters, kernel_size=3, num_layers=1)
    
    fmap2_inter = BiFPN()([fmap3_inter, fmap2])
    fmap2_inter = create_sep_conv_block(fmap2_inter, filters=filters, kernel_size=3, num_layers=1)
    
    fmap1_out   = BiFPN()([fmap2_inter, fmap1])
    fmap1_out   = create_sep_conv_block(fmap1_out, filters=filters, kernel_size=3, num_layers=1)
    
    fmap2_out   = BiFPN()([fmap2, fmap2_inter, fmap1_out])
    fmap2_out   = create_sep_conv_block(fmap2_out, filters=filters, kernel_size=3, num_layers=1)
    
    fmap3_out   = BiFPN()([fmap3, fmap3_inter, fmap2_out])
    fmap3_out   = create_sep_conv_block(fmap3_out, filters=filters, kernel_size=3, num_layers=1)
    
    fmap4_out   = BiFPN()([fmap4, fmap3_out])
    fmap4_out   = create_sep_conv_block(fmap4_out, filters=filters, kernel_size=3, num_layers=1)
    
    return fmap1_out, fmap2_out, fmap3_out, fmap4_out

def create_pixor_det(input_shape=(800, 700, 35), 
                     kernel_regularizer=None,
                     downsample_factor=4,
                     reg_channels=8):
    '''
        This architecture key differences:
            
        # Params: 3,969,031
    '''
    K.clear_session()
    
    KERNEL_REG       = kernel_regularizer
    KERNEL_SIZE      = {
                         "Layer1": 7,
                         "Block1": 3,
                         "Block2": 3,
                         "Block3": 3,
                         "Block4": 3,
                         "Header": 3,
                         "Out": 1
                       }
    PADDING          = 'same'
    FILTERS          = 256
    CLASS_CHANNELS   = 1
    OUT_KERNEL_SIZE  = 1
    REGRESS_CHANNELS = reg_channels
    BiFPN_filters    = 128
    
    inp = Input(shape=input_shape)
    x   = inp
    
    with tf.name_scope("Backbone"):
        with tf.name_scope("Block1"):
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Layer1'], padding=PADDING, kernel_regularizer=KERNEL_REG, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Block1'], num_layers=1)
            block1_out = x
            # Downsize twice
            block1_out = AveragePooling2D()(block1_out)
            block1_out = create_sep_conv_block(block1_out, FILTERS // 8, KERNEL_SIZE['Block1'], num_layers=1)
            block1_out = AveragePooling2D()(block1_out)
            block1_out = create_sep_conv_block(block1_out, FILTERS // 8, KERNEL_SIZE['Block1'], num_layers=1)
            # change number of channels from X -> BiFPN channels            
            block1_out = Conv2D(filters=BiFPN_filters, kernel_size=1, padding=PADDING, use_bias=False)(block1_out)
            block1_out = BatchNormalization()(block1_out)
            block1_out = ReLU()(block1_out)
            
        with tf.name_scope("Block2"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], padding=PADDING, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], num_layers=1)
            block2_out = x
            # Downsize once
            block2_out = AveragePooling2D()(block2_out)
            block2_out = create_sep_conv_block(block2_out, FILTERS // 4, KERNEL_SIZE['Block1'], num_layers=1)
            # change number of channels from X -> BiFPN channels
            block2_out = Conv2D(filters=BiFPN_filters, kernel_size=1, padding=PADDING, use_bias=False)(block2_out)
            block2_out = BatchNormalization()(block2_out)
            block2_out = ReLU()(block2_out)
            
        with tf.name_scope("Block3"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], padding=PADDING, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], num_layers=2)
            block3_out = x
            # change number of channels from X -> BiFPN channels
            block3_out = Conv2D(filters=BiFPN_filters, kernel_size=1, padding=PADDING, use_bias=False)(block3_out)
            block3_out = BatchNormalization()(block3_out)
            block3_out = ReLU()(block3_out)
            
        with tf.name_scope("Block4"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], num_layers=5)
            block4_out = x
            # upsmaple once
            block4_out = tf.image.resize(block4_out, size=get_new_shape(block4_out, 2), method=RESIZE_METHOD)
            block4_out = SeparableConv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING)(block4_out)
            # change number of channels from X -> BiFPN channels
            block4_out = Conv2D(filters=BiFPN_filters, kernel_size=1, padding=PADDING, use_bias=False)(block4_out)
            block4_out = BatchNormalization()(block4_out)
            block4_out = ReLU()(block4_out)
            
    out1, out2, out3, out4 = block1_out, block2_out, block3_out, block4_out
    out1, out2, out3, out4 = build_BiFPN_v2(out1, out2, out3, out4, filters=BiFPN_filters)
    out1, out2, out3, out4 = build_BiFPN_v2(out1, out2, out3, out4, filters=BiFPN_filters)
    # out1, out2, out3, out4 = build_BiFPN_v2(out1, out2, out3, out4, filters=BiFPN_filters)
    # out1, out2, out3, out4 = build_BiFPN_v2(out1, out2, out3, out4, filters=BiFPN_filters)

    with tf.name_scope("FinalOutput"):
      
      concat = Add()([out1, out2, out3, out4])
      x = create_conv_block(concat, filters=BiFPN_filters, kernel_size=3, num_layers=4)
      x = Add()([x, concat])
      
      obj        = Conv2D(CLASS_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, activation='sigmoid', name='objectness_map')(x)
      geo        = Conv2D(REGRESS_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, name='geometric_map')(x)
      out        = Concatenate(axis=-1, name='output_map')([obj, geo])

    model = Model(inp, out)
    return model

# model = create_pixor_det()
# model.summary()
