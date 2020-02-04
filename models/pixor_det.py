
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add, Average, Lambda, UpSampling2D, DepthwiseConv2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def add_skip_conv(last_input, filters, kernel_size, padding):
  x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False)(last_input)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return Add()([x, last_input])

def add_skip_sep_conv(last_input, filters, kernel_size, padding):
  x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False)(last_input)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return Add()([x, last_input])

class BiFPN(Layer):

  def __init__(self, **kwargs):
    super(BiFPN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.w = self.add_weight(name='w',
                             shape=(len(input_shape),),
                             initializer='uniform',
                             trainable=True)
    self.eps = 0.0001
    super(BiFPN, self).build(input_shape)

  def call(self, x):
    assert isinstance(x, list)

    # w = ReLU()(self.w)
    x = tf.reduce_sum([self.w[i] * x[i] for i in range(len(x))], axis=0)
    x = x / (tf.reduce_sum(self.w) + self.eps)
    
    return x

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super(BiFPN, self).get_config()
    return config

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

def sep_conv_block(x, filters, kernel_size, padding):
  x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding=padding, use_bias=False)(x)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return x

def build_BiFPN(fmap1, fmap2, fmap3, filters=192):#fmap4, filters=192):
  with tf.name_scope("BiFPN"):

    fmap3_resized = tf.image.resize(fmap3, size=get_new_shape(fmap3, 2), method='nearest')#UpSampling2D(data_format='channels_last', interpolation='bilinear')(fmap3)
    fmap3_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3)(fmap3_resized)

    fmap2_inter_out = BiFPN()([fmap3_resized, fmap2])
    fmap2_inter_out = sep_conv_block(fmap2_inter_out, filters=filters, kernel_size=3, padding='same')
    fmap2_inter_out_resized = tf.image.resize(fmap2_inter_out, size=get_new_shape(fmap2_inter_out, 2), method='nearest')#UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(fmap2_inter_out)
    fmap2_inter_out_resized = SeparableConv2D(filters=filters, padding='same', kernel_size=3)(fmap2_inter_out_resized)
    
    fmap1_out = BiFPN()([fmap2_inter_out_resized, fmap1])
    fmap1_out = sep_conv_block(fmap1_out, filters=filters, kernel_size=3, padding='same')
    fmap1_out_resized = AveragePooling2D()(fmap1_out)
    
    fmap2_out = BiFPN()([fmap1_out_resized, fmap2_inter_out, fmap2])
    fmap2_out = sep_conv_block(fmap2_out, filters=filters, kernel_size=3, padding='same')
    fmap2_out_resized = AveragePooling2D()(fmap2_out)

    fmap3_out = BiFPN()([fmap2_out_resized, fmap3])
    fmap3_out = sep_conv_block(fmap3_out, filters=filters, kernel_size=3, padding='same')
    
    return fmap1_out, fmap2_out, fmap3_out


def create_conv_block(input_tensor, filters, kernel_size, num_layers):
  last_input = input_tensor
  x = input_tensor
  for _ in range(num_layers):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  return Add()([x, last_input])

def create_sep_conv_block(input_tensor, filters, kernel_size, num_layers):
  last_input = input_tensor
  x = input_tensor
  for _ in range(num_layers):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  return Add()([x, last_input])

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
    MAXPOOL_SIZE     = 2
    MAXPOOL_STRIDES  = None
    DECONV_STRIDES   = 2
    CLASS_CHANNELS   = 1
    ANGLE_CHANNELS   = 2
    OFFSET_CHANNELS  = 2
    SIZE_CHANNELS    = 2
    OUT_KERNEL_SIZE  = 1
    REGRESS_CHANNELS = reg_channels
    BiFPN_filters = 128
    
    inp = Input(shape=input_shape)
    x   = inp
    
    with tf.name_scope("Backbone"):
        with tf.name_scope("Block1"):
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Layer1'], padding=PADDING, kernel_regularizer=KERNEL_REG, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Block1'], num_layers=2)
            block1_out = x
            
        with tf.name_scope("Block2"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], padding=PADDING, use_bias=False)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], num_layers=2)
            block2_out = x
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
            block4_out = Conv2D(filters=BiFPN_filters, kernel_size=1, padding=PADDING, use_bias=False)(block4_out)
            block4_out = BatchNormalization()(block4_out)
            block4_out = ReLU()(block4_out)

    # print(block1_out.shape, block2_out.shape, block3_out.shape, block4_out.shape)
    out2, out3, out4 = block2_out, block3_out, block4_out
    out2, out3, out4 = build_BiFPN(out2, out3, out4, filters=BiFPN_filters)
    out2, out3, out4 = build_BiFPN(out2, out3, out4, filters=BiFPN_filters)
    out2, out3, out4 = build_BiFPN(out2, out3, out4, filters=BiFPN_filters)
    out2, out3, out4 = build_BiFPN(out2, out3, out4, filters=BiFPN_filters)
    out2, out3, out4 = build_BiFPN(out2, out3, out4, filters=BiFPN_filters)
    out2, out3, out4 = build_BiFPN(out2, out3, out4, filters=BiFPN_filters)
    
    with tf.name_scope("FinalOutput"):
      out2 = AveragePooling2D()(out2)

      out4 = tf.image.resize(out4, size=get_new_shape(out4, 2), method='nearest')#UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(out4)
      out4 = SeparableConv2D(filters=BiFPN_filters, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING)(out4)
      
      # print(out2.shape, out3.shape, out4.shape)
      # concat = BiFPN()([out2, out3, out4])
      concat = Concatenate(axis=-1)([out2, out3, out4])
      concat = Conv2D(filters=128, kernel_size=1, padding=PADDING, use_bias=False)(concat)
      concat = BatchNormalization()(concat)
      concat = ReLU()(concat)
      # concat = SeparableConv2D(filters=128, kernel_size=3, padding=PADDING)(concat)
      # concat = BatchNormalization()(concat)
      # concat = ReLU()(concat)
      
      # print(concat.shape)
      obj        = Conv2D(CLASS_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, activation='sigmoid', name='objectness_map')(concat)
      # head_angle = Conv2D(ANGLE_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, name='heading_angle_map')(concat)
      # offset     = Conv2D(OFFSET_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, name='offset_map')(concat)
      # size       = Conv2D(SIZE_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, name='size_map')(concat)
      geo        = Conv2D(REGRESS_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, name='geometric_map')(concat)
      out    = Concatenate(axis=-1, name='output_map')([obj, geo])
      # print(obj.shape, geo.shape, out.shape)

    model = Model(inp, out)
    return model

# model = create_pixor_det()
# model.summary()
