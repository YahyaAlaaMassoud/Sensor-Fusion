
# import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add, Average, Lambda, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

def add_skip_conv(last_input, filters, kernel_size, padding):
  x = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(last_input)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return Add()([x, last_input])

def add_skip_sep_conv(last_input, filters, kernel_size, padding):
  x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding=padding)(last_input)
  x = BatchNormalization()(x)
  x = ReLU()(x)
  return Add()([x, last_input])

class TwoResConv(Layer):

  def __init__(self, filters=128, kernel_size=3, padding='same', **kwargs):
    self.filters = filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.sep_conv = SeparableConv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
    self.change_depth_conv = Conv2D(filters=self.filters, kernel_size=1, padding=self.padding)
    self.bn = BatchNormalization()
    self.relu = ReLU()
    super(TwoResConv, self).__init__(**kwargs)

  def build(self, input_shape):
    self.a = self.add_weight(name='a',
                             shape=(1,),
                             initializer='uniform',
                             trainable=True)
    self.b = self.add_weight(name='b',
                             shape=(1,),
                             initializer='uniform',
                             trainable=True)
    self.eps = 0.0001
    super(TwoResConv, self).build(input_shape)

  def call(self, x):
    assert isinstance(x, list)
    assert len(x) == 2
    for i in range(len(x)):
    #   if x[i].shape[-1] > self.filters:
        # print('abl', fmaps[i].shape[-1])
        x[i] = self.change_depth_conv(x[i])
        x[i] = self.bn(x[i])
        x[i] = self.relu(x[i])
        # print('b3d', fmaps[i].shape[-1])
    fmap1, fmap2 = x
    a, b = self.relu(self.a), self.relu(self.b)
    fmap1 = a * fmap1
    fmap2 = b * fmap2
    # fmap_out = Concatenate(axis=-1)([fmap1, fmap2])
    # print(fmap_out.shape)
    fmap_out = (fmap1 + fmap2) / (a + b + self.eps)
    fmap_out = self.sep_conv(fmap_out)
    fmap_out = self.bn(fmap_out)
    fmap_out = self.relu(fmap_out)
    return fmap_out

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super(TwoResConv, self).get_config()
    config['filters'] = self.filters
    config['kernel_size'] = self.kernel_size
    config['padding'] = self.padding
    return config

class ThreeResConv(Layer):

  def __init__(self, filters=128, kernel_size=3, padding='same', **kwargs):
    self.filters = filters
    self.kernel_size = kernel_size
    self.padding = padding
    self.sep_conv = SeparableConv2D(filters=self.filters, kernel_size=self.kernel_size, padding=self.padding)
    self.change_depth_conv = Conv2D(filters=self.filters, kernel_size=1, padding=self.padding)
    self.bn = BatchNormalization()
    self.relu = ReLU()
    super(ThreeResConv, self).__init__(**kwargs)

  def build(self, input_shape):
    self.a = self.add_weight(name='a',
                             shape=(1,),
                             initializer='uniform',
                             trainable=True)
    self.b = self.add_weight(name='b',
                             shape=(1,),
                             initializer='uniform',
                             trainable=True)
    self.c = self.add_weight(name='c',
                             shape=(1,),
                             initializer='uniform',
                             trainable=True)
    self.eps = 0.0001
    super(ThreeResConv, self).build(input_shape)

  def call(self, x):
    assert isinstance(x, list)
    assert len(x) == 3
    for i in range(len(x)):
    #   if x[i].shape[-1] > self.filters:
        # print('abl', fmaps[i].shape[-1])
        x[i] = self.change_depth_conv(x[i])
        x[i] = self.bn(x[i])
        x[i] = self.relu(x[i])
        # print('b3d', fmaps[i].shape[-1])

    fmap1, fmap2, fmap3 = x
    a, b, c = self.relu(self.a), self.relu(self.b), self.relu(self.c)
    fmap1 = a * fmap1
    fmap2 = b * fmap2
    fmap3 = c * fmap3
    # fmap_out = Concatenate(axis=-1)([fmap1, fmap2, fmap3])
    # print(fmap_out.shape)
    fmap_out = (fmap1 + fmap2 + fmap3) / (self.a + self.b + self.c + self.eps)
    fmap_out = self.sep_conv(fmap_out)
    fmap_out = self.bn(fmap_out)
    fmap_out = self.relu(fmap_out)
    return fmap_out

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super(ThreeResConv, self).get_config()
    config['filters'] = self.filters
    config['kernel_size'] = self.kernel_size
    config['padding'] = self.padding
    return config

def get_new_shape(fmap, resize_factor):
  # print('old shape', (fmap.shape[1], fmap.shape[2]))
  new_w = fmap.shape[1] * resize_factor + 1 if fmap.shape[1] % 5 else fmap.shape[1] * resize_factor
  new_h = fmap.shape[2] * resize_factor + 1 if fmap.shape[2] % 5 else fmap.shape[2] * resize_factor
  # print('new shape', new_w, new_h)
  return (new_w, new_h)

def BiFPN(fmap1, fmap2, fmap3, filters=192):#fmap4, filters=192):
  with tf.name_scope("BiFPN"):

    fmap3_resized = tf.image.resize(fmap3, size=get_new_shape(fmap3, 2))#UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(fmap3)

    fmap2_inter_out = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap3_resized, fmap2])
    fmap2_inter_out_resized = tf.image.resize(fmap2_inter_out, size=get_new_shape(fmap2_inter_out, 2))#UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(fmap2_inter_out)
    
    fmap1_out = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap2_inter_out_resized, fmap1])
    fmap1_out_resized = AveragePooling2D()(fmap1_out)
    
    fmap2_out = ThreeResConv(filters=filters, kernel_size=3, padding='same')([fmap1_out_resized, fmap2_inter_out, fmap2])
    fmap2_out_resized = AveragePooling2D()(fmap2_out)

    fmap3_out = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap2_out_resized, fmap3])

    # fmap4_resized   = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(fmap4) 

    # fmap3_inter_out = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap4_resized, fmap3])
    # fmap3_inter_out_resized = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(fmap3_inter_out) 

    # fmap2_inter_out = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap3_inter_out_resized, fmap2])
    # fmap2_inter_out_resized = UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(fmap2_inter_out) 

    # fmap1_out    = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap2_inter_out_resized, fmap1])
    # fmap1_out_resized = AveragePooling2D()(fmap1_out)#K.resize_images(fmap1_out, .5, .5, 'channels_last')

    # fmap2_out    = ThreeResConv(filters=filters, kernel_size=3, padding='same')([fmap1_out_resized, fmap2_inter_out, fmap2])
    # fmap2_out_resized = AveragePooling2D()(fmap2_out)#K.resize_images(fmap2_out, .5, .5, 'channels_last')

    # fmap3_out    = ThreeResConv(filters=filters, kernel_size=3, padding='same')([fmap2_out_resized, fmap3_inter_out, fmap3])
    # fmap3_out_resized = AveragePooling2D()(fmap3_out)#K.resize_images(fmap3_out, .5, .5, 'channels_last')

    # fmap4_out    = TwoResConv(filters=filters, kernel_size=3, padding='same')([fmap3_out_resized, fmap4])

    # print(fmap1_out.shape, fmap2_out.shape, fmap3_out.shape)
    return fmap1_out, fmap2_out, fmap3_out


def create_conv_block(input_tensor, filters, kernel_size, num_layers):
  last_input = input_tensor
  x = input_tensor
  for _ in range(num_layers):
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  return Add()([x, last_input])

def create_sep_conv_block(input_tensor, filters, kernel_size, num_layers):
  last_input = input_tensor
  x = input_tensor
  for _ in range(num_layers):
    x = SeparableConv2D(filters=filters, kernel_size=kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
  return Add()([x, last_input])

def create_model(input_shape=(1000, 1000, 35), 
                 kernel_regularizer=None,
                 downsample_factor=4):
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
    FILTERS          = 384
    MAXPOOL_SIZE     = 2
    MAXPOOL_STRIDES  = None
    DECONV_STRIDES   = 2
    CLASS_CHANNELS   = 1
    ANGLE_CHANNELS   = 2
    OFFSET_CHANNELS  = 2
    SIZE_CHANNELS    = 2
    OUT_KERNEL_SIZE  = 1
    REGRESS_CHANNELS = 6
    
    inp = Input(shape=input_shape)
    x   = inp
    
    with tf.name_scope("Backbone"):
        with tf.name_scope("Block1"):
            x = Conv2D(filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Layer1'], padding=PADDING, kernel_regularizer=KERNEL_REG)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 8, kernel_size=KERNEL_SIZE['Block1'], num_layers=2)
            block1_out = x
            
        with tf.name_scope("Block2"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], padding=PADDING)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 4, kernel_size=KERNEL_SIZE['Block2'], num_layers=2)
            block2_out = x
            
        with tf.name_scope("Block3"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], padding=PADDING)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 2, kernel_size=KERNEL_SIZE['Block3'], num_layers=2)
            block3_out = x
            
        with tf.name_scope("Block4"):
            #--------Downsampling--------#
            x = MaxPooling2D()(x)
            #--------Downsampling--------#
            x = Conv2D(filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], padding=PADDING)(x)
            x = BatchNormalization()(x)
            x = ReLU()(x)
            x = create_conv_block(x, filters=FILTERS // 1, kernel_size=KERNEL_SIZE['Block4'], num_layers=5)
            block4_out = x

    # print(block1_out.shape, block2_out.shape, block3_out.shape, block4_out.shape)
    BiFPN_filters = 128
    out2, out3, out4 = BiFPN(block2_out, block3_out, block4_out, filters=BiFPN_filters)
    # print(out2.shape, out3.shape, out4.shape)
    out2, out3, out4 = BiFPN(out2, out3, out4, filters=BiFPN_filters)
    out2, out3, out4 = BiFPN(out2, out3, out4, filters=BiFPN_filters)
    
    with tf.name_scope("FinalOutput"):
      out2 = AveragePooling2D()(out2)

      out4 = tf.image.resize(out4, size=get_new_shape(out4, 2))#UpSampling2D(size=(2, 2), data_format='channels_last', interpolation='bilinear')(out4)

      # print(out2.shape, out3.shape, out4.shape)
      concat = Concatenate(axis=-1)([out2, out3, out4])
      # print(concat.shape)
      obj        = Conv2D(CLASS_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, activation='sigmoid', name='objectness_map')(concat)
      head_angle = Conv2D(ANGLE_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, activation='tanh', name='heading_angle_map')(concat)
      offset     = Conv2D(OFFSET_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, name='offset_map')(concat)
      size       = Conv2D(SIZE_CHANNELS, kernel_size=OUT_KERNEL_SIZE, padding=PADDING, activation='relu', name='size_map')(concat)
      out    = Concatenate(axis=-1, name='output_map')([obj, head_angle, offset, size])
      # print(obj.shape, geo.shape, out.shape)
            
    model = Model(inp, out)
    return model
        
# model = create_model()
# model.summary()
