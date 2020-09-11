

import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Conv2D, SeparableConv2D, ReLU, BatchNormalization, Add, DepthwiseConv2D, Activation, Dense, AveragePooling2D, UpSampling2D, \
                                       GlobalAveragePooling2D, Reshape, multiply, Layer, InputSpec, Average, Input, Dropout, Concatenate, Subtract, Multiply, Lambda
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.utils import get_custom_objects

from tensorflow.keras.initializers import RandomNormal

from .layers import BiFPN

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

def inverted_res_block(input_tensor, filters, kernel_size, expansion_factor, width_mul, strides, res=False, act='relu', max_relu_val=None, excite=False, kr=None, group_bn=False, bn_groups=10):
  channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
  # Depth
  tchannel = K.int_shape(input_tensor)[channel_axis] * expansion_factor
  # Width
  cchannel = int(filters * width_mul)

  x = conv_block(input_tensor=input_tensor, filters=tchannel, kernel_size=(1, 1), strides=(1, 1), max_relu_val=max_relu_val, act=act, BN=True)

  x = DepthwiseConv2D(kernel_size, strides=(strides, strides), depth_multiplier=1, padding='same', kernel_regularizer=kr)(x)
  if group_bn:
    x = GroupNormalization(groups=bn_groups, axis=channel_axis, epsilon=0.1)(x)
  else:
    x = BatchNormalization(axis=channel_axis)(x)
  if act == 'relu':
    x = ReLU(max_value=max_relu_val)(x)

  if excite:
    x = squeeze_excite_block(x)

  x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same', kernel_regularizer=kr)(x)
  if group_bn:
    x = GroupNormalization(groups=bn_groups, axis=channel_axis, epsilon=0.1)(x)
  else:
    x = BatchNormalization(axis=channel_axis)(x)

  if res:
    x = Add()([x, input_tensor])

  return x

def conv_block(input_tensor, filters, kernel_size, strides, BN=True, data_format='channels_last', max_relu_val=None, act='relu', kr=None, group_bn=False, bn_groups=10, name=''):
  options = {
    'kernel_initializer': 'glorot_uniform',
    'bias_initializer': 'zeros',
    'kernel_regularizer': kr,
    'name': name,
  }
  x = input_tensor
  with tf.name_scope("ConvBlock"):
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=not BN, data_format=data_format, **options)(x)
    if BN is True:
      if data_format == 'channels_first':
        if group_bn:
          x = GroupNormalization(groups=bn_groups, axis=1, epsilon=0.1)(x)
        else:
          x = BatchNormalization(axis=1, name=name+"_BN")(x)
      else:
        if group_bn:
          x = GroupNormalization(groups=bn_groups, axis=-1, epsilon=0.1)(x)
        else:
          x = BatchNormalization(axis=-1, name=name+"_BN")(x)
    if act == 'relu':
      x = ReLU(max_value=max_relu_val, name=name+"_ReLU")(x)
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

def create_res_conv_block(input_tensor, filters, kernel_size, first_of_block=False, BN=True, data_format='channels_last', max_relu_val=None, name=''):
  # https://github.com/raghakot/keras-resnet

  last_input = input_tensor
  if input_tensor.get_shape()[-1] != filters:
    strides = 1
    if first_of_block:
      strides = 2
    last_input = Conv2D(filters=filters,
                        kernel_size=1,
                        strides=strides,
                        name=name + "_DimChange",
                        kernel_initializer='glorot_uniform')(last_input)

  x = input_tensor

  for j in range(2):
    if BN is True:
        if data_format == 'channels_first':
          x = BatchNormalization(axis=1, name=name+"_BN_{}".format(j))(x)
        else:
          x = BatchNormalization(axis=-1, name=name+"_BN_{}".format(j))(x)
    x = ReLU(max_value=max_relu_val, name=name+"_ReLU_{}".format(j))(x)
    strides = 1
    if first_of_block:
      strides = 2
      first_of_block = False
    x = Conv2D(filters=filters, 
               kernel_size=kernel_size, 
               padding='same',
               use_bias=not BN,
               kernel_initializer='glorot_uniform',
               strides=strides,
               name=name + "_LastConv_{}".format(j),
               data_format=data_format)(x)
  
  return Add(name=name+"_Add")([x, last_input])

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


def create_input_layer(input_shape, name, dtype):
  return tf.keras.layers.Input(
      shape=input_shape,
      name=name,
      dtype=dtype,
  )


def deep_fuse_layer(inputs, filters, kernel_size, strides):
  if type(inputs) is not list:
    raise Exception()
  
  avg = Average()(inputs)

  conv_list = []
  for _ in range(len(inputs)):
    conv_list.append(create_res_conv_block(avg, filters, kernel_size, strides))

  return conv_list



def factorized_bilinear_pooling(F1, F2, init_filters, new_filters):

  F1_expand = Conv2D(filters=new_filters, kernel_size=1, padding='same', strides=1)(F1)
  F1_expand = ReLU()(F1_expand)

  F2_expand = Conv2D(filters=new_filters, kernel_size=1, padding='same', strides=1)(F2)
  F2_expand = ReLU()(F2_expand)

  inter = tf.keras.layers.multiply([F1_expand, F2_expand])
  inter = Dropout(rate=0.1)(inter)

  F = Conv2D(filters=init_filters, kernel_size=1, padding='same', strides=1)(inter)
  F = ReLU()(F)

  power_normalize = tf.sqrt(tf.nn.relu(F)) - tf.sqrt(tf.nn.relu(-F))
  l2_normalize = tf.nn.l2_normalize(power_normalize, axis=-1)

  return l2_normalize


def factorized_bilinear_pooling_new(F1, F2, init_filters, new_filters, name=""):

  F1_expand = Conv2D(filters=new_filters, kernel_size=1, padding='same', strides=1, name=name + "Conv1")(F1)
  F1_expand = ReLU(name=name + "Relu1")(F1_expand)

  F2_expand = Conv2D(filters=new_filters, kernel_size=1, padding='same', strides=1, name=name + "Conv2")(F2)
  F2_expand = ReLU(name=name + "Relu2")(F2_expand)

  F_aux = Add(name=name + "Add1")([F1_expand, F2_expand])

  inter = Multiply(name=name + "Mul1")([F1_expand, F2_expand])
  inter = Dropout(rate=0.1, name=name + "Dropout1")(inter)

  F = Conv2D(filters=init_filters, kernel_size=1, padding='same', strides=1, name=name + "Conv3")(inter)
  F = ReLU(name=name + "Relu3")(F)

  out = Concatenate(name=name + "Concat")([F_aux, F])
  out = Conv2D(filters=init_filters, kernel_size=1, padding='same', strides=1, name=name + "Conv4")(out)
  out = ReLU(name=name + "Relu4")(out)

  power_normalize = Subtract()([Lambda(tf.keras.backend.sqrt)(ReLU(name=name+"Relu5")(out)), Lambda(tf.keras.backend.sqrt)(ReLU(name=name+"Relu6")(-out))])
  # power_normalize = tf.sqrt(tf.nn.relu(out)) - tf.sqrt(tf.nn.relu(-out))
  l2_normalize = Lambda(tf.keras.backend.l2_normalize, arguments={'axis':-1})(power_normalize)

  return l2_normalize


def transform_rangeview_to_bev(bev_map, out_filters, name=''):
  filters = bev_map.shape[-1]

  # x = create_res_conv_block(bev_map, out_filters, 1, False, name=name + "_TransformRV2BEV")
  x = conv_block(bev_map, filters, 1, 1, BN=False, name=name + '_TransformRV2BEV_1')
  x = conv_block(x, filters, 1, 1, BN=False, name=name + '_TransformRV2BEV_2')
  # # x = conv_block(x, out_filters, 1, 1, BN=False, name=name + '_TransformRV2BEV_3')
  x = Conv2D(filters=out_filters, kernel_size=1, strides=1, padding='same', name=name + '_TransformRV2BEV_3')(x)

  return x

def build_BiFPN(f1, f2, f3, filters=128, kernel_size=3, BN=True, max_relu_val=None, weighted=True, name=""):

    def _sep_conv(x, i):
        x = SeparableConv2D(filters=filters, 
                            kernel_size=kernel_size, 
                            padding='same', 
                            use_bias=not BN, 
                            kernel_regularizer=tf.keras.regularizers.l1_l2(l1=5e-4, l2=5e-4),
                            name=name + "sepconv" + str(i))(x)
        if BN is True:
            x = BatchNormalization(name=name + "BN" + str(i))(x)
        x = ReLU(max_value=max_relu_val, name=name + "Relu" + str(i))(x)
        return x

    fusion_fn = Add
    if weighted:
        fusion_fn = BiFPN
    # print(fusion_fn.__name__)

    f1_resized = AveragePooling2D(name=name + "avg_pool1")(f1)
    f2_inter   = fusion_fn(name=name + "fusion1")([f2, f1_resized])
    f2_inter   = _sep_conv(f2_inter, 0)
    # print('f2_inter.shape', f2_inter.shape)

    f2_inter_resized = AveragePooling2D(name=name + "avg_pool2")(f2_inter)
    f3_out           = fusion_fn(name=name + "fusion2")([f2_inter_resized, f3])
    f3_out           = _sep_conv(f3_out, 1)
    # print('f3_out.shape', f3_out.shape)

    f3_out_resized = UpSampling2D(name=name + "upsample1")(f3_out)
    # f3_out_resized = ZeroPadding2D(((0,0), (0,1)))(f3_out_resized)
    f2_out         = fusion_fn(name=name + "fusion3")([f2, f2_inter, f3_out_resized])
    f2_out           = _sep_conv(f2_out, 2)
    # print('f2_out.shape', f2_out.shape)

    f2_out_resized = UpSampling2D(name=name + "upsample2")(f2_out)
    f1_out         = fusion_fn(name=name + "fusion4")([f2_out_resized, f1])
    f1_out           = _sep_conv(f1_out, 3)
    # print('f1_out.shape', f1_out.shape)

    return f1_out, f2_out, f3_out

class GroupNormalization(Layer):
    """Group normalization layer
    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. GN's computation is independent of batch sizes,
    and its accuracy is stable in a wide range of batch sizes
    # Arguments
        groups: Integer, the number of groups for Group Normalization.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        group_reduction_axes = group_reduction_axes[2:]

        mean = K.mean(inputs, axis=group_reduction_axes, keepdims=True)
        variance = K.var(inputs, axis=group_reduction_axes, keepdims=True)

        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)
        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


# get_custom_objects().update({'GroupNormalization': GroupNormalization})


# if __name__ == '__main__':
#     from tensorflow.keras.layers import Input
#     from tensorflow.keras.models import Model
#     ip = Input(shape=(None, None, 4))
#     #ip = Input(batch_shape=(100, None, None, 2))
#     x = GroupNormalization(groups=2, axis=-1, epsilon=0.1)(ip)
#     model = Model(ip, x)
#     model.summary()