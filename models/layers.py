
import tensorflow as tf

from tensorflow.keras.layers import Layer

class BiFPN(Layer):

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

    # w = ReLU()(self.w)
    x = tf.reduce_sum([self.w[i] * x[i] for i in range(len(x))], axis=0)
    x = x / (tf.reduce_sum(self.w) + self.eps)
    
    return x

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = super(BiFPN, self).get_config()
    return config