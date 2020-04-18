import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, ReLU, BatchNormalization, MaxPooling2D, AveragePooling2D, \
                                        Conv2DTranspose, Concatenate, Layer, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils import Sequence
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Concatenate, BatchNormalization, MaxPooling2D, AveragePooling2D, ReLU
from tensorflow.keras.models import Model


def create_pixor_pp(input_shape=(800, 700, 35), kr=None, ki='he_normal', data_format=None):
    def _cbr(t, filters, name, max_pool=False, batch_norm=True):
        with tf.name_scope(name):
            if max_pool:
                t = MaxPooling2D((2, 2), strides=2)(t)
            t = Conv2D(filters, (3, 3), activation=None, padding='same', use_bias=not batch_norm, kernel_regularizer=kr, kernel_initializer=ki)(t)
            if batch_norm:
                t = BatchNormalization(momentum=0.99)(t)
            t = ReLU()(t)
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
        for i in range(1, 6):
            x = _cbr(x, 256, "C{}".format(i))
        b4_out = x

    # Aggregate feature maps
    with tf.name_scope('Concatenation'):
        # Downsample block 1 twice
        b1_out = AveragePooling2D((2, 2), strides=2)(b1_out)
        b1_out = AveragePooling2D((2, 2), strides=2)(b1_out)

        # Downsample block 2
        b2_out = AveragePooling2D((2, 2), strides=2)(b2_out)

        # Upsample block 4
        b4_out = tf.image.resize(b4_out, size=b3_out.shape[1:3])

        x = Concatenate(axis=-1, name='Concat')([b1_out, b2_out, b3_out, b4_out])

    # Head
    with tf.name_scope('Head'):
        x = _cbr(x, 256, "C1")
        for i in range(2, 6):
            x = _cbr(x, 256, "C{}".format(i), batch_norm=False)

        obj_map = Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='obj_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(x)
        geo_map = Conv2D(8, kernel_size=3, padding='same', activation=None, name='geo_map', kernel_regularizer=kr, kernel_initializer='glorot_normal')(x)

    return Model(inputs=input_tensor, outputs=[obj_map, geo_map])



# model = create_pixor_pp((800, 700, 35))
# model.summary()