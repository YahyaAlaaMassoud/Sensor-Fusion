
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D, Add, Conv2D, Flatten, Reshape
from pprint import pprint

from models.blocks import create_res_conv_block, create_input_layer, conv_block
from models.rangeview_branch import create_range_view_branch

def create_sensor_fusion_net():
    bev_in = create_input_layer((800, 700, 10), 'bev')
    x = bev_in

    '''
        Backbone
    '''
    for i in range(2): # same res
        x = create_res_conv_block(x, 24, 3)
    f1 = x
        
    for i in range(4): # 1 / 2 res
        x = create_res_conv_block(x, 48, 3, i==0)
    f2 = x

    for i in range(4): # 1 / 4 res
        x = create_res_conv_block(x, 96, 3, i==0)
    f3 = x

    for i in range(4): # 1 / 8 res
        x = create_res_conv_block(x, 192, 3, i==0)
    f4 = x

    for i in range(4): # 1 / 16 res
        x = create_res_conv_block(x, 256, 3, i==0)
    f5 = x


    rv_net = create_range_view_branch(input_shape=(375, 1242, 3), 
                                  input_names=['depth_map_input', 'intensity_map_input', 'height_map_input'])
    rv_inputs  = rv_net['inputs']
    rv_outputs = rv_net['outputs']

    l2, l3, l4, l5 = rv_outputs

    '''
        RV FPN
    '''
    rv_l2_dw = conv_block(l2, 96, 1, 1)
    rv_l2_dw = AveragePooling2D()(rv_l2_dw)
    rv_l2_dw = ZeroPadding2D(((0,0), (2,0)))(rv_l2_dw)
    rv_l2 = tf.image.resize(rv_l2_dw, 
                                        (200,
                                        175),)

    rv_l3_zp = conv_block(l3, 96, 1, 1)
    rv_l3 = tf.image.resize(rv_l3_zp, 
                                        (200,
                                        175),)

    rv_l4_zp = conv_block(l4, 96, 1, 1)
    rv_l4 = tf.image.resize(rv_l4_zp, 
                                        (200,
                                        175),)

    rv_l5_up = conv_block(l5, 96, 1, 1)
    rv_l5 = tf.image.resize(rv_l5_up, 
                                        (200,
                                        175),)

    # pprint([rv_l2, rv_l3, rv_l4, rv_l5])

    # rv_out = Add()([ rv_l2_dw, rv_l3_zp, rv_l4_zp, rv_l5_up ])
    # print(rv_out)
    # rv_out_f = tf.keras.backend.resize_images(rv_out, 
    #                                           200/94,
    #                                           175/312,
    #                                           'channels_last',
    #                                           'bilinear')

    # rv_out_f

    '''
        FPN + Header
    '''
    f2_dw = conv_block(f2, 96, 1, 1)
    f2_dw = AveragePooling2D()(f2_dw)
    f2_dw = Add()([f2_dw, rv_l2])

    f3_zp = f3
    f3_zp = Add()([f3_zp, rv_l3])

    f4_up = conv_block(f4, 96, 1, 1)
    f4_up = tf.image.resize(f4_up, 
                                        (200,
                                        175))
    f4_up = Add()([f4_up, rv_l4])

    f5_up = conv_block(f5, 96, 1, 1)
    f5_up = tf.image.resize(f5_up, 
                                        (200, 175),)
    f5_up = Add()([f5_up, rv_l5])

    out = Add()([f2_dw, f3_zp, f4_up, f5_up])

    obj_map = Conv2D(filters=1, 
                    kernel_size=1, 
                    padding='same', 
                    activation='sigmoid', 
                    name='obj_map', 
                    kernel_initializer='glorot_normal')(out)
    geo_map = Conv2D(filters=11, 
                    kernel_size=1, 
                    padding='same', 
                    activation=None, 
                    name='geo_map',
                    kernel_initializer='glorot_normal')(out)

    # pprint([f2, f3, f4, f5])
    # print('-----------------')
    # pprint([f2_dw, f3_zp, f4_up, f5_up])
    # print('-----------------')
    # pprint([out])
    # print('-----------------')
    # pprint([obj_map, geo_map])
    # print('-----------------')

    model = Model([bev_in] + rv_inputs, [obj_map, geo_map])
    # print(model.summary())

    return model

# create_sensor_fusion_net()