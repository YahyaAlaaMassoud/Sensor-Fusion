
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D, Add, Conv2D, Flatten, Reshape, MaxPooling2D, Average, Concatenate
from pprint import pprint

from models.blocks import create_res_conv_block, create_input_layer, conv_block, factorized_bilinear_pooling_new, factorized_bilinear_pooling
from models.rangeview_branch import create_range_view_branch

def create_sensor_fusion_net():
    bev_in = create_input_layer((448, 512, 32), 'bev')
    x = bev_in

    '''
        Backbone
    '''
    for i in range(2): # same res
        x = conv_block(x, 64, 3, 1)
    f1 = x
        
    for i in range(4): # 1 / 2 res
        x = create_res_conv_block(x, 128, 3, i==0)
    f2 = x

    for i in range(6): # 1 / 4 res
        x = create_res_conv_block(x, 192, 3, i==0)
    f3 = x

    for i in range(6): # 1 / 8 res
        x = create_res_conv_block(x, 256, 3, i==0)
    f4 = x

    '''
        Range View Net
    '''
    rv_net = create_range_view_branch(input_shape=(375, 1242, 3), 
                                  input_names=['rgb_img_input', 'depth_map_input', 'intensity_map_input', 'height_map_input'])
    rv_inputs  = rv_net['inputs']
    rv_outputs = rv_net['outputs']

    l2, l3, l4 = rv_outputs

    '''
        RV FPN
    '''

    out_map_sz = (112, 128)
    TOP_DOWN_PYRAMID_SIZE = 96

    L2 = conv_block(l2, TOP_DOWN_PYRAMID_SIZE, 1, 1)
    L2 = tf.image.resize(L2, (94, 311), 'nearest')
    L2 = conv_block(L2, TOP_DOWN_PYRAMID_SIZE, 3, 1)

    L3 = conv_block(l3, TOP_DOWN_PYRAMID_SIZE, 1, 1)

    L4 = conv_block(l4, TOP_DOWN_PYRAMID_SIZE, 1, 1)
    L4 = tf.image.resize(L4, (94, 311), 'nearest')
    L4 = conv_block(L4, TOP_DOWN_PYRAMID_SIZE, 3, 1)

    RV_OUT = Add()([L2, L3, L4])
    RV_OUT = tf.image.resize(RV_OUT, out_map_sz, 'nearest')
    RV_OUT = conv_block(RV_OUT, TOP_DOWN_PYRAMID_SIZE, 3, 1)

    # pprint([rv_l2, rv_l3, rv_l4, rv_l5])

    '''
        BEV FPN + Header
    '''
    F2 = conv_block(f2, TOP_DOWN_PYRAMID_SIZE, 1, 1)
    F2 = tf.image.resize(F2, out_map_sz, 'nearest')
    F2 = conv_block(F2, TOP_DOWN_PYRAMID_SIZE, 3, 1)
    F2 = factorized_bilinear_pooling_new(F2, RV_OUT, TOP_DOWN_PYRAMID_SIZE, TOP_DOWN_PYRAMID_SIZE * 2)
    # F2 = conv_block(F2, TOP_DOWN_PYRAMID_SIZE, 3, 1)

    F3 = f3
    F3 = conv_block(F3, TOP_DOWN_PYRAMID_SIZE, 1, 1)
    F3 = factorized_bilinear_pooling_new(F3, RV_OUT, TOP_DOWN_PYRAMID_SIZE, TOP_DOWN_PYRAMID_SIZE * 2)
    # F3 = conv_block(F3, TOP_DOWN_PYRAMID_SIZE, 3, 1)

    F4 = conv_block(f4, TOP_DOWN_PYRAMID_SIZE, 1, 1)
    F4 =  tf.image.resize(F4, out_map_sz, method='nearest')
    F4 = conv_block(F4, TOP_DOWN_PYRAMID_SIZE, 3, 1)
    F4 = factorized_bilinear_pooling_new(F4, RV_OUT, TOP_DOWN_PYRAMID_SIZE, TOP_DOWN_PYRAMID_SIZE * 2)
    # F4 = conv_block(F4, TOP_DOWN_PYRAMID_SIZE, 3, 1)

    BEV_OUT = Add()([F2, F3, F4])
    OUT = create_res_conv_block(BEV_OUT, 96, 3)

    obj_map = Conv2D(filters=1, 
                     kernel_size=1, 
                     padding='same', 
                     activation='sigmoid', 
                     name='obj_map', 
                     kernel_initializer='glorot_normal')(OUT)
    geo_map = Conv2D(filters=8, 
                     kernel_size=1, 
                     padding='same', 
                     activation=None, 
                     name='geo_map',
                     kernel_initializer='glorot_normal')(OUT)


    model = Model([bev_in] + rv_inputs, [obj_map, geo_map])
    # print(model.summary())

    return model

# create_sensor_fusion_net().summary()