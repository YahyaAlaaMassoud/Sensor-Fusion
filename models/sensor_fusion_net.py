
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D, Add, Conv2D, Flatten, Reshape, MaxPooling2D, Average, Concatenate, Multiply, Conv2DTranspose
from pprint import pprint

from models.blocks import create_res_conv_block, create_input_layer, conv_block, factorized_bilinear_pooling_new, factorized_bilinear_pooling
from models.rangeview_branch import create_range_view_branch

def create_sensor_fusion_net():
    bev_in = create_input_layer((448, 512, 32), 'bev', 'float32')
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
    rv_net = create_range_view_branch()
    rv_inputs  = rv_net['inputs']
    rv_outputs = rv_net['outputs']

    RV2BEV_2x, RV2BEV_4x, RV2BEV_8x = rv_outputs

    # print(f2.shape, RV2BEV_2x.shape)
    # print(f3.shape, RV2BEV_4x.shape)
    # print(f4.shape, RV2BEV_8x.shape)

    F2 = factorized_bilinear_pooling_new(f2, RV2BEV_2x, RV2BEV_2x.shape[-1], RV2BEV_2x.shape[-1] * 2, "Bilinear1_")
    F2 = MaxPooling2D()(F2)
    F2 = conv_block(F2, RV2BEV_4x.shape[-1], 1, 1)

    F3 = factorized_bilinear_pooling_new(f3, RV2BEV_4x, RV2BEV_4x.shape[-1], RV2BEV_4x.shape[-1] * 2, "Bilinear2_")

    F4 = factorized_bilinear_pooling_new(f4, RV2BEV_8x, RV2BEV_8x.shape[-1], RV2BEV_8x.shape[-1] * 2, "Bilinear3_")
    F4 = UpSampling2D()(F4)
    F4 = conv_block(F4, RV2BEV_8x.shape[-1], 3, 1)
    F4 = conv_block(F4, RV2BEV_4x.shape[-1], 1, 1)

    # print(F2.shape, F3.shape, F4.shape)

    BEV_OUT = Multiply()([F2, F3, F4])
    OUT = create_res_conv_block(BEV_OUT, RV2BEV_4x.shape[-1], 3)

    obj_map = Conv2D(filters=1, 
                     kernel_size=1, 
                     padding='same', 
                     activation='sigmoid', 
                     name='obj_map', 
                     kernel_initializer='glorot_normal')(OUT)
    geo_map = Conv2D(filters=9, 
                     kernel_size=1, 
                     padding='same', 
                     activation=None, 
                     name='geo_map',
                     kernel_initializer='glorot_normal')(OUT)


    model = Model([bev_in] + rv_inputs, [obj_map, geo_map])
    # print(model.summary())

    return model

# create_sensor_fusion_net().summary()