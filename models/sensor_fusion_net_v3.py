
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D, Add, Conv2D, \
                                    Flatten, Reshape, MaxPooling2D, Average, Concatenate, Multiply, Conv2DTranspose, Lambda
from pprint import pprint

from models.blocks import create_res_conv_block, create_input_layer, conv_block, factorized_bilinear_pooling_new, factorized_bilinear_pooling, transform_rangeview_to_bev, build_BiFPN
from models.rangeview_branch import create_range_view_branch

def create_sensor_fusion_net_v3(
                             input_shapes=[(448, 512, 32), 
                                           (224, 256 , 2),
                                           (112, 128 , 2),
                                           (56 , 64  , 2),
                                           (224, 256 , 3),
                                           (112, 128 , 3),
                                           (56 , 64  , 3),
                                          ], 
                             input_names=['bev_input', 
                                          'mapping_2x',
                                          'mapping_4x',
                                          'mapping_8x',
                                          'geo_2x',
                                          'geo_4x',
                                          'geo_8x',
                                          ],
                             input_types=['float64',
                                          'int32',
                                          'int32',
                                          'int32',
                                          'float64',
                                          'float64',
                                          'float64',
                                          ]
                            ):

    '''
        Construct Input Layers
    '''
    init_inputs = []
    inputs_dict = {}
    for name, shape, dtype in zip(input_names, input_shapes, input_types):
        inp = create_input_layer(input_shape=shape, name=name, dtype=dtype)
        init_inputs.append(inp)  
        inputs_dict[name] = inp

    '''
        Range View Net
    '''
    rv_net = create_range_view_branch()
    rv_inputs  = rv_net.inputs
    rv_outputs = rv_net.outputs

    obj_2d_high, obj_2d_mid, obj_2d_low, rv_high, rv_mid, rv_low = rv_outputs

    '''
        Continious Conv
    '''

    def gather(layer_in):
      params, indices = layer_in
      return tf.gather_nd(params, indices, batch_dims=1)

    print('rv_high.shape', rv_high.shape)
    print("inputs_dict['mapping_2x'].shape", inputs_dict['mapping_2x'].shape)
    rv_high = conv_block(rv_high, 128, 3, 1, name='RV_HIGH_CONV')
    RV2BEV_2x_vis = Lambda(gather)([rv_high, inputs_dict['mapping_2x']])
    RV2BEV_2x = Concatenate()([RV2BEV_2x_vis, inputs_dict['geo_2x']]) # add geometric feature # RV2BEV_2x_vis#
    RV2BEV_2x = transform_rangeview_to_bev(RV2BEV_2x, 128, name='RV2BEV2x')

    print('rv_mid.shape', rv_mid.shape)
    print("inputs_dict['mapping_4x'].shape", inputs_dict['mapping_4x'].shape)
    rv_mid = conv_block(rv_mid, 192, 3, 1, name='RV_MID_CONV')
    RV2BEV_4x_vis = Lambda(gather)([rv_mid, inputs_dict['mapping_4x']])
    RV2BEV_4x = Concatenate()([RV2BEV_4x_vis, inputs_dict['geo_4x']]) # add geometric feature # RV2BEV_4x_vis#
    RV2BEV_4x = transform_rangeview_to_bev(RV2BEV_4x, 192, name='RV2BEV4x')

    print('rv_low.shape', rv_low.shape)
    print("inputs_dict['mapping_8x'].shape", inputs_dict['mapping_8x'].shape)
    rv_low = conv_block(rv_low, 256, 3, 1, name='RV_LOW_CONV')
    RV2BEV_8x_vis = Lambda(gather)([rv_low, inputs_dict['mapping_8x']])
    RV2BEV_8x = Concatenate()([RV2BEV_8x_vis, inputs_dict['geo_8x']]) # add geometric feature # RV2BEV_8x_vis#
    RV2BEV_8x = transform_rangeview_to_bev(RV2BEV_8x, 256, name='RV2BEV8x')

    print(RV2BEV_2x.shape, RV2BEV_4x.shape, RV2BEV_8x.shape)
    print('#####################################################')

    '''
        BEV Backbone
    '''
    x = inputs_dict['bev_input']

    for i in range(2): # same res
        x = conv_block(x, 64, 3, 1, name='BEV_Block1_{}'.format(i))
    F1 = x

    for i in range(4): # 1 / 2 res
        x = create_res_conv_block(x, 128, 3, i==0, name='BEV_Block2_{}'.format(i))
    # x = factorized_bilinear_pooling_new(x, RV2BEV_2x, RV2BEV_2x.shape[-1], RV2BEV_2x.shape[-1] * 2, "Bilinear1_")
    # x = Add(name="Add_Fusion_Block2")([x, RV2BEV_2x])
    BEV_2x = x

    for i in range(4): # 1 / 4 res
        x = create_res_conv_block(x, 192, 3, i==0, name='BEV_Block3_{}'.format(i))
    # x = factorized_bilinear_pooling_new(x, RV2BEV_4x, RV2BEV_4x.shape[-1], RV2BEV_4x.shape[-1] * 2, "Bilinear2_")
    # x = Add(name="Add_Fusion_Block3")([x, RV2BEV_4x])
    BEV_4x = x

    for i in range(4): # 1 / 8 res
        x = create_res_conv_block(x, 256, 3, i==0, name='BEV_Block4_{}'.format(i))
    # x = factorized_bilinear_pooling_new(x, RV2BEV_8x, RV2BEV_8x.shape[-1], RV2BEV_8x.shape[-1] * 2, "Bilinear3_")
    # x = Add(name="Add_Fusion_Block4")([x, RV2BEV_8x])
    BEV_8x = x

    F2 = factorized_bilinear_pooling_new(BEV_2x, RV2BEV_2x, RV2BEV_2x.shape[-1], RV2BEV_2x.shape[-1] * 2, "Bilinear1_")
    F3 = factorized_bilinear_pooling_new(BEV_4x, RV2BEV_4x, RV2BEV_4x.shape[-1], RV2BEV_4x.shape[-1] * 2, "Bilinear2_")
    F4 = factorized_bilinear_pooling_new(BEV_8x, RV2BEV_8x, RV2BEV_8x.shape[-1], RV2BEV_8x.shape[-1] * 2, "Bilinear3_")

    print(F1.shape, F2.shape, F3.shape, F4.shape)
    print('#####################################################')

    print('F2.shape', F2.shape, 'RV2BEV_2x.shape', RV2BEV_2x.shape)
    print('F3.shape', F3.shape, 'RV2BEV_4x.shape', RV2BEV_4x.shape)
    print('F4.shape', F4.shape, 'RV2BEV_8x.shape', RV2BEV_8x.shape)

    BIFPN_CHANNELS = 128
    F2 = conv_block(F2, BIFPN_CHANNELS, 1, 1, BN=False, name='F2_1x1_Conv_preBiFPN')
    F3 = conv_block(F3, BIFPN_CHANNELS, 1, 1, BN=False, name='F3_1x1_Conv_preBiFPN')
    F4 = conv_block(F4, BIFPN_CHANNELS, 1, 1, BN=False, name='F4_1x1_Conv_preBiFPN')

    F2, F3, F4 = build_BiFPN(F2, F3, F4, filters=BIFPN_CHANNELS, BN=False, weighted=True, name="BiFPN1")
    # F2, F3, F4 = build_BiFPN(F2, F3, F4, filters=BIFPN_CHANNELS, BN=False, weighted=True, name="BiFPN2")
    # F2, F3, F4 = build_BiFPN(F2, F3, F4, filters=128, BN=True, weighted=True, name="BiFPN3")

    F2 = MaxPooling2D()(F2)
    # F2 = conv_block(F2, F2.shape[-1], 3, 1, BN=False, name='F2_Downsample')
    # F2 = create_res_conv_block(F2, RV2BEV_4x.shape[-1], 1, False, name='F2_Downsample')

    F3 = conv_block(F3, F3.shape[-1], 1, 1, BN=False, name='F3_1x1_Conv')

    F4 = UpSampling2D()(F4)
    F4 = conv_block(F4, F4.shape[-1], 3, 1, BN=False, name='F4_Downsample')
    # F4 = create_res_conv_block(F4, RV2BEV_4x.shape[-1], 1, False, name='F4_Upsample')

    print('F2.shape', F2.shape)
    print('F3.shape', F3.shape)
    print('F4.shape', F4.shape)

    BEV_OUT = Concatenate(name="BEV_OUT")([F2, F3, F4])
    # BEV_OUT = Add(name="BEV_OUT")([F2, F3, F4])
    BEV_OUT = conv_block(BEV_OUT, BIFPN_CHANNELS, 3, 1, BN=False, name='Header')

    obj3d_map = Conv2D(filters=1, 
                     kernel_size=1, 
                     padding='same', 
                     activation='sigmoid', 
                     name='obj_map', 
                     kernel_initializer='glorot_uniform')(BEV_OUT)
    geo_map = Conv2D(filters=8, 
                     kernel_size=1, 
                     padding='same', 
                     activation=None, 
                     name='geo_map',
                     kernel_initializer='glorot_uniform')(BEV_OUT)


    return Model(inputs=init_inputs + rv_inputs, 
                 outputs=[
                     obj3d_map, 
                     geo_map,
                     rv_high,
                     rv_low,
                     RV2BEV_2x,
                     BEV_2x,
                     RV2BEV_8x,
                     BEV_8x,
                     F2,
                     F4,
                     BEV_OUT,
                 ])

# create_sensor_fusion_net().summary()