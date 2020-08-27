
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import UpSampling2D, Conv2DTranspose, AveragePooling2D, ZeroPadding2D, Add, Conv2D, \
                                    Flatten, Reshape, MaxPooling2D, Average, Concatenate, Multiply, Conv2DTranspose, Lambda
from pprint import pprint

from models.rangeview_branch import create_range_view_branch

def create_car_recognition_net():

    '''
        Range View Net
    '''
    rv_net = create_range_view_branch()
    rv_inputs  = rv_net.inputs
    rv_outputs = rv_net.outputs

    RV_OUT = rv_outputs[0]
    
    obj_map = Conv2D(filters=1, 
                     kernel_size=1, 
                     padding='same', 
                     activation='sigmoid', 
                     name='obj_map', 
                     kernel_initializer='glorot_uniform')(RV_OUT)
    
    model = Model(rv_inputs, obj_map)

    return model

# create_sensor_fusion_net().summary()