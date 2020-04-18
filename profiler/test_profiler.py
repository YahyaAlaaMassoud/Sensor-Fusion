

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import tensorflow.keras.backend as K

from pixor_utils.model_utils import load_model
from models.pixor_det import BiFPN, create_pixor_det
from models.pixor_pp import create_pixor_pp

run_meta = tf.RunMetadata()

with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    # net = create_pixor_pp()#create_pixor_det()
    model = load_model('outputs/ckpts_3bifpn_car_aug_abs_3d/pixor_bifpn_car_noRelu_conv_aug_abs__epoch_8.json', 
                       'outputs/ckpts_3bifpn_car_aug_abs_3d/pixor_bifpn_car_noRelu_conv_aug_abs__epoch_8.h5', 
                       {'BiFPN': BiFPN})

    # sess.run(tf.global_variables_initializer())

#     net.summary()
    net = model(inputs=tf.placeholder('float32', shape=(1,800,700,35)))


    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    # params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)