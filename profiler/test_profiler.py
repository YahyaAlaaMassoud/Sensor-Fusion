
import tensorflow as tf
import tensorflow.keras.backend as K

from pixor_utils.model_utils import load_model
from models.pixor_det import BiFPN, create_pixor_det
from models.pixor_pp import create_pixor_pp

run_meta = tf.RunMetadata()

with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
#     sess.run(tf.global_variables_initializer())
#     model = create_pixor_pp()
    net = load_model('outputs/ckpts_3bifpn_car_aug_abs_3d/pixor_bifpn_car_noRelu_conv_aug_abs__epoch_8.json', 
                       'outputs/ckpts_3bifpn_car_aug_abs_3d/pixor_bifpn_car_noRelu_conv_aug_abs__epoch_8.h5', 
                       {'BiFPN': BiFPN})


#     net.summary()

    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)