import tensorflow as tf
import tensorflow.keras.backend as K

from pixor_utils.model_utils import load_model
from training_utils.losses import total_loss
from training_utils.metrics import objectness_metric, regression_metric
from models.pixor_det import BiFPN
from tensorflow.keras.optimizers import Adam



run_meta = tf.RunMetadata()

with tf.Session(graph=tf.Graph()) as sess:
    K.set_session(sess)
    sess.run(tf.global_variables_initializer())
    model = load_model('outputs/car_3bifpn_concat_aug50random_abs_densify/car_3bifpn_concat_aug50random_abs_densify_epoch_1.json', 
                       'outputs/car_3bifpn_concat_aug50random_abs_densify/car_3bifpn_concat_aug50random_abs_densify_epoch_1.h5', 
                       {'BiFPN': BiFPN})

    optimizer = Adam(lr=0.0001)
    losses = {
            'output_map': total_loss(alpha=0.25, gamma=2.0, reg_loss_name='abs', reg_channels=8, weight=1.0, subsampling_flag=False)
            }
    metrics = {
            'output_map': [objectness_metric(alpha=0.25, gamma=2.0, subsampling_flag=False),
                            regression_metric(reg_loss_name='abs', reg_channels=8, weight=1.0, subsampling_flag=False)],
            }

    for layer in model.layers:
        layer.trainable = False

    model.compile(optimizer=optimizer,
                    loss=losses,
                    metrics=metrics)

    # model.summary()
    net = model(inputs=tf.compat.v1.placeholder('float32', shape=(1,800,700,35)))

    opts = tf.profiler.ProfileOptionBuilder.float_operation()    
    flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    # opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
    # params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

    # print("{:,} --- {:,}".format(flops.total_float_ops, params.total_parameters))