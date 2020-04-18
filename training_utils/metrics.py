
import tensorflow as tf
import tensorflow.keras.backend as K

def reg_metric(value_idx, reg_channels):

    def compute_metric(y_true, y_pred):
        mask = tf.reshape(tf.equal(y_true[...,-1], 1.0), shape=(-1,))
        pos_count = tf.reduce_sum(tf.cast(mask, tf.float32))

        y_true = y_true[:,:,:,:-1]
        y_true_masked = tf.boolean_mask(tf.reshape(y_true, shape=(-1, reg_channels)), 
                                        mask=mask, 
                                        axis=0)
        y_pred_masked = tf.boolean_mask(tf.reshape(y_pred, shape=(-1, reg_channels)),
                                        mask=mask,
                                        axis=0)

        y_true = y_true_masked[:,value_idx]
        y_pred = y_pred_masked[:,value_idx]

        return tf.cond(tf.equal(tf.size(y_true), 0), lambda: tf.constant(0.0), lambda: tf.reduce_mean(tf.abs(y_true - y_pred))) 
    
    if value_idx == 0:
        compute_metric.__name__ = 'cos_theta'
    elif value_idx == 1:
        compute_metric.__name__ = 'sin_theta'
    elif value_idx == 2:
        compute_metric.__name__ = 'z_offset'
    elif value_idx == 3:
        compute_metric.__name__ = 'x_offset'
    elif value_idx == 4:
        compute_metric.__name__ = 'altitude'
    elif value_idx == 5:
        compute_metric.__name__ = 'log_w'
    elif value_idx == 6:
        compute_metric.__name__ = 'log_l'
    elif value_idx == 7:
        compute_metric.__name__ = 'log_h'
    elif value_idx == 8:
        compute_metric.__name__ = 'ratio_log_w'
    elif value_idx == 9:
        compute_metric.__name__ = 'ratio_log_l'
    elif value_idx == 10:
        compute_metric.__name__ = 'ratio_log_h'
    return compute_metric