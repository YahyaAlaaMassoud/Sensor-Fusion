
import tensorflow as tf
import tensorflow.keras.backend as K

from .losses import binary_focal_loss, smooth_l1_loss, absolute_diff_loss, focal_loss

def objectness_metric(alpha, gamma, subsampling_flag):
    
    obj_loss_fn = focal_loss(alpha, gamma, subsampling_flag)
    
    def compute_objectness_metric(y_true, y_pred):
        metric_value = obj_loss_fn(y_true, y_pred[:, :, :, 0])
        return metric_value
    
    return compute_objectness_metric

def regression_metric(reg_loss_name, reg_channels, weight, subsampling_flag):
    
    reg_loss_fn = smooth_l1_loss(weight)
    if reg_loss_name == 'abs':
        reg_loss_fn = absolute_diff_loss
    
    def compute_regression_metric(y_true, y_pred):
        reg_start = 1
        if subsampling_flag:
            reg_start = 2
            
        y_true_cls, y_true_reg = y_true[:, :, :, 0], y_true[:, :, :, reg_start:]
        _         , y_pred_reg = y_pred[:, :, :, 0], y_pred[:, :, :, 1:]

        # Regression Loss
        mask = tf.reshape(tf.equal(y_true_cls, 1.0), shape=(-1,))
        y_true_reg_masked = tf.boolean_mask(tf.reshape(y_true_reg, shape=(-1, reg_channels)), 
                                            mask=mask, 
                                            axis=0)
        y_pred_reg_masked = tf.boolean_mask(tf.reshape(y_pred_reg, shape=(-1, reg_channels)),
                                            mask=mask,
                                            axis=0)
        metric_value = reg_loss_fn(y_true_reg_masked, y_pred_reg_masked)

        return metric_value

    return compute_regression_metric