
import tensorflow as tf
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt
import numpy as np


def focal(alpha, gamma, subsampling_flag, data_format='channels_last'):

    def _focal(y_true, y_pred):
        subsampling_mask = 1.0
        if subsampling_flag:
            subsampling_mask = y_true[...,1:2]

        y_true = y_true[...,0:1]
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=False)

        alpha_factor = K.ones_like(y_true) * alpha
        alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = alpha_factor * (focal_weight ** gamma)

        obj_loss = focal_weight * ce * subsampling_mask
        obj_loss_sum = tf.reduce_sum(obj_loss)

        num_pos = tf.reduce_sum(y_true) + 1.0

        return obj_loss_sum / num_pos
        # return tf.reduce_sum(ce)/ num_pos
    
    _focal.__name__ = 'focal_loss'

    return _focal


def abs_loss(reg_channels, data_format='channels_last'):

    def compute_abs_loss(y_true, y_pred):

        mask = tf.reshape(tf.equal(y_true[...,-1], 1.0), shape=(-1,))
        pos_count = tf.reduce_sum(tf.cast(mask, tf.float32))

        y_true = y_true[...,:-1]
        y_true_masked = tf.boolean_mask(tf.reshape(y_true, shape=(-1, reg_channels)), 
                                        mask=mask, 
                                        axis=0)
        y_pred_masked = tf.boolean_mask(tf.reshape(y_pred, shape=(-1, reg_channels)),
                                        mask=mask,
                                        axis=0)
        if y_true_masked.shape[0] == 0:
            return tf.constant(value=0., dtype=tf.float32)

        abs_diff  = tf.abs(y_true - y_pred)
        sum = tf.reduce_sum(abs_diff, axis=1)
        loss = tf.reduce_mean(sum)
        return loss
    
    compute_abs_loss.__name__ = 'absolute_diff'

    return compute_abs_loss



def smooth_l1_loss(sigma, reg_channels, data_format='channels_last'):

    sigma_squared = sigma ** 2

    def compute_smooth_l1_loss(y_true, y_pred):

        mask = tf.reshape(tf.equal(y_true[...,-1], 1.0), shape=(-1,))
        pos_count = tf.reduce_sum(tf.cast(mask, tf.float32))

        y_true = y_true[...,:-1]
        y_true_masked = tf.boolean_mask(tf.reshape(y_true, shape=(-1, reg_channels)), 
                                        mask=mask, 
                                        axis=0)
        y_pred_masked = tf.boolean_mask(tf.reshape(y_pred, shape=(-1, reg_channels)),
                                        mask=mask,
                                        axis=0)
        if y_true_masked.shape[0] == 0:
            return tf.constant(value=0., dtype=tf.float32)

        abs_diff  = tf.abs(y_true - y_pred)

        smooth_l1 = tf.where(abs_diff < (1. / sigma_squared), 
                             0.5 * sigma_squared * tf.square(abs_diff), 
                             abs_diff - 0.5 / sigma_squared)

        sum = tf.reduce_sum(smooth_l1, axis=1)
        loss = tf.reduce_mean(sum)
        return loss
    
    compute_smooth_l1_loss.__name__ = 'smooth_l1'

    return compute_smooth_l1_loss
