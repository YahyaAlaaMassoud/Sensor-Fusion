
import tensorflow as tf
import tensorflow.keras.backend as K

alpha, gamma = 0.25, 2.0
# FL_ALPHA, FL_GAMMA = 0.75, 0.5

def binary_focal_loss(y_true, y_pred):
    # y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    # p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
    # alpha_t = K.ones_like(y_true) * FL_ALPHA
    # alpha_t = tf.where(K.equal(y_true, 1), alpha_t, 1 - alpha_t)

    # loss = -alpha_t * K.pow((1 - p_t), FL_GAMMA) * K.log(p_t)
    
    # pt_cnt = tf.cast(tf.math.count_nonzero(y_true), tf.float32)
    # # tf.print('# pos ->',pt_cnt)
    # # tf.print('loss(1) ->', tf.divide(K.sum(loss), pt_cnt))
    # # tf.print('loss(2) ->', K.sum(loss))
    # # print('------------')
    # loss = tf.cond(tf.greater(pt_cnt, 0), lambda: tf.divide(K.sum(loss), pt_cnt), lambda: K.sum(loss))
    
    # Get the cross_entropy for each entry
    from_logits = False
    
    ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    alpha_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

    modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
    
def smooth_L1(y_true, y_pred):
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff < 1, 0.5 * tf.square(diff), diff - 0.5)
    return tf.reduce_mean(loss)

def pixor_loss(y_true, y_pred):
    # Split into cls and reg
    y_true_cls, y_true_reg = y_true[:, :, :, 0], y_true[:, :, :, 1:]
    y_pred_cls, y_pred_reg = y_pred[:, :, :, 0], y_pred[:, :, :, 1:]

    # Classification Loss
    obj_loss = binary_focal_loss(y_true_cls, y_pred_cls)

    # Regression Loss
    mask = tf.reshape(K.equal(y_true_cls, 1.0), shape=(-1,))
    
    y_true_reg_masked = tf.boolean_mask(tf.reshape(y_true_reg, shape=(-1, 6)), mask=mask, axis=0)
    y_pred_reg_masked = tf.boolean_mask(tf.reshape(y_pred_reg, shape=(-1, 6)), mask=mask, axis=0)
    geo_loss = smooth_L1(y_true_reg_masked, y_pred_reg_masked)

    # Remove Nan
    geo_loss = tf.where(tf.math.is_nan(geo_loss), tf.zeros_like(geo_loss), geo_loss)
    
    total_loss = obj_loss + 2. * geo_loss
    
    return total_loss



def binary_focal_loss_metric(y_true, y_pred):
    y_true, y_pred = y_true[:, :, :, 0], y_pred[:, :, :, 0]
    return binary_focal_loss(y_true, y_pred)

def smooth_L1_metric(y_true, y_pred):
    # Split into cls and reg
    y_true_cls, y_true_reg = y_true[:, :, :, 0], y_true[:, :, :, 1:]
    _         , y_pred_reg = y_pred[:, :, :, 0], y_pred[:, :, :, 1:]

    # Classification Loss
    # obj_loss = binary_focal_loss(y_true_cls, y_pred_cls)

    # Regression Loss
    mask = tf.reshape(K.equal(y_true_cls, 1.0), shape=(-1,))
    
    y_true_reg_masked = tf.boolean_mask(tf.reshape(y_true_reg, shape=(-1, 6)), mask=mask, axis=0)
    y_pred_reg_masked = tf.boolean_mask(tf.reshape(y_pred_reg, shape=(-1, 6)), mask=mask, axis=0)
    geo_loss = smooth_L1(y_true_reg_masked, y_pred_reg_masked)

    # Remove Nan
    geo_loss = tf.where(tf.math.is_nan(geo_loss), tf.zeros_like(geo_loss), geo_loss)
    
    return 2. * geo_loss