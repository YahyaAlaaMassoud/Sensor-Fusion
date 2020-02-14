
import tensorflow as tf
import tensorflow.keras.backend as K

def binary_focal_loss(alpha, gamma, subsampling_flag):
    
    def compute_binary_focal_loss(y_true, y_pred):
        y_true_cls       = y_true[:, :, :, 0]
        subsampling_mask = 1.0
        if subsampling_flag:
            subsampling_mask = y_true[:, :, :, 1]
        
        ce = K.binary_crossentropy(y_true_cls, y_pred, from_logits=False)
        
        y_true_comp = (1.0 - y_true_cls)
        y_pred_comp = (1.0 - y_pred)
        
        p_t = (y_true_cls * y_pred) + (y_true_comp * y_pred_comp)
        
        alpha_factor = (y_true_cls * alpha + (y_true_comp * (1.0 - alpha)))
        
        modulating_factor = tf.pow((1.0 - p_t), gamma)
        
        return tf.reduce_sum(alpha_factor * modulating_factor * ce * subsampling_mask, axis=-1)
    
    return compute_binary_focal_loss

def absolute_diff_loss(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    return tf.reduce_mean(abs_diff)

def smooth_l1_loss(weight):
    
    def compute_smooth_l1_loss(y_true, y_pred):
        abs_diff  = tf.abs(y_true - y_pred)
        smooth_l1 = tf.where(abs_diff < 1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
        return weight * tf.reduce_mean(smooth_l1)
    
    return compute_smooth_l1_loss

def total_loss(alpha, gamma, reg_loss_name, reg_channels, weight, subsampling_flag):
    '''
        erosion mask is expected to be the last channel in the output
    '''
    obj_loss_fn = binary_focal_loss(alpha, gamma, subsampling_flag)
    
    reg_loss_fn = smooth_l1_loss(weight)
    if reg_loss_name == 'abs':
        reg_loss_fn = absolute_diff_loss
    
    def compute_total_loss(y_true, y_pred):
        reg_start = 1
        if subsampling_flag:
            reg_start = 2
            
        y_true_cls, y_true_reg = y_true[:, :, :, :reg_start], y_true[:, :, :, reg_start:]
        y_pred_cls, y_pred_reg = y_pred[:, :, :, 0], y_pred[:, :, :, 1:]

        # Objectness Loss
        obj_loss = obj_loss_fn(y_true_cls, y_pred_cls)

        # Regression Loss
        mask = tf.reshape(tf.equal(y_true_cls[:, :, :, 0], 1.0), shape=(-1,))
        y_true_reg_masked = tf.boolean_mask(tf.reshape(y_true_reg, shape=(-1, reg_channels)), 
                                            mask=mask, 
                                            axis=0)
        y_pred_reg_masked = tf.boolean_mask(tf.reshape(y_pred_reg, shape=(-1, reg_channels)),
                                            mask=mask,
                                            axis=0)
        reg_loss = reg_loss_fn(y_true_reg_masked, y_pred_reg_masked)
        reg_loss = tf.where(tf.math.is_nan(reg_loss), tf.zeros_like(reg_loss), reg_loss)
        
        return obj_loss + reg_loss
    
    return compute_total_loss
