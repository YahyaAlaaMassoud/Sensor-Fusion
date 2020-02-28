
import tensorflow as tf
import tensorflow.keras.backend as K

_EPSILON = tf.keras.backend.epsilon()

def focal_loss(alpha, gamma, subsampling_flag):
    def focal_loss_fixed(y_true, y_pred):
        y_true_cls       = y_true[:, :, :, 0]
        subsampling_mask = 1.0
        if subsampling_flag:
            subsampling_mask = y_true[:, :, :, 1]
            
        pos_mask = tf.cast(tf.equal(y_true_cls, 1.0), tf.float32)
        neg_mask = tf.cast(tf.less(y_true_cls, 1.0), tf.float32)

        pos_loss = -tf.math.log(tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)) * tf.pow(1 - y_pred, gamma) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1 - y_pred, _EPSILON, 1.0 - _EPSILON)) * tf.pow(y_pred, gamma) * neg_mask

        num_pos = tf.reduce_sum(pos_mask)
        pos_loss = tf.reduce_sum(pos_loss) * alpha
        neg_loss = tf.reduce_sum(neg_loss) * (1 - alpha)

        return (pos_loss + neg_loss) / (num_pos + 1)

    return focal_loss_fixed


def binary_focal_loss(alpha, gamma, subsampling_flag):
    
    def compute_binary_focal_loss(y_true, y_pred):
        y_true_cls       = y_true[:, :, :, 0]
        subsampling_mask = 1.0
        if subsampling_flag:
            subsampling_mask = y_true[:, :, :, 1]
        
        y_pred = tf.clip_by_value(y_pred, _EPSILON, 1.0 - _EPSILON)

        ce = K.binary_crossentropy(y_true_cls, y_pred, from_logits=False)
        
        y_true_comp = (1.0 - y_true_cls)
        y_pred_comp = tf.clip_by_value(1.0 - y_pred, _EPSILON, 1.0 - _EPSILON)
        
        p_t = (y_true_cls * y_pred) + (y_true_comp * y_pred_comp)
        
        alpha_factor = (y_true_cls * alpha + y_true_comp * (1.0 - alpha))
        
        modulating_factor = tf.pow((1.0 - p_t), gamma)

        num_pos = tf.reduce_sum(y_true_cls) + 1.0
        loss = alpha_factor * modulating_factor * ce * subsampling_mask
        loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)

        return tf.reduce_sum(loss) / num_pos

    return compute_binary_focal_loss

@tf.function(experimental_relax_shapes=True)
def absolute_diff_loss(y_true, y_pred):
    abs_diff = tf.abs(y_true - y_pred)
    loss = tf.reduce_mean(abs_diff)
    if tf.math.is_nan(loss):
        return abs_diff
    return loss

def smooth_l1_loss(weight):

    def compute_smooth_l1_loss(y_true, y_pred):
        abs_diff  = tf.abs(y_true - y_pred)
        smooth_l1 = tf.where(abs_diff < 1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
        return smooth_l1

    return compute_smooth_l1_loss

def total_loss(alpha, gamma, reg_loss_name, reg_channels, weight, subsampling_flag):
    '''
        erosion mask is expected to be the second channel in the output
    '''
    obj_loss_fn = focal_loss(alpha, gamma, subsampling_flag)
    
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
        mask = tf.reshape(tf.equal(y_true_cls[:,:,:,0], 1.0), shape=(-1,))
        # print('mask', mask.shape)
        y_true_reg_masked = tf.boolean_mask(tf.reshape(y_true_reg, shape=(-1, reg_channels)), 
                                            mask=mask, 
                                            axis=0)
        # print('gt', y_true_reg_masked.shape)
        y_pred_reg_masked = tf.boolean_mask(tf.reshape(y_pred_reg, shape=(-1, reg_channels)),
                                            mask=mask,
                                            axis=0)
        # print('pred', y_pred_reg_masked.shape)
        reg_loss = reg_loss_fn(y_true_reg_masked, y_pred_reg_masked)

        return obj_loss + reg_loss
    
    return compute_total_loss
