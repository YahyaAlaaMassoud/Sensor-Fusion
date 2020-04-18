
import tensorflow as tf
import tensorflow.keras.backend as K

def Swish():

    def _swish(x):
        return x * K.sigmoid(x)
    
    return _swish