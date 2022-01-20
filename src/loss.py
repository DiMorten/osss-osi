

from tensorflow.keras import backend as K
import tensorflow as tf

def categorical_focal(alpha=0.25,gamma=2):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    #alpha = K.variable(alpha)
    #gamma = K.variable(gamma)
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_pred_softmax = tf.nn.softmax(y_pred) # I should do softmax before the loss
        #log_softmax = tf.nn.log_softmax(y_pred)
        #log_softmax = tf.log(y_pred)
        #log_softmax = K.log(y_pred)
#        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1]+1)
        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1])

#        unpacked = tf.unstack(y_true, axis=-1)
#        y_true = tf.stack(unpacked[:-1], axis=-1)
        focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
        cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax), axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss

def weighted_categorical_focal(weights, alpha=0.25,gamma=2):
    """
    Focal loss implementation
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """

    weights = K.variable(weights)
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        y_pred_softmax = tf.nn.softmax(y_pred)

        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1])

        focal_term = alpha * K.pow(1. - y_pred_softmax, gamma)
        cross_entropy = -K.sum(focal_term * y_true * K.log(y_pred_softmax) * weights, axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        log_softmax = tf.nn.log_softmax(y_pred)
        #log_softmax = tf.log(y_pred)
        #log_softmax = K.log(y_pred)

        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1])

        cross_entropy = -K.sum(y_true * log_softmax * weights , axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss



def categorical_crossentropy():
    """

    Usage:
        loss = categorical_crossentropy()
        model.compile(loss=loss,optimizer='adam')
    """
    
        
    def loss(y_true, y_pred):
        y_pred = K.reshape(y_pred, (-1, K.int_shape(y_pred)[-1]))
        log_softmax = tf.nn.log_softmax(y_pred)

        y_true = K.one_hot(tf.cast(K.flatten(y_true), tf.int32), K.int_shape(y_pred)[-1])

        cross_entropy = -K.sum(y_true * log_softmax, axis=1)
        loss = K.mean(cross_entropy)

        return loss
    
    return loss