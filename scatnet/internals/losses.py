from __future__ import print_function, unicode_literals, absolute_import, division
from six.moves import range, zip, map, reduce, filter
import tensorflow as tf
from ..utils import _raise, backend_channels_last

import numpy as np
import keras.backend as K



def _mean_or_not(mean):
    # return (lambda x: K.mean(x,axis=(-1 if backend_channels_last() else 1))) if mean else (lambda x: x)
    # Keras also only averages over axis=-1, see https://github.com/keras-team/keras/blob/master/keras/losses.py
    return (lambda x: K.mean(x,axis=-1)) if mean else (lambda x: x)


def loss_ce(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def ce(y_true, y_pred):
            n = K.shape(y_true)[-1]
            y_true=(y_true-tf.reduce_min(y_true))/(tf.reduce_max(y_true)-tf.reduce_min(y_true))
            y_pred=tf.divide(y_pred[...,:n]-tf.reduce_min(y_pred[...,:n]),(tf.reduce_max(y_pred[...,:n])-tf.reduce_min(y_pred[...,:n])))
            return R(K.binary_crossentropy(y_true, y_pred[...,:n]), axis=-1)
        return ce
    else:
        def ce(y_true, y_pred):
            n = K.shape(y_true)[1]
            y_true=(y_true-tf.reduce_min(y_true))/(tf.reduce_max(y_true)-tf.reduce_min(y_true))
            y_pred=tf.divide(y_pred[...,:n,...]-tf.reduce_min(y_pred[...,:n,...]),(tf.reduce_max(y_pred[...,:n,...])-tf.reduce_min(y_pred[...,:n,...])))
            return R(K.binary_crossentropy(y_true, y_pred[...,:n,...]))
        return ce

def loss_mae(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mae(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.abs(y_pred[...,:n] - y_true))
        return mae
    else:
        def mae(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.abs(y_pred[:,:n,...] - y_true))
        return mae 

def loss_mse(mean=True):
    R = _mean_or_not(mean)
    if backend_channels_last():
        def mse(y_true, y_pred):
            n = K.shape(y_true)[-1]
            return R(K.square(y_pred[...,:n] - y_true))
        return mse
    else:
        def mse(y_true, y_pred):
            n = K.shape(y_true)[1]
            return R(K.square(y_pred[:,:n,...] - y_true))
        return mse


def loss_thresh_weighted_decay(loss_per_pixel, thresh, w1, w2, alpha):
    def _loss(y_true, y_pred):
        val = loss_per_pixel(y_true, y_pred)
        k1 = alpha * w1 + (1 - alpha)
        k2 = alpha * w2 + (1 - alpha)
        return K.mean(K.tf.where(K.tf.less_equal(y_true, thresh), k1 * val, k2 * val),
                      axis=(-1 if backend_channels_last() else 1))
    return _loss