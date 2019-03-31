import numpy as np
import tensorflow as tf
import tensorlayer as tl


def C(x, n_filter, reuse=False):
    with tf.variable_scope('convolution', reuse=reuse):
        x = tl.layers.Conv2d(x, n_filter, (4, 4), (2, 2))
        x = tl.layers.DownSampling2dLayer(x, 2)
        x = tl.layers.BatchNormLayer(x)
        x = tf.nn.relu(x, name='relu')
        return x


def CD(x, n_filter, reuse=False):
    with tf.variable_scope('convolution dropout', reuse=reuse):
        x = tl.layers.Conv2d(x, n_filter, (4, 4), (2, 2))
        x = tl.layers.DownSampling2dLayer(x, 2)
        x = tl.layers.BatchNormLayer(x)
        x = tl.layers.DropoutLayer(x, 0.7)
        x = tf.nn.relu(x, name='relu')
        return x


def DC(x, n_filter, reuse=False):
    with tf.variable_scope('deconvolution', reuse=reuse):
        x = tl.layers.DeConv2d(x, n_filter)
        x = tl.layers.UpSampling2dLayer(x, 2)
        x = tf.nn.relu(x, name='relu')
        return x


def FC(x, n_units, reuse=False):
    with tf.variable_scope('full connection', reuse=reuse):
        x = tl.layers.DenseLayer(x, n_units)
        return x


def ENet_type1(x, reuse=False):
    with tf.variable_scope('encoder network type 1', reuse=reuse):
        x = CD(x, 16, reuse)
        x = CD(x, 32, reuse)
        x = CD(x, 64, reuse)
        x = FC(x, 100, reuse)
        return x


def ENet_type2(x, reuse=False):
    with tf.variable_scope('encoder network type 2', reuse=reuse):
        x = CD(x, 16, reuse)
        x = CD(x, 32, reuse)
        x = CD(x, 64, reuse)
        x = CD(x, 64, reuse)
        x = FC(x, 100, reuse)
        return x


def ENet_type3(x, reuse=False):
    with tf.variable_scope('encoder network type 3', reuse=reuse):
        x = CD(x, 32, reuse)
        x = CD(x, 64, reuse)
        x = CD(x, 128, reuse)
        x = CD(x, 128, reuse)
        x = CD(x, 128, reuse)
        x = FC(x, 100, reuse)
        return x


def DNet(x, reuse=False):
    with tf.variable_scope('decoder network', reuse=reuse):
        x = DC(x, 128, reuse)
        x = DC(x, 128, reuse)
        x = DC(x, 64, reuse)
        x = DC(x, 32, reuse)
        x = DC(x, 1, reuse)
        return x


def ANet_type1(x, reuse=False):
    with tf.variable_scope('attributes network type 1', reuse=reuse):
        G = FC(x, 64, reuse)
        G = FC(G, 32, reuse)
        G = FC(G, 15, reuse)
        D = FC(x, 256, reuse)
        D = FC(D, 128, reuse)
        D = FC(D, 64, reuse)
        D = FC(x, 1, reuse)
        return G, D


def ANet_type2(x, reuse=False):
    with tf.variable_scope('attributes network type 2', reuse=reuse):
        x = C(x, 64, reuse)
        x = CD(x, 64, reuse)
        x = CD(x, 32, reuse)
        x = CD(x, 1, reuse)
        return x


def PNet(x, reuse=False):
    with tf.variable_scope('adversarial prediction network', reuse=reuse):
        x = FC(x, 256, reuse)
        x = FC(x, 128, reuse)
        x = FC(x, 100, reuse)
        return x
