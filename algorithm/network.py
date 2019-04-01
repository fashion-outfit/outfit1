import numpy as np
import tensorflow as tf
import tensorlayer as tl


def C(x, n_filter, reuse=False):
    with tf.variable_scope('Convolution', reuse=reuse):
        x = tl.layers.Conv2d(x, n_filter, (4, 4), (2, 2))
        x = tl.layers.DownSampling2dLayer(x, 2)
        x = tl.layers.BatchNormLayer(x)
        x = tf.nn.relu(x)
        return x


def CD(x, n_filter, reuse=False):
    with tf.variable_scope('ConvolutionDropout', reuse=reuse):
        x = tl.layers.Conv2d(x, n_filter, (4, 4), (2, 2))
        x = tl.layers.DownSampling2dLayer(x, (2, 2))
        x = tl.layers.BatchNormLayer(x)
        x = tl.layers.DropoutLayer(x, 0.7)
        x = tf.nn.relu(x)
        return x


def DC(x, n_filter, reuse=False):
    with tf.variable_scope('Deconvolution', reuse=reuse):
        x = tl.layers.DeConv2d(x, n_filter)
        x = tl.layers.UpSampling2dLayer(x, (2, 2))
        x = tf.nn.relu(x)
        return x


def FC(x, n_units, reuse=False):
    with tf.variable_scope('FullConnection', reuse=reuse):
        x = tl.layers.DenseLayer(x, n_units)
        return x


def ENet_type1(x, reuse=False):
    with tf.variable_scope('EncoderNetworkType1', reuse=reuse):
        x = CD(x, 16, reuse=reuse)
        x = CD(x, 32, reuse=reuse)
        x = CD(x, 64, reuse=reuse)
        x = FC(x, 100, reuse=reuse)
        return x


def ENet_type2(x, reuse=False):
    with tf.variable_scope('EncoderNetworkType2', reuse=reuse):
        x = CD(x, 16, reuse=reuse)
        x = CD(x, 32, reuse=reuse)
        x = CD(x, 64, reuse=reuse)
        x = CD(x, 64, reuse=reuse)
        x = FC(x, 100, reuse=reuse)
        return x


def ENet_type3(x, reuse=False):
    with tf.variable_scope('EncoderNetworkType3', reuse=reuse):
        x = CD(x, 32, reuse=reuse)
        x = CD(x, 64, reuse=reuse)
        x = CD(x, 128, reuse=reuse)
        x = CD(x, 128, reuse=reuse)
        x = CD(x, 128, reuse=reuse)
        x = FC(x, 100, reuse=reuse)
        return x


def DNet(x, reuse=False):
    with tf.variable_scope('DecoderNetwork', reuse=reuse):
        x = DC(x, 128, reuse=reuse)
        x = DC(x, 128, reuse=reuse)
        x = DC(x, 64, reuse=reuse)
        x = DC(x, 32, reuse=reuse)
        x = DC(x, 1, reuse=reuse)
        return x


def ANet_type1(x, reuse=False):
    with tf.variable_scope('AttributesNetworkType1', reuse=reuse):
        G = FC(x, 64, reuse=reuse)
        G = FC(G, 32, reuse=reuse)
        G = FC(G, 15, reuse=reuse)
        D = FC(x, 256, reuse=reuse)
        D = FC(D, 128, reuse=reuse)
        D = FC(D, 64, reuse=reuse)
        D = FC(x, 1, reuse=reuse)
        return G, D


def ANet_type2(x, reuse=False):
    with tf.variable_scope('AttributesNetworkType2', reuse=reuse):
        x = C(x, 64, reuse=reuse)
        x = CD(x, 64, reuse=reuse)
        x = CD(x, 32, reuse=reuse)
        x = CD(x, 1, reuse=reuse)
        return x


def PNet_1(x, reuse=False):
    with tf.variable_scope('AdversarialPredictionNetwork1', reuse=reuse):
        x = FC(x, 256, reuse=reuse)
        x = FC(x, 128, reuse=reuse)
        x = FC(x, 100, reuse=reuse)
        return x


def PNet_2(x, reuse=False):
    with tf.variable_scope('AdversarialPredictionNetwork2', reuse=reuse):
        x = FC(x, 256, reuse=reuse)
        x = FC(x, 128, reuse=reuse)
        x = FC(x, 100, reuse=reuse)
        return x


def PNet_3(x, reuse=False):
    with tf.variable_scope('AdversarialPredictionNetwork3', reuse=reuse):
        x = FC(x, 256, reuse=reuse)
        x = FC(x, 128, reuse=reuse)
        x = FC(x, 100, reuse=reuse)
        return x


def encoder(I, reuse=False):
    with tf.variable_scope('Encoder', reuse=reuse):
        r1 = ENet_type1(I, reuse=reuse)
        r2 = ENet_type2(I, reuse=reuse)
        r3 = ENet_type3(I, reuse=reuse)
        R = tf.concat([r1, r2, r3], 0)
        return R


def decoder(R, reuse=False):
    with tf.variable_scope('Decoder', reuse=reuse):
        I_ = DNet(R, reuse=reuse)
        return I_


def attributes(part1, part2, reuse=False):
    with tf.variable_scope('Attributes', reuse=reuse):
        label1 = ANet_type1(part1, reuse=reuse)
        label2 = ANet_type2(part2, reuse=reuse)
        return label1, label2


def adversarial_prediction(R, reuse=False):
    with tf.variable_scope('AdversarialPrediction', reuse=reuse):
        part1 = PNet_1(R, reuse=reuse)
        part2 = PNet_2(R, reuse=reuse)
        part3 = PNet_3(R, reuse=reuse)
        return part1, part2, part3
