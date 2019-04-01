import numpy as np
import tensorflow as tf
import tensorlayer as tl
from network import encoder, decoder

I = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
I_ = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
label1 = tf.placeholder(tf.float32, shape=(None, 1))
label2 = tf.placeholder(tf.float32, shape=(None, 1))

I = tl.layers.InputLayer(I)
R = encoder(I, reuse=False)
I_ = decoder(R, reuse=False)

with tf.Session() as sess:
    pass
