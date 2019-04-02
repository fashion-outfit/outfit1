import numpy as np
import tensorflow as tf
import tensorlayer as tl

import network

I = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
I_ = tf.placeholder(tf.float32, shape=(None, 128, 128, 3))
label1 = tf.placeholder(tf.float32, shape=(None, 200))
label2 = tf.placeholder(tf.float32, shape=(None, 300))

r1, r2, r3, R = network.encoder(I, reuse=False)
I_ = network.decoder(R, reuse=False)
part1, part2, part3 = network.adversarialPrediction(R, reuse=False)
label1, label2 = network.attributes(part1, part2, reuse=False)

with tf.Session() as sess:
    pass
