import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm, trange

import network

with tf.Session() as sess:

    # placeholders
    I = tf.placeholder(dtype=tf.float32,
                       shape=(None, 128, 128, 3),
                       name='Images')
    I_ = tf.placeholder(dtype=tf.float32,
                        shape=(None, 128, 128, 3),
                        name='DecodedImages')
    label1 = tf.placeholder(dtype=tf.float32,
                            shape=(None, 200),
                            name='StyleLabels')
    label2 = tf.placeholder(dtype=tf.float32,
                            shape=(None, 300),
                            name='ShapeLabels')

    # network outputs
    r1, eNet1, r2, eNet2, r3, eNet3, R, concatLayer = network.encoder(images=I,
                                                                      reuse=False)
    I_, dNet = network.decoder(embeddings=R,
                               reuse=False)
    part1, pNet1, part2, pNet2, part3, pNet3 = network.adversarialPrediction(embeddings=R,
                                                                             reuse=False)
    label1_, aNet1, label2_, aNet2 = network.attributes(part1=part1,
                                                        part2=part2,
                                                        reuse=False)

    # losses
    # auto-encoder loss
    autoEncoderLoss = tl.cost.mean_squared_error(output=I,
                                                 target=I_,
                                                 name='EncoderLoss')

    # predicting loss
    predicting1Loss = tl.cost.mean_squared_error(output=part1,
                                                 target=r1,
                                                 name='PredictingLossPartOne')
    predicting2Loss = tl.cost.mean_squared_error(output=part2,
                                                 target=r2,
                                                 name='PredictingLossPartTwo')
    predicting3Loss = tl.cost.mean_squared_error(output=part3,
                                                 target=r3,
                                                 name='PredictingLossPartThree')
    predictingLoss = predicting1Loss + predicting2Loss + predicting3Loss

    # encoding loss
    encoding1Loss = -1*tl.cost.mean_squared_error(output=r1,
                                                  target=part1,
                                                  name='EncodingLossPartOne')
    encoding2Loss = -1*tl.cost.mean_squared_error(output=r2,
                                                  target=part2,
                                                  name='EncodingLossPartTwo')
    encoding3Loss = -1*tl.cost.mean_squared_error(output=r3,
                                                  target=part3,
                                                  name='EncodingLossPartThree')
    encodingLoss = encoding1Loss + encoding2Loss + encoding3Loss

    # optimizers
    # auto-encoder optimizer
    autoEncoderOptimizer = tl.optimizers.AMSGrad(learning_rate=0.01,
                                                 beta1=0.9,
                                                 beta2=0.999,
                                                 name='EncoderOptimizer').minimize(loss=autoEncoderLoss,
                                                                                   var_list=dNet.all_params)

    # predicting optimizer
    predictingParams = pNet1.all_params + pNet2.all_params + pNet3.all_params
    predictingOptimizer = tl.optimizers.AMSGrad(learning_rate=0.01,
                                                beta1=0.9,
                                                beta2=0.999,
                                                name='PredictingOptimizer').minimize(loss=predictingLoss,
                                                                                     var_list=predictingParams)

    # encoding optimizer
    encodingParams = eNet1.all_params + eNet2.all_params + eNet3.all_params
    encodingOptimizer = tl.optimizers.AMSGrad(learning_rate=0.01,
                                              beta1=0.9,
                                              beta2=0.999,
                                              name='EncodingOptimizer').minimize(loss=encodingLoss,
                                                                                 var_list=encodingParams)

    tl.layers.initialize_global_variables(sess)
    dNet.print_layers()
    dNet.print_params()

    for epoch in trange(1000):
        pass
