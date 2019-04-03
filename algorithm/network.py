import tensorflow as tf
import tensorlayer as tl


def encoder(images, reuse=False):

    with tf.variable_scope('EncoderModule', reuse=reuse):
        inputLayer = tl.layers.InputLayer(inputs=images,
                                          name='Input')

        with tf.variable_scope('EncoderNetwork1', reuse=reuse):
            eNet1 = tl.layers.Conv2d(prev_layer=inputLayer,
                                     n_filter=16,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu1')
            eNet1 = tl.layers.BatchNormLayer(prev_layer=eNet1,
                                             name='BatchNorm1')
            eNet1 = tl.layers.DropoutLayer(prev_layer=eNet1,
                                           keep=0.7,
                                           name='Dropout1')

            eNet1 = tl.layers.Conv2d(prev_layer=eNet1,
                                     n_filter=32,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu2')
            eNet1 = tl.layers.BatchNormLayer(prev_layer=eNet1,
                                             name='BatchNorm2')
            eNet1 = tl.layers.DropoutLayer(prev_layer=eNet1,
                                           keep=0.7,
                                           name='Dropout2')

            eNet1 = tl.layers.Conv2d(prev_layer=eNet1,
                                     n_filter=64,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu3')
            eNet1 = tl.layers.BatchNormLayer(prev_layer=eNet1,
                                             name='BatchNorm3')
            eNet1 = tl.layers.DropoutLayer(prev_layer=eNet1,
                                           keep=0.7,
                                           name='Dropout3')

            eNet1 = tl.layers.FlattenLayer(prev_layer=eNet1,
                                           name='Flatten')
            eNet1 = tl.layers.DenseLayer(prev_layer=eNet1,
                                         n_units=100,
                                         name='FullConnection')

        with tf.variable_scope('EncoderNetwork2', reuse=reuse):
            eNet2 = tl.layers.Conv2d(prev_layer=inputLayer,
                                     n_filter=16,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu1')
            eNet2 = tl.layers.BatchNormLayer(prev_layer=eNet2,
                                             name='BatchNorm1')
            eNet2 = tl.layers.DropoutLayer(prev_layer=eNet2,
                                           keep=0.7,
                                           name='Dropout1')

            eNet2 = tl.layers.Conv2d(prev_layer=eNet2,
                                     n_filter=32,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu2')
            eNet2 = tl.layers.BatchNormLayer(prev_layer=eNet2,
                                             name='BatchNorm2')
            eNet2 = tl.layers.DropoutLayer(prev_layer=eNet2,
                                           keep=0.7,
                                           name='Dropout2')

            eNet2 = tl.layers.Conv2d(prev_layer=eNet2,
                                     n_filter=64,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu3')
            eNet2 = tl.layers.BatchNormLayer(prev_layer=eNet2,
                                             name='BatchNorm3')
            eNet2 = tl.layers.DropoutLayer(prev_layer=eNet2,
                                           keep=0.7,
                                           name='Dropout3')

            eNet2 = tl.layers.Conv2d(prev_layer=eNet2,
                                     n_filter=64,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu4')
            eNet2 = tl.layers.BatchNormLayer(prev_layer=eNet2,
                                             name='BatchNorm4')
            eNet2 = tl.layers.DropoutLayer(prev_layer=eNet2,
                                           keep=0.7,
                                           name='Dropout4')

            eNet2 = tl.layers.FlattenLayer(prev_layer=eNet2,
                                           name='Flatten')
            eNet2 = tl.layers.DenseLayer(prev_layer=eNet2,
                                         n_units=100,
                                         name='FullConnection')

        with tf.variable_scope('EncoderNetwork3', reuse=reuse):
            eNet3 = tl.layers.Conv2d(prev_layer=inputLayer,
                                     n_filter=32,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu1')
            eNet3 = tl.layers.BatchNormLayer(prev_layer=eNet3,
                                             name='BatchNorm1')
            eNet3 = tl.layers.DropoutLayer(prev_layer=eNet3,
                                           keep=0.7,
                                           name='Dropout1')

            eNet3 = tl.layers.Conv2d(prev_layer=eNet3,
                                     n_filter=64,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu2')
            eNet3 = tl.layers.BatchNormLayer(prev_layer=eNet3,
                                             name='BatchNorm2')
            eNet3 = tl.layers.DropoutLayer(prev_layer=eNet3,
                                           keep=0.7,
                                           name='Dropout2')

            eNet3 = tl.layers.Conv2d(prev_layer=eNet3,
                                     n_filter=128,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu3')
            eNet3 = tl.layers.BatchNormLayer(prev_layer=eNet3,
                                             name='BatchNorm3')
            eNet3 = tl.layers.DropoutLayer(prev_layer=eNet3,
                                           keep=0.7,
                                           name='Dropout3')

            eNet3 = tl.layers.Conv2d(prev_layer=eNet3,
                                     n_filter=128,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu4')
            eNet3 = tl.layers.BatchNormLayer(prev_layer=eNet3,
                                             name='BatchNorm4')
            eNet3 = tl.layers.DropoutLayer(prev_layer=eNet3,
                                           keep=0.7,
                                           name='Dropout4')

            eNet3 = tl.layers.Conv2d(prev_layer=eNet3,
                                     n_filter=128,
                                     filter_size=(4, 4),
                                     strides=(2, 2),
                                     act=tf.nn.relu,
                                     name='ConvolutionRelu5')
            eNet3 = tl.layers.BatchNormLayer(prev_layer=eNet3,
                                             name='BatchNorm5')
            eNet3 = tl.layers.DropoutLayer(prev_layer=eNet3,
                                           keep=0.7,
                                           name='Dropout5')

            eNet3 = tl.layers.FlattenLayer(prev_layer=eNet3,
                                           name='Flatten')
            eNet3 = tl.layers.DenseLayer(prev_layer=eNet3,
                                         n_units=100,
                                         name='FullConnection')

        concatLayer = tl.layers.ConcatLayer(prev_layer=[eNet1, eNet2, eNet3],
                                            concat_dim=1,
                                            name='ConcatFullConnection')

        r1 = eNet1.outputs
        r2 = eNet2.outputs
        r3 = eNet3.outputs
        R = concatLayer.outputs
        return r1, eNet1, r2, eNet2, r3, eNet3, R, concatLayer


def decoder(embeddings, reuse=False):

    with tf.variable_scope('DecoderModule', reuse=reuse):
        inputLayer = tl.layers.InputLayer(inputs=embeddings,
                                          name='Input')
        inputLayer = tl.layers.ReshapeLayer(prev_layer=inputLayer,
                                            shape=(-1, 10, 10, 3),
                                            name='Reshape')

        with tf.variable_scope('DecoderNetwork', reuse=reuse):
            dNet = tl.layers.DeConv2d(prev_layer=inputLayer,
                                      n_filter=128,
                                      filter_size=(4, 4),
                                      strides=(2, 2),
                                      act=tf.nn.relu,
                                      name='DeconvolutionRelu1')

            dNet = tl.layers.DeConv2d(prev_layer=dNet,
                                      n_filter=128,
                                      filter_size=(4, 4),
                                      strides=(2, 2),
                                      act=tf.nn.relu,
                                      name='DeconvolutionRelu2')

            dNet = tl.layers.DeConv2d(prev_layer=dNet,
                                      n_filter=64,
                                      filter_size=(4, 4),
                                      strides=(2, 2),
                                      act=tf.nn.relu,
                                      name='DeconvolutionRelu3')

            dNet = tl.layers.DeConv2d(prev_layer=dNet,
                                      n_filter=32,
                                      filter_size=(4, 4),
                                      strides=(2, 2),
                                      act=tf.nn.relu,
                                      name='DeconvolutionRelu4')

            dNet = tl.layers.DeConv2d(prev_layer=dNet,
                                      n_filter=3,
                                      filter_size=(4, 4),
                                      strides=(2, 2),
                                      act=tf.nn.relu,
                                      name='DeconvolutionRelu5')

        I_ = dNet.outputs
        I_ = tf.image.resize_images(I_, size=(128, 128))
        return I_, dNet


def attributes(part1, part2, reuse=False):

    with tf.variable_scope('SupervisedAttributesModule', reuse=reuse):
        inputLayer1 = tl.layers.InputLayer(inputs=part1,
                                           name='Input1')
        inputLayer2 = tl.layers.InputLayer(inputs=part2,
                                           name='Input2')

        with tf.variable_scope('AttributesNetwork1', reuse=reuse):
            aNet1 = tl.layers.DenseLayer(prev_layer=inputLayer1,
                                         n_units=200,  # this num is just placeholder
                                         act=tf.nn.softmax,
                                         name='FullConnection')

        with tf.variable_scope('AttributesNetwork2', reuse=reuse):
            aNet2 = tl.layers.DenseLayer(prev_layer=inputLayer2,
                                         n_units=300,  # this num is just placeholder
                                         act=tf.nn.softmax,
                                         name='FullConnection')

        label1_ = aNet1.outputs
        label2_ = aNet2.outputs
        return label1_, aNet1, label2_, aNet2


def adversarialPrediction(embeddings, reuse=False):

    with tf.variable_scope('MultiIndependentBlock', reuse=reuse):
        inputLayer = tl.layers.InputLayer(inputs=embeddings,
                                          name='Input')

        with tf.variable_scope('AdversarialPredictionNetwork1', reuse=reuse):
            pNet1 = tl.layers.DenseLayer(prev_layer=inputLayer,
                                         n_units=256,
                                         name='FullConnection1')

            pNet1 = tl.layers.DenseLayer(prev_layer=pNet1,
                                         n_units=128,
                                         name='FullConnection2')

            pNet1 = tl.layers.DenseLayer(prev_layer=pNet1,
                                         n_units=100,
                                         name='FullConnection3')

        with tf.variable_scope('AdversarialPredictionNetwork2', reuse=reuse):
            pNet2 = tl.layers.DenseLayer(prev_layer=inputLayer,
                                         n_units=256,
                                         name='FullConnection1')

            pNet2 = tl.layers.DenseLayer(prev_layer=pNet2,
                                         n_units=128,
                                         name='FullConnection2')

            pNet2 = tl.layers.DenseLayer(prev_layer=pNet2,
                                         n_units=100,
                                         name='FullConnection3')

        with tf.variable_scope('AdversarialPredictionNetwork3', reuse=reuse):
            pNet3 = tl.layers.DenseLayer(prev_layer=inputLayer,
                                         n_units=256,
                                         name='FullConnection1')

            pNet3 = tl.layers.DenseLayer(prev_layer=pNet3,
                                         n_units=128,
                                         name='FullConnection2')

            pNet3 = tl.layers.DenseLayer(prev_layer=pNet3,
                                         n_units=100,
                                         name='FullConnection3')

        part1 = pNet1.outputs
        part2 = pNet2.outputs
        part3 = pNet3.outputs
        return part1, pNet1, part2, pNet2, part3, pNet3
