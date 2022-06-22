# -*- coding: utf-8 -*-
# ResNet101，三个256全连接层，无Dropout

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')


class MyResNet101():
    def __init__(self, inputs):
        self.inputs = inputs

    def CreateMyModel(self):
        myConv1 = keras.layers.Conv2D(3, (1, 1), padding='same', use_bias=False, kernel_initializer=keras.initializers.ones(),
                            trainable=False, name='myConv1')(self.inputs)
        base_model = keras.applications.ResNet101(input_tensor=myConv1, input_shape=(256, 256, 3), include_top=False,
                                                  weights=None)
        base_model.trainable = False
        out1_score = keras.layers.Dense(64, name='out1_score')
        out1_output = keras.layers.Activation('softmax', name='out1')

        x = base_model.output
        x = keras.layers.GlobalAvgPool2D(name='conv5_block3_out_GAP')(x)
        x = out1_score(x)
        self.outputs = out1_output(x)
        return keras.Model(self.inputs, self.outputs)

# inputs = keras.Input(shape=(256, 256, 1), name="images")
# b = MyResNet101(inputs).CreateMyModel()
# b.summary()
