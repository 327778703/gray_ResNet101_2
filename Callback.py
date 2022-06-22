# -*- coding: utf-8 -*-
# 回调函数

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib
matplotlib.rc("font", family='FangSong')
import numpy as np


class MyCallback():
    def CheckPointCallback(self, path=r'D:\MyFiles\ResearchSubject\door4\doorModels/VGG16-ckpt-512/avg'):
        os.makedirs(path, exist_ok=True)
        CheckPoint_Path = path + '/cp-{epoch:03d}-{loss:.3f}-{acc:.3f}-{top2_acc:.3f}-{val_loss:.3f}-' \
                             '{val_acc:.3f}-{val_top2_acc:.3f}.h5'
        cp_callback = keras.callbacks.ModelCheckpoint(CheckPoint_Path, verbose=2, save_weights_only=True)
        return cp_callback

    def TensorboardCallback(self, log_dir=r'D:\MyFiles\ResearchSubject\door4\doorTensorboard/VGG16-ckpt-512/avg'):
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
        return tensorboard_callback

    def decay1(self, epoch):
        step_size = 1600
        iterations = epoch * 41
        base_lr = 1e-05
        max_lr = 3.65e-04
        cycle = np.floor(1 + iterations / (2 * step_size))
        x = np.abs(iterations / step_size - 2 * cycle + 1)
        lr = base_lr + (max_lr - base_lr) * np.maximum(0, (1 - x)) * x
        return lr

    def decay2(self, epoch):
        initial_lrate = 0.1
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * tf.math.pow(drop, tf.math.floor((1 + epoch) / epochs_drop))
        return lrate
