# -*- coding: utf-8 -*-


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import tensorflow.keras as keras
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font", family='FangSong')
import numpy as np

# tensorflow版本
print("tf.version:", tf.__version__)

# 获取类名称
# train_path = r'D:\MyFiles\ResearchSubject\Alldatasets3\Alldatasets3\train'
# cwd = train_path + '\\'
# className = os.listdir(train_path)
# for i in range(len(className)):
#     c = re.split("_", className[i])
#     className[i] = c[1]+"_"+c[2]
# print("64个类：", className)
className = ['1708_CM2', '1708_CM', '1736_Y', '1757_4GSY', '1757_4G', '1757_6G', '1757_8GSY', '1757_8G', '1757_BSY',
             '1757_B', '1757_CM2', '1757_CM', '1757_Y', '1757_橱柜门', '1757_面板', '1757_面板2', '1765_4GSY', '1765_4G',
             '1765_6G', '1765_8GSY', '1765_8G', '1765_BSY', '1765_B', '1765_CM2', '1765_CM', '1765_Y', '1765_橱柜门',
             '1765_面板', '1765_面板2', '1770_4GSY', '1770_4G', '1770_6GSY', '1770_6G', '1770_8GSY', '1770_8G', '1770_BSY',
             '1770_B', '1770_CM2', '1770_CM', '1770_Y', '1770_橱柜门', '1770_面板', '1770_面板2', '1771_4GSY', '1771_4G',
             '1771_6G', '1771_8GSY', '1771_8G', '1771_BSY', '1771_B', '1771_CM2', '1771_CM', '1771_Y', '1771_橱柜门',
             '1771_面板', '1771_面板2', '1773_Y', '1782_Y', '1783_B', '1783_Y', '1783_橱柜门', '1783_面板', '1786_Y', '1787_Y']
# 从tfrecord得到数据集
train_tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\gray_tfrecords\train.tfrecords'
valid_tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\gray_tfrecords\valid.tfrecords'
test_tfrecords_file = r'D:\MyFiles\ResearchSubject\Alldatasets3\gray_tfrecords\test.tfrecords'
train_dataset = tf.data.TFRecordDataset(train_tfrecords_file)
valid_dataset = tf.data.TFRecordDataset(valid_tfrecords_file)
# test_dataset = tf.data.TFRecordDataset(test_tfrecords_file)

features = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'img_raw': tf.io.FixedLenFeature([], tf.string)
}

def read_and_decode(example_string):
    features_dic = tf.io.parse_single_example(example_string, features)  # 解析example序列变成的字符串序列
    img = tf.io.decode_raw(features_dic['img_raw'], tf.uint8)
    img = tf.reshape(img, [256, 256, 1])
    label = tf.cast(features_dic['label'], tf.int32)
    return img, label

from MeanStd import MeanStd
mean, std = MeanStd().Getmean_std()

def standardize(image_data):
    image_data = tf.cast(image_data, tf.float32)
    image_data = (image_data - mean)/std
    # 这里灰度图，不用将RGB转成BGR，因为3通道值一样。符合VGG16预训练模型的输入要求（预处理要求）
    # 在VGG16中预处理要求还有一条要进行中心化，但是如果采用VGG16默认的预处理方法，则中心化是以ImageNet数据集而言的，因此不能采用VGG16
    # 默认的预处理方法
    return image_data

def getBatchDataset(dataset, batch=64):
    dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().repeat(1).shuffle(buffer_size=32000)
    dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def getBatchDataset2(dataset, batch=64):
    dataset = dataset.map(read_and_decode, num_parallel_calls=tf.data.experimental.AUTOTUNE).cache().repeat(1).shuffle(buffer_size=32000, seed=12, reshuffle_each_iteration=False)
    dataset = dataset.map(lambda x, y: (standardize(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=batch)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 32
TEST_BATCH_SIZE = 32
train_batch_dataset = getBatchDataset(train_dataset, TRAIN_BATCH_SIZE)
valid_batch_dataset = getBatchDataset2(valid_dataset, VALID_BATCH_SIZE)
# test_batch_dataset = getBatchDataset(test_dataset, TEST_BATCH_SIZE)

# img, label = next(iter(valid_batch_dataset))
# print(img, label)

def reverse_standardize(image_data):
    image_data = np.clip(image_data * std + mean, 0, 255)
    return image_data

def displayImages(dataset):
    plt.figure(figsize=(10, 10))
    # 整个画布（包括各子图在内）的大小是1000×1000
    images, labels = next(iter(dataset))
    # print(labels)
    # 取一个batch的数据
    for i in range(9):
        # img = tf.squeeze(imags[i], 2)
        plt.subplot(3, 3, i + 1)
        plt.imshow(reverse_standardize(images[i]).astype('uint8'))
        plt.title(className[tf.cast(labels[i], tf.int32)])
        plt.axis('off')
    plt.show()

# displayImages(train_batch_dataset)
# displayImages(valid_batch_dataset)

# 创建模型
inputs = keras.Input(shape=(256, 256, 1), name="my_input")
from MyResNet101 import MyResNet101
model = MyResNet101(inputs).CreateMyModel()

# 加载上次结束训练时的权重
model.load_weights(r"doorModels\cp-028-0.003-0.999-1.000-0.178-0.951-0.998.h5", by_name=True)  # 1.3698-0.6014-0.7758
print('successfully loading weights')
model.summary()

# 模型编译
model.compile(optimizer=keras.optimizers.Adam(1e-04),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=["acc",
              keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_acc')])

# 两张图片卷积层的可视化
import cv2.cv2 as cv2
imgPath = r"7_2_1_1_2_1_9_1.jpg"
imgPath2 = r"5_6_1_1_2_1_3_2.jpg"
imgPath3 = r"4_3_2_1.jpg"
imgPath4 = r"4_5_1_4_1_1_6_3.jpg"
imgPath5 = r"4_5_1_4_1_1_17_8.jpg"
imgPath6 = r"17_2_2_3_3_2_11_8.jpg"
imgPath7 = r"14_2_2_3_2_1_11_7.jpg"
def get(imgPath):
    img = tf.io.read_file(imgPath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [256, 256])
    img = tf.image.rgb_to_grayscale(img)
    img = tf.expand_dims(img, 0)
    img = standardize(img)
    return img
imgPathList = [imgPath, imgPath2, imgPath3, imgPath4, imgPath5, imgPath6, imgPath7]
finalImg = None
for imgPath in imgPathList:
  if imgPath == imgPathList[0]:
    finalImg = get(imgPath)
  else:
    img = get(imgPath)
    finalImg = tf.concat((finalImg, img), axis=0)

# finalImg, labels = next(iter(train_batch_dataset))
B = finalImg.shape[0]

layeroutput = model.get_layer("conv5_block3_out")
class_layer = model.get_layer("out1_score")
sub_model = keras.models.Model(inputs=model.input, outputs=[layeroutput.output, class_layer.output])
with tf.GradientTape() as gtape:
    conv_output, predictions = sub_model(finalImg)
    print("predictions:", predictions)
    # print("conv_output:", conv_output)
    c = tf.concat((tf.reshape(tf.range(0, B), [-1, 1]), tf.cast(tf.reshape(np.argmax(predictions, axis=1), [-1, 1]), tf.int32)), axis=1)
    prob = tf.gather_nd(predictions, c)
    print("prob:", prob)
    gradient = gtape.gradient(prob, conv_output)  # 某具体类别与卷积层的梯度
    print("gradient:", gradient)
    # pooled_grads1 = tf.reshape(keras.backend.mean(gradient, axis=(1, 2)), [B, 1, 1, 2048])
    # resnet101最后一个卷积层输出是(None, 8, 8, 2048)
    pooled_grads = tf.reshape(keras.backend.max(gradient, axis=(1, 2)), [B, 1, 1, 2048])
    print("pooled_grads:", pooled_grads)  # 对卷积层的梯度的全局平均代表每个通道的权重
heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)  # 权重与卷积层输出相乘，512层求和（求和或者求平均都可以不影响）
heatmap = np.maximum(heatmap, 0)  # 取heatmap和0中的最大值，即relu激活
print("heatmap:", heatmap)
max_heat = np.max(np.max(heatmap, axis=-1), axis=-1)
print("max_heat：", max_heat)

# 找出heatmap tensor图中各样本的最大值
# b2 = tf.reshape(heatmap, (B, -1))
# index = tf.argmax(b2, axis=-1)
# a1 = tf.reshape(tf.range(B), (B, 1))
# index = tf.cast(tf.transpose(tf.reshape(tf.concat((index // 16, index % 16), axis=0), (2, B)), [1, 0]), tf.int32)
# index = tf.concat((a1, index), axis=-1)
# print("index:", index)
# print("通过索引找出来的max_heat：", tf.gather_nd(heatmap, index))

max_heat = tf.where(max_heat == 0, 1e-10, max_heat)
b2 = tf.reshape(max_heat, [B, 1, 1])
heatmap = tf.divide(heatmap, b2)  # 其实就是(heatmap-min_heat) / (max_heat - min_heat)，这里min_heat==0
# 这里就是将heatmap每个位置的值归一化到0~1
print("heatmap:", heatmap)
# 选择阈值
heatmap = tf.where(heatmap > 0.6, heatmap, 0)
print("heatmap:", heatmap)

plt.figure()
for i in range(B):
    plt.matshow(heatmap[i], cmap='viridis')

plt.figure()
for i in range(B):
    # 这里是在彩色图片上使用cv2进行热力图显示
    original_img = cv2.imread(imgPathList[i])
    original_img = cv2.resize(np.array(original_img), (256, 256), interpolation=cv2.INTER_CUBIC)
    heatmap1 = cv2.resize(np.array(heatmap[i]), (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_CUBIC)

    heatmap1 = np.uint8(255*heatmap1)
    heatmap1 = cv2.applyColorMap(heatmap1, cv2.COLORMAP_JET)
    frame_out = cv2.addWeighted(original_img, 0.5, heatmap1, 0.5, 0)  # cv2.addWeighted就是把两张图片合在一起
    # cv2.imwrite()
    plt.subplot(1, B, i+1)
    plt.imshow(frame_out)
plt.show()
