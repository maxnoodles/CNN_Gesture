import numpy as np
import pydot
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPool2D, Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image
import os
import random
from keras import backend as K
from keras import regularizers
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
import re

from sklearn.metrics import confusion_matrix     # 混淆矩阵

import itertools


class Training():

    def __init__(self, batch_size, epochs, categories, train_folder, test_folder, model_name, type):
        # 批处理数量
        self.batch_size = batch_size
        # 训练次数
        self.epochs = epochs
        # 分类个数
        self.categories = categories
        # 训练集文件夹
        self.train_folder = train_folder
        # 测试集文件夹
        self.test_folder = test_folder
        # 模型名
        self.model_name = model_name
        # 手势类型
        self.type = type
        # 图片尺寸
        self.shape1 = 100
        self.shape2 = 100

    def read_train_images(self, folder):
        """从文件夹中读取图像和标签，放回图像列表和标签列表"""
        img_list = []
        lable_list = []
        for file in os.listdir(folder):
            img = Image.open(folder + file)
            img = np.array(img).reshape(self.shape1, self.shape2, 1)
            img_list.append(img)
            lable_list.append(int(file.split('_')[1][0]))
        return img_list, lable_list

    def train(self):
        train_img_list, train_lable_list = self.read_train_images(folder=self.train_folder)

        test_img_list, test_lable_list = self.read_train_images(folder=self.test_folder)
        # 测试集图像归一化，并将图像和标签转化为numpy中的array格式
        test_img_list, test_lable_list = np.array(test_img_list).astype('float32') / 255, np.array(test_lable_list)

        # 手动打乱图像顺序
        # index = [i for i in range(len(train_img_list))]
        # random.shuffle(index)
        # for i in range(len(train_img_list)):
        #     j = index[i]
        #     train_img_list[i], train_img_list[j] = train_img_list[j], train_img_list[i]
        #     train_lable_list[i], train_lable_list[j] = train_lable_list[j], train_lable_list[i]

        # 训练集图像归一化
        train_img_list = np.array(train_img_list).astype('float32') / 255
        train_lable_list = np.array(train_lable_list)

        # 训练集和测试集的标签转化独热编码
        train_lable_list = np_utils.to_categorical(train_lable_list, self.categories)
        test_lable_list = np_utils.to_categorical(test_lable_list, self.categories)

        # keras序贯模型，不分叉
        model = Sequential()
        # 卷积层1，个数32，尺寸3*3，填充方式valid，步长默认1*1
        model.add(Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            padding='valid',
            input_shape=(self.shape1, self.shape2, 1),
            name = 'conv2d_1'
        ))
        # 批规范化处理
        model.add(BatchNormalization())
        # 激活函数relu
        model.add(Activation('relu', name='activation_1'))
        # 卷积层2，个数32，尺寸3*3，填充方式valid，步长默认1*1
        model.add(Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            name='conv2d_2'
        ))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_2'))

        # 池化层，尺寸2*2，步长为2*2，填充方式为valid
        model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            name='max_pooling2d_1'
        ))
        # dropout层，失活系数0.5
        model.add(Dropout(0.5, name='dropout_1'))
        # 转化为一维矩阵
        model.add(Flatten(name='flatten_1'))
        # 全连接层，128个神经元
        model.add(Dense(128, name='dense_1'))
        model.add(BatchNormalization())
        model.add(Activation('relu', name='activation_3'))
        model.add(Dropout(0.5, name='dropout_2'))

        # 分类层，L2正则优化
        model.add(Dense(self.categories,
                        kernel_regularizer=regularizers.l2(0.01),
                        name='dense_2'))
        # 分类层，激活函数sofomax
        model.add(Activation('softmax', name='activation_4'))

        # 自适应学习率的算法adam
        adam = Adam(lr=0.001)

        # 配置模型，优化器为adam，损失函数为，指标为准确率
        model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        # 打印出模型概况
        model.summary()
        # 返回包含模型配置信息的Python字典
        model.get_config()
        # 保存模型结构图
        plot_model(model, to_file='model.png', show_shapes=True)
        # 拟合模型
        hist = model.fit(
            x=train_img_list,
            y=train_lable_list,
            epochs=self.epochs,
            validation_split=0.33,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(test_img_list, test_lable_list)
        )
        # 模型可视化参数
        pred_y = model.predict(test_img_list)
        pred_label = np.argmax(pred_y, axis=1)
        true_label = np.argmax(test_lable_list, axis=1)

        # 混淆矩阵数据
        confusion_mat = confusion_matrix(true_label, pred_label)
        # 混淆矩阵可视化
        self.plot_sonfusion_matrix(confusion_mat, classes=range(5))

        self.visualizeHis(hist)
        # 保存模型
        model.save(self.model_name)
        K.clear_session()

    def plot_sonfusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        # 混淆矩阵可视化
        plt.figure(1, figsize=(7, 5))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()

        Gesturetype = self.type

        tick_marks = np.arange(len(classes))
        # plt.xticks(tick_marks, classes, rotation=45)
        plt.xticks(tick_marks, Gesturetype, rotation=45)
        # plt.yticks(tick_marks, classes)
        plt.yticks(tick_marks, Gesturetype)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predict label')

    def visualizeHis(self, hist):
        # 损失函数和准确率可视化

        train_loss = hist.history['loss']
        val_loss = hist.history['val_loss']
        train_acc = hist.history['acc']
        val_acc = hist.history['val_acc']
        xc = range(self.epochs)

        plt.figure(2, figsize=(7, 5))
        plt.plot(xc, train_loss)
        plt.plot(xc, val_loss)
        plt.xlabel('num of Epochs')
        plt.ylabel('loss')
        plt.title('train_loss vs val_loss')
        plt.grid(True)
        plt.legend(['train', 'val'])

        plt.figure(3, figsize=(7, 5))
        plt.plot(xc, train_acc)
        plt.plot(xc, val_acc)
        plt.xlabel('num of Epochs')
        plt.ylabel('accuracy')
        plt.title('train_acc vs val_acc')
        plt.grid(True)
        plt.legend(['train', 'val'], loc=4)

        plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train = Training(batch_size=32, epochs=20, categories=5,
                     train_folder='Gesture_train/', test_folder='Gesture_predict/',
                     model_name='Gesture_2.h5')
    train.train()
    K.clear_session()
