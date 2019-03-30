import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dropout, MaxPool2D, Dense, Activation, BatchNormalization
from keras.optimizers import Adam
from keras.utils import np_utils
from PIL import Image
import os
import random
from keras import backend as K
from keras import regularizers


class Training():

    def __init__(self, batch_size, epochs, categories, train_folder, test_folder, model_name):
        self.batch_size = batch_size
        self.epochs = epochs
        self.categories = categories
        self.train_folder = train_folder
        self.test_folder = test_folder
        self.model_name = model_name
        self.shape1 = 100
        self.shape2 = 100

    def read_train_images(self, folder):
        # 返回图像和标签
        img_list = []
        lable_list = []
        for file in os.listdir(folder):
            img = Image.open(folder + file)
            img = np.array(img).reshape(self.shape1, self.shape2, 1)
            img_list.append(img)
            lable_list.append(int(file.split('_')[0]))
        return img_list, lable_list

    def train(self):
        train_img_list, train_lable_list = self.read_train_images(folder=self.train_folder)

        test_img_list, test_lable_list = self.read_train_images(folder=self.test_folder)
        test_img_list, test_lable_list = np.array(test_img_list).astype('float32') / 255, np.array(test_lable_list)

        index = [i for i in range(len(train_img_list))]
        random.shuffle(index)
        for i in range(len(train_img_list)):
            j = index[i]
            train_img_list[i], train_img_list[j] = train_img_list[j], train_img_list[i]
            train_lable_list[i], train_lable_list[j] = train_lable_list[j], train_lable_list[i]

        train_img_list = np.array(train_img_list).astype('float32') / 255
        train_lable_list = np.array(train_lable_list)

        train_lable_list = np_utils.to_categorical(train_lable_list, self.categories)
        test_lable_list = np_utils.to_categorical(test_lable_list, self.categories)

        model = Sequential()

        model.add(Convolution2D(
            filters=32,
            kernel_size=(3, 3),
            padding='valid',
            input_shape=(self.shape1, self.shape2, 1),
        ))
        model.add(BatchNormalization())

        model.add(Activation('relu'))

        model.add(Convolution2D(
            filters=32,
            kernel_size=(3, 3),

        ))
        model.add(BatchNormalization())

        model.add(Activation('relu'))
        model.add(MaxPool2D(
            pool_size=(2, 2),
            strides=(2, 2),
            padding='valid',
        ))
        model.add(Dropout(0.3))

        model.add(Flatten())

        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.3))

        model.add(Dense(self.categories,
                        kernel_regularizer=regularizers.l2(0.01),
                        # activity_regularizer=regularizers.l1(0.001)
                        ))
        model.add(BatchNormalization())
        model.add(Activation('softmax'))



        adam = Adam(lr=0.001)

        model.compile(
            optimizer=adam,
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

        model.fit(
            x=train_img_list,
            y=train_lable_list,
            epochs=self.epochs,
            validation_split=0.33,
            batch_size=self.batch_size,
            shuffle=True,
            validation_data=(test_img_list, test_lable_list)
        )

        model.save(self.model_name)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train = Training(batch_size=32, epochs=5, categories=5,
                     train_folder='Gesture_train/', test_folder='Gesture_predict/',
                     model_name='2')
    train.train()
    K.clear_session()