import os
import cv2
from PIL import Image
import numpy as np


path = '手势合集/'
Gesturetype = ['666', 'stop', 'yech', 'ok']

def FileRename(path, Gesture_set):
    num_count = 0
    for ges in Gesture_set:
        file_count = 0
        phones = os.listdir(path+ges)
        # print(phones)
        for phone in phones:
            file_count += 1
            print(phone)
            os.rename(path + ges + '/' + phone, path + ges + '/' + str(num_count) + '_' + 'ges{}'.format(file_count) + '.jpg')
        num_count += 1


def File_to_train_folder(path, Gesture_set):
    for ges in Gesture_set:
        phones = os.listdir(path + ges)
        for phone in phones:
            img_open = Image.open(path + ges +'/' + phone)
            # img_open = img_open.resize((150, 150), Image.BILINEAR)
            img_open.save('D:\pycharm-work\opencv-test\CNN_Gesture_categorizer\Gesture_train\\' + phone)


File_to_train_folder(path, Gesturetype)

