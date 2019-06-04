import os
import cv2
from PIL import Image
import numpy as np


path = 'D:\pycharm-work\opencv-visualization_image\CNN_Gesture_categorizer\Gesture_predict\\'
Gesturetype = ['666', 'stop', 'yech', 'ok', 'one']

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
            img_open.save('D:\pycharm-work\opencv-visualization_image\CNN_Gesture_categorizer\Gesture_train\\' + phone)

def lable_rename(path):
    for i in os.listdir(path):
        a = i.split('_')[0]
        os.rename(path + i, path + str(int(i.split('_')[0])-1) + '_' + i.split('_')[1])





lable_rename(path=path)
# File_to_train_folder(path, Gesturetype)

