import cv2
import numpy as np
from keras.models import load_model
from training import Training
import os
from keras import backend
import time
import random


class Gesture():

    def __init__(self, train_path, predict_path, gesture, train_model):
        self.blurValue = 5
        self.bgSubThreshold = 36
        self.train_path = train_path
        self.predict_path = predict_path
        self.threshold = 60
        self.gesture = gesture
        self.train_model = train_model
        self.skinkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.x1 = 380
        self.y1 = 60
        self.x2 = 640
        self.y2 = 350

    def collect_gesture(self, capture, ges, photo_num):
        photo_num = photo_num
        vedeo = False
        predict = False
        count = 0
        # 读取默认摄像头
        cap = cv2.VideoCapture(capture)
        # 设置捕捉模式
        cap.set(10, 200)
        # 背景减法创建及初始化
        bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)

        while True:
            # 读取视频帧
            ret, frame = cap.read()
            # 镜像转换
            frame = cv2.flip(frame, 1)

            cv2.imshow('Original', frame)
            # 双边滤波
            frame = cv2.bilateralFilter(frame, 5, 50,100)

            # 绘制矩形，第一个为左上角坐标(x,y），第二个为右下角坐标
            # rec = cv2.rectangle(frame, (220, 50), (450, 300), (255, 0, 0), 2)
            rec = cv2.rectangle(frame, (self.x1, self.y1), (self.x2, self.y2), (255, 0, 0), 2)

            # 定义roi区域，第一个为y的取值，第2个为x的取值
            # frame = frame[50:300, 220:450]
            frame = frame[self.y1:self.y2, self.x1:self.x2]
            cv2.imshow('bilateralFilter', frame)
            # 背景减法运动检测
            bg = bgModel.apply(frame, learningRate=0)
            # 显示背景减法的窗口
            cv2.imshow('bg', bg)
            # 图像边缘处理--腐蚀
            fgmask = cv2.erode(bg, self.skinkernel, iterations=1)
            # 显示边缘处理后的图像
            cv2.imshow('erode', fgmask)
            # 将原始图像与背景减法+腐蚀处理后的蒙版做"与"操作
            bitwise_and = cv2.bitwise_and(frame, frame, mask=fgmask)
            # 显示与操作后的图像
            cv2.imshow('bitwise_and', bitwise_and)
            # 灰度处理
            gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
            # 高斯滤波
            blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 2)
            cv2.imshow('GaussianBlur', blur)

            # 使用自适应阈值分割(adaptiveThreshold)
            thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            cv2.imshow('thresh', thresh)

            Ges = cv2.resize(thresh, (100, 100))
            # 图像的阈值处理(采用ostu)
            # _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # cv2.imshow('threshold1', thresh)

            if predict == True:

                # img = cv2.resize(thresh, (100, 100))
                img = np.array(Ges).reshape(-1, 100, 100, 1)/255
                prediction = p_model.predict(img)
                final_prediction = [result.argmax() for result in prediction][0]
                ges_type = self.gesture[final_prediction]
                # print(ges_type)
                cv2.putText(rec, ges_type, (self.x1, self.y1), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=2, color=(0, 0, 255))
                # cv2.putText(rec, ges_type, (150, 220), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, thickness=3, color=(0, 0, 255))

            cv2.imshow('Original', rec)
            if vedeo is True and count < photo_num:
                # 录制训练集
                cv2.imencode('.jpg', Ges)[1].tofile(self.train_path + '{}_{}.jpg'.format(str(random.randrange(1000, 100000)),str(ges)))
                count += 1
                print(count)
            elif count == photo_num:
                print('{}张测试集手势录制完毕，3秒后录制此手势测试集，共{}张'.format(photo_num, int(photo_num*0.43)))
                time.sleep(3)
                count += 1
            elif vedeo is True and photo_num < count < int(photo_num*1.43):
                cv2.imencode('.jpg', Ges)[1].tofile(self.predict_path + '{}_{}.jpg'.format(str(random.randrange(1000, 100000)),str(ges)))
                count += 1
                print(count)
            elif vedeo is True and count >= int(photo_num*1.43):
                vedeo = False
                ges += 1
                if ges < len(self.gesture):
                    print('此手势录制完成，按l录制下一个手势')
                else:
                    print('手势录制结束, 按t进行训练')

            k = cv2.waitKey(10)
            if k == 27:
                break

            elif k == ord('l'):  # 录制手势
                vedeo = True
                count = 0

            elif k == ord('p'):  # 预测手势
                predict = True
                while True:
                    model_name = input('请输入模型的名字:\n')
                    if model_name == 'exit':
                        break
                    if model_name in os.listdir('./'):
                        print('正在加载{}模型'.format(model_name))
                        p_model = load_model(model_name)
                        break
                    else:
                        print('模型名字输入错误，请重新输入，或输入exit退出')

            elif k == ord('b'):
                bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)
                print('背景重置完成')

            elif k == ord('t'):
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                train = Training(batch_size=32, epochs=5, categories=len(self.gesture), train_folder=self.train_path,
                                 test_folder=self.predict_path, model_name=self.train_model, type=self.gesture)
                train.train()
                backend.clear_session()
                print(f'{self.train_model}模型训练结束')


if __name__ == '__main__':

    # 要训练的手势类型
    # Gesturetype = input('请输入训练手势(用逗号隔开)：\n')
    # if Gesturetype == "none":
    #     Gesturetype = ['666', 'yech', 'stop', 'punch', 'OK']
    # else:
    #     Gesturetype = Gesturetype.split(',')

    Gesturetype = ['666', 'yech', 'stop', 'punch', 'OK']
    train_path = 'Gesture_train/'
    pridect_path = 'Gesture_predict/'

    # # 训练集路径
    # train_path = 'train_test/'
    # # 测试集路径
    # pridect_path = 'predict_test/'

    for path in [train_path, pridect_path]:
        if not os.path.exists(path):
            os.mkdir(path)
    print(f'训练手势有：{Gesturetype}')

    # 模型保存命名
    # train_model = input('请输入训练模型名：\n')

    train_model = 'Gesture.h5'
    # 初始化手势识别类
    Ges = Gesture(train_path, pridect_path, Gesturetype, train_model)
    # 单个手势要录制的数量
    num = 500
    # 训练手势类别计数器
    x = 0
    # 调用启动函数
    Ges.collect_gesture(capture=0, ges=x, photo_num=num)
