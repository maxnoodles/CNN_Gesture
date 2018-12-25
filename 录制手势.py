import cv2
import numpy as np
import copy
import math


class Gesture():

    def __init__(self, path):
        self.blurValue = 41
        self.bgSubThreshold = 50
        self.path = path
        self.threshold = 60

    def _get_distance(self, a,b):
        # 余弦定理
        distance = math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        return distance

    def printThreshold(self, thr):
        # 定义createTrackbar的回调函数
        print("! Changed threshold to " + str(thr))

    def collect_gesture(self, capture, gesturetype, photo_num):
        photo_num = photo_num
        vedio = False
        count = 0
        frame_index = 0
        # 读取默认摄像头
        cap = cv2.VideoCapture(capture)
        # 设置捕捉模式
        cap.set(10, 200)
        # 背景减法创建及初始化
        bgModel = cv2.createBackgroundSubtractorMOG2(0, self.bgSubThreshold)

        # cv2.namedWindow('trackbar', cv2.WINDOW_NORMAL)
        # 创建阈值轨迹条
        # cv2.createTrackbar('trh1', 'trackbar', self.threshold, 100, self.printThreshold)

        while(True):
            # 读取视频帧
            ret, frame = cap.read()
            # 镜像转换
            frame = cv2.flip(frame, 1)
            # 显示原图像
            cv2.imshow('Original', frame)
            # 获得阈值轨迹条的值
            # threshold = cv2.getTrackbarPos('trh1', 'trackbar')

            # 双边滤波
            frame = cv2.bilateralFilter(frame, 5, 50 ,100)

            # rec = cv2.rectangle(frame, (0, 0), (300,300), (255, 0, 0), 2)
            # frame = frame[0:300, 0:300]
            # 绘制矩形，第一个为左上角坐标(x,y），第二个为右下角坐标
            rec = cv2.rectangle(frame, (200, 0), (500, 300), (255, 0, 0), 2)
            cv2.imshow('Original', rec)
            # 定义roi区域，第一个为y的取值，第2个为x的取值
            frame = frame[0:300, 200:500]

            # 背景减法运动检测
            bg = bgModel.apply(frame, learningRate=0)
            # 显示背景减法的窗口
            cv2.imshow('bg', bg)

            kernel = np.ones((3,3), np.uint8)
            # 图像边缘处理--腐蚀
            fgmask = cv2.erode(bg, kernel, iterations=1)
            # 显示边缘处理后的图像
            # cv2.imshow('erode', fgmask)

            # 将原始图像与背景减法+腐蚀处理后的蒙版做"与"操作
            bitwise_and = cv2.bitwise_and(frame, frame, mask=fgmask)
            # 显示与操作后的图像
            cv2.imshow('bitwise_and', bitwise_and)

            # 灰度处理
            gray = cv2.cvtColor(bitwise_and, cv2.COLOR_BGR2GRAY)
            # 高斯滤波
            blur = cv2.GaussianBlur(gray, (self.blurValue, self.blurValue), 0)
            # cv2.imshow('GaussianBlur', blur)

            # 图像的阈值处理(采用ostu)
            # _, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow('threshold1', thresh)
            if vedio == True and count < photo_num :
                Ges0 = cv2.resize(thresh, (100, 100))
                # Ges0 = thresh
                # 录制训练集
                # cv2.imencode('.jpg', Ges0)[1].tofile(self.path + gesturetype + '\\' + str(x) + '_Ges{}.jpg'.format(count))
                # 录制测试集
                cv2.imencode('.jpg', Ges0)[1].tofile(self.path + '\\' + str(x) + '_Ges{}.jpg'.format(count))
                # cv2.imencode('.jpg', Ges0)[1].tofile('Ges{}.jpg'.format(count))
                count += 1
                print(count)

            elif count >= photo_num:
                break

            k = cv2.waitKey(100)
            if k == 27:
                break
            elif k == ord('l'):
                vedio = True


if __name__ == '__main__':

    Gesturetype = ['666', 'stop', 'yech', 'ok']
    # path = 'D:\\pycharm-work\\opencv-test\\CNN_Gesture_categorizer\\手势合集\\'
    path = 'D:\pycharm-work\opencv-test\CNN_Gesture_categorizer\Gesture_predict\\'
    Ges = Gesture(path)
    num = 20
    x = 3
    Ges.collect_gesture(capture=0, gesturetype=Gesturetype[x], photo_num=num)
