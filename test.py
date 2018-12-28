# Gesturetypr = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五'}
# a = 5
# if a in Gesturetypr.keys():
#     print(Gesturetypr[a])
# print('D:\\pycharm-work\\opencv-test\\CNN_Gesture_categorizer\\手势合集\\一\\ges{}.jpg')
#
# import cv2
# import numpy as np
#
# cap = cv2.VideoCapture(0)
# count = 0
# a = 'D:\\pycharm-work\\opencv-test\\CNN_Gesture_categorizer\\手势合集\\一\\'
# while True:
#     count += 1
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow('gray', frame)
#     cv2.imencode('.jpg', frame)[1].tofile(a+'ges{}.jpg'.format(count))
#     k = cv2.waitKey(10)
#     if k == '27':
#         break
#
# cap.release()
# cv2.destroyAllWindows()

# Gesturetype = {0: '零', 1: '一', 2: '二', 3: '三', 4: '四', 5: '五'}
# print(Gesturetype[0])

# import cv2
# import numpy as np
# from PIL import Image
#
# # a = Image.open('0_ges1.jpg')
#
# print(np.array('').shape)

# import random
# # data = [1, 2, 3, 4, 5]
# # data2 = []
# # index = [i for i in range(len(data))]
# # print(index)
# # print(random.shuffle(index))
# # for i in index:
# #     data2.append(data[i])
# # print

# import os
#
# Gesturetype = ['666', 'stop', 'yech', 'ok']
#
# for i in Gesturetype:
#     os.mkdir('D:\pycharm-work\opencv-test\CNN识别手势\手势合集\\' + i)

# import random
# print(random.randrange(0, 50))
print(5, '哈哈哈')
