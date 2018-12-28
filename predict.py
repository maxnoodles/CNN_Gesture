from keras.models import load_model
import matplotlib.image as processimage
import matplotlib as plt
import numpy as np
from PIL import Image
import os

model = load_model('Gesture1.0.h5')
Gesturetype = ['666', 'stop', 'yech', 'ok', 'one']
path = 'D:\pycharm-work\opencv-test\CNN_Gesture_categorizer\Gesture_predict\\'
file_count = 0

for file in os.listdir(path):
    list = []
    # img = Image.open(path + 'Ges{}.jpg'.format(i))
    img = Image.open(path + file)
    img = np.array(img).reshape(-1, 100, 100, 1) / 255

    prediction = model.predict(img)
    final_prediction = [result.argmax() for result in prediction][0]
    print('第' +str(file_count)+ '次结果:'+Gesturetype[final_prediction])
    count = 0
    for i in prediction[0]:
        percentage = '%.2f%%' % (i*100)
        list.append(str(Gesturetype[count]) + '的概率:'+str(percentage))
        # print(gestype[i], '概率:', percentage)
        count += 1
    file_count += 1

    print(list)