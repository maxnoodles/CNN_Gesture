from keras.models import load_model
# import matplotlib.image as processimage
# import matplotlib as plt
import numpy as np
from PIL import Image
import os

model = load_model('D:\pycharm-work\opencv-visualization_image\CNN_Gesture_categorizer\Gesture_2.h5')
Gesturetype = ['666', 'yech', 'stop', 'punch', 'OK']



# Gesturetype = ['666', 'stop', 'yech', 'ok', 'one']
path = 'D:\pycharm-work\opencv-visualization_image\CNN_Gesture_categorizer\Gesture_predict\\'
file_count = 0

for file in os.listdir(path):
    list = []
    img = Image.open(path + file)
    test = file.split('_')[0]
    table = file.split('_')[1][0]
    img = np.array(img).reshape(-1, 100, 100, 1) / 255

    prediction = model.predict(img)
    final_prediction = [result.argmax() for result in prediction][0]

    if final_prediction != int(table):
        print('第' + test + '次结果:'+Gesturetype[final_prediction])
        count = 0
        for i in prediction[0]:
            percentage = '%.2f%%' % (i*100)
            list.append(str(Gesturetype[count]) + '的概率:'+str(percentage))
            count += 1
        file_count += 1
        print(list)

print(file_count)