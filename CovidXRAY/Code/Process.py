import numpy as np
import cv2
import time
import win32api as wapi
import os
import sys
from PIL import Image
from random import shuffle
import pandas as pd

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print("File exists, loading previous data")
    training_data = list(np.load(file_name))
else:
    print("File does not exist, starting fresh")
    training_data = []

for i in list(range(3))[::-1]:
    print(i+1)
    time.sleep(1)
    
last_time = time.time()

imgs = ['Angry_1', 'Angry_2', 'Angry_3', 'Angry_4', 'Angry_5', 
        'BAS_1', 'BAS_2', 'BAS_3','BAS_4', 'BAS_5', 'download (1)',
        'download', 'Hungry_1', 'Hungry_2', 'Owner_1']
#imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
imagePaths = ['Angry_1.png', 'Angry_2.png', 'Angry_3.png', 'Angry_4.png', 
            'Angry_5.png', 'BAS_1.png', 'BAS_2.png', 'BAS_3.png',
            'BAS_4.png', 'BAS_5.png', 'download (1).png', 'download.png', 
            'Hungry_1.png', 'Hungry_2.png', 'Owner_1.png']
for index,file in enumerate(imagePaths):
    # img = cv2.imread('Images/'+file)
    # img = cv2.resize(img, (80,60))
    # print(np.shape(img))
    # output = imgs[index]
    # training_data.append([img, output])
    # print('Frame took {} seconds'.format(time.time()-last_time))
    # last_time = time.time() 
    try:
        img = cv2.imread('Images/'+file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('img',img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (80,60))
        #img = cv2.imread('Images/'+file)
        output = output = imgs[index]
        training_data.append([img, output])
        print('Frame took {} seconds'.format(time.time()-last_time))
        last_time = time.time()
        if cv2.waitKey(30) & 0xff == 'q' == 27:
            break
    except:
        np.save(file_name, training_data)        
np.save(file_name, training_data)
cv2.destroyAllWindows()


train_data = np.load('training_data.npy')

TOTAL = []

for index, data in enumerate(train_data):
    name = data[1]
    print(np.shape(data[0]))
    if name in imgs:
        TOTAL.append([data[0], index])

shuffle(TOTAL)
np.save('training_data_cleaned.npy', TOTAL)
