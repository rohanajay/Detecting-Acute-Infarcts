from keras.models import load_model
import keras
import cv2
import numpy as np
import os
import glob 
f = open('Acute_infarct.txt','r')
arr_train_1 = f.readlines()
arr_train = []
length = len(arr_train_1)
for i in range(length):
    arr_train_1[i] = arr_train_1[i].split('\n')
for i in range(len(arr_train_1)):
    arr_train.append(arr_train_1[i][0])
#print(arr_train)
f.close()
act_dir = os.getcwd()
model = load_model('mri_predict.h5')
xpredict = []


print('')
print('ACCURACY ON TRAINING IMAGES')
print('')
os.chdir(act_dir)
new_dir = act_dir + '/FILTERED_DATA/train'
os.chdir(new_dir)
x_predict_1 = os.listdir()
#print(x_predict_1)
count = 0
match = 0
for i in x_predict_1:
    temp = i
    cur_path_1 = new_dir
    path1 = cur_path_1 + '/' + temp
    os.chdir(path1)
    temp_imgs = glob.glob('*.jpg')
    length_t_i  = len(temp_imgs)
    for j in range(length_t_i):
        img = temp_imgs[j]
        img = cv2.imread(img)
        img = cv2.resize(img,(112,112))
        img = np.reshape(img, [1,112,112,3])
        classes = model.predict_classes(img)
        for k in classes:
            print('Input class : {}, Predicted class : {}, Predicted class number : {} '.format(i,arr_train[k],classes))
        if(i == arr_train[k]):          
            match +=1
        count +=1
print('')
print('Training Image Accuracy : ',match/count)
print('')
print('--------------------------------------------------------------------------------')
print('ACCURACY ON TEST IMAGES')
print('')
os.chdir(act_dir)
new_dir = act_dir + '/FILTERED_DATA/test'
os.chdir(new_dir)
x_predict_1 = os.listdir()
#print(x_predict_1)
count = 0
match = 0
for i in x_predict_1:
    temp = i
    cur_path_1 = new_dir
    path1 = cur_path_1 + '/' + temp
    os.chdir(path1)
    temp_imgs = glob.glob('*.jpg')
    length_t_i  = len(temp_imgs)
    for j in range(length_t_i):
        img = temp_imgs[j]
        img = cv2.imread(img)
        img = cv2.resize(img,(112,112))
        img = np.reshape(img, [1,112,112,3])
        classes = model.predict_classes(img)
        for k in classes:
            print('Input class : {}, Predicted class : {}, Predicted class number : {} '.format(i,arr_train[k],classes))
        if(i == arr_train[k]):          
            match +=1
        count +=1
print('')
print('Accuracy on test Images : ',match/count)

print('')
print('----------------------------------------------------------------------------------------------')
print('ACCURACY ON INFERENCE  IMAGES')
print('')
new_dir = act_dir + '/INFERENCE_DATA'
os.chdir(new_dir)

x_predict = os.listdir()
count = 0
match = 0
for i in x_predict:
    temp = i
    cur_path_1 = new_dir
    path1 = cur_path_1 + '/' + temp
    os.chdir(path1)
    temp_imgs = glob.glob('*.jpg')
    length_t_i  = len(temp_imgs)
    for j in range(length_t_i):
        img = temp_imgs[j]
        img = cv2.imread(img)
        img = cv2.resize(img,(112,112))
        img = np.reshape(img, [1,112,112,3])
        classes = model.predict_classes(img)
        for k in classes:
            print('Input class : {}, Predicted class : {}, Predicted class number : {} '.format(i,arr_train[k],classes))
        if(i == arr_train[k]):          
            match +=1
        count +=1
print('')
print('Accuracy on Inference Images : ',match/count)
print('')

print('--------------------------------------------------------------------------------')
