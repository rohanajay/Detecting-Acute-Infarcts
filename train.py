import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import os
import cv2
import glob
import numpy as np

#---------------------start of data preparation--------------------------------------------------
xtrain = []
ytrain = []
cur_path = os.getcwd()
new_path = cur_path + '/FILTERED_DATA/train'
os.chdir(new_path)
arr_train = os.listdir()

flag = 0
for i in arr_train:
    
    temp = i
    path1 = new_path + '/' +temp
    os.chdir(path1)
    temp_img = glob.glob('*.jpg')
    length = len(temp_img)
    for j in range(length):
        img = temp_img[0]
        img = cv2.imread(img)
        img = cv2.resize(img,(112,112))
        
        xtrain.append(img)
        ytrain.append(arr_train.index(temp))
        flag +=1
print('flag ', flag)
os.chdir(cur_path)

new_path1 = cur_path + '/FILTERED_DATA/test'
os.chdir(new_path1)
arr_test = os.listdir()

xtest=[]
ytest=[]
for i in arr_test:
    temp = i
    
    path1 = new_path + '/' +temp
    os.chdir(path1)
    temp_img = glob.glob('*.jpg')
    for j in range(length):
        img = temp_img[0]
        img = cv2.imread(img)
        img = cv2.resize(img,(112,112))
        xtest.append(img)
        ytest.append(arr_train.index(temp))

os.chdir(cur_path)
file1 = open('Acute_infarct.txt','w')
arr_train_text = []
for i in range(len(arr_train)):
    arr_train_text.append(arr_train[i] + '\n')
file1.writelines(arr_train_text)
file1.close()

#----------------------end of data preparation---------------------------------------------------------------------------
#----------------------- building model -------------------------------
xtrain = np.asarray(xtrain)
xtest = np.asarray(xtest)
batch_size = 6
num_classes = 42
epoch = 20
img_rows, img_cols = 112, 112
if K.image_data_format() == 'channels_first':
    input_shape = (3,img_width,img_height)
else:
    input_shape=(img_rows,img_cols,3)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255


print('x_train shape:', xtrain.shape)
print(xtrain[0].shape, 'train samples')
print(xtest[0].shape, 'test samples')
ytrain = keras.utils.to_categorical(ytrain, num_classes)
ytest = keras.utils.to_categorical(ytest, num_classes)
model = Sequential()
model.add(Conv2D(64, kernel_size=(14, 14),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(14, 14)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False),
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range = 15,
    featurewise_center=True,
    featurewise_std_normalization=True,
    width_shift_range=0.005,
    height_shift_range=0.005,
    zoom_range = 0.03,
    shear_range = 0.01,
    )
print('length xtrain : ', len(xtrain))
H=model.fit_generator(datagen.flow(xtrain, ytrain, batch_size=batch_size),
                      validation_data = (xtest,ytest),
                    steps_per_epoch=12,
                      epochs = epoch, verbose = 1)

model.save('mri_predict.h5')

