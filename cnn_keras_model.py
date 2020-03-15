#클래스미사용 fit 사용.
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.wrappers.scikit_learn import KerasClassifier,KerasRegressor
from keras.callbacks import ModelCheckpoint
# from kt_utils import *

import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

import tensorflow as tf
from tensorflow.python.framework import ops #graph, tensor등의 정식api가 아닌tensorflow 모듈, 버젼마다 상이할수있음
from cnn_utils import * #cnn 사용
# %matplotlib inline
from keras.layers.core import Lambda, Flatten, Dense

## 파일 경로 설정 및 읽어오기.
import glob
import imageio
import os
import os.path as path
import random
import math #수학계산
import numpy as np
import h5py#대용량 데이터 파일 처리
import matplotlib.pyplot as plt #그래프
import matplotlib.image as mpimg
import cv2
import scipy #데이터분석, 수치미분 계산

## 파라매터 랜덤값 seed
np.random.seed(1)

## keras 모듈

from keras.utils import np_utils #np_utils.
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import KFold, cross_val_score, train_test_split,StratifiedKFold
from tensorflow import keras

from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import random
from image_setting import *


image_path= "H:\workspace\cplusplus_opencv\Casecadenet\dataset\PCA64"
class_names = ["hyeonu", "irene","joy","selgi", "wendy"]



def labeling():
    # x_train = []
    # for i in range(6):
    #     x_train.append(np.load(image_path+"/Ann"+str(i)+".npy"))
    # for i in range(6):
    #     x_train.append(np.load(image_path+"/Edmund"+str(i)+".npy"))
    # for i in range(6):
    #     x_train.append(np.load(image_path+"/Gray"+str(i)+".npy"))
    # for i in range(6):
    #     x_train.append(np.load(image_path+"/Hugo"+str(i)+".npy"))
    # for i in range(6):
    #     x_train.append(np.load(image_path+"/Jacques"+str(i)+".npy"))

    files = glob.glob(path.join(image_path, '*.jpg'))
    img = []
    for i in range(60):
        img.append(files[i][63:])
        # img_test.append(test_files)
        img[i] = img[i].replace(".jpg","")
        img[i] = int(img[i])
    img.sort()
    for i in range(60):
        img[i] = "H:\\workspace\\cplusplus_opencv\\Casecadenet\\dataset\\PCA64\\"+"setting"+str(img[i]) +".jpg"

    images = [imageio.imread(path) for path in img]
    X_train = np.array(images)



    Y_train_orig = []
    for _ in range(12):
        Y_train_orig.append(0)
    for _ in range(12):
            Y_train_orig.append(1)
    for _ in range(12):
        Y_train_orig.append(2)
    for _ in range(12):
        Y_train_orig.append(3)
    for _ in range(12):
        Y_train_orig.append(4)

    Y_train = np.array(Y_train_orig)
    # Y_test_orig = [[0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4]]
    # #test 라벨 저장
    # f = open("Y_test_label.txt", 'w')
    # f.write(str(Y_test_orig))
    # f.close()
    # Y_test_orig = np.array(Y_test_orig)
    Y_train = convert_to_one_hot(Y_train,5).T


    dict = {"X_train" : X_train, "Y_train" : Y_train}

    plt.figure(figsize=(10,20))
    for i in range(0,60):
        plt.subplot(5,20,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[Y_train_orig[i]])
    plt.show()

    return dict


class model_Stock:
    def __init__(Self):
        pass
    def Model(self):
        model = Sequential()

        model.add(Conv2D(8, (5, 5), strides = (1, 1), name = 'conv0', input_shape=(64,64,3)))#60,60,8
        model.add(BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn1'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D((5,5),strides = (5,5)))# 12,12,8
        model.add(Conv2D(16,(2,2),strides=(1,1),padding="SAME")) #11,11,16
        model.add(BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2'))
        model.add(Activation('relu'))

        model.add(Conv2D(32,(2,2), strides=(1,1), padding="SAME"))# 10,10,32
        model.add(Activation('relu'))

        model.add(Conv2D(64,(2,2), strides=(1,1), padding="SAME")) #9,9,64
        model.add(Activation('relu'))

        model.add(MaxPooling2D((3,3))) #3,3,64

        model.add(Conv2D(128,(1,1), strides=(1,1), padding ="SAME")) #3,3,128
        model.add(BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3'))
        model.add(Activation('relu'))

        # model.add(MaxPooling2D((1,4)))#4,4,128
        model.add(Conv2D(256,(1,1),strides=(1,1),padding="SAME")) # 3,3,256
        model.add(Flatten())
        model.add(Dense(128, activation='relu', name='fc'))#relu function 추가 #활성화함수에대해 좀더 보기 !

        model.add(Dense(5, activation = 'softmax'))
        model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics=['accuracy'])


        return model



def predict_Kfold():

    k = labeling()
    # X = argument() #300000 장
    X_train = k["X_train"]
    Y_train = k["Y_train"]
    cv = KFold(n_splits=5, shuffle=True)#4
    accuracy=[]
    hyeonuModel = model_Stock()
    Mol= KerasClassifier(build_fn=hyeonuModel.Model, epochs=5, batch_size = 32, verbose=0) #인스턴스 전달
    results = cross_val_score(Mol, X_train, Y_train, cv = cv)
    print('\nK-fold cross validation Accuracy: {}'.format(results))

    print("완료 \n\n")

def predict_split():
    k = labeling()

    X_train = k["X_train"]
    Y_train = k["Y_train"]
    X_train, X_test, Y_train, Y_test=train_test_split(X_train, Y_train, test_size=0.3, stratify=Y_train, random_state=0)
    hyeonuModel = model_Stock()
    hyeonuModel.Model().fit(X_train, Y_train, batch_size = 32, epochs=5)
    preds =hyeonuModel.Model().evaluate(x=X_test, y=Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    #
    #
    # predictions = hyeonuModel.Model().predict(X_test)
    #
    # print("첫번째 레이블 예측 확인 : ", np.argmax(predictions[0]))
    # model_json = hyeonuModel.Model().to_json()
    # with open("face_batch_model.json", "w") as json_file :
    #     json_file.write(model_json)
    # #model weight save
    # hyeonuModel.Model().save_weights("face_batch_model.h5")
    # print("Saved model to disk")
    #
    # import time
    # saved_model_path = "./saved_models/{}".format(int(time.time()))
    #
    # tf.keras.experimental.export_saved_model(model, saved_model_path)
    # print("완료")


    # pred ={"X_test": X_test, "Y_test": Y_test, "model":hyeonuModel.Model(), "predictions": predictions}
    # return pred

def predict_argument():
    k = labeling()

    X_train = k["X_train"]
    Y_train = k["Y_train"]

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=10,
        rescale = 1./255,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    X_train, X_test, Y_train, Y_test=train_test_split(X_train, Y_train, test_size=0.3, stratify=Y_train, random_state=0)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    hyeonuModel = model_Stock()
    
# fits the model on batches with real-time data augmentation:
    hyeonuModel.Model().fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 64, epochs=5)


    preds =hyeonuModel.Model().evaluate(x=X_test, y=Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))

# # here's a more "manual" example
# for e in range(epochs):
#     print('Epoch', e)
#     batches = 0
#     for x_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=32):
#         model.fit(x_batch, y_batch)
#         batches += 1
#         if batches >= len(x_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break


def plot_image(i, predictions_arrays, true_labels, img):
    true_labels = true_labels.tolist()
    predictions_array, true_label, img = predictions_arrays[i], true_labels[i].index(1), img[i]

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    # print("predicted_label",predicted_label)
    # print("true_label",true_label)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_arrays, true_labels):
    true_labels = true_labels.tolist()
    predictions_array, true_label = predictions_arrays[i], true_labels[i].index(1)

    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(5), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def main():
    pred = predict()
    model = pred["model"]
    X_test =pred["X_test"]
    Y_test = pred["Y_test"]
    predictions = pred["predictions"]
    i = 0

    plt.figure(figsize=(6,3))
    plt.subplot(1,2,1)
    plot_image(i, predictions, Y_test, X_test)
    plt.subplot(1,2,2)
    plot_value_array(i, predictions,  Y_test)
    plt.show()

    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        plot_image(i, predictions, Y_test, X_test)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plot_value_array(i, predictions, Y_test)
    plt.show()



if __name__=='__main__':
    # predict_Kfold()
    # predict_split()
    predict_argument()
