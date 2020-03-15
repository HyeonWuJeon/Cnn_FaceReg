
## 파일 경로 설정 및 읽어오기.

#클래스미사용 fit 사용.

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
from PIL import Image

class_names = ["hyeonu","irene","joy","selgi","wendy"]

lfw_path= "H:/workspace/cplusplus_opencv/FaceDetection/CNN/lfwpeople"
class_lfw = ["Ann", "Edmund","Gray","Hugo", "Jacques"]


def lfw_labeling():
    x_train = []
    for i in range(6):
        x_train.append(np.load(lfw_path+"/Ann"+str(i)+".npy"))
    for i in range(6):
        x_train.append(np.load(lfw_path+"/Edmund"+str(i)+".npy"))
    for i in range(6):
        x_train.append(np.load(lfw_path+"/Gray"+str(i)+".npy"))
    for i in range(6):
        x_train.append(np.load(lfw_path+"/Hugo"+str(i)+".npy"))
    for i in range(6):
        x_train.append(np.load(lfw_path+"/Jacques"+str(i)+".npy"))

    Y_train_orig = []
    for _ in range(6):
        Y_train_orig.append(0)
    for _ in range(6):
            Y_train_orig.append(1)
    for _ in range(6):
        Y_train_orig.append(2)
    for _ in range(6):
        Y_train_orig.append(3)
    for _ in range(6):
        Y_train_orig.append(4)

    print(Y_train_orig)
    Y_train_orig = np.array(Y_train_orig).reshape(1,30)
    # print(Y_train_orig, [...]
    # Y_test_orig = [[0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4]]
    # # #test 라벨 저장
    # # f = open("Y_test_label.txt", 'w')
    # # f.write(str(Y_test_orig))
    # # f.close()
    # # Y_test_orig = np.array(Y_test_orig)
    Y_train = convert_to_one_hot(Y_train_orig,6).T
    #
    # # Y_test = convert_to_one_hot(Y_test_orig,5).T
    #

    # # dict = {"X_train" : X_train, "X_test" : X_test, "Y_train" : Y_train, "Y_test" : Y_test};
    plt.figure(figsize=(10,30))
    for i in range(0,30):
        plt.subplot(5,15,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_lfw[Y_train_orig[0][i]])
    plt.show()

    return dict

def labeling():
    image_path = "H:/workspace/cplusplus_opencv/Casecadenet/dataset/Image"
    test_image_path = "C:/Users/tkawn/Desktop/test"

    files = glob.glob(path.join(image_path, '*.jpg'))
    test_files = glob.glob(path.join(test_image_path,'*.jpg'))
    img = []
    img_test =[]
    for i in range(30):
        img.append(files[i][63:])
        # img_test.append(test_files)
        img[i] = img[i].replace(".jpg","")
        img[i] = int(img[i])
    img.sort()
    for i in range(30):
        img[i] = "H:/workspace/cplusplus_opencv/Casecadenet/dataset/Image\\"+"setting"+str(img[i]) +".jpg"

    images = [imageio.imread(path) for path in img]
    test_images = [imageio.imread(path) for path in test_files]

    X_train = np.array(images)
    X_test = np.asarray(test_images)
    Y_train_orig = []
    for _ in range(6):
        Y_train_orig.append(0)
    for _ in range(6):
            Y_train_orig.append(1)
    for _ in range(6):
        Y_train_orig.append(2)
    for _ in range(6):
        Y_train_orig.append(3)
    for _ in range(6):
        Y_train_orig.append(4)
    # for i in range(300000)
    # #라벨값 저장
    f = open("Y_train_label.txt", 'w')
    f.write(str(Y_train_orig))
    f.close()
    Y_train_orig = np.array(Y_train_orig).reshape(1, 30)
    np.expand_dims(Y_train_orig, axis = 0)
    # print(Y_train_orig, [...])
    Y_test_orig = [[0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4]]
    # #test 라벨 저장
    f = open("Y_test_label.txt", 'w')
    f.write(str(Y_test_orig))
    f.close()
    Y_test_orig = np.array(Y_test_orig)
    Y_train = convert_to_one_hot(Y_train_orig,5).T

    Y_test = convert_to_one_hot(Y_test_orig,5).T
    dict = {"X_train" : X_train, "X_test" : X_test, "Y_train" : Y_train, "Y_test" : Y_test};

    return dict


def test(start,finish):
    k = labeling()    #
    # a = argument()
    X_train = k["X_train"]
    X_test = k["X_test"]
    Y_train = k["Y_train"]
    Y_test = k["Y_test"]

    plt.figure(figsize=(10,15))
    for i in range(start,finish):
        plt.subplot(6,10,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[Y_train[0][i]])
    plt.show()

def argument():

    k = labeling()    #
    X_train = k["X_train"]
    X_test = k["X_test"]



    print ("증가전, number of training examples = " + str(X_train.shape[0]))
    print ("증가전, number of test examples = " + str(X_test.shape[0]))
    trains_x =[]
    for i in range(30):
        samples = expand_dims(X_train[i],0) #차원 늘려준다
        datagen = ImageDataGenerator(
                        rotation_range = 10,
                        rescale = 1./255,
                        horizontal_flip=True,
                        zoom_range =[0.7,1.0], #1.0값 기준
                        featurewise_std_normalization=True#데이터의 집합 std별로 데이터를나눔.
                                )
        #zca, pca란?

        it = datagen.flow(samples, batch_size=10) #실시간데이터 증강을 통해 모델에 fit시킨다.
        for i in range(10): #해당 배수만금 generator = 3000
            batch = it.next()
            # plt.subplot(330+1+i)
            trains_x.append(batch)
            image = batch[0]#batch , convert to unsigned integers for viewing
            # plt.imshow(image)
    X_train = np.array(trains_x)
    X_train = X_train.reshape(300,200,200,3)
    #한번 더 증가 시키기.
    print(" 증가후 : number of argumentation data : ", X_train.shape)

    return X_train




def img_to_encodeing():
    model = predict()
    # model = model["model"]
    # load_weights_from_FaceNet(hyeonuModel.Model())#해당함수가져오기
    img1 = labeling()
    img1 = img1["X_train"][10]
    img = img1[...,::-1]
    img = np.around(np.transpose(img, (2,0,1))/255.0, decimals=12)
    x_train = np.array([img])
    embedding = model.predict_on_batch(x_train)
    return embedding



#cnn용 헤더파일 만들기
def argument_show(img):

    samples = expand_dims(img, 0)
    # create image data augmentation generator
    datagen = ImageDataGenerator(
                            rotation_range= 10,
                            rescale = 1./255,
                            horizontal_flip=True,
                            zoom_range =[0.9,1.0], #1.0값 기준
                            featurewise_std_normalization=True)


# prepare iterator
    it = datagen.flow(samples, batch_size=20)


    for i in range(9):
# # define subplot
        plt.subplot(330 + 1 + i)
# # generate batch of images
        batch = it.next()

# # convert to unsigned integers for viewing
        image = batch[0]
# # plot raw pixel data
        plt.imshow(image)
    plt.imshow(samples[0])
# # show the figure
    plt.show()

def image_resize(source_image, resave_image,h,w, filetype):

    image = Image.open(source_image)
    resize_image = image.resize((h,w))
    resize_image.save(resave_image, filetype, quality=95)

def image_npsave(image_path, images, start, end):
    for i in range(start,end):
        np.save(image_path, images[i])

def image_npload(image_path):
    np = np.load(path)
    plt.imshow(np)
    plt.show()
    return np

def image_numpy():
    lfw_path_ann= "H:\workspace\cplusplus_opencv\FaceDetection\CNN\lfwpeople\lfw_funneled\\Jacques_Rogge"
    files = glob.glob(path.join(lfw_path_ann, '*.jpg'))
    images = [imageio.imread(path) for path in files]
    images = np.array(images)
    print(images.shape)

    a= np.load("H:\workspace\cplusplus_opencv\FaceDetection\CNN\lfwpeople\\test\\Jacques6.npy")
    plt.imshow(a)
    plt.show()
    # for i in range(6,12):
    #     np.save("H:\workspace\cplusplus_opencv\FaceDetection\CNN\lfwpeople\\test\\Jacques"+str(i)+".npy", images[i])

    # for i in range(6):
        # np.save("")
if __name__=='__main__':
    image_numpy()
    # image_numpy()
    # print(dict["X_train"][10])
