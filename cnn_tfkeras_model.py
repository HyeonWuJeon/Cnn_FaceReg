
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
from sklearn.model_selection import KFold, cross_val_score, train_test_split,StratifiedKFold
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
from keras.layers.core import Lambda, Flatten, Dense


class_names=["hyeon","irene","joy","selgi","wendy"]
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
    # list = [0,1,2,3,4]
    # for i in range(6):#6000
    #     out = random.sample(list, 5)
    #     Y_train_orig.append(out)
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

    Y_test_orig = [[0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3, 4,4,4,4,4,4]]
    #test 라벨 저장
    f = open("Y_test_label.txt", 'w')
    f.write(str(Y_test_orig))
    f.close()
    Y_test_orig = np.array(Y_test_orig)
    Y_train = convert_to_one_hot(Y_train_orig,5).T
    Y_test = convert_to_one_hot(Y_test_orig,5).T


    dict = {"X_train" : X_train, "X_test" : X_test, "Y_train" : Y_train, "Y_test" : Y_test};

    return dict


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
                        rotation_range = 90,
                        rescale = 1./255,
                        horizontal_flip=True,
                        zoom_range =[0.7,1.0], #1.0값 기준
                        featurewise_std_normalization=True,#데이터의 집합 std별로 데이터를나눔.
                        brightness_range=[0.2,0.8])
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

#삼중 손실함수.
def triplet_loss(y_true, y_pred, alpha = 0.2):

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss


with tf.Session() as test:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    #3,128 shape 랜덤한 값
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed = 1))
    # print(test.run(y_pred))
    # print(y_pred[0].get_shape())#3,128
    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))

def triplet_loss(y_true, y_pred, alpha = 0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    ### START CODE HERE ### (≈ 4 lines)
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,positive)), axis = -1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor,negative)), axis = -1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    ### END CODE HERE ###

    return loss


accuracy=[]

def Model():
    model = Sequential()
    model.add(Conv2D(8, (5, 5), strides = (1, 1), name = 'conv0'))#200,200,8
    model.add(BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn1'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((5,5),strides = (5,5)))# 40,40,8
    model.add(Conv2D(16,(2,2),strides=(1,1),padding="SAME")) #40,40,16
    model.add(BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2'))
    model.add(Activation('relu'))

    model.add(Conv2D(32,(2,2), strides=(1,1), padding="SAME"))# 40,40,32
    model.add(Activation('relu'))

    model.add(Conv2D(64,(2,2), strides=(1,1), padding="SAME")) #40,40,64
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2,2))) #20,20,64
    model.add(Conv2D(128,(2,2), strides=(1,1), padding ="SAME")) #20,20,128
    model.add(BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((4,4)))#4,4,128
    model.add(Flatten())
    # model.add(Dense(128, activation='relu', name='fc'))#relu function 추가 #활성화함수에대해 좀더 보기 !
            # model.add(Dropout(0.5))
    # model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))
    model.add(Dense(128, activation='relu', name='fc'))#relu function 추가 #활성화함수에대해 좀더 보기 !
    model.add(Dense(5, activation = 'softmax'))

    #dont have softmax
    model.compile(optimizer = 'Adam', loss = "categorical_crossentropy", metrics=['accuracy'])

    return model

#모델 그림그려서 shape 확인해보기!!!


def predict():

    k = labeling()
    # X = argument() #300000 장
    # X_train = X
    X_train = k["X_train"]
    Y_train = k["Y_train"]
    X_test = k["X_test"]
    Y_test = k["Y_test"]

    model = Model()

    accuracy=[]
    model.fit(x=X_train, y= Y_train,epochs=2, batch_size = 10, verbose=0)
    # k_accuracy ="%f" % (model.evaluate(X_train, Y_train))
    # accuracy.append(k_accuracy) #Training 훈련 정확도.
    # #
    # print('\n training Accuracy: {}'.format(accuracy))
    # print('\nK-fold cross validation Accuracy: {}'.format(results))

    preds =model.evaluate(x=X_test, y=Y_test)
    print ("Loss = " + str(preds[0]))
    print ("Test Accuracy = " + str(preds[1]))
    #
    predictions = model.predict(X_test)

    print("첫번째 레이블 예측 확인 : ", np.argmax(predictions[0]))
    # #model save
    # model_json = hyeonuModel.Model().to_json()
    # with open("face_batch_model.json", "w") as json_file :
    #     json_file.write(model_json)
    # #model weight save
    # hyeonuModel.Model().save_weights("face_batch_model.h5")
    # print("Saved model to disk")

    print(model.summary())
    pred ={"X_test": X_test, "Y_test": Y_test, "Model":model, "predictions": predictions}
    return pred


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
    model = pred["Model"]
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
    num_cols = 6
    num_images = num_rows*num_cols
    plt.figure(figsize=(3*3*num_cols, 4*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 3*num_cols, 3*i+1)
        plot_image(i, predictions, Y_test, X_test)
        plt.subplot(num_rows, 3*num_cols, 3*i+2.5)
        plot_value_array(i, predictions, Y_test)
    plt.show()

#     img = X_test[15]
#     img = (np.expand_dims(img,15))#tf.keras모들엔 배치로 예측을 만드는데 최적화됨.
#     #하나의 이미지를 사용헐때도 2차원 배열로 만들어야한다.
#     predictions_single = model.predict(img)
#
#     print(predictions_single)
#
#     plot_value_array(0, predictions_single, Y_test)
# _ = plt.xticks(range(5), class_names, rotation=30)

# drop out 적용하기
# 하이퍼파라매터조정
# Adam 조정하기.


if __name__ == '__main__':
    main()
