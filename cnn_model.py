import math #수학계산
import numpy as np
import h5py#대용량 데이터 파일 처리
import matplotlib.pyplot as plt #그래프
import matplotlib.image as mpimg
import cv2
import scipy #데이터분석, 수치미분 계산

import tensorflow as tf
from tensorflow.python.framework import ops #graph, tensor등의 정식api가 아닌tensorflow 모듈, 버젼마다 상이할수있음
from cnn_utils import * #cnn 사용

np.random.seed(1)

from keras.utils import np_utils #np_utils.
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from tensorflow import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2

from image_setting import *


# //keep_p = tf.placeholder(tf.int32)

# def labeling():
#     image_path = "H:/workspace/cplusplus_opencv/Casecadenet/dataset/Image"
#     test_image_path = "C:/Users/tkawn/Desktop/test"
#
#     files = glob.glob(path.join(image_path, '*.jpg'))
#     test_files = glob.glob(path.join(test_image_path,'*.jpg'))
#     img = []
#     img_test =[]
#     for i in range(30):
#         img.append(files[i][63:])
#         # img_test.append(test_files)
#         img[i] = img[i].replace(".jpg","")
#         img[i] = int(img[i])
#     img.sort()
#     for i in range(30):
#         img[i] = "H:/workspace/cplusplus_opencv/Casecadenet/dataset/Image\\"+"setting"+str(img[i]) +".jpg"
#
#     images = [imageio.imread(path) for path in img]
#     test_images = [imageio.imread(path) for path in test_files]
#
#     X_train = np.array(images)
#     X_test = np.asarray(test_images)
#
#
#     Y_train_orig = []
#     list = [0,1,2,3,4]
#     for i in range(600):
#         out = random.sample(list, 5)
#         Y_train_orig.append(out)
#     # #라벨값 저장
#     f = open("Y_train_label.txt", 'w')
#     f.write(str(Y_train_orig))
#     f.close()
#     Y_train_orig = np.array(Y_train_orig).reshape(1, 3000)
#     np.expand_dims(Y_train_orig, axis = 0)
#
#     Y_test_orig = [[4,1,2,3,0, 4,0,3,1,2, 4,3,1,2,0, 1,3,2,4,0, 0,4,3,1,2, 1,0,2,4,3]]
#     #test 라벨 저장
#     f = open("Y_test_label.txt", 'w')
#     f.write(str(Y_test_orig))
#     f.close()
#     Y_test_orig = np.array(Y_test_orig)
#     Y_train = convert_to_one_hot(Y_train_orig,5).T
#     Y_test = convert_to_one_hot(Y_test_orig,5).T
#
#
#     dict = {"X_train" : X_train, "X_test" : X_test, "Y_train" : Y_train, "Y_test" : Y_test};
#
#     return dict
#
#

#값을 넘겨줌.
def create_placeholders(n_H0, n_W0, n_C0, n_y):

    X = tf.placeholder(tf.float32, shape=(None,n_H0,n_W0,n_C0)) #200,200,3
    Y = tf.placeholder(tf.float32, shape=(None,n_y)) # 6

    return X, Y


# GRADED FUNCTION: initialize_parameters

def initialize_parameters():

    tf.set_random_seed(1)                              # so that your "random" numbers match ours
    #파라메타 초기화
    ### START CODE HERE ### (approx. 2 lines of code) #파라메터값을 입력받는다

    #xavier형식으로 파라메터 초기화.
    #variable vs get_variabele vs constant
    #get_variable : shape에 맞춰서 실행시마다 랜덤값 생성
    #variable : 픽스된 고정값

# 3,3 ;필터 크기, 3 = 이전레이어의 채널수; 8 = 현재레이어에 사용할 필터수;
    W1 = tf.get_variable("W1",[5,5,3,8], initializer=tf.contrib.layers.variance_scaling_initializer(seed = 0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.variance_scaling_initializer(seed = 0))
    W3 = tf.get_variable("W3",[2,2,16,32],initializer=tf.contrib.layers.variance_scaling_initializer(seed = 0))
    W4 = tf.get_variable("W4",[2,2,32,64],initializer=tf.contrib.layers.variance_scaling_initializer(seed = 0))
    W5 = tf.get_variable("W5",[2,2,64,128],initializer=tf.contrib.layers.variance_scaling_initializer(seed = 0))


    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3,
                  "W4": W4,
                  "W5": W5}

    return parameters

# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    keep_prob = 0.5
    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    W4 = parameters['W4']
    W5 = parameters['W5']


    ### START CODE HERE ###
    # CONV2D: stride of 1, padding 'SAME'
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME') #결과값 그대로 200* 200 * 8
    A1 = tf.nn.relu(Z1)
    # A1 = tf.nn.dropout(A1, keep_prob= keep_prob) #0.5
    # MAXPOOL: window 8x8, sride 8, padding 'SAME'
    #8칸씩이동 1차원 8x8
    P1 = tf.nn.max_pool(A1, ksize=[1,5,5,1],strides=[1,5,5,1],padding='SAME')# 40,40,8
    # CONV2D: filters W2, stride 1, padding 'SAME
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')#40,40,16
    A2 = tf.nn.relu(Z2)
    # A2 = tf.nn.dropout(A2, keep_prob = 0.6)

    Z3 = tf.nn.conv2d(A2,W3,strides=[1,1,1,1],padding='SAME')#40,40,32
    A3 = tf.nn.relu(Z3)
    # A3 = tf.nn.dropout(A3, keep_prob = 0.6)

    Z4 = tf.nn.conv2d(A3,W4,strides=[1,1,1,1],padding='SAME')#40,40,64
    A4 = tf.nn.relu(Z4)

    # MAXPOOL: window 4x4, stride 4, padding 'SAME'
    #stride 크기가 4인 1차원 리스트. 4칸씩 이동. Maxpool 전에 드랍아웃 x
    P2 = tf.nn.max_pool(A4,ksize=[1,2,2,1],strides=[1,2,2,1], padding='SAME')#20,20,64
    # FLATTEN

    Z5 = tf.nn.conv2d(P2,W5,strides=[1,1,1,1],padding='SAME')#20,20,128
    A5 = tf.nn.relu(Z5)

    P3 = tf.nn.max_pool(A5,ksize=[1,4,4,1],strides=[1,4,4,1], padding='SAME')#5,5,128
    P = tf.contrib.layers.flatten(P2) # 3200


    # FULLY-CONNECTED without non-linear activation function (not not call softmax).
    #1차원 데이터만 입력받을수 있기 때문에 flatten으로 1차원으로 펼침 3차원데이터의 공간적 정보가 손실된다.
    # 6 neurons in output layer. Hint: one of the arguments should be "activation_fn=None"

    Z3 = tf.contrib.layers.fully_connected(P, 5, activation_fn =None)
    ### END CODE HERE ###

    return Z3


def compute_cost(Z3, Y):


    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))

    return cost
#
# def argument():
#
#     k = labeling()    #
#     X_train = k["X_train"]
#     X_test = k["X_test"]
#     Y_train = k["Y_train"]
#     Y_test = k["Y_test"]
#
#
#     print ("증가전, number of training examples = " + str(X_train.shape[0]))
#     print ("number of test examples = " + str(X_test.shape[0]))
#
#     trains_x =[]
#     for i in range(30):
#         samples = expand_dims(X_train[i],0) #차원 늘려준다
#         datagen = ImageDataGenerator(
#                         rotation_range = 90,
#                         rescale = 1./255,
#                         horizontal_flip=True,
#                         zoom_range =[0.7,1.0], #1.0값 기준
#                         featurewise_std_normalization=True,#데이터의 집합 std별로 데이터를나눔.
#                         brightness_range=[0.2,0.8])
#         #zca, pca란?
#     #
#         it = datagen.flow(samples, batch_size=10) #실시간데이터 증강을 통해 모델에 fit시킨다.
#         for i in range(100): #해당 배수만금 generator = 300
#             batch = it.next()
#             # plt.subplot(330+1+i)
#             trains_x.append(batch)
#             image = batch[0]#batch , convert to unsigned integers for viewing
#             # plt.imshow(image)
#     X_train = np.array(trains_x)
#     print("reshape 전:", X_train.shape)
#     X_train = X_train.reshape(3000,200,200,3)
#     #한번 더 증가 시키기.
#     print(" 증가후 : number of argumentation data", len(X_train[0]))
#
#
#     print(X_train.shape)
#     print(Y_train.shape)
#
#
#     X = {"X_train":X_train, "Y_train":Y_train}
#
#     return X

def model(X_train, Y_train,  X_test, Y_test, learning_rate = 0.01,
          num_epochs = 15, minibatch_size = 128, print_cost = True): #epoches 10(300장일 경우) : 1.607
                                                                    #epcohes 15: 1.5390
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """


    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)

    (m, n_H0, n_W0, n_C0) = X_train.shape   #300,200,200,3

    n_y = Y_train.shape[1]

    costs = []                                        # To keep track of the cost
    # 값전달 : 300,200,200,30, 300
    X, Y = create_placeholders(n_H0,n_W0,n_C0, n_y)
    ### END CODE HERE ###


    # Xavier 파라메터 초기화
    parameters = initialize_parameters()
    ### END CODE HERE ###

    # 순방향 학습, 텐서그래프
    ### START CODE HERE ### (1 line)
    Z3 = forward_propagation(X, parameters)
    ### END CODE HERE ###

    #cost값
    cost = compute_cost(Z3, Y)
    ### END CODE HERE ###

    #Backpropagation = Adam , minibatch 30?
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    ### END CODE HERE ###

    # Initialize all the variables globally
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph


    with tf.Session() as sess:

        # Run the initialization
        sess.run(init) #init
        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.

            num_minibatches = int(m / minibatch_size) # 210개
            seed = seed + 1

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)#minibatch :트레이닝의 과부하를 줄이기위해
            #랜덤한값 몇가지를 전달한다


            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the optimizer and the cost, the feedict should contain a minibatch for (X,Y).
                ### START CODE HERE ### (1 line)
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})#optimizer, cost, 캐글신경망 전달, input, output batch값 전달.
                ### END CODE HERE ###
                minibatch_cost += temp_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)

        r = random.randint(0, 29)
        print("Label: ", sess.run(tf.argmax(Y_test[r:r + 1], 1)))
        print("Prediction: ", sess.run(
            tf.argmax(Z3, 1), feed_dict={X: X_test[r:r + 1]}))




    return train_accuracy, test_accuracy, parameters

if __name__ == '__main__':


    k = labeling()
    X_train = argument() #300장


    Y_train = k["Y_train"]
    X_test = k["X_test"]
    Y_test = k["Y_test"]

    _, _, parameters = model(X_train, Y_train, X_test, Y_test)



#1. validation 셋 이용
# drop out 사용시 학습이진행안됨... 언더피팅
