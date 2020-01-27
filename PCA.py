import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Visualization
import matplotlib.pyplot as plt

#Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
np.random.seed(1)
from sklearn.model_selection import KFold, cross_val_score, train_test_split,StratifiedKFold
from sklearn import linear_model
# from sklearn.kernel_ridge import KernelRidge
# from sklearn.preprocessing import RobustScalerdddd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor,VotingClassifier, AdaBoostClassifier, BaggingClassifier

from cnn_utils import *
#System
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
from numpy import expand_dims


#1번 64 사이즈로해보기  x 사이즈문제는 아니다.
#2번 트레이닝 세트에서 테스트셋 잘라서쓰기
#3번 증강안쓰기 이새키다... + feature가 많아 정확도가 줄어든다 사진크기 64로 바꾸기
#원인을 알았으니 이제... test 와 train을 합쳐서 60장을 만든다음에 나머지 test를뺀다

class_names = ["hyeon", "irene","joy","selgi","wendy"]
import warnings
warnings.filterwarnings('ignore')
print("Warnings ignored!!")

def labeling():
    image_path = "H:/workspace/cplusplus_opencv/Casecadenet/dataset/PCA64"
    files = glob.glob(path.join(image_path, '*.jpg'))
    img = []

    for i in range(60):
        img.append(files[i][63:])
        img[i] = img[i].replace(".jpg","")
        img[i] = int(img[i])


    img.sort()
    for i in range(60):
        img[i] = "H:/workspace/cplusplus_opencv/Casecadenet/dataset/PCA64\\"+"setting"+str(img[i]) +".jpg"

#
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
    Y_train_orig = np.array(Y_train_orig).reshape(1, 60)
    np.expand_dims(Y_train_orig, axis = 0) #원핫 인코딩시만

    Y_train = convert_to_one_hot(Y_train_orig,5).T
    dict = {"X_train" : X_train, "Y_train" : Y_train, "Y_train_orig" : Y_train_orig};
    print(Y_train.shape)

    return dict


    # print("There are {} images in the dataset".format(len(a["X_train"])))
    # print("There are {} unique targets in the dataset".format(len(np.unique(a["Y_train"])))) #중복성분 제외 40명 one hot x
    # print("Size of each image is {}x{}".format(a["X_train"].shape[1],a["X_train"].shape[2]))
    # print("Pixel values were scaled to [0,1] interval. e.g:{}".format(a["X_train"][1][0,:4])) #인터벌해주기







def argument():
    k = labeling()    #
    X_train = k["X_train"]
    # X_test = k["X_test"]



    print ("증가전, number of training examples = " + str(X_train.shape[0]))
    # print ("증가전, number of test examples = " + str(X_test.shape[0]))
    trains_x =[]
    for i in range(30):
        samples = expand_dims(X_train[i],0) #차원 늘려준다
        datagen = ImageDataGenerator(
                        rotation_range = 10,
                        rescale = 1./255,
                        horizontal_flip=True,
                        # zoom_range =[0.7,1.0], #1.0값 기준
                        featurewise_std_normalization=True#데이터의 집합 std별로 데이터를나눔.
                                )
        #zca, pca란?

        it = datagen.flow(samples, batch_size=10) #실시간데이터 증강을 통해 모델에 fit시킨다.
        for i in range(3): #인당 3장씩,
            batch = it.next()
            # plt.subplot(330+1+i)
            trains_x.append(batch)
            # image = batch[0]#batch , convert to unsigned integers for viewing
            # plt.imshow(image)
    X_train = np.array(trains_x)
    X_train = X_train.reshape(90,64,64,3)
    print("X_train.shape : ", X_train.shape)


    #한번 더 증가 시키기.
    print(" 증가후 : number of argumentation data : ", X_train.shape)


    return X_train







def black():
    X = labeling()
    X_train = X["X_train"]
    Y_train_orig = X["Y_train_orig"]

    # X_train = argument()
    # X_train = np.asarray(X_train)
    # Y_test = X["Y_test"]

    X = []
    for i in range(60):
        X.append(cv2.cvtColor(X_train[i], cv2.COLOR_BGR2GRAY))
    X_train = np.asarray(X)
    plt.figure(figsize=(10,20))
    for i in range(0,60):
        plt.subplot(5,20,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[Y_train_orig[0][i]])
    plt.show()
    plt.imshow(X_train[6])
    plt.show()
    #We reshape images for machine learnig  model
    X_train=X_train.reshape((X_train.shape[0], X_train.shape[1]*X_train.shape[2]))
    # X_test=X_test.reshape((X_test.shape[0], X_test.shape[1]*X_test.shape[2]))
    # print("흑백 변경 후: X_train shape:",X_train.shape)
    X_train = X_train/255

    return X_train






from sklearn.decomposition import PCA
import mglearn
def main():
    print("\n\n-----------------PCA-----------------\n\n")
    X_train = black()
    x = labeling()
    Y_train = x["Y_train"]
    Y_train_orig = x["Y_train_orig"]
    pca=PCA(n_components=2)
    pca.fit(X_train)
    X_pca=pca.transform(X_train)
    number_of_people=5
    index_range=number_of_people*5
    fig=plt.figure(figsize=(10,8))
    ax=fig.add_subplot(1,1,1)
    scatter=ax.scatter(X_pca[:index_range,0], #data 위치
            X_pca[:index_range,1], #data 위치
            c=Y_train_orig[0][:index_range], #색상, 순서 또는 색상 순서 (선택 사항) 마커 색상입니다. 가능한 값 : 단일 색상 형식 문자열 길이 n의 일련의 색상 사양. cmap 및 norm을 사용하여 색상에 매핑되는 n 개의 시퀀스입니다. 행이 RGB 또는 RGBA 인 2 차원 배열입니다.
            s=5, #scalar, 어레이
           cmap=plt.get_cmap('jet', number_of_people)
           )

    ax.set_xlabel("First Principle Component")
    ax.set_ylabel("Second Principle Component")
    ax.set_title("PCA projection of {} people".format(number_of_people))

    fig.colorbar(scatter)#25개산출
    plt.show()


    pca=PCA()
    pca.fit(X_train)

    plt.figure(1, figsize=(12,8))

    plt.plot(pca.explained_variance_, linewidth=2)

    plt.xlabel('Components')
    plt.ylabel('Explained Variaces')
    plt.show()


    n_components=25
    pca=PCA(n_components=n_components, whiten=True)
    pca.fit(X_train)

    fig,ax=plt.subplots(1,1,figsize=(8,8))
    ax.imshow(pca.mean_.reshape((64,64)), cmap="gray")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Average Face')
    plt.show()

    number_of_eigenfaces=len(pca.components_)
    print(number_of_eigenfaces)
    eigen_faces=pca.components_.reshape((number_of_eigenfaces, 64, 64))

    cols=5
    rows=int(number_of_eigenfaces/cols)
    fig, axarr=plt.subplots(nrows=rows, ncols=cols, figsize=(15,15))
    axarr=axarr.flatten()
    for i in range(number_of_eigenfaces):
        axarr[i].imshow(eigen_faces[i],cmap="gray")
        axarr[i].set_xticks([])
        axarr[i].set_yticks([])
        axarr[i].set_title("eigen id:{}".format(i))
    plt.suptitle("All Eigen Faces".format(10*"=", 10*"="))
    plt.show()

    X_train, X_test, Y_train, Y_test=train_test_split(X_train, Y_train_orig[0], test_size=0.3, stratify=Y_train, random_state=0)

    X_train_pca=pca.transform(X_train)
    X_test_pca=pca.transform(X_test)

    clf = SVC()
    clf.fit(X_train_pca, Y_train)

    y_pred = clf.predict(X_test_pca)
    print("accuracy score:{:.2f}".format(metrics.accuracy_score(Y_test, y_pred)))
    import seaborn as sns
    plt.figure(1, figsize=(12,8))
    sns.heatmap(metrics.confusion_matrix(Y_test, y_pred))
    plt.show()

    print(metrics.classification_report(Y_test, y_pred))

    models=[]
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(("LR",LogisticRegression()))
    models.append(("NB",GaussianNB()))
    models.append(("KNN",KNeighborsClassifier(n_neighbors=5)))
    models.append(("DT",DecisionTreeClassifier()))
    models.append(("SVM",SVC()))
    models.append((("Percept"),linear_model.Perceptron(penalty='l2')))
    models.append(("RandomFo", RandomForestClassifier(n_estimators=13)))


    for name, model in models:

        clf=model

        clf.fit(X_train_pca, Y_train)

        y_pred=clf.predict(X_test_pca)
        print(10*"=","{} Result".format(name).upper(),10*"=")
        print("Accuracy score:{:0.2f}".format(metrics.accuracy_score(Y_test, y_pred)))
        print()

    # labeling()
    #
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold

    pca=PCA(n_components=40, whiten=True)
    pca.fit(X_train)
    X_pca=pca.transform(X_train)

    for name, model in models:
        kfold=KFold(n_splits=5, shuffle=True, random_state=0)
        cv_scores=cross_val_score(model, X_pca, Y_train, cv=kfold)
        print("{} mean cross validations score:{:.2f}".format(name, cv_scores.mean()))


if __name__ == '__main__':
    main()
